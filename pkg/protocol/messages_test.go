package protocol

import (
	"bytes"
	"testing"
	"time"
)

func TestHeader_Serialization(t *testing.T) {
	header := Header{
		Type:      MsgLookup,
		RequestID: 12345,
		Timestamp: time.Now().UnixNano(),
	}

	buf := SerializeHeader(header)
	if len(buf) != HeaderSize {
		t.Errorf("Expected buffer size %d, got %d", HeaderSize, len(buf))
	}

	decoded, err := DeserializeHeader(buf)
	if err != nil {
		t.Fatalf("DeserializeHeader failed: %v", err)
	}

	if decoded.Type != header.Type {
		t.Errorf("Type mismatch: %d vs %d", decoded.Type, header.Type)
	}
	if decoded.RequestID != header.RequestID {
		t.Errorf("RequestID mismatch: %d vs %d", decoded.RequestID, header.RequestID)
	}
	if decoded.Timestamp != header.Timestamp {
		t.Errorf("Timestamp mismatch: %d vs %d", decoded.Timestamp, header.Timestamp)
	}
}

func TestHeader_DeserializeTooSmall(t *testing.T) {
	buf := make([]byte, HeaderSize-1)
	_, err := DeserializeHeader(buf)
	if err == nil {
		t.Error("Expected error for too small buffer")
	}
}

func TestWriteReadMessage_LookupRequest(t *testing.T) {
	var buf bytes.Buffer

	req := LookupRequest{
		Keys: []CacheKeyWire{
			{
				ModelID:   "llama-7b",
				LayerID:   5,
				TokenHash: make([]byte, 32),
				SeqStart:  0,
				SeqEnd:    256,
			},
		},
	}

	reqID := uint64(42)
	if err := WriteMessage(&buf, MsgLookup, reqID, req); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	header, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	if header.Type != MsgLookup {
		t.Errorf("Expected MsgLookup, got %d", header.Type)
	}
	if header.RequestID != reqID {
		t.Errorf("Expected RequestID %d, got %d", reqID, header.RequestID)
	}

	var decoded LookupRequest
	if err := DecodePayload(payload, &decoded); err != nil {
		t.Fatalf("DecodePayload failed: %v", err)
	}

	if len(decoded.Keys) != 1 {
		t.Errorf("Expected 1 key, got %d", len(decoded.Keys))
	}
	if decoded.Keys[0].ModelID != "llama-7b" {
		t.Errorf("ModelID mismatch: %s", decoded.Keys[0].ModelID)
	}
}

func TestWriteReadMessage_LookupResponse(t *testing.T) {
	var buf bytes.Buffer

	resp := LookupResponse{
		Found: []bool{true, false, true},
		Sizes: []int64{1000, 0, 2000},
	}

	if err := WriteMessage(&buf, MsgLookupAck, 1, resp); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	_, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	var decoded LookupResponse
	if err := DecodePayload(payload, &decoded); err != nil {
		t.Fatalf("DecodePayload failed: %v", err)
	}

	if len(decoded.Found) != 3 {
		t.Errorf("Expected 3 found entries, got %d", len(decoded.Found))
	}
	if !decoded.Found[0] || decoded.Found[1] || !decoded.Found[2] {
		t.Error("Found values mismatch")
	}
}

func TestWriteReadMessage_GetRequest(t *testing.T) {
	var buf bytes.Buffer

	req := GetRequest{
		Keys: []CacheKeyWire{
			{ModelID: "model1", LayerID: 0},
			{ModelID: "model1", LayerID: 1},
		},
	}

	if err := WriteMessage(&buf, MsgGet, 1, req); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	header, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	if header.Type != MsgGet {
		t.Errorf("Expected MsgGet, got %d", header.Type)
	}

	var decoded GetRequest
	if err := DecodePayload(payload, &decoded); err != nil {
		t.Fatalf("DecodePayload failed: %v", err)
	}

	if len(decoded.Keys) != 2 {
		t.Errorf("Expected 2 keys, got %d", len(decoded.Keys))
	}
}

func TestWriteReadMessage_GetResponse(t *testing.T) {
	var buf bytes.Buffer

	resp := GetResponse{
		Entries: []CacheEntryWire{
			{
				Key:       CacheKeyWire{ModelID: "model", LayerID: 0},
				K:         []byte("key-data"),
				V:         []byte("value-data"),
				CreatedAt: time.Now().UnixNano(),
			},
		},
		Errors: []string{"", "not found"},
	}

	if err := WriteMessage(&buf, MsgGetAck, 1, resp); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	_, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	var decoded GetResponse
	if err := DecodePayload(payload, &decoded); err != nil {
		t.Fatalf("DecodePayload failed: %v", err)
	}

	if len(decoded.Entries) != 1 {
		t.Errorf("Expected 1 entry, got %d", len(decoded.Entries))
	}
	if string(decoded.Entries[0].K) != "key-data" {
		t.Error("K data mismatch")
	}
}

func TestWriteReadMessage_PutRequest(t *testing.T) {
	var buf bytes.Buffer

	req := PutRequest{
		Entries: []CacheEntryWire{
			{
				Key: CacheKeyWire{ModelID: "model", LayerID: 0},
				K:   make([]byte, 1000),
				V:   make([]byte, 1000),
			},
		},
	}

	if err := WriteMessage(&buf, MsgPut, 1, req); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	header, _, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	if header.Type != MsgPut {
		t.Errorf("Expected MsgPut, got %d", header.Type)
	}
}

func TestWriteReadMessage_PingPong(t *testing.T) {
	var buf bytes.Buffer

	ping := PingRequest{
		SentAt: time.Now().UnixNano(),
	}

	if err := WriteMessage(&buf, MsgPing, 1, ping); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	header, payload, err := ReadMessage(&buf)
	if err != nil {
		t.Fatalf("ReadMessage failed: %v", err)
	}

	if header.Type != MsgPing {
		t.Errorf("Expected MsgPing, got %d", header.Type)
	}

	var decoded PingRequest
	if err := DecodePayload(payload, &decoded); err != nil {
		t.Fatalf("DecodePayload failed: %v", err)
	}

	if decoded.SentAt != ping.SentAt {
		t.Error("SentAt mismatch")
	}
}

func TestCacheKeyWire(t *testing.T) {
	key := CacheKeyWire{
		ModelID:   "llama-70b",
		LayerID:   31,
		TokenHash: make([]byte, 32),
		SeqStart:  100,
		SeqEnd:    356,
	}

	// Fill token hash with test data
	for i := range key.TokenHash {
		key.TokenHash[i] = byte(i)
	}

	var buf bytes.Buffer
	req := LookupRequest{Keys: []CacheKeyWire{key}}

	if err := WriteMessage(&buf, MsgLookup, 1, req); err != nil {
		t.Fatalf("WriteMessage failed: %v", err)
	}

	_, payload, _ := ReadMessage(&buf)

	var decoded LookupRequest
	DecodePayload(payload, &decoded)

	decodedKey := decoded.Keys[0]
	if decodedKey.ModelID != key.ModelID {
		t.Errorf("ModelID: got %s, want %s", decodedKey.ModelID, key.ModelID)
	}
	if decodedKey.LayerID != key.LayerID {
		t.Errorf("LayerID: got %d, want %d", decodedKey.LayerID, key.LayerID)
	}
	if decodedKey.SeqStart != key.SeqStart {
		t.Errorf("SeqStart: got %d, want %d", decodedKey.SeqStart, key.SeqStart)
	}
	if decodedKey.SeqEnd != key.SeqEnd {
		t.Errorf("SeqEnd: got %d, want %d", decodedKey.SeqEnd, key.SeqEnd)
	}
	if !bytes.Equal(decodedKey.TokenHash, key.TokenHash) {
		t.Error("TokenHash mismatch")
	}
}

func TestAllMessageTypes(t *testing.T) {
	types := []MessageType{
		MsgLookup,
		MsgLookupAck,
		MsgGet,
		MsgGetAck,
		MsgPut,
		MsgPutAck,
		MsgEvict,
		MsgPing,
		MsgPong,
	}

	for _, mt := range types {
		if mt < MsgLookup || mt > MsgPong {
			t.Errorf("Message type %d out of expected range", mt)
		}
	}
}
