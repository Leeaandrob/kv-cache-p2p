// Package protocol defines P2P message types for KV cache sharing.
package protocol

import (
	"encoding/binary"
	"errors"
	"io"
	"time"

	"github.com/vmihailenco/msgpack/v5"
)

// ProtocolID is the libp2p protocol identifier.
const ProtocolID = "/neurogrid/kvcache/1.0.0"

// MessageType defines P2P message types.
type MessageType uint8

const (
	MsgLookup    MessageType = 1 // Query if peer has cache
	MsgLookupAck MessageType = 2 // Lookup response
	MsgGet       MessageType = 3 // Request cache data
	MsgGetAck    MessageType = 4 // Response with data
	MsgPut       MessageType = 5 // Push cache to peer
	MsgPutAck    MessageType = 6 // Put confirmation
	MsgEvict     MessageType = 7 // Notify eviction
	MsgPing      MessageType = 8 // Health check
	MsgPong      MessageType = 9 // Health response
)

// Header is the common message header.
type Header struct {
	Type      MessageType
	RequestID uint64
	Timestamp int64 // Unix nano
}

// HeaderSize is the size of serialized header.
const HeaderSize = 1 + 8 + 8 // type + request_id + timestamp

// SerializeHeader writes header to buffer.
func SerializeHeader(h Header) []byte {
	buf := make([]byte, HeaderSize)
	buf[0] = byte(h.Type)
	binary.BigEndian.PutUint64(buf[1:9], h.RequestID)
	binary.BigEndian.PutUint64(buf[9:17], uint64(h.Timestamp))
	return buf
}

// DeserializeHeader reads header from buffer.
func DeserializeHeader(buf []byte) (Header, error) {
	if len(buf) < HeaderSize {
		return Header{}, errors.New("buffer too small for header")
	}
	return Header{
		Type:      MessageType(buf[0]),
		RequestID: binary.BigEndian.Uint64(buf[1:9]),
		Timestamp: int64(binary.BigEndian.Uint64(buf[9:17])),
	}, nil
}

// CacheKeyWire is the wire format for CacheKey.
type CacheKeyWire struct {
	ModelID   string `msgpack:"m"`
	LayerID   int    `msgpack:"l"`
	TokenHash []byte `msgpack:"h"` // 32 bytes
	SeqStart  int    `msgpack:"s"`
	SeqEnd    int    `msgpack:"e"`
}

// CacheEntryWire is the wire format for CacheEntry.
type CacheEntryWire struct {
	Key       CacheKeyWire `msgpack:"k"`
	K         []byte       `msgpack:"K"` // K tensor data
	V         []byte       `msgpack:"V"` // V tensor data
	CreatedAt int64        `msgpack:"c"` // Unix nano
}

// LookupRequest asks if peer has specified keys.
type LookupRequest struct {
	Keys []CacheKeyWire `msgpack:"keys"`
}

// LookupResponse indicates which keys the peer has.
type LookupResponse struct {
	Found []bool  `msgpack:"found"` // Parallel to Keys
	Sizes []int64 `msgpack:"sizes"` // Size of each found entry
}

// GetRequest requests cache data for keys.
type GetRequest struct {
	Keys []CacheKeyWire `msgpack:"keys"`
}

// GetResponse returns cache data.
type GetResponse struct {
	Entries []CacheEntryWire `msgpack:"entries"`
	Errors  []string         `msgpack:"errors"` // Empty if success
}

// PutRequest pushes cache to peer.
type PutRequest struct {
	Entries []CacheEntryWire `msgpack:"entries"`
}

// PutResponse confirms put operation.
type PutResponse struct {
	Accepted []bool   `msgpack:"accepted"` // Parallel to entries
	Errors   []string `msgpack:"errors"`
}

// EvictNotification notifies peers of eviction.
type EvictNotification struct {
	Keys []CacheKeyWire `msgpack:"keys"`
}

// PingRequest is a health check.
type PingRequest struct {
	SentAt int64 `msgpack:"sent_at"`
}

// PongResponse is the ping reply.
type PongResponse struct {
	SentAt     int64 `msgpack:"sent_at"`     // Echo back
	ReceivedAt int64 `msgpack:"received_at"` // When received
	CacheStats Stats `msgpack:"stats"`       // Current stats
}

// Stats is wire format for cache statistics.
type Stats struct {
	Entries   int64 `msgpack:"entries"`
	SizeBytes int64 `msgpack:"size"`
	HitRate   float64 `msgpack:"hit_rate"`
}

// WriteMessage writes a message to a writer.
func WriteMessage(w io.Writer, msgType MessageType, reqID uint64, payload interface{}) error {
	header := Header{
		Type:      msgType,
		RequestID: reqID,
		Timestamp: time.Now().UnixNano(),
	}

	// Write header
	if _, err := w.Write(SerializeHeader(header)); err != nil {
		return err
	}

	// Serialize payload
	data, err := msgpack.Marshal(payload)
	if err != nil {
		return err
	}

	// Write payload length (4 bytes)
	lenBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(lenBuf, uint32(len(data)))
	if _, err := w.Write(lenBuf); err != nil {
		return err
	}

	// Write payload
	_, err = w.Write(data)
	return err
}

// ReadMessage reads a message from a reader.
func ReadMessage(r io.Reader) (Header, []byte, error) {
	// Read header
	headerBuf := make([]byte, HeaderSize)
	if _, err := io.ReadFull(r, headerBuf); err != nil {
		return Header{}, nil, err
	}

	header, err := DeserializeHeader(headerBuf)
	if err != nil {
		return Header{}, nil, err
	}

	// Read payload length
	lenBuf := make([]byte, 4)
	if _, err := io.ReadFull(r, lenBuf); err != nil {
		return Header{}, nil, err
	}
	payloadLen := binary.BigEndian.Uint32(lenBuf)

	// Read payload
	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return Header{}, nil, err
	}

	return header, payload, nil
}

// DecodePayload unmarshals payload into target.
func DecodePayload(data []byte, target interface{}) error {
	return msgpack.Unmarshal(data, target)
}
