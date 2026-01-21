package cache

import (
	"testing"
)

func TestCacheKey_String(t *testing.T) {
	key := CacheKey{
		ModelID:   "llama-7b",
		LayerID:   5,
		TokenHash: [32]byte{0x01, 0x02, 0x03},
		SeqStart:  0,
		SeqEnd:    256,
	}

	str := key.String()
	if str == "" {
		t.Error("CacheKey.String() returned empty string")
	}

	// Should contain model ID
	if len(str) < len("llama-7b") {
		t.Error("String representation too short")
	}
}

func TestTokenHasher_HashPrefix(t *testing.T) {
	hasher := NewTokenHasher(256)

	tokens1 := []int32{1, 2, 3, 4, 5}
	tokens2 := []int32{1, 2, 3, 4, 5}
	tokens3 := []int32{1, 2, 3, 4, 6} // Different last token

	hash1 := hasher.HashPrefix(tokens1)
	hash2 := hasher.HashPrefix(tokens2)
	hash3 := hasher.HashPrefix(tokens3)

	// Same tokens should produce same hash
	if hash1 != hash2 {
		t.Error("Same tokens produced different hashes")
	}

	// Different tokens should produce different hash
	if hash1 == hash3 {
		t.Error("Different tokens produced same hash")
	}
}

func TestTokenHasher_GenerateKeys(t *testing.T) {
	hasher := NewTokenHasher(256)

	// Use exactly 256 tokens for 1 chunk
	tokens := make([]int32, 256)
	for i := range tokens {
		tokens[i] = int32(i)
	}

	keys := hasher.GenerateKeys("llama-7b", 32, tokens)

	// With 1 chunk and 32 layers: 1 * 32 = 32 keys
	if len(keys) != 32 {
		t.Errorf("Expected 32 keys, got %d", len(keys))
	}

	// All keys should have same token hash (same chunk)
	firstHash := keys[0].TokenHash
	for i, key := range keys {
		if key.ModelID != "llama-7b" {
			t.Errorf("Key %d has wrong ModelID: %s", i, key.ModelID)
		}
		if key.LayerID != i {
			t.Errorf("Key %d has wrong LayerID: %d", i, key.LayerID)
		}
		if key.TokenHash != firstHash {
			t.Errorf("Key %d has different TokenHash", i)
		}
	}
}

func TestTokenHasher_GenerateKeys_MultipleChunks(t *testing.T) {
	hasher := NewTokenHasher(256)

	// Use 512 tokens for 2 chunks
	tokens := make([]int32, 512)
	for i := range tokens {
		tokens[i] = int32(i)
	}

	keys := hasher.GenerateKeys("llama-7b", 32, tokens)

	// With 2 chunks and 32 layers: 2 * 32 = 64 keys
	if len(keys) != 64 {
		t.Errorf("Expected 64 keys (2 chunks * 32 layers), got %d", len(keys))
	}

	// First 32 keys are chunk 0, next 32 are chunk 1
	chunk0Hash := keys[0].TokenHash
	chunk1Hash := keys[32].TokenHash

	if chunk0Hash == chunk1Hash {
		t.Error("Different chunks should have different hashes")
	}
}

func TestCacheEntry_RefCounting(t *testing.T) {
	key := CacheKey{ModelID: "test", LayerID: 0}
	entry := NewCacheEntry(key, make([]byte, 100), make([]byte, 100))

	// NewCacheEntry starts with refCount=1 (one reference from creator)
	if entry.RefCount() != 1 {
		t.Errorf("New entry should have 1 ref, got %d", entry.RefCount())
	}

	entry.Ref()
	if entry.RefCount() != 2 {
		t.Errorf("After Ref(), should have 2 refs, got %d", entry.RefCount())
	}

	entry.Ref()
	if entry.RefCount() != 3 {
		t.Errorf("After second Ref(), should have 3 refs, got %d", entry.RefCount())
	}

	entry.Unref()
	if entry.RefCount() != 2 {
		t.Errorf("After Unref(), should have 2 refs, got %d", entry.RefCount())
	}

	entry.Unref()
	if entry.RefCount() != 1 {
		t.Errorf("After second Unref(), should have 1 ref, got %d", entry.RefCount())
	}

	// Final unref to release creator's reference
	entry.Unref()
	if entry.RefCount() != 0 {
		t.Errorf("After third Unref(), should have 0 refs, got %d", entry.RefCount())
	}
}

func TestCacheEntry_Pinning(t *testing.T) {
	key := CacheKey{ModelID: "test", LayerID: 0}
	entry := NewCacheEntry(key, make([]byte, 100), make([]byte, 100))

	if entry.IsPinned() {
		t.Error("New entry should not be pinned")
	}

	entry.Pin()
	if !entry.IsPinned() {
		t.Error("Entry should be pinned after Pin()")
	}

	entry.Unpin()
	if entry.IsPinned() {
		t.Error("Entry should not be pinned after Unpin()")
	}
}

func TestCacheEntry_Size(t *testing.T) {
	key := CacheKey{ModelID: "test", LayerID: 0}
	k := make([]byte, 1000)
	v := make([]byte, 2000)
	entry := NewCacheEntry(key, k, v)

	size := entry.Size()
	if size < 3000 {
		t.Errorf("Entry size should be at least 3000, got %d", size)
	}
}
