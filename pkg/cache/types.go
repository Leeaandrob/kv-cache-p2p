// Package cache provides KV cache management for distributed LLM inference.
package cache

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"time"
)

// CacheKey uniquely identifies a KV cache block.
type CacheKey struct {
	ModelID   string   // Model identifier (e.g., "llama-7b")
	LayerID   int      // Layer number (0 to num_layers-1)
	TokenHash [32]byte // SHA256 of token prefix
	SeqStart  int      // Start position in sequence
	SeqEnd    int      // End position in sequence
}

// String returns a human-readable key representation.
func (k CacheKey) String() string {
	return fmt.Sprintf("%s:L%02d:%x:%d-%d",
		k.ModelID, k.LayerID, k.TokenHash[:8], k.SeqStart, k.SeqEnd)
}

// Bytes returns a binary representation for hashing/comparison.
func (k CacheKey) Bytes() []byte {
	buf := make([]byte, len(k.ModelID)+4+32+8)
	copy(buf, k.ModelID)
	offset := len(k.ModelID)
	binary.BigEndian.PutUint32(buf[offset:], uint32(k.LayerID))
	offset += 4
	copy(buf[offset:], k.TokenHash[:])
	offset += 32
	binary.BigEndian.PutUint32(buf[offset:], uint32(k.SeqStart))
	binary.BigEndian.PutUint32(buf[offset+4:], uint32(k.SeqEnd))
	return buf
}

// CacheEntry stores K and V tensors for a token range.
type CacheEntry struct {
	Key        CacheKey
	K          []byte    // [seq_len, num_kv_heads, head_dim] in FP16
	V          []byte    // [seq_len, num_kv_heads, head_dim] in FP16
	CreatedAt  time.Time // When entry was created
	AccessedAt time.Time // Last access time
	refCount   int32     // Atomic reference count
	pinned     bool      // Prevent eviction
}

// NewCacheEntry creates a new cache entry.
func NewCacheEntry(key CacheKey, k, v []byte) *CacheEntry {
	now := time.Now()
	return &CacheEntry{
		Key:        key,
		K:          k,
		V:          v,
		CreatedAt:  now,
		AccessedAt: now,
		refCount:   1,
	}
}

// Size returns the total size in bytes.
func (e *CacheEntry) Size() int {
	return len(e.K) + len(e.V)
}

// Ref increments the reference count.
func (e *CacheEntry) Ref() {
	atomic.AddInt32(&e.refCount, 1)
}

// Unref decrements the reference count.
func (e *CacheEntry) Unref() int32 {
	return atomic.AddInt32(&e.refCount, -1)
}

// RefCount returns current reference count.
func (e *CacheEntry) RefCount() int32 {
	return atomic.LoadInt32(&e.refCount)
}

// Pin marks entry as non-evictable.
func (e *CacheEntry) Pin() {
	e.pinned = true
}

// Unpin allows eviction.
func (e *CacheEntry) Unpin() {
	e.pinned = false
}

// IsPinned returns true if entry cannot be evicted.
func (e *CacheEntry) IsPinned() bool {
	return e.pinned
}

// Touch updates the access time.
func (e *CacheEntry) Touch() {
	e.AccessedAt = time.Now()
}

// StorageStats contains backend statistics.
type StorageStats struct {
	Entries    int64 // Number of entries
	SizeBytes  int64 // Total size in bytes
	HitCount   int64 // Cache hits
	MissCount  int64 // Cache misses
	EvictCount int64 // Evictions
}

// Storage is the interface for cache backends.
type Storage interface {
	// Contains checks if key exists.
	Contains(ctx context.Context, key CacheKey) bool

	// Get retrieves an entry (blocks until available or timeout).
	Get(ctx context.Context, key CacheKey) (*CacheEntry, error)

	// Put stores an entry.
	Put(ctx context.Context, entry *CacheEntry) error

	// Delete removes an entry.
	Delete(ctx context.Context, key CacheKey) error

	// List returns all keys.
	List(ctx context.Context) ([]CacheKey, error)

	// Stats returns backend statistics.
	Stats() StorageStats

	// Close releases resources.
	Close() error
}

// TokenChunk represents a chunk of tokens with its hash.
type TokenChunk struct {
	Start int      // Start position in token sequence
	End   int      // End position in token sequence
	Hash  [32]byte // SHA256 of prefix up to End
}

// TokenHasher generates hashes for token prefixes.
type TokenHasher struct {
	ChunkSize int // Chunk size (default: 256)
}

// NewTokenHasher creates a new hasher with default chunk size.
func NewTokenHasher(chunkSize int) *TokenHasher {
	if chunkSize <= 0 {
		chunkSize = 256
	}
	return &TokenHasher{ChunkSize: chunkSize}
}

// HashPrefix generates SHA256 for a token prefix.
func (h *TokenHasher) HashPrefix(tokens []int32) [32]byte {
	buf := make([]byte, len(tokens)*4)
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(t))
	}
	return sha256.Sum256(buf)
}

// ChunkTokens divides tokens into chunks and returns hashes.
func (h *TokenHasher) ChunkTokens(tokens []int32) []TokenChunk {
	var chunks []TokenChunk
	for i := 0; i < len(tokens); i += h.ChunkSize {
		end := i + h.ChunkSize
		if end > len(tokens) {
			end = len(tokens)
		}
		prefix := tokens[:end]
		chunks = append(chunks, TokenChunk{
			Start: i,
			End:   end,
			Hash:  h.HashPrefix(prefix),
		})
	}
	return chunks
}

// GenerateKeys generates cache keys for all layers and chunks.
func (h *TokenHasher) GenerateKeys(modelID string, numLayers int, tokens []int32) []CacheKey {
	chunks := h.ChunkTokens(tokens)
	keys := make([]CacheKey, 0, len(chunks)*numLayers)

	for _, chunk := range chunks {
		for layer := 0; layer < numLayers; layer++ {
			keys = append(keys, CacheKey{
				ModelID:   modelID,
				LayerID:   layer,
				TokenHash: chunk.Hash,
				SeqStart:  chunk.Start,
				SeqEnd:    chunk.End,
			})
		}
	}
	return keys
}
