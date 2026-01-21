// Package storage provides cache storage backends.
package storage

import (
	"container/list"
	"context"
	"errors"
	"sync"
	"sync/atomic"

	"github.com/neurogrid/kv-cache-p2p/pkg/cache"
)

var (
	ErrNotFound     = errors.New("cache entry not found")
	ErrStorageFull  = errors.New("storage full, eviction needed")
	ErrEntryClosed  = errors.New("storage closed")
	ErrEntryPinned  = errors.New("cannot evict pinned entry")
	ErrEntryTooLarge = errors.New("entry too large for storage")
)

// LocalStorage implements in-memory cache with LRU eviction.
type LocalStorage struct {
	maxSize   int64 // Maximum size in bytes
	entries   map[string]*list.Element
	lruList   *list.List
	mu        sync.RWMutex
	closed    bool

	// Stats
	currentSize int64
	hitCount    atomic.Int64
	missCount   atomic.Int64
	evictCount  atomic.Int64
}

// lruEntry wraps cache entry with key for LRU list.
type lruEntry struct {
	key   string
	entry *cache.CacheEntry
}

// NewLocalStorage creates a new local storage with given max size.
func NewLocalStorage(maxSizeBytes int64) *LocalStorage {
	return &LocalStorage{
		maxSize: maxSizeBytes,
		entries: make(map[string]*list.Element),
		lruList: list.New(),
	}
}

// Contains checks if key exists.
func (s *LocalStorage) Contains(ctx context.Context, key cache.CacheKey) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	_, exists := s.entries[key.String()]
	return exists
}

// Get retrieves an entry.
func (s *LocalStorage) Get(ctx context.Context, key cache.CacheKey) (*cache.CacheEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, ErrEntryClosed
	}

	keyStr := key.String()
	elem, exists := s.entries[keyStr]
	if !exists {
		s.missCount.Add(1)
		return nil, ErrNotFound
	}

	// Move to front (most recently used)
	s.lruList.MoveToFront(elem)

	entry := elem.Value.(*lruEntry).entry
	entry.Touch()
	entry.Ref()

	s.hitCount.Add(1)
	return entry, nil
}

// Put stores an entry.
func (s *LocalStorage) Put(ctx context.Context, entry *cache.CacheEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrEntryClosed
	}

	keyStr := entry.Key.String()
	entrySize := int64(entry.Size())

	// Check if entry is too large
	if entrySize > s.maxSize {
		return ErrEntryTooLarge
	}

	// If key exists, update it
	if elem, exists := s.entries[keyStr]; exists {
		oldEntry := elem.Value.(*lruEntry).entry
		s.currentSize -= int64(oldEntry.Size())
		s.currentSize += entrySize
		elem.Value.(*lruEntry).entry = entry
		s.lruList.MoveToFront(elem)
		return nil
	}

	// Evict if needed
	for s.currentSize+entrySize > s.maxSize {
		if err := s.evictOne(); err != nil {
			return err
		}
	}

	// Add new entry
	le := &lruEntry{key: keyStr, entry: entry}
	elem := s.lruList.PushFront(le)
	s.entries[keyStr] = elem
	s.currentSize += entrySize

	return nil
}

// evictOne removes the least recently used entry (must hold lock).
func (s *LocalStorage) evictOne() error {
	// Start from back (least recently used)
	for elem := s.lruList.Back(); elem != nil; elem = elem.Prev() {
		le := elem.Value.(*lruEntry)
		entry := le.entry

		// Skip pinned entries
		if entry.IsPinned() {
			continue
		}

		// Skip entries with active references
		if entry.RefCount() > 0 {
			continue
		}

		// Evict this entry
		s.lruList.Remove(elem)
		delete(s.entries, le.key)
		s.currentSize -= int64(entry.Size())
		s.evictCount.Add(1)
		return nil
	}

	return ErrStorageFull
}

// Delete removes an entry.
func (s *LocalStorage) Delete(ctx context.Context, key cache.CacheKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrEntryClosed
	}

	keyStr := key.String()
	elem, exists := s.entries[keyStr]
	if !exists {
		return nil // Already deleted
	}

	entry := elem.Value.(*lruEntry).entry
	if entry.IsPinned() {
		return ErrEntryPinned
	}

	s.lruList.Remove(elem)
	delete(s.entries, keyStr)
	s.currentSize -= int64(entry.Size())

	return nil
}

// List returns all keys.
func (s *LocalStorage) List(ctx context.Context) ([]cache.CacheKey, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, ErrEntryClosed
	}

	keys := make([]cache.CacheKey, 0, len(s.entries))
	for elem := s.lruList.Front(); elem != nil; elem = elem.Next() {
		le := elem.Value.(*lruEntry)
		keys = append(keys, le.entry.Key)
	}
	return keys, nil
}

// Stats returns storage statistics.
func (s *LocalStorage) Stats() cache.StorageStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return cache.StorageStats{
		Entries:    int64(len(s.entries)),
		SizeBytes:  s.currentSize,
		HitCount:   s.hitCount.Load(),
		MissCount:  s.missCount.Load(),
		EvictCount: s.evictCount.Load(),
	}
}

// Close releases resources.
func (s *LocalStorage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.closed = true
	s.entries = nil
	s.lruList = nil
	return nil
}

// CurrentSize returns current used size.
func (s *LocalStorage) CurrentSize() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentSize
}

// MaxSize returns maximum size.
func (s *LocalStorage) MaxSize() int64 {
	return s.maxSize
}

// NeedsEviction returns true if storage is near capacity.
func (s *LocalStorage) NeedsEviction() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return float64(s.currentSize) >= float64(s.maxSize)*0.9
}
