// Package cache provides KV cache management for distributed LLM inference.
package cache

import (
	"context"
	"errors"
	"log"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
)

var (
	ErrCacheMiss = errors.New("cache miss")
	ErrNoStorage = errors.New("no storage configured")
)

// P2PClient is the interface for P2P cache operations.
type P2PClient interface {
	// Lookup queries peers for cache keys. Returns map of peer -> indices of found keys.
	Lookup(ctx context.Context, keys []CacheKey) (map[peer.ID][]int, error)

	// FetchFromPeer retrieves entries from a specific peer.
	FetchFromPeer(ctx context.Context, pid peer.ID, keys []CacheKey) ([]*CacheEntry, error)
}

// ManagerConfig holds cache manager configuration.
type ManagerConfig struct {
	// LocalMaxSize is the maximum local cache size in bytes.
	LocalMaxSize int64

	// ChunkSize is the token chunk size for hashing.
	ChunkSize int

	// EnableP2P enables peer-to-peer cache sharing.
	EnableP2P bool

	// P2PLookupTimeout is the timeout for P2P lookups.
	P2PLookupTimeout time.Duration

	// P2PFetchTimeout is the timeout for P2P data transfers.
	P2PFetchTimeout time.Duration

	// PrefetchEnabled enables background prefetching.
	PrefetchEnabled bool
}

// DefaultConfig returns default configuration.
func DefaultConfig() ManagerConfig {
	return ManagerConfig{
		LocalMaxSize:     5 * 1024 * 1024 * 1024, // 5GB
		ChunkSize:        256,
		EnableP2P:        true,
		P2PLookupTimeout: 5 * time.Second,
		P2PFetchTimeout:  30 * time.Second,
		PrefetchEnabled:  true,
	}
}

// Manager coordinates local and P2P cache.
type Manager struct {
	config       ManagerConfig
	localStorage Storage
	p2pClient    P2PClient
	hasher       *TokenHasher

	// Stats
	localHits   atomic.Int64
	p2pHits     atomic.Int64
	misses      atomic.Int64
	prefetches  atomic.Int64
}

// NewManager creates a new cache manager.
func NewManager(config ManagerConfig, localStorage Storage, p2pClient P2PClient) *Manager {
	return &Manager{
		config:       config,
		localStorage: localStorage,
		p2pClient:    p2pClient,
		hasher:       NewTokenHasher(config.ChunkSize),
	}
}

// Lookup searches for a cache entry locally, then P2P.
func (m *Manager) Lookup(ctx context.Context, key CacheKey) (*CacheEntry, error) {
	// 1. Try local first
	if m.localStorage != nil {
		entry, err := m.localStorage.Get(ctx, key)
		if err == nil {
			m.localHits.Add(1)
			return entry, nil
		}
	}

	// 2. Try P2P if enabled
	if m.config.EnableP2P && m.p2pClient != nil {
		entry, err := m.lookupP2P(ctx, key)
		if err == nil {
			m.p2pHits.Add(1)
			// Store locally for next access
			if m.localStorage != nil {
				entryCopy := *entry
				go m.localStorage.Put(ctx, &entryCopy)
			}
			return entry, nil
		}
	}

	m.misses.Add(1)
	return nil, ErrCacheMiss
}

// lookupP2P searches for key in P2P network.
func (m *Manager) lookupP2P(ctx context.Context, key CacheKey) (*CacheEntry, error) {
	ctx, cancel := context.WithTimeout(ctx, m.config.P2PLookupTimeout)
	defer cancel()

	// Find peers that have this key
	results, err := m.p2pClient.Lookup(ctx, []CacheKey{key})
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, ErrCacheMiss
	}

	// Fetch from first peer that has it
	for pid := range results {
		fetchCtx, fetchCancel := context.WithTimeout(ctx, m.config.P2PFetchTimeout)
		entries, err := m.p2pClient.FetchFromPeer(fetchCtx, pid, []CacheKey{key})
		fetchCancel()

		if err != nil {
			log.Printf("Failed to fetch from peer %s: %v", pid, err)
			continue
		}

		if len(entries) > 0 && entries[0] != nil {
			return entries[0], nil
		}
	}

	return nil, ErrCacheMiss
}

// LookupBatch searches for multiple keys.
func (m *Manager) LookupBatch(ctx context.Context, keys []CacheKey) ([]*CacheEntry, []int, error) {
	entries := make([]*CacheEntry, len(keys))
	var missedIdxs []int

	// Try local first
	for i, key := range keys {
		if m.localStorage != nil {
			entry, err := m.localStorage.Get(ctx, key)
			if err == nil {
				entries[i] = entry
				m.localHits.Add(1)
				continue
			}
		}
		missedIdxs = append(missedIdxs, i)
	}

	// If no misses or P2P disabled, return early
	if len(missedIdxs) == 0 || !m.config.EnableP2P || m.p2pClient == nil {
		for range missedIdxs {
			m.misses.Add(1)
		}
		return entries, missedIdxs, nil
	}

	// Build list of missed keys
	missedKeys := make([]CacheKey, len(missedIdxs))
	for i, idx := range missedIdxs {
		missedKeys[i] = keys[idx]
	}

	// Query P2P for missed keys
	ctx, cancel := context.WithTimeout(ctx, m.config.P2PLookupTimeout)
	defer cancel()

	results, err := m.p2pClient.Lookup(ctx, missedKeys)
	if err != nil {
		for range missedIdxs {
			m.misses.Add(1)
		}
		return entries, missedIdxs, nil
	}

	// Fetch from peers
	for pid, foundIdxs := range results {
		// Build keys to fetch
		keysToFetch := make([]CacheKey, len(foundIdxs))
		originalIdxs := make([]int, len(foundIdxs))
		for i, idx := range foundIdxs {
			keysToFetch[i] = missedKeys[idx]
			originalIdxs[i] = missedIdxs[idx]
		}

		fetchCtx, fetchCancel := context.WithTimeout(ctx, m.config.P2PFetchTimeout)
		fetched, err := m.p2pClient.FetchFromPeer(fetchCtx, pid, keysToFetch)
		fetchCancel()

		if err != nil {
			continue
		}

		// Store fetched entries
		for i, entry := range fetched {
			if entry != nil {
				entries[originalIdxs[i]] = entry
				m.p2pHits.Add(1)

				// Remove from missed list
				for j, midx := range missedIdxs {
					if midx == originalIdxs[i] {
						missedIdxs = append(missedIdxs[:j], missedIdxs[j+1:]...)
						break
					}
				}

				// Store locally
				if m.localStorage != nil {
					entryCopy := *entry
					go m.localStorage.Put(ctx, &entryCopy)
				}
			}
		}
	}

	for range missedIdxs {
		m.misses.Add(1)
	}

	return entries, missedIdxs, nil
}

// Store saves an entry to local cache.
func (m *Manager) Store(ctx context.Context, entry *CacheEntry) error {
	if m.localStorage == nil {
		return ErrNoStorage
	}
	return m.localStorage.Put(ctx, entry)
}

// StoreBatch saves multiple entries.
func (m *Manager) StoreBatch(ctx context.Context, entries []*CacheEntry) error {
	if m.localStorage == nil {
		return ErrNoStorage
	}

	for _, entry := range entries {
		if err := m.localStorage.Put(ctx, entry); err != nil {
			return err
		}
	}
	return nil
}

// Prefetch fetches keys in background for future use.
func (m *Manager) Prefetch(ctx context.Context, keys []CacheKey) {
	if !m.config.PrefetchEnabled {
		return
	}

	go func() {
		// Filter keys not in local cache
		var toFetch []CacheKey
		for _, key := range keys {
			if m.localStorage != nil && !m.localStorage.Contains(ctx, key) {
				toFetch = append(toFetch, key)
			}
		}

		if len(toFetch) == 0 {
			return
		}

		m.prefetches.Add(int64(len(toFetch)))

		// Lookup in P2P
		if m.config.EnableP2P && m.p2pClient != nil {
			results, err := m.p2pClient.Lookup(ctx, toFetch)
			if err != nil {
				return
			}

			// Fetch from peers
			for pid, foundIdxs := range results {
				keysToFetch := make([]CacheKey, len(foundIdxs))
				for i, idx := range foundIdxs {
					keysToFetch[i] = toFetch[idx]
				}

				entries, err := m.p2pClient.FetchFromPeer(ctx, pid, keysToFetch)
				if err != nil {
					continue
				}

				// Store locally
				for _, entry := range entries {
					if entry != nil && m.localStorage != nil {
						m.localStorage.Put(ctx, entry)
					}
				}
			}
		}
	}()
}

// GenerateKeysForRequest generates cache keys for a model request.
func (m *Manager) GenerateKeysForRequest(modelID string, numLayers int, tokens []int32) []CacheKey {
	return m.hasher.GenerateKeys(modelID, numLayers, tokens)
}

// Stats returns cache statistics.
type ManagerStats struct {
	LocalHits  int64
	P2PHits    int64
	Misses     int64
	Prefetches int64
	HitRate    float64
	LocalStats StorageStats
}

// Stats returns manager statistics.
func (m *Manager) Stats() ManagerStats {
	localHits := m.localHits.Load()
	p2pHits := m.p2pHits.Load()
	misses := m.misses.Load()
	total := localHits + p2pHits + misses

	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(localHits+p2pHits) / float64(total)
	}

	var localStats StorageStats
	if m.localStorage != nil {
		localStats = m.localStorage.Stats()
	}

	return ManagerStats{
		LocalHits:  localHits,
		P2PHits:    p2pHits,
		Misses:     misses,
		Prefetches: m.prefetches.Load(),
		HitRate:    hitRate,
		LocalStats: localStats,
	}
}

// Invalidate removes entries matching a prefix.
func (m *Manager) Invalidate(ctx context.Context, modelID string, tokenPrefix []int32) error {
	if m.localStorage == nil {
		return nil
	}

	prefixHash := m.hasher.HashPrefix(tokenPrefix)

	keys, err := m.localStorage.List(ctx)
	if err != nil {
		return err
	}

	for _, key := range keys {
		if key.ModelID == modelID && key.TokenHash == prefixHash {
			m.localStorage.Delete(ctx, key)
		}
	}

	return nil
}

// Close releases resources.
func (m *Manager) Close() error {
	if m.localStorage != nil {
		return m.localStorage.Close()
	}
	return nil
}
