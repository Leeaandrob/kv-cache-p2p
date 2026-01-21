// Package integration provides NeuroGrid integration interfaces.
package integration

import (
	"context"
	"sync"

	"github.com/neurogrid/kv-cache-p2p/pkg/cache"
	"github.com/neurogrid/kv-cache-p2p/pkg/storage"
	"github.com/neurogrid/kv-cache-p2p/pkg/transport"
)

// KVCacheProvider is the interface that NeuroGrid uses to interact with KV cache.
// This interface can be implemented by different backends (local only, P2P, remote, etc.)
type KVCacheProvider interface {
	// LookupKV looks up KV cache for the given model, layer, and token sequence.
	// Returns nil if not found.
	LookupKV(ctx context.Context, modelID string, layerID int, tokens []int32) (k, v []byte, found bool)

	// StoreKV stores KV cache for the given model, layer, and token sequence.
	StoreKV(ctx context.Context, modelID string, layerID int, tokens []int32, k, v []byte) error

	// PrefetchKV initiates background prefetch for the next layers.
	PrefetchKV(ctx context.Context, modelID string, layerIDs []int, tokens []int32)

	// Stats returns cache statistics.
	Stats() KVCacheStats

	// Close releases resources.
	Close() error
}

// KVCacheStats provides cache performance metrics.
type KVCacheStats struct {
	LocalHits   int64
	P2PHits     int64
	Misses      int64
	HitRate     float64
	CacheSize   int64 // bytes
	EntryCount  int64
	PeerCount   int
}

// NeuroGridAdapter adapts the cache.Manager to the KVCacheProvider interface.
type NeuroGridAdapter struct {
	manager *cache.Manager
	p2pNode *transport.P2PNode
	hasher  *cache.TokenHasher
	mu      sync.RWMutex
}

// Config holds adapter configuration.
type Config struct {
	// MaxLocalCacheSize is the maximum local cache size in bytes.
	MaxLocalCacheSize int64

	// ChunkSize is the token chunk size for hashing.
	ChunkSize int

	// P2PPort is the libp2p listen port.
	P2PPort int

	// EnableMDNS enables mDNS peer discovery.
	EnableMDNS bool

	// BootstrapPeers are initial peers to connect to.
	BootstrapPeers []string

	// EnableP2P enables P2P cache sharing.
	EnableP2P bool
}

// DefaultConfig returns default adapter configuration.
func DefaultConfig() Config {
	return Config{
		MaxLocalCacheSize: 5 * 1024 * 1024 * 1024, // 5GB
		ChunkSize:         256,
		P2PPort:           9100,
		EnableMDNS:        true,
		EnableP2P:         true,
	}
}

// NewNeuroGridAdapter creates a new adapter with full P2P support.
func NewNeuroGridAdapter(ctx context.Context, cfg Config) (*NeuroGridAdapter, error) {
	// Create local storage
	localStorage := storage.NewLocalStorage(cfg.MaxLocalCacheSize)

	var p2pNode *transport.P2PNode
	var err error

	if cfg.EnableP2P {
		// Create P2P node
		p2pCfg := transport.Config{
			ListenPort:     cfg.P2PPort,
			EnableMDNS:     cfg.EnableMDNS,
			BootstrapPeers: cfg.BootstrapPeers,
		}
		p2pNode, err = transport.NewP2PNode(ctx, p2pCfg, localStorage)
		if err != nil {
			localStorage.Close()
			return nil, err
		}
	}

	// Create manager
	managerCfg := cache.ManagerConfig{
		LocalMaxSize:     cfg.MaxLocalCacheSize,
		ChunkSize:        cfg.ChunkSize,
		EnableP2P:        cfg.EnableP2P,
		PrefetchEnabled:  true,
	}
	manager := cache.NewManager(managerCfg, localStorage, p2pNode)

	return &NeuroGridAdapter{
		manager: manager,
		p2pNode: p2pNode,
		hasher:  cache.NewTokenHasher(cfg.ChunkSize),
	}, nil
}

// NewLocalOnlyAdapter creates an adapter with local cache only (no P2P).
func NewLocalOnlyAdapter(maxCacheSize int64) *NeuroGridAdapter {
	localStorage := storage.NewLocalStorage(maxCacheSize)

	managerCfg := cache.ManagerConfig{
		LocalMaxSize:    maxCacheSize,
		ChunkSize:       256,
		EnableP2P:       false,
		PrefetchEnabled: false,
	}
	manager := cache.NewManager(managerCfg, localStorage, nil)

	return &NeuroGridAdapter{
		manager: manager,
		hasher:  cache.NewTokenHasher(256),
	}
}

// LookupKV implements KVCacheProvider.
func (a *NeuroGridAdapter) LookupKV(ctx context.Context, modelID string, layerID int, tokens []int32) (k, v []byte, found bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	hash := a.hasher.HashPrefix(tokens)
	key := cache.CacheKey{
		ModelID:   modelID,
		LayerID:   layerID,
		TokenHash: hash,
		SeqStart:  0,
		SeqEnd:    len(tokens),
	}

	entry, err := a.manager.Lookup(ctx, key)
	if err != nil {
		return nil, nil, false
	}

	// Copy data to avoid reference issues
	kCopy := make([]byte, len(entry.K))
	vCopy := make([]byte, len(entry.V))
	copy(kCopy, entry.K)
	copy(vCopy, entry.V)

	entry.Unref()
	return kCopy, vCopy, true
}

// StoreKV implements KVCacheProvider.
func (a *NeuroGridAdapter) StoreKV(ctx context.Context, modelID string, layerID int, tokens []int32, k, v []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	hash := a.hasher.HashPrefix(tokens)
	key := cache.CacheKey{
		ModelID:   modelID,
		LayerID:   layerID,
		TokenHash: hash,
		SeqStart:  0,
		SeqEnd:    len(tokens),
	}

	entry := cache.NewCacheEntry(key, k, v)
	return a.manager.Store(ctx, entry)
}

// PrefetchKV implements KVCacheProvider.
func (a *NeuroGridAdapter) PrefetchKV(ctx context.Context, modelID string, layerIDs []int, tokens []int32) {
	hash := a.hasher.HashPrefix(tokens)

	keys := make([]cache.CacheKey, len(layerIDs))
	for i, layerID := range layerIDs {
		keys[i] = cache.CacheKey{
			ModelID:   modelID,
			LayerID:   layerID,
			TokenHash: hash,
			SeqStart:  0,
			SeqEnd:    len(tokens),
		}
	}

	a.manager.Prefetch(ctx, keys)
}

// Stats implements KVCacheProvider.
func (a *NeuroGridAdapter) Stats() KVCacheStats {
	stats := a.manager.Stats()

	peerCount := 0
	if a.p2pNode != nil {
		peerCount = len(a.p2pNode.Peers())
	}

	return KVCacheStats{
		LocalHits:  stats.LocalHits,
		P2PHits:    stats.P2PHits,
		Misses:     stats.Misses,
		HitRate:    stats.HitRate,
		CacheSize:  stats.LocalStats.SizeBytes,
		EntryCount: stats.LocalStats.Entries,
		PeerCount:  peerCount,
	}
}

// Close implements KVCacheProvider.
func (a *NeuroGridAdapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.p2pNode != nil {
		a.p2pNode.Close()
	}
	return a.manager.Close()
}

// P2PNode returns the underlying P2P node (for advanced usage).
func (a *NeuroGridAdapter) P2PNode() *transport.P2PNode {
	return a.p2pNode
}

// Manager returns the underlying cache manager (for advanced usage).
func (a *NeuroGridAdapter) Manager() *cache.Manager {
	return a.manager
}

// BatchLookup performs batch lookup for multiple layers.
func (a *NeuroGridAdapter) BatchLookup(ctx context.Context, modelID string, numLayers int, tokens []int32) ([]*LayerKV, []int) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	keys := a.manager.GenerateKeysForRequest(modelID, numLayers, tokens)
	entries, missedIdxs, _ := a.manager.LookupBatch(ctx, keys)

	results := make([]*LayerKV, len(entries))
	for i, entry := range entries {
		if entry != nil {
			results[i] = &LayerKV{
				LayerID: i,
				K:       entry.K,
				V:       entry.V,
			}
			entry.Unref()
		}
	}

	return results, missedIdxs
}

// LayerKV holds K/V data for a single layer.
type LayerKV struct {
	LayerID int
	K       []byte
	V       []byte
}

// BatchStore stores K/V data for multiple layers.
func (a *NeuroGridAdapter) BatchStore(ctx context.Context, modelID string, tokens []int32, layers []*LayerKV) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	hash := a.hasher.HashPrefix(tokens)

	entries := make([]*cache.CacheEntry, len(layers))
	for i, layer := range layers {
		key := cache.CacheKey{
			ModelID:   modelID,
			LayerID:   layer.LayerID,
			TokenHash: hash,
			SeqStart:  0,
			SeqEnd:    len(tokens),
		}
		entries[i] = cache.NewCacheEntry(key, layer.K, layer.V)
	}

	return a.manager.StoreBatch(ctx, entries)
}
