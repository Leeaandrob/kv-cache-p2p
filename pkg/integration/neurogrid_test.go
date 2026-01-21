package integration

import (
	"context"
	"testing"
)

func TestNewLocalOnlyAdapter(t *testing.T) {
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	if adapter == nil {
		t.Fatal("NewLocalOnlyAdapter returned nil")
	}

	stats := adapter.Stats()
	if stats.PeerCount != 0 {
		t.Error("Local-only adapter should have 0 peers")
	}
}

func TestAdapter_StoreAndLookup(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	modelID := "llama-7b"
	layerID := 5
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	k := []byte("key-tensor-data")
	v := []byte("value-tensor-data")

	// Store
	err := adapter.StoreKV(ctx, modelID, layerID, tokens, k, v)
	if err != nil {
		t.Fatalf("StoreKV failed: %v", err)
	}

	// Lookup
	kResult, vResult, found := adapter.LookupKV(ctx, modelID, layerID, tokens)
	if !found {
		t.Fatal("LookupKV should find stored entry")
	}

	if string(kResult) != string(k) {
		t.Errorf("K mismatch: got %s, want %s", kResult, k)
	}
	if string(vResult) != string(v) {
		t.Errorf("V mismatch: got %s, want %s", vResult, v)
	}
}

func TestAdapter_LookupMiss(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	_, _, found := adapter.LookupKV(ctx, "nonexistent", 0, []int32{1, 2, 3})
	if found {
		t.Error("LookupKV should return found=false for missing entry")
	}
}

func TestAdapter_DifferentTokensProduceDifferentKeys(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	modelID := "llama-7b"
	layerID := 0

	tokens1 := []int32{1, 2, 3}
	tokens2 := []int32{1, 2, 4} // Different last token

	k1 := []byte("k1")
	v1 := []byte("v1")
	k2 := []byte("k2")
	v2 := []byte("v2")

	// Store both
	adapter.StoreKV(ctx, modelID, layerID, tokens1, k1, v1)
	adapter.StoreKV(ctx, modelID, layerID, tokens2, k2, v2)

	// Lookup first
	kResult1, vResult1, found1 := adapter.LookupKV(ctx, modelID, layerID, tokens1)
	if !found1 {
		t.Fatal("tokens1 entry not found")
	}
	if string(kResult1) != string(k1) {
		t.Errorf("tokens1 K mismatch: got %s, want %s", kResult1, k1)
	}

	// Lookup second
	kResult2, vResult2, found2 := adapter.LookupKV(ctx, modelID, layerID, tokens2)
	if !found2 {
		t.Fatal("tokens2 entry not found")
	}
	if string(kResult2) != string(k2) {
		t.Errorf("tokens2 K mismatch: got %s, want %s", kResult2, k2)
	}

	_ = vResult1
	_ = vResult2
}

func TestAdapter_Stats(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	// Initial stats
	stats := adapter.Stats()
	if stats.EntryCount != 0 {
		t.Error("Initial entry count should be 0")
	}

	// Store entry
	adapter.StoreKV(ctx, "model", 0, []int32{1}, []byte("k"), []byte("v"))

	stats = adapter.Stats()
	if stats.EntryCount != 1 {
		t.Errorf("Entry count should be 1, got %d", stats.EntryCount)
	}

	// Hit
	adapter.LookupKV(ctx, "model", 0, []int32{1})
	stats = adapter.Stats()
	if stats.LocalHits != 1 {
		t.Errorf("Local hits should be 1, got %d", stats.LocalHits)
	}

	// Miss
	adapter.LookupKV(ctx, "model", 99, []int32{1})
	stats = adapter.Stats()
	if stats.Misses != 1 {
		t.Errorf("Misses should be 1, got %d", stats.Misses)
	}

	// Hit rate
	if stats.HitRate < 0.4 || stats.HitRate > 0.6 {
		t.Errorf("Hit rate should be ~0.5, got %f", stats.HitRate)
	}
}

func TestAdapter_BatchLookup(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	modelID := "llama-7b"
	tokens := []int32{1, 2, 3, 4, 5}

	// Store some layers
	adapter.StoreKV(ctx, modelID, 0, tokens, []byte("k0"), []byte("v0"))
	adapter.StoreKV(ctx, modelID, 2, tokens, []byte("k2"), []byte("v2"))

	// Batch lookup for 4 layers
	results, missed := adapter.BatchLookup(ctx, modelID, 4, tokens)

	if len(results) != 4 {
		t.Errorf("Expected 4 results, got %d", len(results))
	}

	// Layer 0 and 2 should be found
	if results[0] == nil {
		t.Error("Layer 0 should be found")
	}
	if results[2] == nil {
		t.Error("Layer 2 should be found")
	}

	// Layer 1 and 3 should be missed
	if results[1] != nil {
		t.Error("Layer 1 should not be found")
	}
	if results[3] != nil {
		t.Error("Layer 3 should not be found")
	}

	// Check missed indices
	if len(missed) != 2 {
		t.Errorf("Expected 2 missed, got %d", len(missed))
	}
}

func TestAdapter_BatchStore(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	modelID := "llama-7b"
	tokens := []int32{1, 2, 3, 4, 5}

	layers := []*LayerKV{
		{LayerID: 0, K: []byte("k0"), V: []byte("v0")},
		{LayerID: 1, K: []byte("k1"), V: []byte("v1")},
		{LayerID: 2, K: []byte("k2"), V: []byte("v2")},
	}

	// Batch store
	err := adapter.BatchStore(ctx, modelID, tokens, layers)
	if err != nil {
		t.Fatalf("BatchStore failed: %v", err)
	}

	// Verify all stored
	for _, layer := range layers {
		k, v, found := adapter.LookupKV(ctx, modelID, layer.LayerID, tokens)
		if !found {
			t.Errorf("Layer %d not found after BatchStore", layer.LayerID)
			continue
		}
		if string(k) != string(layer.K) {
			t.Errorf("Layer %d K mismatch", layer.LayerID)
		}
		if string(v) != string(layer.V) {
			t.Errorf("Layer %d V mismatch", layer.LayerID)
		}
	}
}

func TestAdapter_Prefetch(t *testing.T) {
	ctx := context.Background()
	adapter := NewLocalOnlyAdapter(1024 * 1024)
	defer adapter.Close()

	// Prefetch should not panic even with empty cache
	adapter.PrefetchKV(ctx, "model", []int{0, 1, 2, 3}, []int32{1, 2, 3})
}

func TestAdapter_Close(t *testing.T) {
	adapter := NewLocalOnlyAdapter(1024 * 1024)

	err := adapter.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Operations after close should fail gracefully
	ctx := context.Background()
	_, _, found := adapter.LookupKV(ctx, "model", 0, []int32{1})
	if found {
		t.Error("Should not find entries after close")
	}
}

func TestConfig_Default(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.MaxLocalCacheSize != 5*1024*1024*1024 {
		t.Error("Default MaxLocalCacheSize should be 5GB")
	}
	if cfg.ChunkSize != 256 {
		t.Error("Default ChunkSize should be 256")
	}
	if cfg.P2PPort != 9100 {
		t.Error("Default P2PPort should be 9100")
	}
	if !cfg.EnableMDNS {
		t.Error("Default should enable mDNS")
	}
	if !cfg.EnableP2P {
		t.Error("Default should enable P2P")
	}
}
