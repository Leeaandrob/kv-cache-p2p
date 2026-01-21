// Package gpu provides GPU memory management for KV cache P2P transfers.
package gpu

import (
	"context"
	"errors"
)

var (
	ErrNotInitialized   = errors.New("GPU connector not initialized")
	ErrInvalidConfig    = errors.New("invalid KV config")
	ErrAllocationFailed = errors.New("memory allocation failed")
	ErrPoolExhausted    = errors.New("pinned memory pool exhausted")
	ErrPoolClosed       = errors.New("pinned memory pool closed")
)

// DType represents data types.
type DType int

const (
	FP16 DType = 0
	INT8 DType = 1
)

// KVConfig describes KV cache tensor configuration.
type KVConfig struct {
	NumTokens  int
	NumKVHeads int
	HeadDim    int
	DType      DType
}

// Size returns total bytes for K+V tensors.
func (c KVConfig) Size() int {
	elemSize := 2 // FP16
	if c.DType == INT8 {
		elemSize = 1
	}
	return 2 * c.NumTokens * c.NumKVHeads * c.HeadDim * elemSize
}

// Validate checks if config is valid.
func (c KVConfig) Validate() error {
	if c.NumTokens <= 0 || c.NumKVHeads <= 0 || c.HeadDim <= 0 {
		return ErrInvalidConfig
	}
	return nil
}

// GPUConnector bridges GPU KV cache with host memory for P2P transfer.
type GPUConnector interface {
	// ToHost copies KV from GPU to host memory.
	// Returns a byte slice that can be sent over network.
	ToHost(ctx context.Context, gpuPtr uintptr, cfg KVConfig) ([]byte, error)

	// ToDevice copies KV from host to GPU memory.
	ToDevice(ctx context.Context, data []byte, gpuPtr uintptr, cfg KVConfig) error

	// Sync waits for async operations to complete.
	Sync() error

	// Close releases resources.
	Close() error
}

// PoolStats contains pool statistics.
type PoolStats struct {
	MaxSize     int64
	CurrentSize int64
	BufferCount int
	FreeCount   int
}
