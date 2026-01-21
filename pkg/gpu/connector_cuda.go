//go:build cuda
// +build cuda

package gpu

import (
	"context"
	"sync"
	"unsafe"

	"github.com/neurogrid/kv-cache-p2p/gpu/bindings"
)

// CUDAConnector implements GPUConnector using CUDA.
type CUDAConnector struct {
	stream bindings.Stream
	pool   *PinnedPool
	mu     sync.Mutex
}

// NewCUDAConnector creates a new CUDA connector.
func NewCUDAConnector(poolSize int64) (*CUDAConnector, error) {
	stream, err := bindings.CreateStream()
	if err != nil {
		return nil, err
	}

	pool, err := NewPinnedPool(poolSize)
	if err != nil {
		bindings.DestroyStream(stream)
		return nil, err
	}

	return &CUDAConnector{
		stream: stream,
		pool:   pool,
	}, nil
}

// ToHost copies KV from GPU to host memory.
func (c *CUDAConnector) ToHost(ctx context.Context, gpuPtr uintptr, cfg KVConfig) ([]byte, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	size := cfg.Size()

	// Allocate pinned buffer
	buf, err := c.pool.Alloc(size)
	if err != nil {
		return nil, err
	}

	// Async copy GPU → Host
	bindingsCfg := bindings.KVConfig{
		NumTokens:  cfg.NumTokens,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		DType:      bindings.DType(cfg.DType),
	}

	pinnedPtr := unsafe.Pointer(&buf[0])
	srcPtr := unsafe.Pointer(gpuPtr)

	if err := bindings.CopyKVToHost(pinnedPtr, srcPtr, bindingsCfg, c.stream); err != nil {
		c.pool.Free(buf)
		return nil, err
	}

	// Sync to ensure copy complete
	if err := bindings.SyncStream(c.stream); err != nil {
		c.pool.Free(buf)
		return nil, err
	}

	// Copy to regular Go slice (so pinned memory can be reused)
	result := make([]byte, size)
	copy(result, buf)
	c.pool.Free(buf)

	return result, nil
}

// ToDevice copies KV from host to GPU memory.
func (c *CUDAConnector) ToDevice(ctx context.Context, data []byte, gpuPtr uintptr, cfg KVConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	expectedSize := cfg.Size()
	if len(data) != expectedSize {
		return ErrInvalidConfig
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Allocate pinned buffer and copy data
	buf, err := c.pool.Alloc(len(data))
	if err != nil {
		return err
	}
	defer c.pool.Free(buf)

	copy(buf, data)

	// Async copy Host → GPU
	bindingsCfg := bindings.KVConfig{
		NumTokens:  cfg.NumTokens,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		DType:      bindings.DType(cfg.DType),
	}

	pinnedPtr := unsafe.Pointer(&buf[0])
	dstPtr := unsafe.Pointer(gpuPtr)

	if err := bindings.CopyKVToDevice(dstPtr, pinnedPtr, bindingsCfg, c.stream); err != nil {
		return err
	}

	// Sync to ensure copy complete
	return bindings.SyncStream(c.stream)
}

// Sync waits for async operations to complete.
func (c *CUDAConnector) Sync() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return bindings.SyncStream(c.stream)
}

// Close releases resources.
func (c *CUDAConnector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pool != nil {
		c.pool.Close()
		c.pool = nil
	}

	if c.stream != nil {
		bindings.DestroyStream(c.stream)
		c.stream = nil
	}

	return nil
}

// DeviceMemInfo returns GPU memory info.
func DeviceMemInfo(deviceID int) (totalMem, freeMem int64, err error) {
	return bindings.GetDeviceMemInfo(deviceID)
}
