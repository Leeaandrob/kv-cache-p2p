// Package gpu provides GPU memory management for KV cache P2P transfers.
package gpu

import (
	"context"
	"sync"
)

// MockConnector implements GPUConnector for testing without a real GPU.
type MockConnector struct {
	memory map[uintptr][]byte
	mu     sync.Mutex
}

// NewMockConnector creates a new mock connector.
func NewMockConnector() *MockConnector {
	return &MockConnector{
		memory: make(map[uintptr][]byte),
	}
}

// SetGPUData simulates data stored on GPU at given pointer.
func (c *MockConnector) SetGPUData(ptr uintptr, data []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.memory[ptr] = append([]byte(nil), data...)
}

// GetGPUData returns simulated GPU data.
func (c *MockConnector) GetGPUData(ptr uintptr) []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.memory[ptr]
}

// ToHost copies KV from "GPU" to host memory.
func (c *MockConnector) ToHost(ctx context.Context, gpuPtr uintptr, cfg KVConfig) ([]byte, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	data, ok := c.memory[gpuPtr]
	if !ok {
		// Return zeros if no data set
		return make([]byte, cfg.Size()), nil
	}

	// Copy to avoid returning internal reference
	result := make([]byte, len(data))
	copy(result, data)
	return result, nil
}

// ToDevice copies KV from host to "GPU" memory.
func (c *MockConnector) ToDevice(ctx context.Context, data []byte, gpuPtr uintptr, cfg KVConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Store copy of data
	c.memory[gpuPtr] = append([]byte(nil), data...)
	return nil
}

// Sync is a no-op for mock.
func (c *MockConnector) Sync() error {
	return nil
}

// Close clears internal state.
func (c *MockConnector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.memory = nil
	return nil
}

// Verify interface compliance.
var _ GPUConnector = (*MockConnector)(nil)
