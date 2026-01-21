//go:build !cuda
// +build !cuda

package gpu

import (
	"sync"
)

// PinnedPool manages a pool of memory buffers.
// In mock mode, this uses regular Go memory instead of pinned GPU memory.
type PinnedPool struct {
	maxSize     int64
	currentSize int64
	buffers     map[*byte]int // pointer -> size
	freeList    [][]byte
	mu          sync.Mutex
	closed      bool
}

// NewPinnedPool creates a new memory pool with given max size.
func NewPinnedPool(maxSize int64) (*PinnedPool, error) {
	return &PinnedPool{
		maxSize: maxSize,
		buffers: make(map[*byte]int),
	}, nil
}

// Alloc allocates a buffer from the pool.
func (p *PinnedPool) Alloc(size int) ([]byte, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, ErrPoolClosed
	}

	// Check if we have a suitable free buffer
	for i, buf := range p.freeList {
		if len(buf) >= size {
			// Remove from free list
			p.freeList = append(p.freeList[:i], p.freeList[i+1:]...)
			return buf[:size], nil
		}
	}

	// Check if we have capacity for new allocation
	if p.currentSize+int64(size) > p.maxSize {
		// Try to compact
		p.compactFreeList()
		if p.currentSize+int64(size) > p.maxSize {
			return nil, ErrPoolExhausted
		}
	}

	// Allocate regular Go memory
	data := make([]byte, size)
	p.buffers[&data[0]] = size
	p.currentSize += int64(size)

	return data, nil
}

// Free returns a buffer to the pool.
func (p *PinnedPool) Free(data []byte) {
	if len(data) == 0 {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}

	// Check if from this pool
	if _, ok := p.buffers[&data[0]]; !ok {
		return
	}

	// Add to free list
	p.freeList = append(p.freeList, data)
}

// compactFreeList releases buffers from free list to make room (must hold lock).
func (p *PinnedPool) compactFreeList() {
	for len(p.freeList) > 0 && p.currentSize > p.maxSize/2 {
		buf := p.freeList[len(p.freeList)-1]
		p.freeList = p.freeList[:len(p.freeList)-1]

		if size, ok := p.buffers[&buf[0]]; ok {
			delete(p.buffers, &buf[0])
			p.currentSize -= int64(size)
		}
	}
}

// Stats returns pool statistics.
func (p *PinnedPool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	return PoolStats{
		MaxSize:     p.maxSize,
		CurrentSize: p.currentSize,
		BufferCount: len(p.buffers),
		FreeCount:   len(p.freeList),
	}
}

// Close releases all memory.
func (p *PinnedPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}

	p.closed = true
	p.buffers = nil
	p.freeList = nil
	p.currentSize = 0

	return nil
}
