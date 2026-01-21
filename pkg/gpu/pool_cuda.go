//go:build cuda
// +build cuda

package gpu

import (
	"sync"
	"unsafe"

	"github.com/neurogrid/kv-cache-p2p/gpu/bindings"
)

// PinnedPool manages a pool of pinned (page-locked) host memory buffers.
// Pinned memory enables zero-copy DMA transfers between GPU and host.
type PinnedPool struct {
	maxSize     int64
	currentSize int64
	buffers     map[uintptr]*pinnedBuffer
	freeList    []*pinnedBuffer
	mu          sync.Mutex
	closed      bool
}

// pinnedBuffer represents a single pinned memory allocation.
type pinnedBuffer struct {
	ptr  unsafe.Pointer
	size int
	data []byte
}

// NewPinnedPool creates a new pinned memory pool with given max size.
func NewPinnedPool(maxSize int64) (*PinnedPool, error) {
	return &PinnedPool{
		maxSize: maxSize,
		buffers: make(map[uintptr]*pinnedBuffer),
	}, nil
}

// Alloc allocates a buffer from the pool.
// Returns a byte slice backed by pinned memory.
func (p *PinnedPool) Alloc(size int) ([]byte, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, ErrPoolClosed
	}

	// Check if we have a suitable free buffer
	for i, buf := range p.freeList {
		if buf.size >= size {
			// Remove from free list
			p.freeList = append(p.freeList[:i], p.freeList[i+1:]...)
			return buf.data[:size], nil
		}
	}

	// Check if we have capacity for new allocation
	if p.currentSize+int64(size) > p.maxSize {
		// Try to free smallest suitable buffer
		p.compactFreeList()
		if p.currentSize+int64(size) > p.maxSize {
			return nil, ErrPoolExhausted
		}
	}

	// Allocate new pinned memory
	ptr, err := bindings.AllocPinned(size)
	if err != nil {
		return nil, err
	}

	// Create byte slice from pinned memory
	data := unsafe.Slice((*byte)(ptr), size)

	buf := &pinnedBuffer{
		ptr:  ptr,
		size: size,
		data: data,
	}

	p.buffers[uintptr(ptr)] = buf
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

	// Find the buffer
	ptr := uintptr(unsafe.Pointer(&data[0]))
	buf, ok := p.buffers[ptr]
	if !ok {
		// Not from this pool, ignore
		return
	}

	// Add to free list
	p.freeList = append(p.freeList, buf)
}

// compactFreeList releases buffers from free list to make room (must hold lock).
func (p *PinnedPool) compactFreeList() {
	// Sort by size (smallest first) and free some
	for len(p.freeList) > 0 && p.currentSize > p.maxSize/2 {
		// Remove last (largest) buffer
		buf := p.freeList[len(p.freeList)-1]
		p.freeList = p.freeList[:len(p.freeList)-1]

		// Actually free the pinned memory
		bindings.FreePinned(buf.ptr)
		delete(p.buffers, uintptr(buf.ptr))
		p.currentSize -= int64(buf.size)
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

// Close releases all pinned memory.
func (p *PinnedPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil
	}

	p.closed = true

	// Free all buffers
	for _, buf := range p.buffers {
		bindings.FreePinned(buf.ptr)
	}

	p.buffers = nil
	p.freeList = nil
	p.currentSize = 0

	return nil
}
