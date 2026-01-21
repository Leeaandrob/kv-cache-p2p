//go:build cuda
// +build cuda

// Package bindings provides CGO bindings for CUDA KV cache operations.
package bindings

/*
#cgo CFLAGS: -I${SRCDIR}/../cuda

// x86_64 with standard CUDA install
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../cuda -lkvcache -L/usr/local/cuda/lib64 -lcudart -lstdc++

// arm64 with system CUDA install (apt)
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/../cuda -lkvcache -L/usr/lib/aarch64-linux-gnu -lcudart -lstdc++

#include "kvcache.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAError wraps CUDA error codes.
type CUDAError int

func (e CUDAError) Error() string {
	return fmt.Sprintf("CUDA error: %d", int(e))
}

// Stream represents a CUDA stream handle.
type Stream C.cudaStream_t

// DType represents data types.
type DType int

const (
	FP16 DType = 0
	INT8 DType = 1
)

// =============================================================================
// Memory Management
// =============================================================================

// AllocPinned allocates pinned (page-locked) host memory.
func AllocPinned(size int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	ret := C.allocPinnedMemory(&ptr, C.size_t(size))
	if ret != 0 {
		return nil, CUDAError(ret)
	}
	return ptr, nil
}

// FreePinned frees pinned host memory.
func FreePinned(ptr unsafe.Pointer) error {
	ret := C.freePinnedMemory(ptr)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// AllocDevice allocates GPU device memory.
func AllocDevice(size int) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	ret := C.allocDeviceMemory(&ptr, C.size_t(size))
	if ret != 0 {
		return nil, CUDAError(ret)
	}
	return ptr, nil
}

// FreeDevice frees GPU device memory.
func FreeDevice(ptr unsafe.Pointer) error {
	ret := C.freeDeviceMemory(ptr)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// =============================================================================
// Stream Management
// =============================================================================

// CreateStream creates a new CUDA stream.
func CreateStream() (Stream, error) {
	var stream C.cudaStream_t
	ret := C.createStream(&stream)
	if ret != 0 {
		return Stream(nil), CUDAError(ret)
	}
	return Stream(stream), nil
}

// DestroyStream destroys a CUDA stream.
func DestroyStream(stream Stream) error {
	ret := C.destroyStream(C.cudaStream_t(stream))
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// SyncStream synchronizes a CUDA stream (blocks until complete).
func SyncStream(stream Stream) error {
	ret := C.syncStream(C.cudaStream_t(stream))
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// =============================================================================
// KV Cache Transfers
// =============================================================================

// KVConfig describes KV cache tensor configuration.
type KVConfig struct {
	NumTokens  int
	NumKVHeads int
	HeadDim    int
	DType      DType
}

// Size returns the total size in bytes for K+V tensors.
func (c KVConfig) Size() int {
	elemSize := 2 // FP16
	if c.DType == INT8 {
		elemSize = 1
	}
	return 2 * c.NumTokens * c.NumKVHeads * c.HeadDim * elemSize
}

// CopyKVToHost copies KV cache from GPU to pinned host memory.
func CopyKVToHost(dst, src unsafe.Pointer, cfg KVConfig, stream Stream) error {
	ret := C.copyKVToHost(
		dst,
		src,
		C.size_t(cfg.NumTokens),
		C.size_t(cfg.NumKVHeads),
		C.size_t(cfg.HeadDim),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// CopyKVToDevice copies KV cache from pinned host to GPU memory.
func CopyKVToDevice(dst, src unsafe.Pointer, cfg KVConfig, stream Stream) error {
	ret := C.copyKVToDevice(
		dst,
		src,
		C.size_t(cfg.NumTokens),
		C.size_t(cfg.NumKVHeads),
		C.size_t(cfg.HeadDim),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// =============================================================================
// KV Cache Operations
// =============================================================================

// AppendToKVCache appends new KV data to existing cache.
func AppendToKVCache(
	kvCache, newKV unsafe.Pointer,
	currentLen, newTokens int,
	numKVHeads, headDim, maxSeqLen int,
	stream Stream,
) error {
	ret := C.appendToKVCache(
		kvCache,
		newKV,
		C.size_t(currentLen),
		C.size_t(newTokens),
		C.size_t(numKVHeads),
		C.size_t(headDim),
		C.size_t(maxSeqLen),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// ExtractKVSlice extracts a slice from KV cache.
func ExtractKVSlice(
	dst, kvCache unsafe.Pointer,
	startPos, endPos int,
	numKVHeads, headDim, maxSeqLen int,
	stream Stream,
) error {
	ret := C.extractKVSlice(
		dst,
		kvCache,
		C.size_t(startPos),
		C.size_t(endPos),
		C.size_t(numKVHeads),
		C.size_t(headDim),
		C.size_t(maxSeqLen),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// MergeKVCache merges cached and computed KV.
func MergeKVCache(
	dst, cached, computed unsafe.Pointer,
	cachedLen, computedLen int,
	numKVHeads, headDim int,
	stream Stream,
) error {
	ret := C.mergeKVCache(
		dst,
		cached,
		computed,
		C.size_t(cachedLen),
		C.size_t(computedLen),
		C.size_t(numKVHeads),
		C.size_t(headDim),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// ClearKVCache fills KV cache with zeros.
func ClearKVCache(
	kvCache unsafe.Pointer,
	numTokens, numKVHeads, headDim int,
	stream Stream,
) error {
	ret := C.clearKVCache(
		kvCache,
		C.size_t(numTokens),
		C.size_t(numKVHeads),
		C.size_t(headDim),
		C.cudaStream_t(stream),
	)
	if ret != 0 {
		return CUDAError(ret)
	}
	return nil
}

// =============================================================================
// Device Info
// =============================================================================

// GetDeviceMemInfo returns total and free GPU memory.
func GetDeviceMemInfo(deviceID int) (totalMem, freeMem int64, err error) {
	var total, free C.size_t
	ret := C.getDeviceMemInfo(C.int(deviceID), &total, &free)
	if ret != 0 {
		return 0, 0, CUDAError(ret)
	}
	return int64(total), int64(free), nil
}

// CalculateKVSize calculates KV cache size in bytes.
func CalculateKVSize(numTokens, numKVHeads, headDim int, dtype DType) int {
	return int(C.calculateKVSize(
		C.size_t(numTokens),
		C.size_t(numKVHeads),
		C.size_t(headDim),
		C.int(dtype),
	))
}
