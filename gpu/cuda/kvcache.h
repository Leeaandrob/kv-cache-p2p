/**
 * KV Cache CUDA Kernels Header
 *
 * C-compatible header for CGO bindings.
 */

#ifndef KVCACHE_H
#define KVCACHE_H

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Memory Management
// =============================================================================

/**
 * Allocates pinned (page-locked) host memory.
 */
cudaError_t allocPinnedMemory(void** ptr, size_t size);

/**
 * Frees pinned host memory.
 */
cudaError_t freePinnedMemory(void* ptr);

/**
 * Allocates GPU device memory.
 */
cudaError_t allocDeviceMemory(void** ptr, size_t size);

/**
 * Frees GPU device memory.
 */
cudaError_t freeDeviceMemory(void* ptr);

/**
 * Copies KV from GPU to pinned host memory (async).
 */
cudaError_t copyKVToHost(
    void* dst,
    const void* src,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
);

/**
 * Copies KV from pinned host to GPU memory (async).
 */
cudaError_t copyKVToDevice(
    void* dst,
    const void* src,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
);

// =============================================================================
// Stream Management
// =============================================================================

/**
 * Creates a CUDA stream.
 */
cudaError_t createStream(cudaStream_t* stream);

/**
 * Destroys a CUDA stream.
 */
cudaError_t destroyStream(cudaStream_t stream);

/**
 * Synchronizes a CUDA stream.
 */
cudaError_t syncStream(cudaStream_t stream);

// =============================================================================
// Device Info
// =============================================================================

/**
 * Gets GPU memory info.
 */
cudaError_t getDeviceMemInfo(int deviceId, size_t* totalMem, size_t* freeMem);

/**
 * Calculates KV cache size in bytes.
 * dtype: 0=FP16, 1=INT8
 */
size_t calculateKVSize(size_t numTokens, size_t numKVHeads, size_t headDim, int dtype);

// =============================================================================
// KV Cache Operations
// =============================================================================

/**
 * Appends new KV to existing cache.
 */
cudaError_t appendToKVCache(
    void* kvCache,
    const void* newKV,
    size_t currentLen,
    size_t newTokens,
    size_t numKVHeads,
    size_t headDim,
    size_t maxSeqLen,
    cudaStream_t stream
);

/**
 * Extracts a slice from KV cache.
 */
cudaError_t extractKVSlice(
    void* dst,
    const void* kvCache,
    size_t startPos,
    size_t endPos,
    size_t numKVHeads,
    size_t headDim,
    size_t maxSeqLen,
    cudaStream_t stream
);

/**
 * Merges cached and computed KV.
 */
cudaError_t mergeKVCache(
    void* dst,
    const void* cached,
    const void* computed,
    size_t cachedLen,
    size_t computedLen,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
);

/**
 * Clears KV cache (fills with zeros).
 */
cudaError_t clearKVCache(
    void* kvCache,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // KVCACHE_H
