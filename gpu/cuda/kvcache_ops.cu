/**
 * KV Cache Operations Kernels
 *
 * GPU kernels for manipulating KV cache tensors:
 * - Append new KV to existing cache
 * - Extract slices for P2P transfer
 * - Merge cached and computed KV
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

// Block size for kernels
#define BLOCK_SIZE 256

/**
 * Kernel: Copy slice from KV cache to output buffer.
 *
 * Each thread copies one element.
 * Grid dims: (elementsPerSlice + BLOCK_SIZE - 1) / BLOCK_SIZE
 */
__global__ void extractSliceKernel(
    __half* dst,
    const __half* src,
    size_t startPos,
    size_t sliceLen,
    size_t numKVHeads,
    size_t headDim,
    size_t maxSeqLen  // Source tensor max sequence length
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total elements per token position: numKVHeads * headDim
    size_t elementsPerPos = numKVHeads * headDim;
    // Total elements to copy: 2 (K,V) * sliceLen * elementsPerPos
    size_t totalElements = 2 * sliceLen * elementsPerPos;

    if (idx >= totalElements) return;

    // Decode position in output
    size_t kvIdx = idx / (sliceLen * elementsPerPos);  // 0=K, 1=V
    size_t remaining = idx % (sliceLen * elementsPerPos);
    size_t posInSlice = remaining / elementsPerPos;
    size_t elemIdx = remaining % elementsPerPos;

    // Calculate source position
    size_t srcPos = startPos + posInSlice;
    size_t srcOffset = kvIdx * maxSeqLen * elementsPerPos + srcPos * elementsPerPos + elemIdx;

    // Calculate destination offset
    size_t dstOffset = kvIdx * sliceLen * elementsPerPos + posInSlice * elementsPerPos + elemIdx;

    dst[dstOffset] = src[srcOffset];
}

/**
 * Kernel: Append new KV to existing cache.
 *
 * Copies newKV data to kvCache starting at position currentLen.
 */
__global__ void appendKernel(
    __half* kvCache,      // [2, maxSeqLen, numKVHeads, headDim]
    const __half* newKV,  // [2, newTokens, numKVHeads, headDim]
    size_t currentLen,
    size_t newTokens,
    size_t numKVHeads,
    size_t headDim,
    size_t maxSeqLen
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t elementsPerPos = numKVHeads * headDim;
    size_t totalElements = 2 * newTokens * elementsPerPos;

    if (idx >= totalElements) return;

    // Decode position
    size_t kvIdx = idx / (newTokens * elementsPerPos);
    size_t remaining = idx % (newTokens * elementsPerPos);
    size_t posInNew = remaining / elementsPerPos;
    size_t elemIdx = remaining % elementsPerPos;

    // Source offset in newKV
    size_t srcOffset = kvIdx * newTokens * elementsPerPos + posInNew * elementsPerPos + elemIdx;

    // Destination offset in kvCache
    size_t dstPos = currentLen + posInNew;
    size_t dstOffset = kvIdx * maxSeqLen * elementsPerPos + dstPos * elementsPerPos + elemIdx;

    kvCache[dstOffset] = newKV[srcOffset];
}

/**
 * Kernel: Merge cached and computed KV into destination.
 *
 * Output = [cached | computed]
 */
__global__ void mergeKernel(
    __half* dst,          // [2, cachedLen + computedLen, numKVHeads, headDim]
    const __half* cached, // [2, cachedLen, numKVHeads, headDim]
    const __half* computed, // [2, computedLen, numKVHeads, headDim]
    size_t cachedLen,
    size_t computedLen,
    size_t numKVHeads,
    size_t headDim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t elementsPerPos = numKVHeads * headDim;
    size_t totalLen = cachedLen + computedLen;
    size_t totalElements = 2 * totalLen * elementsPerPos;

    if (idx >= totalElements) return;

    // Decode position
    size_t kvIdx = idx / (totalLen * elementsPerPos);
    size_t remaining = idx % (totalLen * elementsPerPos);
    size_t pos = remaining / elementsPerPos;
    size_t elemIdx = remaining % elementsPerPos;

    __half value;

    if (pos < cachedLen) {
        // From cached portion
        size_t srcOffset = kvIdx * cachedLen * elementsPerPos + pos * elementsPerPos + elemIdx;
        value = cached[srcOffset];
    } else {
        // From computed portion
        size_t computedPos = pos - cachedLen;
        size_t srcOffset = kvIdx * computedLen * elementsPerPos + computedPos * elementsPerPos + elemIdx;
        value = computed[srcOffset];
    }

    dst[idx] = value;
}

extern "C" {

/**
 * Appends new KV data to existing cache.
 *
 * @param kvCache Existing KV cache [2, maxSeqLen, numKVHeads, headDim]
 * @param newKV New KV data [2, newTokens, numKVHeads, headDim]
 * @param currentLen Current sequence length in cache
 * @param newTokens Number of new tokens to append
 * @param numKVHeads Number of KV heads
 * @param headDim Head dimension
 * @param maxSeqLen Maximum sequence length of cache
 * @param stream CUDA stream
 * @return cudaSuccess or error code
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
) {
    // Bounds check
    if (currentLen + newTokens > maxSeqLen) {
        fprintf(stderr, "Error: append would exceed maxSeqLen (%zu + %zu > %zu)\n",
                currentLen, newTokens, maxSeqLen);
        return cudaErrorInvalidValue;
    }

    size_t elementsPerPos = numKVHeads * headDim;
    size_t totalElements = 2 * newTokens * elementsPerPos;
    int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    appendKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)kvCache,
        (const __half*)newKV,
        currentLen,
        newTokens,
        numKVHeads,
        headDim,
        maxSeqLen
    );

    return cudaGetLastError();
}

/**
 * Extracts a slice from KV cache.
 *
 * @param dst Output buffer [2, sliceLen, numKVHeads, headDim]
 * @param kvCache Source cache [2, maxSeqLen, numKVHeads, headDim]
 * @param startPos Start position (inclusive)
 * @param endPos End position (exclusive)
 * @param numKVHeads Number of KV heads
 * @param headDim Head dimension
 * @param maxSeqLen Max sequence length of source
 * @param stream CUDA stream
 * @return cudaSuccess or error code
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
) {
    if (startPos >= endPos || endPos > maxSeqLen) {
        return cudaErrorInvalidValue;
    }

    size_t sliceLen = endPos - startPos;
    size_t elementsPerPos = numKVHeads * headDim;
    size_t totalElements = 2 * sliceLen * elementsPerPos;
    int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    extractSliceKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)dst,
        (const __half*)kvCache,
        startPos,
        sliceLen,
        numKVHeads,
        headDim,
        maxSeqLen
    );

    return cudaGetLastError();
}

/**
 * Merges cached and computed KV into a single tensor.
 *
 * @param dst Output [2, cachedLen + computedLen, numKVHeads, headDim]
 * @param cached Cached portion [2, cachedLen, numKVHeads, headDim]
 * @param computed Computed portion [2, computedLen, numKVHeads, headDim]
 * @param cachedLen Length of cached portion
 * @param computedLen Length of computed portion
 * @param numKVHeads Number of KV heads
 * @param headDim Head dimension
 * @param stream CUDA stream
 * @return cudaSuccess or error code
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
) {
    size_t elementsPerPos = numKVHeads * headDim;
    size_t totalLen = cachedLen + computedLen;
    size_t totalElements = 2 * totalLen * elementsPerPos;
    int numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mergeKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)dst,
        (const __half*)cached,
        (const __half*)computed,
        cachedLen,
        computedLen,
        numKVHeads,
        headDim
    );

    return cudaGetLastError();
}

/**
 * Fills KV cache with zeros.
 *
 * @param kvCache Cache to clear
 * @param numTokens Number of tokens
 * @param numKVHeads Number of KV heads
 * @param headDim Head dimension
 * @param stream CUDA stream
 * @return cudaSuccess or error code
 */
cudaError_t clearKVCache(
    void* kvCache,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
) {
    size_t size = 2 * numTokens * numKVHeads * headDim * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(kvCache, 0, size, stream));
    return cudaSuccess;
}

} // extern "C"
