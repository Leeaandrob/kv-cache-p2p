/**
 * KV Cache Memory Management Kernels
 *
 * Provides zero-copy memory operations for efficient GPU ↔ Host transfers
 * used in P2P KV cache sharing.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

extern "C" {

/**
 * Allocates pinned (page-locked) host memory for zero-copy DMA transfers.
 *
 * Pinned memory enables:
 * - Direct DMA transfers without staging buffers
 * - Higher bandwidth GPU ↔ Host transfers
 * - Async copy operations overlapping with compute
 *
 * @param ptr Output pointer to allocated memory
 * @param size Size in bytes to allocate
 * @return cudaSuccess or error code
 */
cudaError_t allocPinnedMemory(void** ptr, size_t size) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    return cudaSuccess;
}

/**
 * Frees previously allocated pinned memory.
 *
 * @param ptr Pointer to pinned memory
 * @return cudaSuccess or error code
 */
cudaError_t freePinnedMemory(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
    return cudaSuccess;
}

/**
 * Allocates GPU device memory.
 *
 * @param ptr Output pointer to allocated memory
 * @param size Size in bytes to allocate
 * @return cudaSuccess or error code
 */
cudaError_t allocDeviceMemory(void** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(ptr, size));
    return cudaSuccess;
}

/**
 * Frees GPU device memory.
 *
 * @param ptr Pointer to device memory
 * @return cudaSuccess or error code
 */
cudaError_t freeDeviceMemory(void* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
    return cudaSuccess;
}

/**
 * Copies KV cache from GPU to pinned host memory (async).
 *
 * Memory layout: [2, numTokens, numKVHeads, headDim] in FP16
 * - First dimension: K=0, V=1
 * - Total size: 2 * numTokens * numKVHeads * headDim * sizeof(half)
 *
 * @param dst Destination pinned host memory
 * @param src Source GPU KV cache
 * @param numTokens Number of tokens in sequence
 * @param numKVHeads Number of KV attention heads
 * @param headDim Dimension per head
 * @param stream CUDA stream for async operation
 * @return cudaSuccess or error code
 */
cudaError_t copyKVToHost(
    void* dst,
    const void* src,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
) {
    // Calculate total size: K + V tensors
    size_t sizePerTensor = numTokens * numKVHeads * headDim * sizeof(__half);
    size_t totalSize = 2 * sizePerTensor;

    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        src,
        totalSize,
        cudaMemcpyDeviceToHost,
        stream
    ));

    return cudaSuccess;
}

/**
 * Copies KV cache from pinned host memory to GPU (async).
 *
 * @param dst Destination GPU KV cache
 * @param src Source pinned host memory
 * @param numTokens Number of tokens in sequence
 * @param numKVHeads Number of KV attention heads
 * @param headDim Dimension per head
 * @param stream CUDA stream for async operation
 * @return cudaSuccess or error code
 */
cudaError_t copyKVToDevice(
    void* dst,
    const void* src,
    size_t numTokens,
    size_t numKVHeads,
    size_t headDim,
    cudaStream_t stream
) {
    size_t sizePerTensor = numTokens * numKVHeads * headDim * sizeof(__half);
    size_t totalSize = 2 * sizePerTensor;

    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        src,
        totalSize,
        cudaMemcpyHostToDevice,
        stream
    ));

    return cudaSuccess;
}

/**
 * Creates a CUDA stream for async operations.
 *
 * @param stream Output stream handle
 * @return cudaSuccess or error code
 */
cudaError_t createStream(cudaStream_t* stream) {
    CUDA_CHECK(cudaStreamCreate(stream));
    return cudaSuccess;
}

/**
 * Destroys a CUDA stream.
 *
 * @param stream Stream handle to destroy
 * @return cudaSuccess or error code
 */
cudaError_t destroyStream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(stream));
    return cudaSuccess;
}

/**
 * Synchronizes a CUDA stream (blocks until all operations complete).
 *
 * @param stream Stream to synchronize
 * @return cudaSuccess or error code
 */
cudaError_t syncStream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return cudaSuccess;
}

/**
 * Gets GPU device properties.
 *
 * @param deviceId GPU device ID
 * @param totalMem Output total memory in bytes
 * @param freeMem Output free memory in bytes
 * @return cudaSuccess or error code
 */
cudaError_t getDeviceMemInfo(int deviceId, size_t* totalMem, size_t* freeMem) {
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaMemGetInfo(freeMem, totalMem));
    return cudaSuccess;
}

/**
 * Calculates KV cache size for given configuration.
 *
 * @param numTokens Number of tokens
 * @param numKVHeads Number of KV heads
 * @param headDim Head dimension
 * @param dtype 0=FP16, 1=INT8
 * @return Size in bytes
 */
size_t calculateKVSize(size_t numTokens, size_t numKVHeads, size_t headDim, int dtype) {
    size_t elemSize = (dtype == 0) ? sizeof(__half) : sizeof(int8_t);
    // K + V tensors
    return 2 * numTokens * numKVHeads * headDim * elemSize;
}

} // extern "C"
