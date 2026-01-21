/**
 * Sparsification CUDA Kernels
 *
 * Implements top-k sparsification for attention scores.
 * Keeps only the most important attention values to reduce data size.
 */

#include "compression.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// ============================================================================
// Helper structures
// ============================================================================

struct ValueIndex {
    float value;
    uint32_t index;
};

__device__ __forceinline__ bool operator<(const ValueIndex& a, const ValueIndex& b) {
    return fabsf(a.value) > fabsf(b.value);  // Descending by absolute value
}

// ============================================================================
// Sparsification Kernels
// ============================================================================

/**
 * Compute absolute values and create index pairs
 */
__global__ void create_value_index_pairs_kernel(
    const __half* __restrict__ input,
    ValueIndex* __restrict__ pairs,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    pairs[idx].value = __half2float(input[idx]);
    pairs[idx].index = idx;
}

/**
 * Extract top-k values after sorting
 */
__global__ void extract_topk_kernel(
    const __half* __restrict__ input,
    const ValueIndex* __restrict__ sorted_pairs,
    __half* __restrict__ output_values,
    uint32_t* __restrict__ output_indices,
    int num_keep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keep) return;

    uint32_t orig_idx = sorted_pairs[idx].index;
    output_values[idx] = input[orig_idx];
    output_indices[idx] = orig_idx;
}

/**
 * Reconstruct dense tensor from sparse representation
 */
__global__ void desparsify_kernel(
    const __half* __restrict__ values,
    const uint32_t* __restrict__ indices,
    __half* __restrict__ output,
    int num_values,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    uint32_t out_idx = indices[idx];
    if (out_idx < output_size) {
        output[out_idx] = values[idx];
    }
}

/**
 * Zero initialize output tensor
 */
__global__ void zero_init_kernel(
    __half* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    output[idx] = __float2half(0.0f);
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

int sparsify_topk(
    const void* input,
    void* output,
    uint32_t* indices,
    int* num_kept,
    int num_elements,
    float top_k,
    int min_keep,
    cudaStream_t stream
) {
    if (!input || !output || !indices || !num_kept) {
        return -1;
    }
    if (top_k <= 0.0f || top_k > 1.0f) {
        return -2;
    }

    // Calculate how many to keep
    int keep_count = (int)(num_elements * top_k);
    keep_count = max(keep_count, min_keep);
    keep_count = min(keep_count, num_elements);

    // Allocate temporary buffer for value-index pairs
    ValueIndex* d_pairs;
    cudaError_t err = cudaMalloc(&d_pairs, num_elements * sizeof(ValueIndex));
    if (err != cudaSuccess) return -3;

    // Create value-index pairs
    int blocks = (num_elements + 255) / 256;
    create_value_index_pairs_kernel<<<blocks, 256, 0, stream>>>(
        (const __half*)input,
        d_pairs,
        num_elements
    );

    // Sort by absolute value (descending)
    thrust::device_ptr<ValueIndex> d_ptr(d_pairs);
    thrust::sort(thrust::cuda::par.on(stream), d_ptr, d_ptr + num_elements);

    // Extract top-k
    int extract_blocks = (keep_count + 255) / 256;
    extract_topk_kernel<<<extract_blocks, 256, 0, stream>>>(
        (const __half*)input,
        d_pairs,
        (__half*)output,
        indices,
        keep_count
    );

    cudaFree(d_pairs);

    *num_kept = keep_count;

    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -4;
}

int desparsify(
    const void* values,
    const uint32_t* indices,
    void* output,
    int num_values,
    int output_size,
    cudaStream_t stream
) {
    if (!values || !indices || !output) {
        return -1;
    }

    // Zero initialize output
    int zero_blocks = (output_size + 255) / 256;
    zero_init_kernel<<<zero_blocks, 256, 0, stream>>>(
        (__half*)output,
        output_size
    );

    // Fill in sparse values
    int blocks = (num_values + 255) / 256;
    desparsify_kernel<<<blocks, 256, 0, stream>>>(
        (const __half*)values,
        indices,
        (__half*)output,
        num_values,
        output_size
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -2;
}

} // extern "C"
