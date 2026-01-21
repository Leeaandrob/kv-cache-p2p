/**
 * INT4 Quantization CUDA Kernels
 *
 * Implements per-group symmetric quantization for KV cache compression.
 * Based on techniques from GPTQ, AWQ, and KVQuant papers.
 */

#include "compression.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <float.h>

// ============================================================================
// Constants
// ============================================================================

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024

// INT4 range: [-8, 7] for signed
#define INT4_MIN -8
#define INT4_MAX 7
#define INT4_RANGE 16

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

// Pack two INT4 values into one byte
__device__ __forceinline__ uint8_t pack_int4(int4_t low, int4_t high) {
    return ((high & 0xF) << 4) | (low & 0xF);
}

// Unpack byte into two INT4 values
__device__ __forceinline__ void unpack_int4(uint8_t packed, int* low, int* high) {
    *low = (packed & 0xF);
    if (*low > 7) *low -= 16;  // Sign extend
    *high = ((packed >> 4) & 0xF);
    if (*high > 7) *high -= 16;  // Sign extend
}

// ============================================================================
// Quantization Kernels
// ============================================================================

/**
 * Kernel to find min/max per group for scale computation
 * Each block handles one group
 */
__global__ void find_group_minmax_kernel(
    const __half* __restrict__ input,
    float* __restrict__ group_max,
    int num_elements,
    int group_size
) {
    int group_idx = blockIdx.x;
    int group_start = group_idx * group_size;
    int group_end = min(group_start + group_size, num_elements);

    // Shared memory for reduction
    __shared__ float s_max[MAX_BLOCK_SIZE];

    float thread_max = 0.0f;

    // Each thread finds max absolute value in its portion
    for (int i = group_start + threadIdx.x; i < group_end; i += blockDim.x) {
        float val = fabsf(half_to_float(input[i]));
        thread_max = fmaxf(thread_max, val);
    }

    s_max[threadIdx.x] = thread_max;
    __syncthreads();

    // Block reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        group_max[group_idx] = s_max[0];
    }
}

/**
 * Main quantization kernel
 * Converts FP16 to INT4 using pre-computed scales
 */
__global__ void quantize_kernel(
    const __half* __restrict__ input,
    uint8_t* __restrict__ output,
    __half* __restrict__ scales,
    const float* __restrict__ group_max,
    int num_elements,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx;  // Each thread handles 2 values (one byte)
    int element_idx = pair_idx * 2;

    if (element_idx >= num_elements) return;

    int group_idx = element_idx / group_size;
    float max_val = group_max[group_idx];

    // Compute scale: max_val / 7.0 (symmetric quantization)
    float scale = (max_val > 0.0f) ? (max_val / 7.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale (only first thread in group)
    if (element_idx % group_size == 0) {
        scales[group_idx] = float_to_half(scale);
    }

    // Quantize two values
    int q0 = 0, q1 = 0;

    float v0 = half_to_float(input[element_idx]);
    q0 = __float2int_rn(v0 * inv_scale);
    q0 = max(INT4_MIN, min(INT4_MAX, q0));

    if (element_idx + 1 < num_elements) {
        float v1 = half_to_float(input[element_idx + 1]);
        q1 = __float2int_rn(v1 * inv_scale);
        q1 = max(INT4_MIN, min(INT4_MAX, q1));
    }

    // Pack and store
    output[pair_idx] = pack_int4(q0, q1);
}

/**
 * Dequantization kernel
 * Converts INT4 back to FP16
 */
__global__ void dequantize_kernel(
    const uint8_t* __restrict__ input,
    const __half* __restrict__ scales,
    __half* __restrict__ output,
    int num_elements,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_idx = idx;
    int element_idx = pair_idx * 2;

    if (element_idx >= num_elements) return;

    int group_idx = element_idx / group_size;
    float scale = half_to_float(scales[group_idx]);

    // Unpack INT4 values
    int q0, q1;
    unpack_int4(input[pair_idx], &q0, &q1);

    // Dequantize and store
    output[element_idx] = float_to_half((float)q0 * scale);

    if (element_idx + 1 < num_elements) {
        output[element_idx + 1] = float_to_half((float)q1 * scale);
    }
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

int quantize_fp16_to_int4(
    const void* input,
    void* output,
    void* scales,
    void* zeros,  // Unused for symmetric quantization
    int num_elements,
    int group_size,
    cudaStream_t stream
) {
    if (!input || !output || !scales || num_elements <= 0 || group_size <= 0) {
        return -1;
    }

    int num_groups = (num_elements + group_size - 1) / group_size;

    // Allocate temporary buffer for group max values
    float* d_group_max;
    cudaError_t err = cudaMalloc(&d_group_max, num_groups * sizeof(float));
    if (err != cudaSuccess) return -2;

    // Step 1: Find min/max per group
    int block_size = min(group_size, MAX_BLOCK_SIZE);
    find_group_minmax_kernel<<<num_groups, block_size, 0, stream>>>(
        (const __half*)input,
        d_group_max,
        num_elements,
        group_size
    );

    // Step 2: Quantize
    int num_pairs = (num_elements + 1) / 2;
    int quant_blocks = (num_pairs + 255) / 256;
    quantize_kernel<<<quant_blocks, 256, 0, stream>>>(
        (const __half*)input,
        (uint8_t*)output,
        (__half*)scales,
        d_group_max,
        num_elements,
        group_size
    );

    cudaFree(d_group_max);

    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -3;
}

int dequantize_int4_to_fp16(
    const void* input,
    const void* scales,
    const void* zeros,  // Unused for symmetric quantization
    void* output,
    int num_elements,
    int group_size,
    cudaStream_t stream
) {
    if (!input || !scales || !output || num_elements <= 0 || group_size <= 0) {
        return -1;
    }

    int num_pairs = (num_elements + 1) / 2;
    int blocks = (num_pairs + 255) / 256;

    dequantize_kernel<<<blocks, 256, 0, stream>>>(
        (const uint8_t*)input,
        (const __half*)scales,
        (__half*)output,
        num_elements,
        group_size
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -2;
}

int calc_int4_output_size(int num_elements) {
    return (num_elements + 1) / 2;  // 2 values per byte
}

int calc_scales_size(int num_elements, int group_size) {
    int num_groups = (num_elements + group_size - 1) / group_size;
    return num_groups * sizeof(__half);
}

int calc_zeros_size(int num_elements, int group_size) {
    // For symmetric quantization, no zero points needed
    return 0;
}

} // extern "C"
