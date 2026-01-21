/**
 * Delta Encoding and Quality Metrics CUDA Kernels
 *
 * Delta encoding computes XOR between current and reference tensors.
 * Quality metrics help validate compression doesn't degrade inference.
 */

#include "compression.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ============================================================================
// Delta Encoding Kernels
// ============================================================================

/**
 * XOR delta for byte-level data (works for quantized INT4)
 */
__global__ void xor_delta_kernel(
    const uint8_t* __restrict__ current,
    const uint8_t* __restrict__ reference,
    uint8_t* __restrict__ output,
    int num_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bytes) return;

    output[idx] = current[idx] ^ reference[idx];
}

/**
 * Apply XOR delta to reconstruct
 */
__global__ void apply_xor_delta_kernel(
    const uint8_t* __restrict__ delta,
    const uint8_t* __restrict__ reference,
    uint8_t* __restrict__ output,
    int num_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bytes) return;

    output[idx] = delta[idx] ^ reference[idx];
}

// ============================================================================
// Quality Metrics Kernels
// ============================================================================

/**
 * Compute squared error per element
 */
__global__ void squared_error_kernel(
    const __half* __restrict__ original,
    const __half* __restrict__ reconstructed,
    float* __restrict__ errors,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float orig = __half2float(original[idx]);
    float recon = __half2float(reconstructed[idx]);
    float diff = orig - recon;
    errors[idx] = diff * diff;
}

/**
 * Compute absolute error per element
 */
__global__ void abs_error_kernel(
    const __half* __restrict__ original,
    const __half* __restrict__ reconstructed,
    float* __restrict__ errors,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float orig = __half2float(original[idx]);
    float recon = __half2float(reconstructed[idx]);
    errors[idx] = fabsf(orig - recon);
}

/**
 * Sum reduction kernel
 */
__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < num_elements) ? input[idx] : 0.0f;

    float block_sum = BlockReduce(temp_storage).Sum(val);

    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

/**
 * Max reduction kernel
 */
__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < num_elements) ? input[idx] : 0.0f;

    float block_max = BlockReduce(temp_storage).Reduce(val, cub::Max());

    if (threadIdx.x == 0) {
        // Atomic max for floats using int representation
        int* address_as_int = (int*)output;
        int old = *address_as_int;
        int expected;
        do {
            expected = old;
            float old_val = __int_as_float(expected);
            float new_val = fmaxf(old_val, block_max);
            old = atomicCAS(address_as_int, expected, __float_as_int(new_val));
        } while (expected != old);
    }
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

int compute_delta(
    const void* current,
    const void* reference,
    void* output,
    int num_bytes,
    cudaStream_t stream
) {
    if (!current || !reference || !output || num_bytes <= 0) {
        return -1;
    }

    int blocks = (num_bytes + 255) / 256;
    xor_delta_kernel<<<blocks, 256, 0, stream>>>(
        (const uint8_t*)current,
        (const uint8_t*)reference,
        (uint8_t*)output,
        num_bytes
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -2;
}

int apply_delta(
    const void* delta,
    const void* reference,
    void* output,
    int num_bytes,
    cudaStream_t stream
) {
    if (!delta || !reference || !output || num_bytes <= 0) {
        return -1;
    }

    int blocks = (num_bytes + 255) / 256;
    apply_xor_delta_kernel<<<blocks, 256, 0, stream>>>(
        (const uint8_t*)delta,
        (const uint8_t*)reference,
        (uint8_t*)output,
        num_bytes
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -2;
}

int compute_mse(
    const void* original,
    const void* reconstructed,
    float* mse,
    int num_elements,
    cudaStream_t stream
) {
    if (!original || !reconstructed || !mse || num_elements <= 0) {
        return -1;
    }

    // Allocate temporary buffer for errors
    float* d_errors;
    float* d_sum;
    cudaError_t err = cudaMalloc(&d_errors, num_elements * sizeof(float));
    if (err != cudaSuccess) return -2;
    err = cudaMalloc(&d_sum, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_errors);
        return -2;
    }
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);

    // Compute squared errors
    int blocks = (num_elements + 255) / 256;
    squared_error_kernel<<<blocks, 256, 0, stream>>>(
        (const __half*)original,
        (const __half*)reconstructed,
        d_errors,
        num_elements
    );

    // Sum reduction
    sum_reduce_kernel<<<blocks, 256, 0, stream>>>(
        d_errors,
        d_sum,
        num_elements
    );

    // Copy result back
    float sum;
    cudaMemcpyAsync(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    *mse = sum / num_elements;

    cudaFree(d_errors);
    cudaFree(d_sum);

    return 0;
}

int compute_max_error(
    const void* original,
    const void* reconstructed,
    float* max_error,
    int num_elements,
    cudaStream_t stream
) {
    if (!original || !reconstructed || !max_error || num_elements <= 0) {
        return -1;
    }

    // Allocate temporary buffer for errors
    float* d_errors;
    float* d_max;
    cudaError_t err = cudaMalloc(&d_errors, num_elements * sizeof(float));
    if (err != cudaSuccess) return -2;
    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_errors);
        return -2;
    }
    cudaMemsetAsync(d_max, 0, sizeof(float), stream);

    // Compute absolute errors
    int blocks = (num_elements + 255) / 256;
    abs_error_kernel<<<blocks, 256, 0, stream>>>(
        (const __half*)original,
        (const __half*)reconstructed,
        d_errors,
        num_elements
    );

    // Max reduction
    max_reduce_kernel<<<blocks, 256, 0, stream>>>(
        d_errors,
        d_max,
        num_elements
    );

    // Copy result back
    cudaMemcpyAsync(max_error, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_errors);
    cudaFree(d_max);

    return 0;
}

} // extern "C"
