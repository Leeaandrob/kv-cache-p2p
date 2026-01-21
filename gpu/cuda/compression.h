#ifndef KVCACHE_COMPRESSION_H
#define KVCACHE_COMPRESSION_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// INT4 Quantization
// ============================================================================

/**
 * Quantize FP16 tensor to INT4 with per-group scaling
 *
 * @param input      FP16 input tensor (device pointer)
 * @param output     INT4 output (packed, 2 values per byte) (device pointer)
 * @param scales     FP16 per-group scales (device pointer)
 * @param zeros      INT4 per-group zero points (packed) (device pointer)
 * @param num_elements Total number of elements
 * @param group_size Number of elements per quantization group (typically 128)
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int quantize_fp16_to_int4(
    const void* input,
    void* output,
    void* scales,
    void* zeros,
    int num_elements,
    int group_size,
    cudaStream_t stream
);

/**
 * Dequantize INT4 tensor back to FP16
 *
 * @param input      INT4 input (packed) (device pointer)
 * @param scales     FP16 per-group scales (device pointer)
 * @param zeros      INT4 per-group zero points (packed) (device pointer)
 * @param output     FP16 output tensor (device pointer)
 * @param num_elements Total number of elements
 * @param group_size Number of elements per quantization group
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int dequantize_int4_to_fp16(
    const void* input,
    const void* scales,
    const void* zeros,
    void* output,
    int num_elements,
    int group_size,
    cudaStream_t stream
);

// ============================================================================
// Sparsification (Top-K Attention)
// ============================================================================

/**
 * Sparsify tensor by keeping only top-k% values
 *
 * @param input      FP16 input tensor (device pointer)
 * @param output     FP16 sparse values (device pointer)
 * @param indices    UINT32 indices of kept values (device pointer)
 * @param num_kept   Output: number of values kept (host pointer)
 * @param num_elements Total number of elements
 * @param top_k      Fraction to keep (0.0-1.0)
 * @param min_keep   Minimum elements to keep
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int sparsify_topk(
    const void* input,
    void* output,
    uint32_t* indices,
    int* num_kept,
    int num_elements,
    float top_k,
    int min_keep,
    cudaStream_t stream
);

/**
 * Reconstruct dense tensor from sparse representation
 *
 * @param values     FP16 sparse values (device pointer)
 * @param indices    UINT32 indices (device pointer)
 * @param output     FP16 dense output (device pointer)
 * @param num_values Number of sparse values
 * @param output_size Total output size (for bounds checking)
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int desparsify(
    const void* values,
    const uint32_t* indices,
    void* output,
    int num_values,
    int output_size,
    cudaStream_t stream
);

// ============================================================================
// Delta Encoding
// ============================================================================

/**
 * Compute delta between two tensors (XOR for quantized data)
 *
 * @param current    Current tensor (device pointer)
 * @param reference  Reference tensor (device pointer)
 * @param output     Delta output (device pointer)
 * @param num_bytes  Number of bytes
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int compute_delta(
    const void* current,
    const void* reference,
    void* output,
    int num_bytes,
    cudaStream_t stream
);

/**
 * Apply delta to reference to reconstruct current
 *
 * @param delta      Delta tensor (device pointer)
 * @param reference  Reference tensor (device pointer)
 * @param output     Reconstructed output (device pointer)
 * @param num_bytes  Number of bytes
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int apply_delta(
    const void* delta,
    const void* reference,
    void* output,
    int num_bytes,
    cudaStream_t stream
);

// ============================================================================
// Quality Metrics (for validation)
// ============================================================================

/**
 * Compute Mean Squared Error between original and reconstructed
 *
 * @param original   FP16 original tensor (device pointer)
 * @param reconstructed FP16 reconstructed tensor (device pointer)
 * @param mse        Output MSE value (host pointer)
 * @param num_elements Number of elements
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int compute_mse(
    const void* original,
    const void* reconstructed,
    float* mse,
    int num_elements,
    cudaStream_t stream
);

/**
 * Compute max absolute error between original and reconstructed
 *
 * @param original   FP16 original tensor (device pointer)
 * @param reconstructed FP16 reconstructed tensor (device pointer)
 * @param max_error  Output max error value (host pointer)
 * @param num_elements Number of elements
 * @param stream     CUDA stream
 * @return           0 on success, error code otherwise
 */
int compute_max_error(
    const void* original,
    const void* reconstructed,
    float* max_error,
    int num_elements,
    cudaStream_t stream
);

// ============================================================================
// Memory helpers
// ============================================================================

/**
 * Calculate output sizes for compression
 */
int calc_int4_output_size(int num_elements);
int calc_scales_size(int num_elements, int group_size);
int calc_zeros_size(int num_elements, int group_size);

#ifdef __cplusplus
}
#endif

#endif // KVCACHE_COMPRESSION_H
