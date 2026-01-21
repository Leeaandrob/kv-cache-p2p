//go:build cuda
// +build cuda

package bindings

/*
#cgo CFLAGS: -I${SRCDIR}/../cuda
#cgo LDFLAGS: -L${SRCDIR}/../cuda -lkvcache_compression -lcudart -lstdc++

#include "compression.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// QuantizeConfig holds INT4 quantization parameters
type QuantizeConfig struct {
	GroupSize int
}

// DefaultQuantizeConfig returns sensible defaults
func DefaultQuantizeConfig() QuantizeConfig {
	return QuantizeConfig{
		GroupSize: 128,
	}
}

// QuantizeFP16ToINT4 quantizes FP16 data to INT4 on GPU
func QuantizeFP16ToINT4(input unsafe.Pointer, numElements int, config QuantizeConfig, stream unsafe.Pointer) (
	output unsafe.Pointer, scales unsafe.Pointer, err error,
) {
	// Calculate output sizes
	outputSize := C.calc_int4_output_size(C.int(numElements))
	scalesSize := C.calc_scales_size(C.int(numElements), C.int(config.GroupSize))

	// Allocate output buffers
	var d_output, d_scales unsafe.Pointer
	if err := cudaMalloc(&d_output, int(outputSize)); err != nil {
		return nil, nil, fmt.Errorf("failed to allocate output: %w", err)
	}
	if err := cudaMalloc(&d_scales, int(scalesSize)); err != nil {
		cudaFree(d_output)
		return nil, nil, fmt.Errorf("failed to allocate scales: %w", err)
	}

	// Call CUDA kernel
	ret := C.quantize_fp16_to_int4(
		input,
		d_output,
		d_scales,
		nil, // zeros (unused for symmetric)
		C.int(numElements),
		C.int(config.GroupSize),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_output)
		cudaFree(d_scales)
		return nil, nil, fmt.Errorf("quantization failed with code %d", ret)
	}

	return d_output, d_scales, nil
}

// DequantizeINT4ToFP16 dequantizes INT4 data back to FP16 on GPU
func DequantizeINT4ToFP16(input, scales unsafe.Pointer, numElements int, config QuantizeConfig, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	// Allocate output buffer (FP16 = 2 bytes per element)
	outputSize := numElements * 2

	var d_output unsafe.Pointer
	if err := cudaMalloc(&d_output, outputSize); err != nil {
		return nil, fmt.Errorf("failed to allocate output: %w", err)
	}

	// Call CUDA kernel
	ret := C.dequantize_int4_to_fp16(
		input,
		scales,
		nil, // zeros (unused for symmetric)
		d_output,
		C.int(numElements),
		C.int(config.GroupSize),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_output)
		return nil, fmt.Errorf("dequantization failed with code %d", ret)
	}

	return d_output, nil
}

// SparsifyTopK sparsifies tensor keeping top-k% values
func SparsifyTopK(input unsafe.Pointer, numElements int, topK float32, minKeep int, stream unsafe.Pointer) (
	values unsafe.Pointer, indices unsafe.Pointer, numKept int, err error,
) {
	// Allocate maximum possible output (worst case: keep all)
	maxOutputSize := numElements * 2 // FP16 values
	maxIndicesSize := numElements * 4 // uint32 indices

	var d_values, d_indices unsafe.Pointer
	if err := cudaMalloc(&d_values, maxOutputSize); err != nil {
		return nil, nil, 0, fmt.Errorf("failed to allocate values: %w", err)
	}
	if err := cudaMalloc(&d_indices, maxIndicesSize); err != nil {
		cudaFree(d_values)
		return nil, nil, 0, fmt.Errorf("failed to allocate indices: %w", err)
	}

	var cNumKept C.int

	// Call CUDA kernel
	ret := C.sparsify_topk(
		input,
		d_values,
		(*C.uint32_t)(d_indices),
		&cNumKept,
		C.int(numElements),
		C.float(topK),
		C.int(minKeep),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_values)
		cudaFree(d_indices)
		return nil, nil, 0, fmt.Errorf("sparsification failed with code %d", ret)
	}

	return d_values, d_indices, int(cNumKept), nil
}

// Desparsify reconstructs dense tensor from sparse representation
func Desparsify(values, indices unsafe.Pointer, numValues, outputSize int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	// Allocate output buffer
	var d_output unsafe.Pointer
	if err := cudaMalloc(&d_output, outputSize*2); err != nil { // FP16
		return nil, fmt.Errorf("failed to allocate output: %w", err)
	}

	// Call CUDA kernel
	ret := C.desparsify(
		values,
		(*C.uint32_t)(indices),
		d_output,
		C.int(numValues),
		C.int(outputSize),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_output)
		return nil, fmt.Errorf("desparsification failed with code %d", ret)
	}

	return d_output, nil
}

// ComputeDelta computes XOR delta between current and reference
func ComputeDelta(current, reference unsafe.Pointer, numBytes int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	var d_output unsafe.Pointer
	if err := cudaMalloc(&d_output, numBytes); err != nil {
		return nil, fmt.Errorf("failed to allocate output: %w", err)
	}

	ret := C.compute_delta(
		current,
		reference,
		d_output,
		C.int(numBytes),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_output)
		return nil, fmt.Errorf("delta computation failed with code %d", ret)
	}

	return d_output, nil
}

// ApplyDelta applies delta to reference to reconstruct original
func ApplyDelta(delta, reference unsafe.Pointer, numBytes int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	var d_output unsafe.Pointer
	if err := cudaMalloc(&d_output, numBytes); err != nil {
		return nil, fmt.Errorf("failed to allocate output: %w", err)
	}

	ret := C.apply_delta(
		delta,
		reference,
		d_output,
		C.int(numBytes),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		cudaFree(d_output)
		return nil, fmt.Errorf("delta apply failed with code %d", ret)
	}

	return d_output, nil
}

// ComputeMSE computes mean squared error between original and reconstructed
func ComputeMSE(original, reconstructed unsafe.Pointer, numElements int, stream unsafe.Pointer) (float32, error) {
	var mse C.float

	ret := C.compute_mse(
		original,
		reconstructed,
		&mse,
		C.int(numElements),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		return 0, fmt.Errorf("MSE computation failed with code %d", ret)
	}

	return float32(mse), nil
}

// ComputeMaxError computes maximum absolute error between original and reconstructed
func ComputeMaxError(original, reconstructed unsafe.Pointer, numElements int, stream unsafe.Pointer) (float32, error) {
	var maxErr C.float

	ret := C.compute_max_error(
		original,
		reconstructed,
		&maxErr,
		C.int(numElements),
		(C.cudaStream_t)(stream),
	)

	if ret != 0 {
		return 0, fmt.Errorf("max error computation failed with code %d", ret)
	}

	return float32(maxErr), nil
}

// CalcINT4OutputSize calculates output size for INT4 quantization
func CalcINT4OutputSize(numElements int) int {
	return int(C.calc_int4_output_size(C.int(numElements)))
}

// CalcScalesSize calculates scales buffer size
func CalcScalesSize(numElements, groupSize int) int {
	return int(C.calc_scales_size(C.int(numElements), C.int(groupSize)))
}

// Helper functions for CUDA memory
func cudaMalloc(ptr *unsafe.Pointer, size int) error {
	// This would call the actual cudaMalloc from kvcache.go
	// For now, using the existing implementation
	return CudaMalloc(ptr, size)
}

func cudaFree(ptr unsafe.Pointer) {
	CudaFree(ptr)
}
