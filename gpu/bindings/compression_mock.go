//go:build !cuda
// +build !cuda

package bindings

import (
	"encoding/binary"
	"math"
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

// Mock implementation for CPU-based quantization (for testing)

// QuantizeFP16ToINT4 mock implementation
func QuantizeFP16ToINT4(input unsafe.Pointer, numElements int, config QuantizeConfig, stream unsafe.Pointer) (
	output unsafe.Pointer, scales unsafe.Pointer, err error,
) {
	// Mock: allocate output buffers
	outputSize := CalcINT4OutputSize(numElements)
	scalesSize := CalcScalesSize(numElements, config.GroupSize)

	outputBuf := make([]byte, outputSize)
	scalesBuf := make([]byte, scalesSize)

	// Simple mock quantization
	inputSlice := unsafe.Slice((*uint16)(input), numElements)
	numGroups := (numElements + config.GroupSize - 1) / config.GroupSize

	for g := 0; g < numGroups; g++ {
		start := g * config.GroupSize
		end := start + config.GroupSize
		if end > numElements {
			end = numElements
		}

		// Find max absolute value in group
		var maxVal float32
		for i := start; i < end; i++ {
			val := float32FromFP16(inputSlice[i])
			absVal := float32(math.Abs(float64(val)))
			if absVal > maxVal {
				maxVal = absVal
			}
		}

		// Compute scale
		scale := maxVal / 7.0
		if scale == 0 {
			scale = 1.0
		}

		// Store scale as FP16
		binary.LittleEndian.PutUint16(scalesBuf[g*2:], fp16FromFloat32(scale))

		// Quantize values
		for i := start; i < end; i += 2 {
			v0 := float32FromFP16(inputSlice[i])
			q0 := int(math.Round(float64(v0 / scale)))
			q0 = clamp(q0, -8, 7)

			var q1 int
			if i+1 < end {
				v1 := float32FromFP16(inputSlice[i+1])
				q1 = int(math.Round(float64(v1 / scale)))
				q1 = clamp(q1, -8, 7)
			}

			// Pack two INT4 values
			outputBuf[i/2] = byte((q1&0xF)<<4) | byte(q0&0xF)
		}
	}

	return unsafe.Pointer(&outputBuf[0]), unsafe.Pointer(&scalesBuf[0]), nil
}

// DequantizeINT4ToFP16 mock implementation
func DequantizeINT4ToFP16(input, scales unsafe.Pointer, numElements int, config QuantizeConfig, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	outputBuf := make([]uint16, numElements)
	inputSlice := unsafe.Slice((*byte)(input), CalcINT4OutputSize(numElements))
	scalesSlice := unsafe.Slice((*uint16)(scales), (numElements+config.GroupSize-1)/config.GroupSize)

	for i := 0; i < numElements; i += 2 {
		groupIdx := i / config.GroupSize
		scale := float32FromFP16(scalesSlice[groupIdx])

		packed := inputSlice[i/2]
		q0 := int(packed & 0xF)
		if q0 > 7 {
			q0 -= 16
		}
		q1 := int((packed >> 4) & 0xF)
		if q1 > 7 {
			q1 -= 16
		}

		outputBuf[i] = fp16FromFloat32(float32(q0) * scale)
		if i+1 < numElements {
			outputBuf[i+1] = fp16FromFloat32(float32(q1) * scale)
		}
	}

	return unsafe.Pointer(&outputBuf[0]), nil
}

// SparsifyTopK mock implementation
func SparsifyTopK(input unsafe.Pointer, numElements int, topK float32, minKeep int, stream unsafe.Pointer) (
	values unsafe.Pointer, indices unsafe.Pointer, numKept int, err error,
) {
	keepCount := int(float32(numElements) * topK)
	if keepCount < minKeep {
		keepCount = minKeep
	}
	if keepCount > numElements {
		keepCount = numElements
	}

	// Mock: just keep first N values
	inputSlice := unsafe.Slice((*uint16)(input), numElements)
	valuesBuf := make([]uint16, keepCount)
	indicesBuf := make([]uint32, keepCount)

	for i := 0; i < keepCount; i++ {
		valuesBuf[i] = inputSlice[i]
		indicesBuf[i] = uint32(i)
	}

	return unsafe.Pointer(&valuesBuf[0]), unsafe.Pointer(&indicesBuf[0]), keepCount, nil
}

// Desparsify mock implementation
func Desparsify(values, indices unsafe.Pointer, numValues, outputSize int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	outputBuf := make([]uint16, outputSize)
	valuesSlice := unsafe.Slice((*uint16)(values), numValues)
	indicesSlice := unsafe.Slice((*uint32)(indices), numValues)

	for i := 0; i < numValues; i++ {
		idx := indicesSlice[i]
		if int(idx) < outputSize {
			outputBuf[idx] = valuesSlice[i]
		}
	}

	return unsafe.Pointer(&outputBuf[0]), nil
}

// ComputeDelta mock implementation
func ComputeDelta(current, reference unsafe.Pointer, numBytes int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	outputBuf := make([]byte, numBytes)
	currentSlice := unsafe.Slice((*byte)(current), numBytes)
	referenceSlice := unsafe.Slice((*byte)(reference), numBytes)

	for i := 0; i < numBytes; i++ {
		outputBuf[i] = currentSlice[i] ^ referenceSlice[i]
	}

	return unsafe.Pointer(&outputBuf[0]), nil
}

// ApplyDelta mock implementation
func ApplyDelta(delta, reference unsafe.Pointer, numBytes int, stream unsafe.Pointer) (
	output unsafe.Pointer, err error,
) {
	return ComputeDelta(delta, reference, numBytes, stream) // XOR is symmetric
}

// ComputeMSE mock implementation
func ComputeMSE(original, reconstructed unsafe.Pointer, numElements int, stream unsafe.Pointer) (float32, error) {
	origSlice := unsafe.Slice((*uint16)(original), numElements)
	reconSlice := unsafe.Slice((*uint16)(reconstructed), numElements)

	var sumSqErr float64
	for i := 0; i < numElements; i++ {
		orig := float64(float32FromFP16(origSlice[i]))
		recon := float64(float32FromFP16(reconSlice[i]))
		diff := orig - recon
		sumSqErr += diff * diff
	}

	return float32(sumSqErr / float64(numElements)), nil
}

// ComputeMaxError mock implementation
func ComputeMaxError(original, reconstructed unsafe.Pointer, numElements int, stream unsafe.Pointer) (float32, error) {
	origSlice := unsafe.Slice((*uint16)(original), numElements)
	reconSlice := unsafe.Slice((*uint16)(reconstructed), numElements)

	var maxErr float64
	for i := 0; i < numElements; i++ {
		orig := float64(float32FromFP16(origSlice[i]))
		recon := float64(float32FromFP16(reconSlice[i]))
		absErr := math.Abs(orig - recon)
		if absErr > maxErr {
			maxErr = absErr
		}
	}

	return float32(maxErr), nil
}

// CalcINT4OutputSize calculates output size for INT4 quantization
func CalcINT4OutputSize(numElements int) int {
	return (numElements + 1) / 2
}

// CalcScalesSize calculates scales buffer size
func CalcScalesSize(numElements, groupSize int) int {
	numGroups := (numElements + groupSize - 1) / groupSize
	return numGroups * 2 // FP16 = 2 bytes
}

// Helper functions
func clamp(val, min, max int) int {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

func float32FromFP16(fp16 uint16) float32 {
	sign := (fp16 >> 15) & 0x1
	exp := (fp16 >> 10) & 0x1F
	frac := fp16 & 0x3FF

	if exp == 0 {
		if frac == 0 {
			if sign == 1 {
				return -0.0
			}
			return 0.0
		}
		// Denormalized
		return float32(math.Pow(-1, float64(sign))) * float32(math.Pow(2, -14)) * float32(frac) / 1024.0
	}
	if exp == 31 {
		if frac == 0 {
			if sign == 1 {
				return float32(math.Inf(-1))
			}
			return float32(math.Inf(1))
		}
		return float32(math.NaN())
	}

	return float32(math.Pow(-1, float64(sign))) * float32(math.Pow(2, float64(exp)-15)) * (1.0 + float32(frac)/1024.0)
}

func fp16FromFloat32(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x1
	exp := int((bits>>23)&0xFF) - 127 + 15
	frac := (bits >> 13) & 0x3FF

	if exp <= 0 {
		return uint16(sign << 15)
	}
	if exp >= 31 {
		return uint16((sign << 15) | (0x1F << 10))
	}

	return uint16((sign << 15) | (uint32(exp) << 10) | frac)
}
