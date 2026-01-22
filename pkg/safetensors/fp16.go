// Package safetensors provides FP16 (IEEE 754 half-precision) conversion utilities.
package safetensors

import (
	"math"
	"unsafe"
)

// FP16ToFloat32 converts IEEE 754 half-precision to single-precision.
// This handles all cases: normal, subnormal, zero, infinity, and NaN.
func FP16ToFloat32(fp16 uint16) float32 {
	sign := uint32(fp16>>15) & 0x1
	exp := uint32(fp16>>10) & 0x1F
	frac := uint32(fp16) & 0x3FF

	var f32Bits uint32

	if exp == 0 {
		if frac == 0 {
			// Zero (positive or negative)
			f32Bits = sign << 31
		} else {
			// Subnormal FP16 -> normalize to FP32
			exp = 1
			for (frac & 0x400) == 0 {
				frac <<= 1
				exp--
			}
			frac &= 0x3FF
			f32Bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
		}
	} else if exp == 31 {
		// Infinity or NaN
		f32Bits = (sign << 31) | (0xFF << 23) | (frac << 13)
	} else {
		// Normal number
		f32Bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
	}

	return math.Float32frombits(f32Bits)
}

// Float32ToFP16 converts float32 to IEEE 754 half-precision.
// This handles overflow to infinity and underflow to zero.
func Float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)

	sign := uint16((bits >> 31) & 0x1)
	exp := int((bits >> 23) & 0xFF)
	frac := bits & 0x7FFFFF

	if exp == 0 {
		// Zero or subnormal float32 -> zero in FP16
		return sign << 15
	} else if exp == 0xFF {
		// Infinity or NaN
		if frac == 0 {
			return (sign << 15) | (0x1F << 10)
		}
		return (sign << 15) | (0x1F << 10) | uint16(frac>>13)
	}

	// Normal number
	newExp := exp - 127 + 15
	if newExp <= 0 {
		// Underflow to zero
		return sign << 15
	} else if newExp >= 31 {
		// Overflow to infinity
		return (sign << 15) | (0x1F << 10)
	}

	return (sign << 15) | (uint16(newExp) << 10) | uint16(frac>>13)
}

// FP16ToFloat64 converts IEEE 754 half-precision to double-precision.
// Useful for higher precision calculations.
func FP16ToFloat64(fp16 uint16) float64 {
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
		// Subnormal
		return math.Pow(-1, float64(sign)) * math.Pow(2, -14) * float64(frac) / 1024.0
	}
	if exp == 31 {
		if frac == 0 {
			if sign == 1 {
				return math.Inf(-1)
			}
			return math.Inf(1)
		}
		return math.NaN()
	}

	return math.Pow(-1, float64(sign)) * math.Pow(2, float64(exp)-15) * (1.0 + float64(frac)/1024.0)
}

// BF16ToFloat32 converts brain floating-point (BF16) to single-precision.
// BF16 is simply the top 16 bits of FP32.
func BF16ToFloat32(bf16 uint16) float32 {
	return math.Float32frombits(uint32(bf16) << 16)
}

// Float32ToBits converts float32 to its bit representation.
func Float32ToBits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}

// Float32FromBits reinterprets a uint32 as float32.
func Float32FromBits(bits uint32) float32 {
	return *(*float32)(unsafe.Pointer(&bits))
}

// Float32ToBytes converts a float32 slice to bytes (little-endian).
func Float32ToBytes(data []float32) []byte {
	bytes := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		bytes[i*4] = byte(bits)
		bytes[i*4+1] = byte(bits >> 8)
		bytes[i*4+2] = byte(bits >> 16)
		bytes[i*4+3] = byte(bits >> 24)
	}
	return bytes
}

// Float32ToFP16Bytes converts float32 slice to FP16 bytes (little-endian).
func Float32ToFP16Bytes(data []float32) []byte {
	bytes := make([]byte, len(data)*2)
	for i, v := range data {
		bits := Float32ToFP16(v)
		bytes[i*2] = byte(bits)
		bytes[i*2+1] = byte(bits >> 8)
	}
	return bytes
}

// ClampInt clamps an integer value to the specified range.
func ClampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// ClampFloat32 clamps a float32 value to the specified range.
func ClampFloat32(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
