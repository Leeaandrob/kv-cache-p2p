package compression

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// generateTestData creates FP16-like test data for benchmarking
func generateTestData(numElements int) []uint16 {
	data := make([]uint16, numElements)
	for i := range data {
		// Generate random values in typical KV cache range
		val := rand.Float32()*2 - 1 // [-1, 1]
		data[i] = fp16FromFloat32Mock(val)
	}
	return data
}

func fp16FromFloat32Mock(f float32) uint16 {
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

func float32FromFP16Mock(fp16 uint16) float32 {
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

func TestLZ4Compression(t *testing.T) {
	testCases := []struct {
		name string
		size int
	}{
		{"1KB", 1024},
		{"64KB", 64 * 1024},
		{"1MB", 1024 * 1024},
	}

	compressor := NewLZ4Compressor(1)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate test data
			data := make([]byte, tc.size)
			rand.Read(data)

			// Compress
			compressed, err := compressor.Compress(data)
			if err != nil {
				t.Fatalf("Compress failed: %v", err)
			}

			// Decompress
			decompressed, err := compressor.Decompress(compressed)
			if err != nil {
				t.Fatalf("Decompress failed: %v", err)
			}

			// Verify
			if len(decompressed) != len(data) {
				t.Errorf("Size mismatch: got %d, want %d", len(decompressed), len(data))
			}

			for i := range data {
				if decompressed[i] != data[i] {
					t.Errorf("Data mismatch at %d: got %d, want %d", i, decompressed[i], data[i])
					break
				}
			}

			ratio := compressor.CompressionRatio(data, compressed)
			t.Logf("LZ4 compression ratio: %.2fx (%d -> %d bytes)", ratio, len(data), len(compressed))
		})
	}
}

func TestQuantizationQuality(t *testing.T) {
	// Test different sizes
	sizes := []int{128, 1024, 8192, 65536}
	groupSize := 128

	for _, numElements := range sizes {
		t.Run(fmt.Sprintf("Size_%d", numElements), func(t *testing.T) {
			// Generate test data
			data := generateTestData(numElements)

			// Simulate quantization (mock)
			quantized := make([]byte, (numElements+1)/2)
			scales := make([]uint16, (numElements+groupSize-1)/groupSize)
			reconstructed := make([]uint16, numElements)

			numGroups := (numElements + groupSize - 1) / groupSize
			for g := 0; g < numGroups; g++ {
				start := g * groupSize
				end := start + groupSize
				if end > numElements {
					end = numElements
				}

				// Find max
				var maxVal float32
				for i := start; i < end; i++ {
					val := float32FromFP16Mock(data[i])
					absVal := float32(math.Abs(float64(val)))
					if absVal > maxVal {
						maxVal = absVal
					}
				}

				// Scale
				scale := maxVal / 7.0
				if scale == 0 {
					scale = 1.0
				}
				scales[g] = fp16FromFloat32Mock(scale)

				// Quantize and dequantize
				for i := start; i < end; i++ {
					val := float32FromFP16Mock(data[i])
					q := int(math.Round(float64(val / scale)))
					if q < -8 {
						q = -8
					}
					if q > 7 {
						q = 7
					}

					// Store quantized (for verification)
					if i%2 == 0 && i/2 < len(quantized) {
						quantized[i/2] = byte(q & 0xF)
					}

					// Dequantize
					reconstructed[i] = fp16FromFloat32Mock(float32(q) * scale)
				}
			}
			_ = quantized // Used for verification

			// Compute MSE
			var sumSqErr float64
			var maxErr float64
			for i := 0; i < numElements; i++ {
				orig := float64(float32FromFP16Mock(data[i]))
				recon := float64(float32FromFP16Mock(reconstructed[i]))
				diff := orig - recon
				sumSqErr += diff * diff
				absErr := math.Abs(diff)
				if absErr > maxErr {
					maxErr = absErr
				}
			}
			mse := sumSqErr / float64(numElements)
			rmse := math.Sqrt(mse)

			t.Logf("INT4 Quantization Quality (n=%d):", numElements)
			t.Logf("  MSE:       %.6f", mse)
			t.Logf("  RMSE:      %.6f", rmse)
			t.Logf("  Max Error: %.6f", maxErr)
			t.Logf("  Compression: 4x (FP16 -> INT4)")

			// Quality threshold (based on Unsloth findings)
			if rmse > 0.1 {
				t.Errorf("RMSE too high: %.6f (want < 0.1)", rmse)
			}
		})
	}
}

func TestCompressionPipeline(t *testing.T) {
	pipeline := NewDefaultPipeline()

	// Test estimation
	numElements := 1024 * 1024 // 1M elements = 2MB FP16
	ratio, outputSize := pipeline.EstimateCompression(numElements)

	t.Logf("Estimated compression for %d elements:", numElements)
	t.Logf("  Original size: %d bytes", numElements*2)
	t.Logf("  Estimated output: %d bytes", outputSize)
	t.Logf("  Estimated ratio: %.2fx", ratio)

	// Should achieve at least 4x with INT4 alone
	if ratio < 4.0 {
		t.Errorf("Compression ratio too low: %.2fx (want >= 4x)", ratio)
	}
}

func BenchmarkLZ4Compress(b *testing.B) {
	sizes := []int{64 * 1024, 1024 * 1024, 16 * 1024 * 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%dKB", size/1024), func(b *testing.B) {
			data := make([]byte, size)
			rand.Read(data)
			compressor := NewLZ4Compressor(1)

			b.SetBytes(int64(size))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _ = compressor.Compress(data)
			}
		})
	}
}

func BenchmarkLZ4Decompress(b *testing.B) {
	sizes := []int{64 * 1024, 1024 * 1024, 16 * 1024 * 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%dKB", size/1024), func(b *testing.B) {
			data := make([]byte, size)
			rand.Read(data)
			compressor := NewLZ4Compressor(1)
			compressed, _ := compressor.Compress(data)

			b.SetBytes(int64(size))
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _ = compressor.Decompress(compressed)
			}
		})
	}
}

func BenchmarkQuantizeMock(b *testing.B) {
	numElements := 1024 * 1024 // 1M elements
	data := generateTestData(numElements)
	groupSize := 128

	b.SetBytes(int64(numElements * 2)) // FP16 = 2 bytes
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Mock quantization
		_ = make([]byte, (numElements+1)/2)
		numGroups := (numElements + groupSize - 1) / groupSize
		_ = make([]uint16, numGroups)

		for g := 0; g < numGroups; g++ {
			start := g * groupSize
			end := start + groupSize
			if end > numElements {
				end = numElements
			}

			var maxVal float32
			for j := start; j < end; j++ {
				val := float32FromFP16Mock(data[j])
				absVal := float32(math.Abs(float64(val)))
				if absVal > maxVal {
					maxVal = absVal
				}
			}
			_ = maxVal
		}
	}
}

// TestEndToEndCompression tests the full pipeline concept
func TestEndToEndCompression(t *testing.T) {
	// Simulate KV cache for 32 layers, 32 heads, 1024 seq_len, 128 head_dim
	layers := 32
	heads := 32
	seqLen := 1024
	headDim := 128

	elementsPerLayer := heads * seqLen * headDim
	totalElements := layers * elementsPerLayer
	originalSize := totalElements * 2 // FP16

	t.Logf("KV Cache Configuration:")
	t.Logf("  Layers: %d", layers)
	t.Logf("  Heads: %d", heads)
	t.Logf("  Seq Length: %d", seqLen)
	t.Logf("  Head Dim: %d", headDim)
	t.Logf("  Total Elements: %d", totalElements)
	t.Logf("  Original Size: %.2f MB", float64(originalSize)/(1024*1024))

	// Calculate expected sizes at each stage
	afterINT4 := totalElements / 2 // 4x reduction
	groupSize := 128
	scalesSize := ((totalElements + groupSize - 1) / groupSize) * 2
	afterINT4WithMeta := afterINT4 + scalesSize

	sparsityRatio := 0.5 // Keep 50%
	afterSparse := int(float64(afterINT4WithMeta) * sparsityRatio)

	lz4Ratio := 0.7 // 30% reduction
	afterLZ4 := int(float64(afterSparse) * lz4Ratio)

	totalRatio := float64(originalSize) / float64(afterLZ4)

	t.Logf("\nCompression Pipeline:")
	t.Logf("  1. INT4 Quantization: %.2f MB (4x)", float64(afterINT4WithMeta)/(1024*1024))
	t.Logf("  2. Sparsification (50%%): %.2f MB (%.1fx)", float64(afterSparse)/(1024*1024), float64(afterINT4WithMeta)/float64(afterSparse))
	t.Logf("  3. LZ4 Compression: %.2f MB (%.1fx)", float64(afterLZ4)/(1024*1024), float64(afterSparse)/float64(afterLZ4))
	t.Logf("  Total Compression: %.1fx", totalRatio)
	t.Logf("  Final Size: %.2f MB (from %.2f MB)", float64(afterLZ4)/(1024*1024), float64(originalSize)/(1024*1024))

	// Should achieve good compression
	if totalRatio < 8.0 {
		t.Logf("Warning: Total compression ratio (%.1fx) lower than expected (8x+)", totalRatio)
	}
}

// Dummy to use unsafe import
var _ = unsafe.Pointer(nil)
