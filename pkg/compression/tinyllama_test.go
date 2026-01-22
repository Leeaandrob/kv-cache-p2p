package compression

import (
	"math"
	"math/rand"
	"testing"

	"github.com/neurogrid/kv-cache-p2p/pkg/safetensors"
)

// TinyLLaMA Model Configuration
// Reference: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
const (
	TinyLLaMALayers     = 22
	TinyLLaMAHeads      = 32
	TinyLLaMAHeadDim    = 64  // 2048 / 32
	TinyLLaMAHiddenSize = 2048
	TinyLLaMAVocabSize  = 32000
)

// TestTinyLLaMAKVCacheCompression validates compression quality for TinyLLaMA model
func TestTinyLLaMAKVCacheCompression(t *testing.T) {
	t.Log("=== TinyLLaMA KV Cache Compression Validation ===")
	t.Log("")
	t.Log("Model: TinyLlama-1.1B-Chat-v1.0")
	t.Logf("Config: %d layers, %d heads, %d head_dim", TinyLLaMALayers, TinyLLaMAHeads, TinyLLaMAHeadDim)

	// Test multiple sequence lengths
	seqLengths := []int{128, 256, 512, 1024, 2048}

	for _, seqLen := range seqLengths {
		t.Run(formatSeqLen(seqLen), func(t *testing.T) {
			testTinyLLaMACompression(t, seqLen)
		})
	}
}

func formatSeqLen(seqLen int) string {
	if seqLen >= 1024 {
		return string(rune('0'+seqLen/1024)) + "K_tokens"
	}
	return string(rune('0'+seqLen/100)) + "00_tokens"
}

func testTinyLLaMACompression(t *testing.T, seqLen int) {
	// KV cache size for TinyLLaMA
	// Size = layers × heads × seq_len × head_dim × 2 (K+V) × 2 bytes (FP16)
	elementsPerKV := TinyLLaMALayers * TinyLLaMAHeads * seqLen * TinyLLaMAHeadDim
	totalElements := elementsPerKV * 2 // K and V caches
	sizeBytes := totalElements * 2     // FP16

	t.Logf("Sequence length: %d tokens", seqLen)
	t.Logf("KV cache size: %.2f MB (FP16)", float64(sizeBytes)/(1024*1024))

	// Generate synthetic KV cache with realistic distribution
	// KV cache values follow approximately Gaussian distribution after attention
	original := generateTinyLLaMAKVCache(totalElements)

	// Test with different group sizes
	groupSizes := []int{32, 64, 128}

	t.Log("")
	t.Logf("%-10s %-10s %-10s %-12s %-12s %-10s",
		"GroupSize", "RMSE", "SNR(dB)", "CosSim", "Compression", "Quality")

	for _, groupSize := range groupSizes {
		reconstructed := simulateINT4Quantization(original, groupSize)

		validator := NewDefaultQualityValidator()
		metrics, err := validator.ComputeTensorMetrics(original, reconstructed)
		if err != nil {
			t.Fatalf("Failed to compute metrics: %v", err)
		}

		level := GetQualityLevel(metrics)

		// Estimate compression ratio
		// INT4 = 4x compression from FP16
		// With scales overhead: ~3.8x effective
		compressionRatio := estimateCompressionRatio(totalElements, groupSize)

		t.Logf("%-10d %-10.6f %-10.2f %-12.6f %-12.2fx %-10s",
			groupSize, metrics.RMSE, metrics.SNR, metrics.CosineSimilarity,
			compressionRatio, level)

		// Validate quality meets thresholds
		passed, violations := validator.ValidateMetrics(metrics)
		if !passed && groupSize <= 64 {
			t.Errorf("GroupSize %d failed validation: %v", groupSize, violations)
		}
	}
}

// TestTinyLLaMAAttentionPatterns tests compression with realistic attention patterns
func TestTinyLLaMAAttentionPatterns(t *testing.T) {
	t.Log("=== TinyLLaMA Attention Pattern Compression ===")

	// Test with attention-like patterns (sparse, heavy-tailed)
	seqLen := 512
	elementsPerHead := seqLen * TinyLLaMAHeadDim

	// Simulate different attention patterns
	patterns := []struct {
		name      string
		generator func(int) []uint16
	}{
		{"Uniform", generateUniformKVCache},
		{"Gaussian", generateGaussianKVCache},
		{"Heavy-Tailed", generateHeavyTailedKVCache},
		{"Sparse", generateSparseKVCache},
	}

	t.Log("")
	t.Logf("%-15s %-10s %-10s %-12s %-10s",
		"Pattern", "RMSE", "SNR(dB)", "CosSim", "Quality")

	groupSize := 64 // Default
	validator := NewDefaultQualityValidator()

	for _, p := range patterns {
		original := p.generator(elementsPerHead * TinyLLaMAHeads * TinyLLaMALayers)
		reconstructed := simulateINT4Quantization(original, groupSize)

		metrics, _ := validator.ComputeTensorMetrics(original, reconstructed)
		level := GetQualityLevel(metrics)

		t.Logf("%-15s %-10.6f %-10.2f %-12.6f %-10s",
			p.name, metrics.RMSE, metrics.SNR, metrics.CosineSimilarity, level)

		// All patterns should be acceptable with GroupSize=64
		passed, violations := validator.ValidateMetrics(metrics)
		if !passed {
			t.Errorf("Pattern %s failed: %v", p.name, violations)
		}
	}
}

// TestTinyLLaMAPerLayerQuality validates per-layer compression quality
func TestTinyLLaMAPerLayerQuality(t *testing.T) {
	t.Log("=== TinyLLaMA Per-Layer Quality Analysis ===")

	seqLen := 512
	elementsPerLayer := TinyLLaMAHeads * seqLen * TinyLLaMAHeadDim * 2 // K+V
	groupSize := 64

	t.Log("")
	t.Logf("%-8s %-10s %-10s %-12s %-10s",
		"Layer", "RMSE", "SNR(dB)", "CosSim", "Quality")

	validator := NewDefaultQualityValidator()
	var worstLayer int
	var worstSNR float64 = math.Inf(1)

	for layer := 0; layer < TinyLLaMALayers; layer++ {
		// Generate layer-specific KV cache
		// Earlier layers tend to have different distributions than later layers
		original := generateLayerSpecificKVCache(elementsPerLayer, layer, TinyLLaMALayers)
		reconstructed := simulateINT4Quantization(original, groupSize)

		metrics, _ := validator.ComputeTensorMetrics(original, reconstructed)
		level := GetQualityLevel(metrics)

		// Only print every 5th layer to reduce output
		if layer%5 == 0 || layer == TinyLLaMALayers-1 {
			t.Logf("%-8d %-10.6f %-10.2f %-12.6f %-10s",
				layer, metrics.RMSE, metrics.SNR, metrics.CosineSimilarity, level)
		}

		if metrics.SNR < worstSNR {
			worstSNR = metrics.SNR
			worstLayer = layer
		}
	}

	t.Logf("")
	t.Logf("Worst layer: %d (SNR=%.2fdB)", worstLayer, worstSNR)

	// Verify worst layer still meets threshold
	if worstSNR < 18.0 {
		t.Errorf("Worst layer %d has SNR %.2fdB < 18dB threshold", worstLayer, worstSNR)
	}
}

// TestTinyLLaMACompressionSummary provides a summary for documentation
func TestTinyLLaMACompressionSummary(t *testing.T) {
	t.Log("╔══════════════════════════════════════════════════════════╗")
	t.Log("║       TinyLLaMA KV Cache Compression Summary             ║")
	t.Log("╠══════════════════════════════════════════════════════════╣")

	seqLen := 512
	elementsPerKV := TinyLLaMALayers * TinyLLaMAHeads * seqLen * TinyLLaMAHeadDim
	totalElements := elementsPerKV * 2
	originalSize := totalElements * 2 // FP16 bytes

	// Generate realistic data
	original := generateTinyLLaMAKVCache(totalElements)

	// Test with recommended GroupSize=64
	groupSize := 64
	reconstructed := simulateINT4Quantization(original, groupSize)

	validator := NewDefaultQualityValidator()
	metrics, _ := validator.ComputeTensorMetrics(original, reconstructed)
	passed, _ := validator.ValidateMetrics(metrics)
	level := GetQualityLevel(metrics)
	compressionRatio := estimateCompressionRatio(totalElements, groupSize)
	compressedSize := float64(originalSize) / compressionRatio

	status := "✅ PASSED"
	if !passed {
		status = "❌ FAILED"
	}

	t.Logf("║ Model: TinyLlama-1.1B-Chat-v1.0                         ║")
	t.Logf("║ Config: %d layers, %d heads, %d head_dim               ║",
		TinyLLaMALayers, TinyLLaMAHeads, TinyLLaMAHeadDim)
	t.Logf("║ Sequence Length: %d tokens                              ║", seqLen)
	t.Log("╠══════════════════════════════════════════════════════════╣")
	t.Logf("║ Original KV Cache:  %.2f MB                            ║", float64(originalSize)/(1024*1024))
	t.Logf("║ Compressed Size:    %.2f MB                             ║", compressedSize/(1024*1024))
	t.Logf("║ Compression Ratio:  %.2fx                               ║", compressionRatio)
	t.Log("╠══════════════════════════════════════════════════════════╣")
	t.Logf("║ Quality Metrics (GroupSize=%d):                         ║", groupSize)
	t.Logf("║   RMSE:             %.6f                              ║", metrics.RMSE)
	t.Logf("║   SNR:              %.2f dB                            ║", metrics.SNR)
	t.Logf("║   Cosine Sim:       %.6f                              ║", metrics.CosineSimilarity)
	t.Logf("║   Quality Level:    %s                            ║", level)
	t.Log("╠══════════════════════════════════════════════════════════╣")
	t.Logf("║ Validation: %s                                       ║", status)
	t.Log("╚══════════════════════════════════════════════════════════╝")
}

// Helper functions

func generateTinyLLaMAKVCache(numElements int) []uint16 {
	data := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		// KV cache follows roughly Gaussian distribution
		// Centered around 0 with std ~0.3 (typical for normalized hidden states)
		val := float32(rand.NormFloat64() * 0.3)
		val = safetensors.ClampFloat32(val, -1, 1)
		data[i] = safetensors.Float32ToFP16(val)
	}
	return data
}

func generateUniformKVCache(numElements int) []uint16 {
	data := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		val := float32(rand.Float64()*2 - 1)
		data[i] = safetensors.Float32ToFP16(val)
	}
	return data
}

func generateGaussianKVCache(numElements int) []uint16 {
	return generateTinyLLaMAKVCache(numElements)
}

func generateHeavyTailedKVCache(numElements int) []uint16 {
	data := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		// Heavy-tailed distribution (more outliers)
		val := float32(rand.NormFloat64() * 0.5)
		if rand.Float32() < 0.1 {
			val *= 2 // 10% chance of outlier
		}
		val = safetensors.ClampFloat32(val, -1, 1)
		data[i] = safetensors.Float32ToFP16(val)
	}
	return data
}

func generateSparseKVCache(numElements int) []uint16 {
	data := make([]uint16, numElements)
	for i := 0; i < numElements; i++ {
		if rand.Float32() < 0.3 {
			// 30% non-zero values
			val := float32(rand.NormFloat64() * 0.4)
			val = safetensors.ClampFloat32(val, -1, 1)
			data[i] = safetensors.Float32ToFP16(val)
		} else {
			data[i] = safetensors.Float32ToFP16(0)
		}
	}
	return data
}

func generateLayerSpecificKVCache(numElements, layer, totalLayers int) []uint16 {
	data := make([]uint16, numElements)

	// Earlier layers tend to have larger magnitudes
	// Later layers are more focused/sparse
	layerProgress := float64(layer) / float64(totalLayers)
	std := 0.4 - layerProgress*0.15 // 0.4 -> 0.25 across layers

	for i := 0; i < numElements; i++ {
		val := float32(rand.NormFloat64() * std)
		val = safetensors.ClampFloat32(val, -1, 1)
		data[i] = safetensors.Float32ToFP16(val)
	}
	return data
}

func simulateINT4Quantization(original []uint16, groupSize int) []uint16 {
	reconstructed := make([]uint16, len(original))
	numGroups := (len(original) + groupSize - 1) / groupSize

	for g := 0; g < numGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > len(original) {
			end = len(original)
		}

		// Find max absolute value in group
		var maxVal float32
		for i := start; i < end; i++ {
			val := safetensors.FP16ToFloat32(original[i])
			absVal := float32(math.Abs(float64(val)))
			if absVal > maxVal {
				maxVal = absVal
			}
		}

		// Compute scale for symmetric quantization
		scale := maxVal / 7.0 // INT4 range is [-8, 7]
		if scale == 0 {
			scale = 1e-7
		}

		// Quantize and dequantize
		for i := start; i < end; i++ {
			val := safetensors.FP16ToFloat32(original[i])
			q := int(math.Round(float64(val / scale)))
			q = safetensors.ClampInt(q, -8, 7)
			reconstructed[i] = safetensors.Float32ToFP16(float32(q) * scale)
		}
	}

	return reconstructed
}

func estimateCompressionRatio(numElements, groupSize int) float64 {
	// Original: FP16 = 2 bytes per element
	originalBytes := numElements * 2

	// Compressed: INT4 = 0.5 bytes per element + scales
	// Scales: FP16 = 2 bytes per group
	numGroups := (numElements + groupSize - 1) / groupSize
	compressedBytes := numElements/2 + numGroups*2

	return float64(originalBytes) / float64(compressedBytes)
}

// BenchmarkTinyLLaMACompression benchmarks compression for TinyLLaMA
func BenchmarkTinyLLaMACompression(b *testing.B) {
	seqLen := 512
	elementsPerKV := TinyLLaMALayers * TinyLLaMAHeads * seqLen * TinyLLaMAHeadDim
	totalElements := elementsPerKV * 2

	original := generateTinyLLaMAKVCache(totalElements)
	groupSize := 64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = simulateINT4Quantization(original, groupSize)
	}
}
