package compression

import (
	"math"
	"math/rand"
	"testing"

	"github.com/neurogrid/kv-cache-p2p/pkg/safetensors"
)

func TestQualityValidator(t *testing.T) {
	validator := NewDefaultQualityValidator()

	// Generate test data
	numElements := 65536
	original := make([]uint16, numElements)
	reconstructed := make([]uint16, numElements)

	for i := 0; i < numElements; i++ {
		// Random values in [-1, 1] range
		val := rand.Float32()*2 - 1
		original[i] = safetensors.Float32ToFP16(val)

		// Add small quantization error
		quantized := int(math.Round(float64(val) * 7 / 1.0)) // INT4 simulation
		quantized = safetensors.ClampInt(quantized, -8, 7)
		reconstructedVal := float32(quantized) * 1.0 / 7
		reconstructed[i] = safetensors.Float32ToFP16(reconstructedVal)
	}

	// Compute metrics
	metrics, err := validator.ComputeTensorMetrics(original, reconstructed)
	if err != nil {
		t.Fatalf("ComputeTensorMetrics failed: %v", err)
	}

	// Print report
	t.Logf("\n%s", validator.QualityReport(metrics))

	// Validate
	passed, violations := validator.ValidateMetrics(metrics)
	if !passed {
		t.Logf("Quality violations: %v", violations)
	}

	// Check quality level
	level := GetQualityLevel(metrics)
	t.Logf("Quality Level: %s", level)

	// Assertions
	if metrics.RMSE > 0.2 {
		t.Errorf("RMSE too high: %.4f", metrics.RMSE)
	}

	if metrics.CosineSimilarity < 0.9 {
		t.Errorf("Cosine similarity too low: %.4f", metrics.CosineSimilarity)
	}
}

func TestQualityLevels(t *testing.T) {
	testCases := []struct {
		name     string
		rmse     float64
		snr      float64
		expected CompressionQualityLevel
	}{
		{"Excellent", 0.005, 45, QualityExcellent},
		{"Good", 0.03, 35, QualityGood},
		{"Acceptable", 0.08, 25, QualityAcceptable},
		{"Poor", 0.15, 15, QualityPoor},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			metrics := &QualityMetrics{
				RMSE: tc.rmse,
				SNR:  tc.snr,
			}

			level := GetQualityLevel(metrics)
			if level != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, level)
			}
		})
	}
}

func TestFullCompressionQuality(t *testing.T) {
	// Simulate a realistic KV cache compression scenario
	t.Log("=== Full Compression Quality Test ===")

	// Configuration similar to LLaMA 7B
	// GroupSize=64 is the documented default for good quality/compression tradeoff
	config := struct {
		Layers    int
		Heads     int
		SeqLen    int
		HeadDim   int
		GroupSize int
	}{
		Layers:    32,
		Heads:     32,
		SeqLen:    512,
		HeadDim:   128,
		GroupSize: 64, // Default per docs/COMPRESSION.md
	}

	elementsPerLayer := config.Heads * config.SeqLen * config.HeadDim
	totalElements := config.Layers * elementsPerLayer

	t.Logf("Configuration: %d layers, %d heads, %d seq_len, %d head_dim",
		config.Layers, config.Heads, config.SeqLen, config.HeadDim)
	t.Logf("Total elements: %d (%.2f MB FP16)", totalElements, float64(totalElements*2)/(1024*1024))

	// Generate synthetic KV cache data with realistic distribution
	// KV cache values typically follow a roughly Gaussian distribution
	original := make([]uint16, totalElements)
	for i := 0; i < totalElements; i++ {
		// Gaussian-like distribution centered at 0
		val := float32(rand.NormFloat64() * 0.3) // std=0.3
		val = safetensors.ClampFloat32(val, -1, 1)
		original[i] = safetensors.Float32ToFP16(val)
	}

	// Simulate INT4 quantization with per-group scaling
	reconstructed := make([]uint16, totalElements)
	numGroups := (totalElements + config.GroupSize - 1) / config.GroupSize

	for g := 0; g < numGroups; g++ {
		start := g * config.GroupSize
		end := start + config.GroupSize
		if end > totalElements {
			end = totalElements
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

		// Compute scale
		scale := maxVal / 7.0
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

	// Validate quality
	validator := NewDefaultQualityValidator()
	metrics, err := validator.ComputeTensorMetrics(original, reconstructed)
	if err != nil {
		t.Fatalf("Failed to compute metrics: %v", err)
	}

	// Print detailed report
	t.Logf("\n%s", validator.QualityReport(metrics))

	// Check quality
	passed, violations := validator.ValidateMetrics(metrics)
	level := GetQualityLevel(metrics)

	t.Logf("\nQuality Summary:")
	t.Logf("  Level: %s", level)
	t.Logf("  Passed: %v", passed)

	if !passed {
		t.Logf("  Violations: %v", violations)
	}

	// Per-layer quality analysis
	t.Logf("\nPer-Layer Quality Analysis (first 5 layers):")
	for layer := 0; layer < 5 && layer < config.Layers; layer++ {
		start := layer * elementsPerLayer
		end := start + elementsPerLayer

		layerOrig := original[start:end]
		layerRecon := reconstructed[start:end]

		layerMetrics, _ := validator.ComputeTensorMetrics(layerOrig, layerRecon)
		t.Logf("  Layer %d: RMSE=%.6f, CosSim=%.6f, SNR=%.2fdB",
			layer, layerMetrics.RMSE, layerMetrics.CosineSimilarity, layerMetrics.SNR)
	}

	// Quality must be at least acceptable
	if level == QualityPoor {
		t.Errorf("Quality level is POOR, compression may degrade inference")
	}
}

func TestCompressionQualityByGroupSize(t *testing.T) {
	// Test how group size affects quality
	groupSizes := []int{32, 64, 128, 256, 512}
	numElements := 65536

	t.Log("=== Group Size vs Quality Analysis ===")
	t.Logf("%-12s %-10s %-10s %-10s %-15s", "GroupSize", "RMSE", "MaxError", "SNR(dB)", "Quality")

	for _, groupSize := range groupSizes {
		original := make([]uint16, numElements)
		reconstructed := make([]uint16, numElements)

		// Generate data
		for i := 0; i < numElements; i++ {
			val := float32(rand.NormFloat64() * 0.3)
			val = safetensors.ClampFloat32(val, -1, 1)
			original[i] = safetensors.Float32ToFP16(val)
		}

		// Quantize with given group size
		numGroups := (numElements + groupSize - 1) / groupSize
		for g := 0; g < numGroups; g++ {
			start := g * groupSize
			end := start + groupSize
			if end > numElements {
				end = numElements
			}

			var maxVal float32
			for i := start; i < end; i++ {
				val := safetensors.FP16ToFloat32(original[i])
				absVal := float32(math.Abs(float64(val)))
				if absVal > maxVal {
					maxVal = absVal
				}
			}

			scale := maxVal / 7.0
			if scale == 0 {
				scale = 1e-7
			}

			for i := start; i < end; i++ {
				val := safetensors.FP16ToFloat32(original[i])
				q := int(math.Round(float64(val / scale)))
				q = safetensors.ClampInt(q, -8, 7)
				reconstructed[i] = safetensors.Float32ToFP16(float32(q) * scale)
			}
		}

		validator := NewDefaultQualityValidator()
		metrics, _ := validator.ComputeTensorMetrics(original, reconstructed)
		level := GetQualityLevel(metrics)

		t.Logf("%-12d %-10.6f %-10.6f %-10.2f %-15s",
			groupSize, metrics.RMSE, metrics.MaxError, metrics.SNR, level)
	}
}

// BenchmarkQualityValidation benchmarks the quality computation
func BenchmarkQualityValidation(b *testing.B) {
	numElements := 1024 * 1024 // 1M elements
	original := make([]uint16, numElements)
	reconstructed := make([]uint16, numElements)

	for i := 0; i < numElements; i++ {
		original[i] = safetensors.Float32ToFP16(rand.Float32()*2 - 1)
		reconstructed[i] = safetensors.Float32ToFP16(rand.Float32()*2 - 1)
	}

	validator := NewDefaultQualityValidator()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = validator.ComputeTensorMetrics(original, reconstructed)
	}
}
