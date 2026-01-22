package compression

import (
	"fmt"
	"math"

	"github.com/neurogrid/kv-cache-p2p/pkg/safetensors"
)

// QualityMetrics holds all quality measurements
type QualityMetrics struct {
	// Tensor-level metrics
	MSE      float64 // Mean Squared Error
	RMSE     float64 // Root Mean Squared Error
	MaxError float64 // Maximum absolute error
	MAE      float64 // Mean Absolute Error
	SNR      float64 // Signal-to-Noise Ratio (dB)
	PSNR     float64 // Peak Signal-to-Noise Ratio (dB)

	// Distribution metrics
	CosineSimilarity float64 // Cosine similarity between original and reconstructed
	KLDivergence     float64 // KL divergence (if probability distributions)

	// Inference-level metrics (populated after inference test)
	TokenMatchRate   float64 // Percentage of matching tokens
	PerplexityDelta  float64 // Change in perplexity
	Top1Accuracy     float64 // Top-1 prediction accuracy
	Top5Accuracy     float64 // Top-5 prediction accuracy
}

// QualityThresholds defines acceptable quality limits
type QualityThresholds struct {
	MaxRMSE            float64
	MaxMaxError        float64
	MinCosineSimilarity float64
	MinTokenMatchRate  float64
	MaxPerplexityDelta float64
	MinSNR             float64
}

// DefaultQualityThresholds returns thresholds based on research
// References: KVQuant, KIVI, GPTQ papers show 18dB SNR is typical for INT4
func DefaultQualityThresholds() QualityThresholds {
	return QualityThresholds{
		MaxRMSE:             0.1,   // Based on KVQuant findings
		MaxMaxError:         0.5,   // Allow some outliers
		MinCosineSimilarity: 0.99,  // Very high similarity required
		MinTokenMatchRate:   0.95,  // 95% token match
		MaxPerplexityDelta:  0.5,   // Max 0.5 perplexity increase (KVQuant < 0.1)
		MinSNR:              18.0,  // 18dB typical for 4-bit quantization
	}
}

// QualityValidator validates compression quality
type QualityValidator struct {
	thresholds QualityThresholds
}

// NewQualityValidator creates a validator with given thresholds
func NewQualityValidator(thresholds QualityThresholds) *QualityValidator {
	return &QualityValidator{thresholds: thresholds}
}

// NewDefaultQualityValidator creates a validator with default thresholds
func NewDefaultQualityValidator() *QualityValidator {
	return NewQualityValidator(DefaultQualityThresholds())
}

// ComputeTensorMetrics computes quality metrics from FP16 tensors
func (v *QualityValidator) ComputeTensorMetrics(original, reconstructed []uint16) (*QualityMetrics, error) {
	if len(original) != len(reconstructed) {
		return nil, fmt.Errorf("tensor size mismatch: %d vs %d", len(original), len(reconstructed))
	}

	n := len(original)
	if n == 0 {
		return nil, fmt.Errorf("empty tensors")
	}

	var sumSqErr, sumAbsErr, maxErr float64
	var sumOrig, sumRecon float64
	var dotProduct, normOrig, normRecon float64

	for i := 0; i < n; i++ {
		orig := safetensors.FP16ToFloat64(original[i])
		recon := safetensors.FP16ToFloat64(reconstructed[i])

		diff := orig - recon
		sumSqErr += diff * diff
		sumAbsErr += math.Abs(diff)

		if math.Abs(diff) > maxErr {
			maxErr = math.Abs(diff)
		}

		sumOrig += orig
		sumRecon += recon

		// For cosine similarity
		dotProduct += orig * recon
		normOrig += orig * orig
		normRecon += recon * recon
	}

	mse := sumSqErr / float64(n)
	rmse := math.Sqrt(mse)
	mae := sumAbsErr / float64(n)

	// Signal-to-Noise Ratio
	var signalPower float64
	meanOrig := sumOrig / float64(n)
	for i := 0; i < n; i++ {
		orig := safetensors.FP16ToFloat64(original[i])
		signalPower += (orig - meanOrig) * (orig - meanOrig)
	}
	signalPower /= float64(n)

	var snr, psnr float64
	if mse > 0 {
		snr = 10 * math.Log10(signalPower/mse)
		// PSNR using max value of 1.0 for normalized data
		psnr = 10 * math.Log10(1.0/mse)
	} else {
		snr = math.Inf(1)
		psnr = math.Inf(1)
	}

	// Cosine similarity
	var cosineSim float64
	if normOrig > 0 && normRecon > 0 {
		cosineSim = dotProduct / (math.Sqrt(normOrig) * math.Sqrt(normRecon))
	}

	return &QualityMetrics{
		MSE:              mse,
		RMSE:             rmse,
		MaxError:         maxErr,
		MAE:              mae,
		SNR:              snr,
		PSNR:             psnr,
		CosineSimilarity: cosineSim,
	}, nil
}

// ValidateMetrics checks if metrics meet thresholds
func (v *QualityValidator) ValidateMetrics(metrics *QualityMetrics) (bool, []string) {
	var violations []string

	if metrics.RMSE > v.thresholds.MaxRMSE {
		violations = append(violations, fmt.Sprintf("RMSE %.4f > %.4f", metrics.RMSE, v.thresholds.MaxRMSE))
	}

	if metrics.MaxError > v.thresholds.MaxMaxError {
		violations = append(violations, fmt.Sprintf("MaxError %.4f > %.4f", metrics.MaxError, v.thresholds.MaxMaxError))
	}

	if metrics.CosineSimilarity < v.thresholds.MinCosineSimilarity {
		violations = append(violations, fmt.Sprintf("CosineSim %.4f < %.4f", metrics.CosineSimilarity, v.thresholds.MinCosineSimilarity))
	}

	if metrics.SNR < v.thresholds.MinSNR {
		violations = append(violations, fmt.Sprintf("SNR %.2fdB < %.2fdB", metrics.SNR, v.thresholds.MinSNR))
	}

	// Inference metrics (if available)
	if metrics.TokenMatchRate > 0 && metrics.TokenMatchRate < v.thresholds.MinTokenMatchRate {
		violations = append(violations, fmt.Sprintf("TokenMatch %.2f%% < %.2f%%",
			metrics.TokenMatchRate*100, v.thresholds.MinTokenMatchRate*100))
	}

	if metrics.PerplexityDelta > v.thresholds.MaxPerplexityDelta {
		violations = append(violations, fmt.Sprintf("PerplexityDelta %.4f > %.4f",
			metrics.PerplexityDelta, v.thresholds.MaxPerplexityDelta))
	}

	return len(violations) == 0, violations
}

// QualityReport generates a human-readable quality report
func (v *QualityValidator) QualityReport(metrics *QualityMetrics) string {
	passed, violations := v.ValidateMetrics(metrics)

	status := "✅ PASSED"
	if !passed {
		status = "❌ FAILED"
	}

	report := fmt.Sprintf(`
╔══════════════════════════════════════════════════════════╗
║            KV Cache Compression Quality Report           ║
╠══════════════════════════════════════════════════════════╣
║ Status: %s                                         ║
╠══════════════════════════════════════════════════════════╣
║ Tensor Metrics:                                          ║
║   MSE:              %.6f                              ║
║   RMSE:             %.6f (threshold: < %.2f)         ║
║   Max Error:        %.6f (threshold: < %.2f)         ║
║   MAE:              %.6f                              ║
║   SNR:              %.2f dB (threshold: > %.0f dB)     ║
║   PSNR:             %.2f dB                            ║
║   Cosine Sim:       %.6f (threshold: > %.2f)         ║
╠══════════════════════════════════════════════════════════╣`,
		status,
		metrics.MSE,
		metrics.RMSE, v.thresholds.MaxRMSE,
		metrics.MaxError, v.thresholds.MaxMaxError,
		metrics.MAE,
		metrics.SNR, v.thresholds.MinSNR,
		metrics.PSNR,
		metrics.CosineSimilarity, v.thresholds.MinCosineSimilarity,
	)

	if metrics.TokenMatchRate > 0 {
		report += fmt.Sprintf(`
║ Inference Metrics:                                       ║
║   Token Match:      %.2f%% (threshold: > %.0f%%)          ║
║   Perplexity Δ:     %.4f (threshold: < %.2f)           ║
║   Top-1 Accuracy:   %.2f%%                               ║
║   Top-5 Accuracy:   %.2f%%                               ║
╠══════════════════════════════════════════════════════════╣`,
			metrics.TokenMatchRate*100, v.thresholds.MinTokenMatchRate*100,
			metrics.PerplexityDelta, v.thresholds.MaxPerplexityDelta,
			metrics.Top1Accuracy*100,
			metrics.Top5Accuracy*100,
		)
	}

	if !passed {
		report += "\n║ Violations:                                              ║\n"
		for _, v := range violations {
			report += fmt.Sprintf("║   - %-52s ║\n", v)
		}
	}

	report += "╚══════════════════════════════════════════════════════════╝"

	return report
}

// CompressionQualityLevel represents quality grades
type CompressionQualityLevel string

const (
	QualityExcellent  CompressionQualityLevel = "EXCELLENT"  // RMSE < 0.01, SNR > 40dB
	QualityGood       CompressionQualityLevel = "GOOD"       // RMSE < 0.05, SNR > 30dB
	QualityAcceptable CompressionQualityLevel = "ACCEPTABLE" // RMSE < 0.1, SNR > 18dB
	QualityPoor       CompressionQualityLevel = "POOR"       // RMSE >= 0.1 or SNR < 18dB
)

// GetQualityLevel returns quality grade based on metrics
// Based on research: 18dB SNR is typical for INT4 quantization
func GetQualityLevel(metrics *QualityMetrics) CompressionQualityLevel {
	if metrics.RMSE < 0.01 && metrics.SNR > 40 {
		return QualityExcellent
	}
	if metrics.RMSE < 0.05 && metrics.SNR > 30 {
		return QualityGood
	}
	if metrics.RMSE < 0.1 && metrics.SNR > 18 {
		return QualityAcceptable
	}
	return QualityPoor
}
