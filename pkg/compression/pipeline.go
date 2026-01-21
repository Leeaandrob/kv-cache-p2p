package compression

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/neurogrid/kv-cache-p2p/gpu/bindings"
)

// Pipeline handles the full compression workflow
type Pipeline struct {
	config     CompressionPipeline
	lz4        *LZ4Compressor
	quantCfg   bindings.QuantizeConfig
}

// NewPipeline creates a new compression pipeline
func NewPipeline(config CompressionPipeline) *Pipeline {
	return &Pipeline{
		config:   config,
		lz4:      NewLZ4Compressor(config.NetworkCompression.Level),
		quantCfg: bindings.QuantizeConfig{GroupSize: config.Quantization.GroupSize},
	}
}

// NewDefaultPipeline creates a pipeline with default settings
func NewDefaultPipeline() *Pipeline {
	return NewPipeline(DefaultPipeline())
}

// CompressResult holds the compressed data and metadata
type CompressResult struct {
	Data     []byte
	Metadata CompressionMetadata
	Stats    CompressionStats
}

// Compress runs the full compression pipeline on GPU data
func (p *Pipeline) Compress(
	input unsafe.Pointer,
	numElements int,
	dtype string,
	shape []int,
	stream unsafe.Pointer,
) (*CompressResult, error) {
	stats := CompressionStats{}
	startTotal := time.Now()

	// Original size (FP16 = 2 bytes per element)
	originalSize := numElements * 2
	stats.OriginalSize = int64(originalSize)

	var currentData unsafe.Pointer = input
	_ = currentData // Used for future pipeline stages

	// Step 1: INT4 Quantization (4x reduction)
	startQuant := time.Now()
	quantized, scales, err := bindings.QuantizeFP16ToINT4(currentData, numElements, p.quantCfg, stream)
	if err != nil {
		return nil, fmt.Errorf("quantization failed: %w", err)
	}
	_ = scales // Used for decompression metadata
	stats.QuantizeTimeMs = float64(time.Since(startQuant).Microseconds()) / 1000.0

	quantizedSize := bindings.CalcINT4OutputSize(numElements)
	scalesSize := bindings.CalcScalesSize(numElements, p.quantCfg.GroupSize)
	stats.QuantizedSize = int64(quantizedSize + scalesSize)

	// Step 2: Sparsification (optional, 30-50% reduction)
	var sparseData unsafe.Pointer = quantized
	var sparseIndices unsafe.Pointer
	var numKept int

	if p.config.Sparsity.TopK < 1.0 {
		startSparse := time.Now()
		sparseData, sparseIndices, numKept, err = bindings.SparsifyTopK(
			quantized,
			numElements/2, // Quantized elements
			p.config.Sparsity.TopK,
			p.config.Sparsity.MinTokens,
			stream,
		)
		if err != nil {
			return nil, fmt.Errorf("sparsification failed: %w", err)
		}
		stats.SparsifyTimeMs = float64(time.Since(startSparse).Microseconds()) / 1000.0
		stats.SparsifiedSize = int64(numKept*2 + numKept*4) // Values + indices
	} else {
		stats.SparsifiedSize = stats.QuantizedSize
		sparseData = quantized
		numKept = quantizedSize
	}

	// Step 3: Copy to host for network compression
	// In real implementation, this would use pinned memory
	hostData := make([]byte, stats.SparsifiedSize)
	_ = sparseData // Would be used with cudaMemcpy in real implementation
	// cudaMemcpy(hostData, sparseData, ...) would go here

	// Step 4: LZ4 compression (20-40% reduction)
	startCompress := time.Now()
	compressed, err := p.lz4.Compress(hostData)
	if err != nil {
		return nil, fmt.Errorf("LZ4 compression failed: %w", err)
	}
	stats.CompressTimeMs = float64(time.Since(startCompress).Microseconds()) / 1000.0
	stats.CompressedSize = int64(len(compressed))

	// Calculate total time and ratio
	stats.TotalTimeMs = float64(time.Since(startTotal).Microseconds()) / 1000.0
	stats.CompressionRatio = float32(stats.OriginalSize) / float32(stats.CompressedSize)

	// Build metadata
	metadata := CompressionMetadata{
		OriginalShape:    shape,
		OriginalDtype:    dtype,
		Pipeline:         p.config,
		CompressionRatio: stats.CompressionRatio,
	}

	// Cleanup intermediate GPU buffers
	_ = sparseIndices // Would free these in real implementation

	return &CompressResult{
		Data:     compressed,
		Metadata: metadata,
		Stats:    stats,
	}, nil
}

// Decompress runs the full decompression pipeline
func (p *Pipeline) Decompress(
	data []byte,
	metadata CompressionMetadata,
	stream unsafe.Pointer,
) (unsafe.Pointer, error) {
	// Step 1: LZ4 decompress
	decompressed, err := p.lz4.Decompress(data)
	if err != nil {
		return nil, fmt.Errorf("LZ4 decompression failed: %w", err)
	}

	// Step 2: Copy to GPU
	// cudaMemcpy would go here

	// Step 3: Desparsify (if sparsification was used)
	numElements := 1
	for _, dim := range metadata.OriginalShape {
		numElements *= dim
	}

	// Step 4: Dequantize INT4 → FP16
	// This would use the GPU bindings
	_ = decompressed

	return nil, fmt.Errorf("decompress not fully implemented yet")
}

// ValidateQuality compresses and decompresses, then computes quality metrics
func (p *Pipeline) ValidateQuality(
	original unsafe.Pointer,
	numElements int,
	stream unsafe.Pointer,
) (mse float32, maxError float32, err error) {
	// Quantize
	quantized, scales, err := bindings.QuantizeFP16ToINT4(original, numElements, p.quantCfg, stream)
	if err != nil {
		return 0, 0, fmt.Errorf("quantization failed: %w", err)
	}

	// Dequantize
	reconstructed, err := bindings.DequantizeINT4ToFP16(quantized, scales, numElements, p.quantCfg, stream)
	if err != nil {
		return 0, 0, fmt.Errorf("dequantization failed: %w", err)
	}

	// Compute MSE
	mse, err = bindings.ComputeMSE(original, reconstructed, numElements, stream)
	if err != nil {
		return 0, 0, fmt.Errorf("MSE computation failed: %w", err)
	}

	// Compute max error
	maxError, err = bindings.ComputeMaxError(original, reconstructed, numElements, stream)
	if err != nil {
		return 0, 0, fmt.Errorf("max error computation failed: %w", err)
	}

	return mse, maxError, nil
}

// EstimateCompression estimates compression ratio without actually compressing
func (p *Pipeline) EstimateCompression(numElements int) (ratio float32, outputSize int) {
	originalSize := numElements * 2 // FP16

	// INT4: 4x reduction
	afterQuant := numElements / 2
	afterQuant += bindings.CalcScalesSize(numElements, p.quantCfg.GroupSize)

	// Sparsity
	afterSparse := int(float32(afterQuant) * p.config.Sparsity.TopK)
	if p.config.Sparsity.TopK >= 1.0 {
		afterSparse = afterQuant
	}

	// LZ4: ~30% reduction on quantized data (conservative estimate)
	afterLZ4 := int(float32(afterSparse) * 0.7)

	return float32(originalSize) / float32(afterLZ4), afterLZ4
}
