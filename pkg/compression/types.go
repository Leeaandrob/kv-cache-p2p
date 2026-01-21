package compression

import "errors"

// Errors
var (
	ErrInvalidInput    = errors.New("invalid input data")
	ErrQuantizeFailed  = errors.New("quantization failed")
	ErrSparsifyFailed  = errors.New("sparsification failed")
	ErrCompressFailed  = errors.New("compression failed")
	ErrDecompressFailed = errors.New("decompression failed")
)

// QuantizationConfig holds INT4 quantization parameters
type QuantizationConfig struct {
	// GroupSize is the number of values per quantization group (typically 64 or 128)
	GroupSize int
	// Symmetric uses symmetric quantization around zero
	Symmetric bool
}

// DefaultQuantizationConfig returns sensible defaults for KV cache
func DefaultQuantizationConfig() QuantizationConfig {
	return QuantizationConfig{
		GroupSize: 128, // Good balance between quality and overhead
		Symmetric: true,
	}
}

// SparsityConfig holds sparsification parameters
type SparsityConfig struct {
	// TopK percentage of attention scores to keep (0.0-1.0)
	TopK float32
	// MinTokens minimum tokens to always keep (for stability)
	MinTokens int
}

// DefaultSparsityConfig returns sensible defaults
func DefaultSparsityConfig() SparsityConfig {
	return SparsityConfig{
		TopK:      0.5,  // Keep top 50% of attention scores
		MinTokens: 4,    // Always keep at least 4 tokens
	}
}

// DeltaConfig holds delta encoding parameters
type DeltaConfig struct {
	// Enabled turns delta encoding on/off
	Enabled bool
	// BaseKey identifies the reference cache for delta computation
	BaseKey string
}

// NetworkCompressionConfig holds LZ4/zstd parameters
type NetworkCompressionConfig struct {
	// Algorithm: "lz4", "zstd", or "none"
	Algorithm string
	// Level: compression level (1-12 for zstd, 1-9 for lz4hc)
	Level int
}

// DefaultNetworkCompressionConfig returns LZ4 fast compression
func DefaultNetworkCompressionConfig() NetworkCompressionConfig {
	return NetworkCompressionConfig{
		Algorithm: "lz4",
		Level:     1, // Fast compression
	}
}

// CompressionPipeline combines all compression stages
type CompressionPipeline struct {
	Quantization       QuantizationConfig
	Sparsity           SparsityConfig
	Delta              DeltaConfig
	NetworkCompression NetworkCompressionConfig
}

// DefaultPipeline returns the recommended compression pipeline
func DefaultPipeline() CompressionPipeline {
	return CompressionPipeline{
		Quantization:       DefaultQuantizationConfig(),
		Sparsity:           DefaultSparsityConfig(),
		Delta:              DeltaConfig{Enabled: false}, // Disabled by default
		NetworkCompression: DefaultNetworkCompressionConfig(),
	}
}

// QuantizedTensor holds INT4 quantized data with metadata
type QuantizedTensor struct {
	// Data holds packed INT4 values (2 values per byte)
	Data []byte
	// Scales holds per-group scaling factors (FP16)
	Scales []uint16
	// ZeroPoints holds per-group zero points (INT4, packed)
	ZeroPoints []byte
	// Shape original tensor shape
	Shape []int
	// GroupSize used for quantization
	GroupSize int
}

// SparseTensor holds sparsified attention data
type SparseTensor struct {
	// Values holds the non-zero values (quantized)
	Values []byte
	// Indices holds the positions of non-zero values
	Indices []uint32
	// Shape original tensor shape
	Shape []int
	// Density actual density after sparsification
	Density float32
}

// CompressedKVCache is the final compressed format for transfer
type CompressedKVCache struct {
	// Keys compressed key tensors per layer
	Keys [][]byte
	// Values compressed value tensors per layer
	Values [][]byte
	// Metadata for decompression
	Metadata CompressionMetadata
}

// CompressionMetadata stores info needed for decompression
type CompressionMetadata struct {
	// OriginalShape [layers, heads, seq_len, head_dim]
	OriginalShape []int
	// OriginalDtype "fp16", "bf16", "fp32"
	OriginalDtype string
	// Pipeline configuration used
	Pipeline CompressionPipeline
	// CompressionRatio achieved ratio
	CompressionRatio float32
	// Checksum for integrity verification
	Checksum uint32
}

// CompressionStats holds metrics from compression
type CompressionStats struct {
	OriginalSize     int64
	QuantizedSize    int64
	SparsifiedSize   int64
	CompressedSize   int64
	QuantizeTimeMs   float64
	SparsifyTimeMs   float64
	CompressTimeMs   float64
	TotalTimeMs      float64
	CompressionRatio float32
}
