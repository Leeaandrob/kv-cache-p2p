# ADR-001: Cross-Language Architecture via Safetensors and CLI Subprocess

## Status
Accepted

## Date
2026-01-22

## Context
The TinyLLaMA Perplexity Validation system requires integration between:
- **Python**: For ML inference, model loading, KV cache extraction (HuggingFace Transformers ecosystem)
- **Go**: For high-performance INT4 quantization, compression pipeline, and quality metrics

We needed to establish a communication mechanism between these two language ecosystems that would:
1. Support efficient binary tensor data exchange
2. Avoid complex build dependencies (CGO, Python C extensions)
3. Enable independent testing and development of each component
4. Maintain type safety for numerical precision

## Decision
Implement cross-language communication using:
1. **Safetensors file format** for binary tensor interoperability
2. **Go CLI subprocess calls** from Python for compression operations
3. **JSON output** from Go CLI for metrics and status reporting

### Key Design Choices

#### Safetensors Format
- Uses `safetensors` file format (HuggingFace standard)
- Supports F16, F32, BF16 data types natively
- Little-endian byte order for cross-platform compatibility
- Header contains tensor metadata (shape, dtype, offsets)

#### CLI Interface
```bash
# Compression
kv-compress -input kv_cache.safetensors -output compressed.safetensors -group-size 64

# Metrics only
kv-compress -input kv_cache.safetensors -metrics-only

# Decompression
kv-compress -decompress -input compressed.safetensors -output decompressed.safetensors
```

#### JSON Output Protocol
```json
{
  "compression_ratio": 3.76,
  "original_size_bytes": 1048576,
  "compressed_size_bytes": 278836,
  "rmse": 0.045,
  "snr_db": 25.2,
  "cosine_similarity": 0.9995,
  "quality_passed": true
}
```

## Alternatives Considered

### Alternative 1: CGO with Python C Extension
- **Pros**: Direct memory sharing, no serialization overhead
- **Cons**:
  - Complex build requirements (CGO, Python headers)
  - Platform-specific compilation issues
  - Debugging across language boundaries is difficult
- **Why rejected**: Build complexity outweighs performance benefits for this use case

### Alternative 2: gRPC/Protocol Buffers
- **Pros**: Type-safe interface, bidirectional streaming
- **Cons**:
  - Requires running a separate server process
  - Protocol buffer schema maintenance
  - Overkill for batch file processing
- **Why rejected**: Adds unnecessary infrastructure for file-based workflow

### Alternative 3: Shared Memory (mmap)
- **Pros**: Zero-copy data transfer
- **Cons**:
  - Platform-specific implementation
  - Complex synchronization requirements
  - No metadata handling built-in
- **Why rejected**: Safetensors provides similar benefits with better portability

### Alternative 4: Apache Arrow/Parquet
- **Pros**: Industry standard for columnar data
- **Cons**:
  - Not optimized for ML tensor operations
  - Less common in ML tooling
- **Why rejected**: Safetensors has better HuggingFace ecosystem integration

## Consequences

### Positive
- **Simple builds**: Each language compiles independently
- **Easy testing**: Components can be tested in isolation
- **Clear interface**: File format and JSON schema serve as contract
- **Ecosystem alignment**: Safetensors is the standard for HuggingFace models
- **Portability**: Works on Linux, macOS, Windows without changes

### Negative
- **I/O overhead**: File writes for inter-process communication
- **No streaming**: Must process entire files (suitable for batch validation)
- **Process spawn cost**: Subprocess startup adds latency (~10-50ms)

### Risks
- **Version drift**: Safetensors format could change (mitigated by pinning versions)
- **Large file handling**: Memory constraints for very large KV caches (mitigated by chunking if needed)

## Implementation Notes

### Go Safetensors Package
```go
// pkg/safetensors/reader.go
func ReadFile(path string) (*File, error)
func (f *File) GetTensorFloat32(name string) ([]float32, error)

// pkg/safetensors/writer.go
func WriteFile(path string, tensors []*TensorData, metadata map[string]string) error
```

### Python Bridge
```python
# validation/compression_bridge.py
def compress_kv_cache(input_path, output_path, group_size=64) -> CompressionMetrics
def decompress_kv_cache(input_path, output_path) -> CompressionMetrics
```

## Related
- PRP: `docs/prps/tinyllama-perplexity-validation.md`
- Implementation: `pkg/safetensors/`, `cmd/compress/`, `validation/`
