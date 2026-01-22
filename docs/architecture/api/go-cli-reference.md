# Go CLI Reference: kv-compress

## Overview
The `kv-compress` CLI provides KV cache compression and quality validation using INT4 quantization. It reads and writes safetensors files and outputs quality metrics as JSON.

## Installation

```bash
# Build from source
cd /path/to/kv-cache-p2p
make build-compress

# Binary location
./bin/kv-compress
```

## Usage

```bash
kv-compress [options]
```

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-input` | string | (required) | Input safetensors file path |
| `-output` | string | (required*) | Output safetensors file path |
| `-group-size` | int | 64 | Quantization group size (32, 64, or 128) |
| `-sparsity` | float | 0.0 | Sparsification ratio (0.0-1.0, 0=disabled) |
| `-decompress` | bool | false | Enable decompression mode |
| `-metrics-only` | bool | false | Compute metrics without writing output |
| `-verbose` | bool | false | Enable verbose logging |

*Required unless `-metrics-only` is set

## Examples

### Compress a KV Cache File
```bash
kv-compress \
  -input kv_cache.safetensors \
  -output kv_cache_compressed.safetensors \
  -group-size 64
```

### Compress with Custom Parameters
```bash
kv-compress \
  -input kv_cache.safetensors \
  -output compressed.safetensors \
  -group-size 128 \
  -sparsity 0.5 \
  -verbose
```

### Get Quality Metrics Only (Dry Run)
```bash
kv-compress \
  -input kv_cache.safetensors \
  -metrics-only \
  -group-size 64
```

### Decompress a File
```bash
kv-compress \
  -decompress \
  -input compressed.safetensors \
  -output decompressed.safetensors
```

## Output Format

### JSON Metrics (stdout)
```json
{
  "compression_ratio": 3.76,
  "original_size_bytes": 1048576,
  "compressed_size_bytes": 278836,
  "rmse": 0.045623,
  "snr_db": 25.18,
  "cosine_similarity": 0.999534,
  "max_error": 0.187500,
  "quality_level": "GOOD",
  "processing_time_ms": 45.32,
  "quality_passed": true,
  "violations": [],
  "tensor_metrics": {
    "layer_0_key_rmse": 0.042,
    "layer_0_value_rmse": 0.048
  }
}
```

### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `compression_ratio` | float | Theoretical compression ratio (original/compressed) |
| `original_size_bytes` | int | Original tensor data size in bytes |
| `compressed_size_bytes` | int | Compressed output size in bytes |
| `rmse` | float | Root Mean Squared Error (averaged across tensors) |
| `snr_db` | float | Signal-to-Noise Ratio in decibels |
| `cosine_similarity` | float | Cosine similarity between original and reconstructed |
| `max_error` | float | Maximum absolute error across all values |
| `quality_level` | string | Quality grade: EXCELLENT, GOOD, ACCEPTABLE, POOR |
| `processing_time_ms` | float | Total processing time in milliseconds |
| `quality_passed` | bool | Whether all quality thresholds were met |
| `violations` | string[] | List of failed threshold checks |
| `tensor_metrics` | object | Per-tensor metrics (optional, verbose mode) |

### Quality Levels

| Level | RMSE | SNR | Description |
|-------|------|-----|-------------|
| EXCELLENT | < 0.01 | > 40 dB | Near-lossless compression |
| GOOD | < 0.05 | > 30 dB | High quality, suitable for most uses |
| ACCEPTABLE | < 0.10 | > 18 dB | Standard INT4 quality (KVQuant threshold) |
| POOR | >= 0.10 | < 18 dB | Quality may impact inference |

### Quality Thresholds

| Metric | Threshold | Source |
|--------|-----------|--------|
| Max RMSE | 0.10 | KVQuant paper |
| Max Max Error | 0.50 | Conservative limit |
| Min Cosine Similarity | 0.99 | High similarity required |
| Min SNR | 18 dB | Typical INT4 quantization |
| Max Perplexity Delta | 0.5 | KVQuant standard |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, invalid input, etc.) |

## Input File Format

### Safetensors Structure
```
[header_size: 8 bytes, little-endian uint64]
[header_json: variable length]
[tensor_data: concatenated binary data]
```

### Expected Tensor Names
```
layer_0_key    # Key cache for layer 0
layer_0_value  # Value cache for layer 0
layer_1_key    # Key cache for layer 1
layer_1_value  # Value cache for layer 1
...
layer_N_key    # Key cache for layer N
layer_N_value  # Value cache for layer N
```

### Supported Data Types
| DType | Bytes | Description |
|-------|-------|-------------|
| F16 | 2 | IEEE 754 half-precision |
| F32 | 4 | IEEE 754 single-precision |
| BF16 | 2 | Brain floating-point |

## Compression Algorithm

### INT4 Per-Group Quantization
1. Split tensor into groups of `group_size` elements
2. For each group:
   - Find `max_abs = max(|values|)`
   - Compute `scale = max_abs / 7`
   - Quantize: `q = round(value / scale)`, clamped to [-8, 7]
   - Store INT4 values and FP16 scale

### Compression Ratio Formula
```
Original:   N * 2 bytes (FP16)
Compressed: N/2 bytes (INT4) + N/group_size * 2 bytes (scales)
Ratio:      N*2 / (N/2 + N/group_size*2)
          = 2N / (N/2 + N/32)  [group_size=64]
          = 2N / (17N/32)
          = 64/17 = 3.76x
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Error: -input is required" | Missing input file | Provide `-input` flag |
| "Error: -output is required" | Missing output file | Provide `-output` or use `-metrics-only` |
| "read input: open file: no such file" | Input file not found | Check file path |
| "parse header: invalid JSON" | Corrupted safetensors file | Verify file integrity |
| "unsupported data type: X" | Unknown dtype in tensor | Use F16, F32, or BF16 |

## Integration with Python

```python
from compression_bridge import compress_kv_cache, CompressionMetrics

metrics: CompressionMetrics = compress_kv_cache(
    input_path="kv_cache.safetensors",
    output_path="compressed.safetensors",
    group_size=64,
    sparsity=0.0,
    verbose=False
)

print(f"Compression ratio: {metrics.compression_ratio:.2f}x")
print(f"RMSE: {metrics.rmse:.6f}")
print(f"Quality: {metrics.quality_level}")
```

## Related Documentation
- ADR-001: Cross-Language Architecture
- ADR-002: INT4 Quantization Strategy
- Python API: `validation/compression_bridge.py`
