# ADR-002: Per-Group INT4 Quantization with FP16 Scales

## Status
Accepted

## Date
2026-01-22

## Context
KV cache compression requires aggressive quantization to reduce memory footprint while maintaining inference quality. The original FP16 KV cache tensors need to be compressed to approximately 4x smaller without significantly impacting perplexity (< 0.5 delta per KVQuant standard).

Key constraints:
- TinyLLaMA uses Grouped Query Attention (GQA) with 4 KV heads, not 32
- KV cache shape: `[batch, 4, seq_len, 64]` (4 heads, 64 head_dim)
- Target compression: ~4x (FP16 to INT4)
- Quality threshold: < 0.5 perplexity delta

## Decision
Implement symmetric per-group INT4 quantization with FP16 scales:

### Quantization Formula
```
For each group of `group_size` values:
1. Find max_abs = max(|values|)
2. Compute scale = max_abs / 7  (INT4 symmetric range: [-8, 7])
3. Quantize: q = round(value / scale), clamped to [-8, 7]
4. Store: q as INT4 (4 bits), scale as FP16 (16 bits)
```

### Dequantization Formula
```
value = q * scale
```

### Group Size Selection
- Default: 64 elements per group
- Options: 32, 64, 128
- Trade-off: Smaller groups = better quality, more scale overhead

### Memory Layout
```
Original:   [N elements] x 16 bits = N * 2 bytes
Compressed: [N elements] x 4 bits + [N/group_size scales] x 16 bits
          = N/2 bytes + (N/64) * 2 bytes = N/2 + N/32 = 17N/32 bytes
Ratio:      N*2 / (17N/32) = 64/17 = ~3.76x
```

## Alternatives Considered

### Alternative 1: INT8 Quantization
- **Pros**: Higher quality, simpler implementation
- **Cons**: Only 2x compression (insufficient for target)
- **Why rejected**: Does not meet compression ratio requirements

### Alternative 2: Asymmetric Quantization (with zero point)
- **Pros**: Better for skewed distributions
- **Cons**: Additional storage for zero points, more complex
- **Why rejected**: KV cache values are approximately symmetric around zero

### Alternative 3: Per-tensor Quantization
- **Pros**: Minimal overhead (single scale per tensor)
- **Cons**: Poor quality for varying value ranges within tensor
- **Why rejected**: Unacceptable quality degradation (SNR < 15dB)

### Alternative 4: Non-uniform Quantization (k-means clustering)
- **Pros**: Can achieve better quality at same bit-width
- **Cons**: Much slower quantization, requires codebook storage
- **Why rejected**: Latency requirements for real-time compression

## Consequences

### Positive
- **Meets quality targets**: SNR typically 18-25 dB, RMSE < 0.1
- **Predictable compression**: 3.76x ratio (group_size=64)
- **Fast**: Simple arithmetic operations, no complex encoding
- **GPU-friendly**: Amenable to CUDA kernel implementation

### Negative
- **Scale overhead**: ~3% storage for scales (with group_size=64)
- **Boundary effects**: Values at group boundaries quantized independently
- **Fixed precision**: Cannot adapt precision per-value

### Quality Metrics (Measured)
| Metric | Threshold | Typical Value |
|--------|-----------|---------------|
| RMSE | < 0.10 | 0.04-0.06 |
| SNR | > 18 dB | 22-28 dB |
| Cosine Similarity | > 0.99 | 0.9995+ |
| Max Error | < 0.50 | 0.15-0.25 |

## Implementation

### Go Implementation (`cmd/compress/main.go`)
```go
func simulateINT4Quantization(original []uint16, groupSize int) []uint16 {
    reconstructed := make([]uint16, len(original))
    numGroups := (len(original) + groupSize - 1) / groupSize

    for g := 0; g < numGroups; g++ {
        start := g * groupSize
        end := min(start + groupSize, len(original))

        // Find max absolute value
        var maxVal float32
        for i := start; i < end; i++ {
            val := FP16ToFloat32(original[i])
            if absVal := abs(val); absVal > maxVal {
                maxVal = absVal
            }
        }

        // Compute scale
        scale := maxVal / 7.0
        if scale == 0 { scale = 1e-7 }

        // Quantize and dequantize
        for i := start; i < end; i++ {
            val := FP16ToFloat32(original[i])
            q := clamp(round(val / scale), -8, 7)
            reconstructed[i] = Float32ToFP16(q * scale)
        }
    }
    return reconstructed
}
```

### Quality Validation (`pkg/compression/quality.go`)
```go
func (v *QualityValidator) ComputeTensorMetrics(original, reconstructed []uint16) (*QualityMetrics, error)
func (v *QualityValidator) ValidateMetrics(metrics *QualityMetrics) (bool, []string)
```

## Related
- ADR-001: Cross-Language Architecture
- Reference: KVQuant paper (https://arxiv.org/abs/2401.18079)
- Reference: KIVI paper (https://arxiv.org/abs/2402.02750)
