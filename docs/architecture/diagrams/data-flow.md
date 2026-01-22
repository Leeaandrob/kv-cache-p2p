# Data Flow Diagram: KV Cache Compression Pipeline

## Overview
This diagram shows how KV cache data flows through the validation system, from model inference through compression to quality validation.

## Complete Validation Flow

```mermaid
flowchart TB
    subgraph Input["Input Phase"]
        A[TinyLLaMA Model] --> B[Forward Pass]
        C[Sample Text] --> B
        B --> D[DynamicCache Object]
    end

    subgraph Extraction["KV Cache Extraction"]
        D --> E[Extract key_cache]
        D --> F[Extract value_cache]
        E --> G{bfloat16?}
        F --> G
        G -->|Yes| H[Convert to float32]
        G -->|No| I[Keep as-is]
        H --> J[CPU Transfer]
        I --> J
    end

    subgraph Serialization["Safetensors Serialization"]
        J --> K["save_file(tensors)"]
        K --> L[("kv_cache.safetensors")]
        K --> M[("metadata.json")]
    end

    subgraph GoCompression["Go CLI Compression"]
        L --> N[ReadFile]
        N --> O[Parse Header JSON]
        O --> P[Extract Tensor Data]
        P --> Q[FP16/FP32 to float32]
        Q --> R[INT4 Quantization]
        R --> S[Compute Scales per Group]
        S --> T[Quantize Values]
        T --> U[Dequantize for Output]
        U --> V[Quality Metrics]
        V --> W[WriteFile]
        W --> X[("compressed.safetensors")]
        V --> Y{{"JSON Metrics"}}
    end

    subgraph Validation["Quality Validation"]
        X --> Z[load_file]
        Y --> AA[Parse Metrics]
        Z --> AB[Reconstruct DynamicCache]
        AA --> AC{RMSE < 0.1?}
        AA --> AD{SNR > 18dB?}
        AA --> AE{CosSim > 0.99?}
        AC & AD & AE --> AF{All Pass?}
        AF -->|Yes| AG[PASSED]
        AF -->|No| AH[FAILED]
    end

    style L fill:#f9f,stroke:#333
    style X fill:#f9f,stroke:#333
    style AG fill:#9f9,stroke:#333
    style AH fill:#f99,stroke:#333
```

## INT4 Quantization Detail

```mermaid
flowchart LR
    subgraph Input["Input Tensor"]
        A["float32 values<br/>[N elements]"]
    end

    subgraph Grouping["Per-Group Processing"]
        A --> B["Split into groups<br/>group_size=64"]
        B --> C["Group 0"]
        B --> D["Group 1"]
        B --> E["..."]
        B --> F["Group N/64"]
    end

    subgraph Quantize["Quantization (per group)"]
        C --> G["max_abs = max(|vals|)"]
        G --> H["scale = max_abs / 7"]
        H --> I["q = round(val/scale)"]
        I --> J["clamp(q, -8, 7)"]
    end

    subgraph Output["Output"]
        J --> K["INT4 values<br/>(4 bits each)"]
        H --> L["FP16 scale<br/>(16 bits)"]
    end

    subgraph Dequant["Dequantization"]
        K --> M["q * scale"]
        L --> M
        M --> N["reconstructed<br/>float32"]
    end
```

## Data Flow Steps

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1 | Prompt text | Tokenization | Token IDs |
| 2 | Token IDs | Model forward pass | DynamicCache |
| 3 | DynamicCache | Extract key/value tensors | PyTorch tensors |
| 4 | Tensors (bfloat16/float16) | Convert to float32 | float32 tensors |
| 5 | float32 tensors | safetensors.save_file() | .safetensors file |
| 6 | .safetensors | Go ReadFile() | Go tensor structs |
| 7 | Go tensors | Per-group quantization | INT4 + scales |
| 8 | INT4 + scales | Dequantization | Reconstructed float32 |
| 9 | Original + reconstructed | Quality metrics | RMSE, SNR, CosSim |
| 10 | Reconstructed | Go WriteFile() | compressed.safetensors |
| 11 | Metrics | JSON encode | stdout JSON |
| 12 | JSON output | Python parse | CompressionMetrics |
| 13 | CompressionMetrics | Threshold check | PASS/FAIL |

## Data Transformations

### Tensor Shapes (TinyLLaMA)
```
Original KV Cache per layer:
  Key:   [1, 4, seq_len, 64]  # batch=1, kv_heads=4, head_dim=64
  Value: [1, 4, seq_len, 64]

Total for 22 layers:
  Keys:   22 * 4 * seq_len * 64 elements
  Values: 22 * 4 * seq_len * 64 elements

Example (seq_len=128):
  Total elements: 22 * 2 * 4 * 128 * 64 = 1,409,024
  Original size (FP16): 2.75 MB
  Compressed size (INT4 + scales): ~0.73 MB
  Compression ratio: ~3.76x
```

### Precision Conversions
| From | To | Method |
|------|-----|--------|
| bfloat16 | float32 | `tensor.float()` in Python |
| float32 | FP16 bytes | `Float32ToFP16()` in Go |
| FP16 bytes | float32 | `FP16ToFloat32()` in Go |
| float32 | INT4 | Symmetric quantization |
| INT4 | float32 | `q * scale` |
