# Architecture Documentation: TinyLLaMA Perplexity Validation System

## Overview
This documentation describes the architecture of the TinyLLaMA Perplexity Validation System, which validates that INT4 compression of KV cache maintains inference quality within acceptable thresholds.

## System Purpose
Validate KV cache compression quality by:
1. Extracting KV cache from TinyLLaMA inference
2. Compressing using INT4 quantization (Go)
3. Measuring perplexity delta against baseline
4. Ensuring delta < 0.5 (KVQuant standard)

## Quick Navigation

### Architecture Decision Records (ADRs)
- [ADR-001: Cross-Language Architecture](decisions/ADR-001-cross-language-architecture.md) - Safetensors + CLI subprocess design
- [ADR-002: INT4 Quantization Strategy](decisions/ADR-002-int4-quantization-strategy.md) - Per-group symmetric quantization

### C4 Diagrams
- [C4 Context](diagrams/c4-context.md) - System context and external dependencies
- [C4 Container](diagrams/c4-container.md) - Python and Go containers
- [C4 Component](diagrams/c4-component.md) - Internal component structure

### Data Flow & Sequences
- [Data Flow Diagram](diagrams/data-flow.md) - Complete data transformation pipeline
- [Sequence Diagram](diagrams/sequence-validation.md) - Validation workflow sequences

### API Documentation
- [Go CLI Reference](api/go-cli-reference.md) - kv-compress command-line interface
- [Python Module API](api/python-modules.md) - Python module reference
- [Safetensors Format](api/safetensors-format.md) - Binary format specification

## Architecture Summary

### High-Level Design
```
+------------------+     safetensors     +------------------+
|                  |  ----------------->  |                  |
|  Python Module   |                      |    Go CLI        |
|  (ML Ecosystem)  |  <-----------------  |  (Compression)   |
|                  |     JSON metrics     |                  |
+------------------+                      +------------------+
        |                                         |
        v                                         v
+------------------+                      +------------------+
|  HuggingFace     |                      |  pkg/safetensors |
|  Transformers    |                      |  pkg/compression |
+------------------+                      +------------------+
```

### Key Design Decisions
1. **Cross-language via safetensors**: Binary format for tensor exchange (no CGO)
2. **CLI subprocess calls**: Simple Python-to-Go communication
3. **Per-group INT4 quantization**: ~3.76x compression with SNR > 18dB
4. **JSON metrics output**: Machine-readable quality reporting

### Technology Stack
| Layer | Python | Go |
|-------|--------|-----|
| Entry Point | validate_perplexity.py | cmd/compress/main.go |
| Core Logic | kv_cache_utils.py | pkg/compression/ |
| I/O Bridge | compression_bridge.py | pkg/safetensors/ |
| Format | safetensors (HuggingFace) | safetensors (custom) |

## Quality Metrics

### Thresholds (KVQuant Standard)
| Metric | Threshold | Typical Value |
|--------|-----------|---------------|
| RMSE | < 0.10 | 0.04-0.06 |
| SNR | > 18 dB | 22-28 dB |
| Cosine Similarity | > 0.99 | 0.9995+ |
| Perplexity Delta | < 0.5 | 0.05-0.15 |
| Compression Ratio | ~3.76x | 3.7-3.8x |

### Quality Levels
| Level | RMSE | SNR | Description |
|-------|------|-----|-------------|
| EXCELLENT | < 0.01 | > 40 dB | Near-lossless |
| GOOD | < 0.05 | > 30 dB | High quality |
| ACCEPTABLE | < 0.10 | > 18 dB | Standard INT4 |
| POOR | >= 0.10 | < 18 dB | May impact inference |

## Usage

### Quick Test (No Model Download)
```bash
cd validation
pip install -r requirements.txt
python validate_perplexity.py --quick-test
```

### Full Validation
```bash
# Build Go CLI
make build-compress

# Run validation
cd validation
python validate_perplexity.py
```

### Expected Output
```
============================================================
  TinyLLaMA KV Cache Compression Validation
============================================================
[1/6] Loading TinyLlama/TinyLlama-1.1B-Chat-v1.0...
    Model loaded on cuda
    Config: 22 layers, 4 KV heads
[2/6] Generating KV cache...
    Generated: 22 layers, 95 tokens, 2.12 MB
[3/6] Computing baseline perplexity...
    Baseline perplexity: 16.4532
[4/6] Compressing KV cache...
    Compression ratio: 3.76x
    RMSE: 0.045623
    SNR: 25.18 dB
[5/6] Loading reconstructed KV cache...
[6/6] Computing perplexity with compressed KV cache...
    Compressed perplexity: 16.5443
============================================================
  RESULTS
============================================================
  Baseline Perplexity:   16.4532
  Compressed Perplexity: 16.5443
  Delta:                 +0.0911 (+0.55%)
  Compression Ratio:     3.76x
------------------------------------------------------------
  RMSE:                  0.045623
  SNR:                   25.18 dB
  Cosine Similarity:     0.999534
============================================================
  [OK] PASSED: Delta 0.0911 < 0.5 threshold
============================================================
```

## File Structure
```
kv-cache-p2p/
├── cmd/compress/
│   └── main.go              # CLI entry point
├── pkg/
│   ├── safetensors/
│   │   ├── types.go         # Data structures
│   │   ├── reader.go        # File parsing
│   │   ├── writer.go        # File generation
│   │   └── fp16.go          # FP16 conversion
│   └── compression/
│       ├── types.go         # Config structures
│       ├── pipeline.go      # Compression pipeline
│       ├── quality.go       # Quality metrics
│       └── lz4.go           # LZ4 compression
├── validation/
│   ├── validate_perplexity.py
│   ├── kv_cache_utils.py
│   ├── compression_bridge.py
│   └── requirements.txt
└── docs/architecture/
    ├── README.md            # This file
    ├── decisions/           # ADRs
    ├── diagrams/            # C4, data flow, sequence
    └── api/                 # API documentation
```

## Related Documents
- [PRP: TinyLLaMA Perplexity Validation](../prps/tinyllama-perplexity-validation.md)
- [Compression Strategy](../COMPRESSION.md)

## References
- [KVQuant Paper](https://arxiv.org/abs/2401.18079) - INT4 quality thresholds
- [KIVI Paper](https://arxiv.org/abs/2402.02750) - KV cache compression
- [Safetensors](https://github.com/huggingface/safetensors) - Format specification
- [TinyLLaMA](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) - Model architecture
