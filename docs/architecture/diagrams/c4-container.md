# C4 Container Diagram: TinyLLaMA Perplexity Validation System

## Overview
The system consists of two main containers: a Python validation module for ML operations and a Go CLI for compression operations. They communicate via safetensors files and JSON output.

## Diagram

```mermaid
C4Container
    title Container Diagram - TinyLLaMA Perplexity Validation

    Person(user, "ML Engineer", "Runs validation")

    Container_Boundary(system, "Perplexity Validation System") {
        Container(pythonValidation, "Python Validation Module", "Python 3.10+", "KV cache extraction, perplexity computation, validation orchestration")
        Container(goCLI, "Go Compression CLI", "Go 1.21+", "INT4 quantization, quality metrics, safetensors I/O")
        Container_Ext(tempFiles, "Temporary Files", "Safetensors + JSON", "Inter-process data exchange")
    }

    System_Ext(huggingface, "HuggingFace", "Model/dataset source")
    System_Ext(filesystem, "Local Filesystem", "Model cache, temp files")

    Rel(user, pythonValidation, "Runs", "CLI")
    Rel(pythonValidation, huggingface, "Downloads", "HTTPS")
    Rel(pythonValidation, tempFiles, "Writes KV cache", "safetensors")
    Rel(pythonValidation, goCLI, "Invokes", "subprocess")
    Rel(goCLI, tempFiles, "Reads/writes", "safetensors")
    Rel(goCLI, pythonValidation, "Returns metrics", "JSON stdout")
    Rel(tempFiles, filesystem, "Stored in", "temp directory")
```

## Containers

| Container | Technology | Purpose | Scaling |
|-----------|------------|---------|---------|
| Python Validation Module | Python 3.10+, PyTorch, Transformers | ML inference, KV cache extraction, perplexity | Single process, GPU optional |
| Go Compression CLI | Go 1.21+, safetensors lib | INT4 quantization, quality metrics | Single process, CPU-bound |
| Temporary Files | Safetensors format | Binary tensor exchange | Limited by disk I/O |

## Communication Flow

```
Python                    Files                      Go CLI
  |                         |                          |
  |-- save_kv_cache() ----->| kv_cache.safetensors     |
  |                         |                          |
  |-- subprocess.run() --------------------------->   |
  |                         |                          |
  |                         |<---- ReadFile() --------|
  |                         |                          |
  |                         |     [Quantize/Decompress]|
  |                         |                          |
  |                         |<---- WriteFile() -------|
  |                         | compressed.safetensors   |
  |                         |                          |
  |<----------------------- JSON metrics -------------|
  |                         |                          |
  |-- load_kv_cache() <-----|                          |
```

## Technology Choices

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Inter-process format | Safetensors | HuggingFace standard, safe (no pickle), efficient |
| CLI invocation | subprocess.run() | Simple, no CGO complexity |
| Metrics format | JSON stdout | Easy parsing, human-readable |
| Temp file location | tempfile.TemporaryDirectory | Auto-cleanup, cross-platform |
