# C4 Component Diagram: TinyLLaMA Perplexity Validation System

## Overview
Detailed breakdown of the Python and Go components, showing internal structure and dependencies.

## Python Validation Module Components

```mermaid
C4Component
    title Component Diagram - Python Validation Module

    Container_Boundary(pythonModule, "Python Validation Module") {
        Component(validatePerplexity, "validate_perplexity.py", "Python", "Main entry point, orchestrates validation workflow")
        Component(kvCacheUtils, "kv_cache_utils.py", "Python", "KV cache extraction, save/load operations")
        Component(compressionBridge, "compression_bridge.py", "Python", "Go CLI subprocess interface")
    }

    Container_Ext(transformers, "HuggingFace Transformers", "Python library")
    Container_Ext(datasets, "HuggingFace Datasets", "Python library")
    Container_Ext(safetensorsPy, "safetensors", "Python library")
    Container_Ext(goCLI, "Go CLI", "kv-compress binary")

    Rel(validatePerplexity, kvCacheUtils, "Uses")
    Rel(validatePerplexity, compressionBridge, "Uses")
    Rel(validatePerplexity, transformers, "Loads model")
    Rel(validatePerplexity, datasets, "Loads WikiText-2")
    Rel(kvCacheUtils, transformers, "DynamicCache API")
    Rel(kvCacheUtils, safetensorsPy, "Save/load tensors")
    Rel(compressionBridge, goCLI, "subprocess.run()")
```

## Go Compression CLI Components

```mermaid
C4Component
    title Component Diagram - Go Compression CLI

    Container_Boundary(goCLI, "Go Compression CLI") {
        Component(mainCmd, "cmd/compress/main.go", "Go", "CLI entry point, flag parsing, workflow orchestration")
        Component(safetensorsGo, "pkg/safetensors/", "Go", "Safetensors file I/O, FP16/FP32 conversion")
        Component(compressionPkg, "pkg/compression/", "Go", "Quality metrics, quantization logic")
    }

    Component(reader, "reader.go", "Go", "Parse safetensors header, read tensors")
    Component(writer, "writer.go", "Go", "Generate safetensors format output")
    Component(fp16, "fp16.go", "Go", "IEEE 754 half-precision conversions")
    Component(types, "types.go", "Go", "DType, TensorInfo, File structs")

    Component(quality, "quality.go", "Go", "RMSE, SNR, cosine similarity")
    Component(pipeline, "pipeline.go", "Go", "Compression pipeline orchestration")
    Component(lz4, "lz4.go", "Go", "LZ4 network compression")

    Rel(mainCmd, safetensorsGo, "Uses")
    Rel(mainCmd, compressionPkg, "Uses")
    Rel(safetensorsGo, reader, "Contains")
    Rel(safetensorsGo, writer, "Contains")
    Rel(safetensorsGo, fp16, "Contains")
    Rel(safetensorsGo, types, "Contains")
    Rel(compressionPkg, quality, "Contains")
    Rel(compressionPkg, pipeline, "Contains")
    Rel(compressionPkg, lz4, "Contains")
```

## Python Components

| Component | Responsibility | Key Functions |
|-----------|---------------|---------------|
| validate_perplexity.py | Main orchestration | `run_full_validation()`, `compute_perplexity()`, `load_model_and_tokenizer()` |
| kv_cache_utils.py | KV cache operations | `extract_kv_cache()`, `save_kv_cache()`, `load_kv_cache()`, `validate_kv_cache_shape()` |
| compression_bridge.py | Go CLI interface | `compress_kv_cache()`, `decompress_kv_cache()`, `find_cli_binary()` |

## Go Components

| Component | Responsibility | Key Functions |
|-----------|---------------|---------------|
| cmd/compress/main.go | CLI interface | `main()`, `runCompress()`, `runDecompress()`, `simulateINT4Quantization()` |
| pkg/safetensors/reader.go | File parsing | `ReadFile()`, `Read()`, `bytesToFloat32()` |
| pkg/safetensors/writer.go | File generation | `WriteFile()`, `Write()`, `WriteFloat32Tensors()` |
| pkg/safetensors/fp16.go | FP16 conversion | `FP16ToFloat32()`, `Float32ToFP16()`, `BF16ToFloat32()` |
| pkg/compression/quality.go | Quality metrics | `ComputeTensorMetrics()`, `ValidateMetrics()`, `GetQualityLevel()` |
