# Sequence Diagram: Full Validation Workflow

## Overview
This diagram shows the complete sequence of operations for validating KV cache compression quality using TinyLLaMA.

## Full Validation Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Main as validate_perplexity.py
    participant KVUtils as kv_cache_utils.py
    participant Bridge as compression_bridge.py
    participant HF as HuggingFace
    participant Model as TinyLLaMA
    participant CLI as kv-compress CLI
    participant FS as Filesystem

    User->>Main: python validate_perplexity.py

    rect rgb(240, 248, 255)
        Note over Main,HF: Phase 1: Load Model
        Main->>HF: AutoModelForCausalLM.from_pretrained()
        HF-->>Main: model
        Main->>HF: AutoTokenizer.from_pretrained()
        HF-->>Main: tokenizer
    end

    rect rgb(255, 248, 240)
        Note over Main,Model: Phase 2: Generate KV Cache
        Main->>KVUtils: extract_kv_cache(model, tokenizer, prompt)
        KVUtils->>Model: model(**inputs, past_key_values=cache)
        Model-->>KVUtils: DynamicCache
        KVUtils-->>Main: cache, metadata
    end

    rect rgb(240, 255, 240)
        Note over Main,HF: Phase 3: Baseline Perplexity
        Main->>HF: load_dataset("wikitext", "wikitext-2-raw-v1")
        HF-->>Main: test dataset
        Main->>Model: model(input_ids, labels=target_ids)
        Note right of Main: Sliding window, stride=512
        Model-->>Main: loss per window
        Main->>Main: perplexity = exp(mean(losses))
    end

    rect rgb(255, 240, 245)
        Note over Main,CLI: Phase 4: Compress KV Cache
        Main->>KVUtils: save_kv_cache(cache, path)
        KVUtils->>FS: safetensors.save_file(tensors)
        FS-->>KVUtils: kv_cache.safetensors

        Main->>Bridge: compress_kv_cache(input, output, group_size=64)
        Bridge->>CLI: subprocess.run([kv-compress, -input, -output, ...])
        CLI->>FS: ReadFile(input_path)
        FS-->>CLI: File struct

        Note over CLI: INT4 Quantization
        CLI->>CLI: simulateINT4Quantization()
        CLI->>CLI: ComputeTensorMetrics()
        CLI->>CLI: ValidateMetrics()

        CLI->>FS: WriteFile(output_path)
        CLI-->>Bridge: JSON metrics (stdout)
        Bridge-->>Main: CompressionMetrics
    end

    rect rgb(245, 245, 255)
        Note over Main,FS: Phase 5: Load Compressed Cache
        Main->>KVUtils: load_kv_cache(compressed_path)
        KVUtils->>FS: safetensors.load_file()
        FS-->>KVUtils: tensors
        KVUtils->>KVUtils: reconstruct DynamicCache
        KVUtils-->>Main: compressed_cache, metadata
    end

    rect rgb(255, 255, 240)
        Note over Main: Phase 6: Validate Results
        Main->>Main: delta = compressed_ppl - baseline_ppl
        Main->>Main: passed = delta < 0.5

        alt Passed
            Main-->>User: PASSED: Delta X.XX < 0.5 threshold
        else Failed
            Main-->>User: FAILED: Delta X.XX >= 0.5 threshold
        end
    end
```

## Quick Test Sequence (Synthetic Data)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Main as validate_perplexity.py
    participant Bridge as compression_bridge.py
    participant CLI as kv-compress CLI
    participant FS as Filesystem

    User->>Main: python validate_perplexity.py --quick-test

    rect rgb(240, 255, 240)
        Note over Main,FS: Create Synthetic KV Cache
        Main->>Main: Generate random tensors (22 layers, 4 heads)
        Main->>FS: safetensors.save_file(synthetic_tensors)
        FS-->>Main: synthetic_kv_cache.safetensors
    end

    rect rgb(255, 240, 245)
        Note over Main,CLI: Compress Synthetic Cache
        Main->>Bridge: compress_kv_cache(input, output)
        Bridge->>CLI: subprocess.run([kv-compress, ...])
        CLI->>CLI: Quantize + Compute Metrics
        CLI-->>Bridge: JSON metrics
        Bridge-->>Main: CompressionMetrics
    end

    rect rgb(255, 255, 240)
        Note over Main: Validate Quality Metrics
        Main->>Main: Check RMSE < 0.1
        Main->>Main: Check SNR > 18dB
        Main->>Main: Check CosSim > 0.99
        Main-->>User: PASSED/FAILED with metrics
    end
```

## Error Handling Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Main as validate_perplexity.py
    participant Bridge as compression_bridge.py
    participant CLI as kv-compress CLI

    User->>Main: python validate_perplexity.py

    Main->>Bridge: check_cli_available()

    alt CLI Not Found
        Bridge->>Bridge: find_cli_binary() raises FileNotFoundError
        Bridge-->>Main: False
        Main-->>User: RuntimeError: Go CLI not found. Build with: make build-compress
    else CLI Found
        Bridge-->>Main: True
        Main->>Main: Continue validation...
    end

    Main->>Bridge: compress_kv_cache(input, output)

    alt Subprocess Fails
        CLI-->>Bridge: Non-zero exit code
        Bridge-->>Main: subprocess.CalledProcessError
        Main-->>User: Error: Compression failed
    else Invalid JSON Output
        CLI-->>Bridge: Malformed stdout
        Bridge-->>Main: ValueError: Invalid JSON from CLI
        Main-->>User: Error: CLI output parsing failed
    else Success
        CLI-->>Bridge: JSON metrics
        Bridge-->>Main: CompressionMetrics
    end
```

## Steps Summary

| Step | Actor | Action | Description |
|------|-------|--------|-------------|
| 1 | User | Run script | Execute `python validate_perplexity.py` |
| 2 | Main | Load model | Download/cache TinyLLaMA from HuggingFace |
| 3 | KVUtils | Extract cache | Run forward pass with DynamicCache |
| 4 | Main | Baseline PPL | Compute perplexity on WikiText-2 |
| 5 | KVUtils | Save cache | Write safetensors file |
| 6 | Bridge | Invoke CLI | Run `kv-compress` subprocess |
| 7 | CLI | Quantize | Apply INT4 per-group quantization |
| 8 | CLI | Metrics | Compute RMSE, SNR, cosine similarity |
| 9 | CLI | Write output | Save compressed safetensors |
| 10 | Bridge | Parse JSON | Extract CompressionMetrics |
| 11 | Main | Validate | Check delta < threshold |
| 12 | User | View result | PASSED/FAILED with metrics |
