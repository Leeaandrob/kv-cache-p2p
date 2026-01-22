# C4 Context Diagram: TinyLLaMA Perplexity Validation System

## Overview
The TinyLLaMA Perplexity Validation System validates that INT4 compression of KV cache maintains inference quality. It bridges the Python ML ecosystem with Go compression pipelines, measuring perplexity delta against the KVQuant threshold of < 0.5.

## Diagram

```mermaid
C4Context
    title System Context Diagram - TinyLLaMA Perplexity Validation

    Person(mlEngineer, "ML Engineer", "Validates compression quality before deployment")
    Person(ciSystem, "CI/CD Pipeline", "Automated quality gates")

    System(validationSystem, "Perplexity Validation System", "Validates KV cache compression quality using TinyLLaMA model")

    System_Ext(huggingface, "HuggingFace Hub", "Model weights, tokenizers, datasets")
    System_Ext(wikitext, "WikiText-2 Dataset", "Standard perplexity benchmark")
    System_Ext(tinyllama, "TinyLLaMA Model", "1.1B parameter LLM for validation")

    Rel(mlEngineer, validationSystem, "Runs validation", "CLI")
    Rel(ciSystem, validationSystem, "Executes tests", "make/pytest")

    Rel(validationSystem, huggingface, "Downloads models/datasets", "HTTPS")
    Rel(validationSystem, wikitext, "Loads test corpus", "datasets API")
    Rel(validationSystem, tinyllama, "Generates KV cache", "transformers API")
```

## Actors

| Actor | Description | Interactions |
|-------|-------------|--------------|
| ML Engineer | Developer validating compression before deployment | Runs `python validate_perplexity.py` to check quality metrics |
| CI/CD Pipeline | Automated testing system | Executes `pytest` and `go test` for quality gates |

## External Systems

| System | Purpose | Integration |
|--------|---------|-------------|
| HuggingFace Hub | Model registry and dataset hosting | Python `transformers` and `datasets` libraries |
| WikiText-2 | Standard perplexity benchmark dataset | Loaded via `datasets.load_dataset()` |
| TinyLLaMA | Reference model for validation (1.1B params, GQA) | Loaded via `AutoModelForCausalLM.from_pretrained()` |

## Key Constraints

- TinyLLaMA uses Grouped Query Attention (GQA): 4 KV heads, not 32
- WikiText-2 perplexity requires stride=512 for accurate measurement
- Perplexity delta threshold: < 0.5 per KVQuant standard
