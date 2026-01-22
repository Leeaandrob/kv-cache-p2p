# Python Module API Reference

## Overview
The Python validation modules provide KV cache extraction from TinyLLaMA, compression bridge to the Go CLI, and perplexity validation.

## Module: kv_cache_utils.py

### Classes

#### KVCacheMetadata
```python
@dataclass
class KVCacheMetadata:
    """Metadata for KV cache tensors."""
    num_layers: int      # 22 for TinyLlama
    num_kv_heads: int    # 4 for TinyLlama (GQA)
    seq_len: int         # Variable
    head_dim: int        # 64 for TinyLlama
    dtype: str           # "float16" or "float32"
    model_name: str      # Model identifier
```

**Methods:**
- `to_dict() -> dict` - Convert to dictionary for JSON serialization
- `from_dict(d: dict) -> KVCacheMetadata` - Create from dictionary

### Functions

#### extract_kv_cache
```python
def extract_kv_cache(
    model,
    tokenizer,
    prompt: str,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[DynamicCache, KVCacheMetadata]:
    """
    Extract KV cache from a model forward pass.

    CRITICAL: Uses DynamicCache for Transformers 4.35+, NOT tuple indexing.

    Args:
        model: HuggingFace model (TinyLlama or similar)
        tokenizer: Corresponding tokenizer
        prompt: Input text to generate KV cache from
        max_length: Optional maximum sequence length
        device: Device to run on (auto-detected if None)

    Returns:
        Tuple of (DynamicCache, KVCacheMetadata)

    Raises:
        ValueError: If batch size != 1 or cache is empty

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> cache, metadata = extract_kv_cache(model, tokenizer, "Hello, world!")
        >>> print(f"Extracted {metadata.num_layers} layers with {metadata.seq_len} tokens")
    """
```

#### save_kv_cache
```python
def save_kv_cache(
    cache,
    path: str,
    metadata_path: Optional[str] = None,
) -> KVCacheMetadata:
    """
    Save KV cache to safetensors file.

    CRITICAL: Converts bfloat16 to float32 since NumPy doesn't support bfloat16.

    Args:
        cache: DynamicCache from Transformers
        path: Output path for safetensors file
        metadata_path: Optional path for JSON metadata (defaults to path.metadata.json)

    Returns:
        KVCacheMetadata describing the saved cache

    Tensor naming convention:
        - layer_{i}_key: Key cache for layer i
        - layer_{i}_value: Value cache for layer i
    """
```

#### load_kv_cache
```python
def load_kv_cache(
    path: str,
    metadata_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[DynamicCache, KVCacheMetadata]:
    """
    Load KV cache from safetensors file.

    Args:
        path: Path to safetensors file
        metadata_path: Optional path to metadata JSON (defaults to path.metadata.json)
        device: Device to load tensors to

    Returns:
        Tuple of (DynamicCache, KVCacheMetadata)
    """
```

#### validate_kv_cache_shape
```python
def validate_kv_cache_shape(
    cache,
    expected_layers: int = 22,
    expected_kv_heads: int = 4,
    expected_head_dim: int = 64,
) -> Tuple[bool, List[str]]:
    """
    Validate KV cache shape matches expected TinyLlama configuration.

    CRITICAL: TinyLlama uses GQA with 4 KV heads, not 32.

    Args:
        cache: DynamicCache to validate
        expected_layers: Expected number of layers (22 for TinyLlama)
        expected_kv_heads: Expected number of KV heads (4 for TinyLlama GQA)
        expected_head_dim: Expected head dimension (64 for TinyLlama)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
```

#### compute_kv_cache_stats
```python
def compute_kv_cache_stats(cache) -> dict:
    """
    Compute statistics about a KV cache.

    Args:
        cache: DynamicCache to analyze

    Returns:
        Dictionary with statistics:
        - num_layers: Number of transformer layers
        - num_kv_heads: Number of key-value heads
        - seq_len: Sequence length
        - head_dim: Head dimension
        - total_elements: Total number of elements
        - total_bytes: Total size in bytes
        - total_mb: Total size in megabytes
        - dtype: Tensor data type
        - value_mean, value_std, value_min, value_max: Value statistics
    """
```

### Constants

```python
TINYLLAMA_NUM_LAYERS = 22
TINYLLAMA_NUM_KV_HEADS = 4  # GQA: 4 KV heads, not 32
TINYLLAMA_NUM_QUERY_HEADS = 32
TINYLLAMA_HEAD_DIM = 64
TINYLLAMA_HIDDEN_SIZE = 2048
TINYLLAMA_VOCAB_SIZE = 32000
```

---

## Module: compression_bridge.py

### Classes

#### CompressionMetrics
```python
@dataclass
class CompressionMetrics:
    """Metrics returned by the Go CLI."""
    compression_ratio: float        # e.g., 3.76
    original_size_bytes: int        # Original tensor size
    compressed_size_bytes: int      # Compressed output size
    rmse: float                     # Root Mean Squared Error
    snr_db: float                   # Signal-to-Noise Ratio (dB)
    cosine_similarity: float        # Cosine similarity (0-1)
    max_error: float                # Maximum absolute error
    quality_level: str              # EXCELLENT, GOOD, ACCEPTABLE, POOR
    processing_time_ms: float       # Processing time
    passed: bool                    # Quality validation result
    violations: list                # List of threshold violations
```

**Methods:**
- `from_dict(d: dict) -> CompressionMetrics` - Create from JSON dictionary

### Functions

#### compress_kv_cache
```python
def compress_kv_cache(
    input_path: str,
    output_path: str,
    group_size: int = 64,
    sparsity: float = 0.0,
    verbose: bool = False,
) -> CompressionMetrics:
    """
    Compress a KV cache file using the Go CLI.

    Args:
        input_path: Path to input safetensors file
        output_path: Path for output compressed file
        group_size: Quantization group size (32, 64, or 128)
        sparsity: Sparsification ratio (0.0-1.0, 0=disabled)
        verbose: Enable verbose output

    Returns:
        CompressionMetrics with quality and performance data

    Raises:
        subprocess.CalledProcessError: If compression fails
        FileNotFoundError: If CLI binary not found
        ValueError: If CLI returns invalid JSON

    Example:
        >>> metrics = compress_kv_cache(
        ...     "kv_cache.safetensors",
        ...     "compressed.safetensors",
        ...     group_size=64
        ... )
        >>> print(f"Compression: {metrics.compression_ratio:.2f}x, RMSE: {metrics.rmse:.6f}")
    """
```

#### decompress_kv_cache
```python
def decompress_kv_cache(
    input_path: str,
    output_path: str,
    verbose: bool = False,
) -> CompressionMetrics:
    """
    Decompress a KV cache file using the Go CLI.

    Args:
        input_path: Path to compressed safetensors file
        output_path: Path for output decompressed file
        verbose: Enable verbose output

    Returns:
        CompressionMetrics (limited, mainly timing info)
    """
```

#### get_compression_metrics
```python
def get_compression_metrics(
    input_path: str,
    group_size: int = 64,
    verbose: bool = False,
) -> CompressionMetrics:
    """
    Get compression metrics without writing output file.

    Args:
        input_path: Path to input safetensors file
        group_size: Quantization group size
        verbose: Enable verbose output

    Returns:
        CompressionMetrics with quality data
    """
```

#### find_cli_binary
```python
def find_cli_binary() -> str:
    """
    Find the kv-compress CLI binary.

    Searches in order:
    1. Environment variable KV_COMPRESS_PATH
    2. ../bin/kv-compress (relative to this file)
    3. ./bin/kv-compress (relative to cwd)
    4. System PATH

    Returns:
        Path to the CLI binary

    Raises:
        FileNotFoundError: If binary not found
    """
```

#### check_cli_available
```python
def check_cli_available() -> bool:
    """
    Check if the Go CLI is available.

    Returns:
        True if CLI is found and executable
    """
```

#### print_metrics_report
```python
def print_metrics_report(metrics: CompressionMetrics, title: str = "Compression Metrics"):
    """
    Print a formatted report of compression metrics.

    Args:
        metrics: CompressionMetrics to display
        title: Report title
    """
```

---

## Module: validate_perplexity.py

### Classes

#### ValidationResult
```python
@dataclass
class ValidationResult:
    """Final validation result."""
    baseline_perplexity: float      # PPL without compression
    compressed_perplexity: float    # PPL with compressed cache
    delta: float                    # compressed - baseline
    delta_percent: float            # (delta / baseline) * 100
    compression_ratio: float        # From compression metrics
    rmse: float                     # Root Mean Squared Error
    snr_db: float                   # Signal-to-Noise Ratio
    cosine_similarity: float        # Cosine similarity
    passed: bool                    # delta < threshold
    threshold: float = 0.5          # KVQuant standard
```

### Functions

#### load_model_and_tokenizer
```python
def load_model_and_tokenizer(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load TinyLlama model and tokenizer.

    Args:
        model_name: HuggingFace model name
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
```

#### compute_perplexity
```python
def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    stride: int = 512,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> float:
    """
    Compute perplexity on a dataset using sliding window.

    CRITICAL: Use stride < max_length for accurate perplexity.
    CRITICAL: Set context tokens to -100 for proper loss calculation.

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        dataset_name: Dataset name
        dataset_config: Dataset configuration
        split: Dataset split
        stride: Sliding window stride (512 recommended)
        max_samples: Limit number of windows (for quick testing)
        verbose: Show progress

    Returns:
        Perplexity score
    """
```

#### run_full_validation
```python
def run_full_validation(
    device: Optional[str] = None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> ValidationResult:
    """
    Run full validation pipeline.

    Steps:
    1. Load TinyLLaMA model
    2. Generate KV cache from sample text
    3. Compute baseline perplexity
    4. Compress KV cache with Go CLI
    5. Load reconstructed KV cache
    6. Compute perplexity with compressed cache

    Args:
        device: Target device (auto-detected if None)
        threshold: Maximum acceptable perplexity delta
        verbose: Show progress

    Returns:
        ValidationResult
    """
```

#### run_quick_test
```python
def run_quick_test(verbose: bool = True) -> ValidationResult:
    """
    Run a quick test with synthetic data (no model download required).

    This validates the pipeline without requiring GPU or model download.

    Returns:
        ValidationResult with synthetic data
    """
```

### CLI Usage

```bash
# Full validation (requires GPU and model download)
python validate_perplexity.py

# Quick test with synthetic data
python validate_perplexity.py --quick-test

# Force CPU mode
python validate_perplexity.py --cpu

# Custom threshold
python validate_perplexity.py --threshold 0.3

# Minimal output
python validate_perplexity.py --quiet
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--quick-test` | flag | false | Run quick test with synthetic data |
| `--cpu` | flag | false | Force CPU mode |
| `--threshold` | float | 0.5 | Maximum acceptable perplexity delta |
| `--quiet` | flag | false | Minimal output |

---

## Dependencies

### requirements.txt
```
torch>=2.0.0
transformers>=4.35.0
safetensors>=0.4.0
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
pytest>=7.0.0
```

### Installation
```bash
cd validation
pip install -r requirements.txt
```

---

## Usage Examples

### Full Validation Pipeline
```python
from validate_perplexity import run_full_validation

result = run_full_validation(device="cuda", threshold=0.5)

print(f"Baseline PPL: {result.baseline_perplexity:.4f}")
print(f"Compressed PPL: {result.compressed_perplexity:.4f}")
print(f"Delta: {result.delta:+.4f} ({result.delta_percent:+.2f}%)")
print(f"Passed: {result.passed}")
```

### KV Cache Extraction Only
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache_utils import extract_kv_cache, save_kv_cache

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

cache, metadata = extract_kv_cache(model, tokenizer, "Hello, world!")
save_kv_cache(cache, "kv_cache.safetensors")

print(f"Saved {metadata.num_layers} layers, {metadata.seq_len} tokens")
```

### Compression Only
```python
from compression_bridge import compress_kv_cache, print_metrics_report

metrics = compress_kv_cache(
    "kv_cache.safetensors",
    "compressed.safetensors",
    group_size=64
)

print_metrics_report(metrics, "KV Cache Compression Results")
```
