"""
KV Cache Utilities for TinyLLaMA Validation.

This module provides functions for extracting, saving, and loading KV cache
from TinyLLaMA models using HuggingFace Transformers.

CRITICAL NOTES:
- TinyLlama uses Grouped Query Attention (GQA) with 4 KV heads (NOT 32)
- Transformers 4.35+ uses DynamicCache, not tuple-based past_key_values
- NumPy doesn't support bfloat16, convert to float32 before saving
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from safetensors.torch import save_file, load_file


@dataclass
class KVCacheMetadata:
    """Metadata for KV cache tensors."""
    num_layers: int      # 22 for TinyLlama
    num_kv_heads: int    # 4 for TinyLlama (GQA)
    seq_len: int         # Variable
    head_dim: int        # 64 for TinyLlama
    dtype: str           # "float16" or "float32"
    model_name: str      # Model identifier

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "KVCacheMetadata":
        """Create from dictionary."""
        return cls(**d)


def extract_kv_cache(
    model,
    tokenizer,
    prompt: str,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[any, KVCacheMetadata]:
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

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> cache, metadata = extract_kv_cache(model, tokenizer, "Hello, world!")
        >>> print(f"Extracted {metadata.num_layers} layers with {metadata.seq_len} tokens")
    """
    from transformers import DynamicCache

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True if max_length else False,
    ).to(device)

    # Create empty cache
    cache = DynamicCache()

    # Forward pass with cache
    with torch.no_grad():
        _ = model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
        )

    # Extract metadata from cache
    # CRITICAL: TinyLlama has 4 KV heads (GQA), not 32
    num_layers = len(cache.key_cache)

    if num_layers == 0:
        raise ValueError("No KV cache generated. Check model configuration.")

    first_key = cache.key_cache[0]
    # Shape: [batch, num_kv_heads, seq_len, head_dim]
    batch_size, num_kv_heads, seq_len, head_dim = first_key.shape

    if batch_size != 1:
        raise ValueError(f"Expected batch size 1, got {batch_size}")

    # Determine dtype string
    dtype_str = str(first_key.dtype).replace("torch.", "")

    metadata = KVCacheMetadata(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype_str,
        model_name=model.config.name_or_path if hasattr(model.config, "name_or_path") else "unknown",
    )

    return cache, metadata


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
    path = Path(path)
    if metadata_path is None:
        metadata_path = path.with_suffix(".metadata.json")
    else:
        metadata_path = Path(metadata_path)

    num_layers = len(cache.key_cache)
    if num_layers == 0:
        raise ValueError("Empty cache, nothing to save")

    # Build tensors dict
    tensors = {}

    for i in range(num_layers):
        key = cache.key_cache[i]
        value = cache.value_cache[i]

        # CRITICAL: Convert bfloat16 to float32 (NumPy doesn't support bfloat16)
        if key.dtype == torch.bfloat16:
            key = key.float()
            value = value.float()

        # Move to CPU for saving
        tensors[f"layer_{i}_key"] = key.cpu()
        tensors[f"layer_{i}_value"] = value.cpu()

    # Get metadata from first layer
    first_key = cache.key_cache[0]
    _, num_kv_heads, seq_len, head_dim = first_key.shape

    # Determine saved dtype
    saved_key = tensors["layer_0_key"]
    dtype_str = str(saved_key.dtype).replace("torch.", "")

    metadata = KVCacheMetadata(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=dtype_str,
        model_name="",  # Will be set by caller if needed
    )

    # Save tensors
    save_file(tensors, str(path))

    # Save metadata as JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return metadata


def load_kv_cache(
    path: str,
    metadata_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[any, KVCacheMetadata]:
    """
    Load KV cache from safetensors file.

    Args:
        path: Path to safetensors file
        metadata_path: Optional path to metadata JSON (defaults to path.metadata.json)
        device: Device to load tensors to

    Returns:
        Tuple of (DynamicCache, KVCacheMetadata)
    """
    from transformers import DynamicCache

    path = Path(path)
    if metadata_path is None:
        metadata_path = path.with_suffix(".metadata.json")
    else:
        metadata_path = Path(metadata_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = KVCacheMetadata.from_dict(json.load(f))

    # Load tensors
    tensors = load_file(str(path), device=device)

    # Create DynamicCache
    cache = DynamicCache()

    for i in range(metadata.num_layers):
        key = tensors[f"layer_{i}_key"]
        value = tensors[f"layer_{i}_value"]

        # Update cache
        # DynamicCache expects (key, value) to be added
        cache.update(key, value, i)

    return cache, metadata


def create_kv_cache_from_tensors(
    key_tensors: List[torch.Tensor],
    value_tensors: List[torch.Tensor],
) -> any:
    """
    Create a DynamicCache from lists of key and value tensors.

    Args:
        key_tensors: List of key tensors, one per layer
        value_tensors: List of value tensors, one per layer

    Returns:
        DynamicCache populated with the tensors
    """
    from transformers import DynamicCache

    if len(key_tensors) != len(value_tensors):
        raise ValueError(
            f"Mismatch: {len(key_tensors)} key tensors vs {len(value_tensors)} value tensors"
        )

    cache = DynamicCache()

    for i, (key, value) in enumerate(zip(key_tensors, value_tensors)):
        cache.update(key, value, i)

    return cache


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
    errors = []

    num_layers = len(cache.key_cache)
    if num_layers != expected_layers:
        errors.append(f"Expected {expected_layers} layers, got {num_layers}")

    if num_layers == 0:
        return False, ["Cache is empty"]

    for i in range(num_layers):
        key = cache.key_cache[i]
        value = cache.value_cache[i]

        # Check key shape: [batch, num_kv_heads, seq_len, head_dim]
        if len(key.shape) != 4:
            errors.append(f"Layer {i} key: expected 4D tensor, got {len(key.shape)}D")
            continue

        batch, kv_heads, seq_len, head_dim = key.shape

        if kv_heads != expected_kv_heads:
            errors.append(
                f"Layer {i}: expected {expected_kv_heads} KV heads, got {kv_heads}"
            )

        if head_dim != expected_head_dim:
            errors.append(
                f"Layer {i}: expected head_dim {expected_head_dim}, got {head_dim}"
            )

        # Check key and value shapes match
        if key.shape != value.shape:
            errors.append(
                f"Layer {i}: key shape {key.shape} != value shape {value.shape}"
            )

    return len(errors) == 0, errors


def compute_kv_cache_stats(cache) -> dict:
    """
    Compute statistics about a KV cache.

    Args:
        cache: DynamicCache to analyze

    Returns:
        Dictionary with statistics
    """
    num_layers = len(cache.key_cache)
    if num_layers == 0:
        return {"error": "Empty cache"}

    first_key = cache.key_cache[0]
    batch, num_kv_heads, seq_len, head_dim = first_key.shape

    # Count total elements
    elements_per_layer = batch * num_kv_heads * seq_len * head_dim
    total_elements = elements_per_layer * num_layers * 2  # K and V

    # Estimate size in bytes
    dtype_size = first_key.element_size()
    total_bytes = total_elements * dtype_size

    # Compute value statistics
    all_values = []
    for i in range(num_layers):
        all_values.append(cache.key_cache[i].flatten())
        all_values.append(cache.value_cache[i].flatten())

    all_values = torch.cat(all_values)

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "dtype": str(first_key.dtype),
        "value_mean": float(all_values.mean()),
        "value_std": float(all_values.std()),
        "value_min": float(all_values.min()),
        "value_max": float(all_values.max()),
    }


# TinyLlama configuration constants
TINYLLAMA_NUM_LAYERS = 22
TINYLLAMA_NUM_KV_HEADS = 4  # GQA: 4 KV heads, not 32
TINYLLAMA_NUM_QUERY_HEADS = 32
TINYLLAMA_HEAD_DIM = 64
TINYLLAMA_HIDDEN_SIZE = 2048
TINYLLAMA_VOCAB_SIZE = 32000


if __name__ == "__main__":
    # Quick test/demo
    print("KV Cache Utilities - Quick Test")
    print("================================")
    print(f"TinyLlama config: {TINYLLAMA_NUM_LAYERS} layers, "
          f"{TINYLLAMA_NUM_KV_HEADS} KV heads (GQA), "
          f"{TINYLLAMA_HEAD_DIM} head_dim")

    # Test metadata
    metadata = KVCacheMetadata(
        num_layers=22,
        num_kv_heads=4,
        seq_len=128,
        head_dim=64,
        dtype="float32",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    print(f"\nMetadata test: {metadata.to_dict()}")
    print("\nAll utilities loaded successfully!")
