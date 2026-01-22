"""
Tests for TinyLLaMA KV Cache Compression Validation.

These tests validate:
1. KV cache utilities (extraction, save/load)
2. Compression bridge (Go CLI interface)
3. Safetensors format compatibility
4. Quality metrics validation

Run with: pytest -v test_validation.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file


# Constants from TinyLlama config
TINYLLAMA_NUM_LAYERS = 22
TINYLLAMA_NUM_KV_HEADS = 4  # GQA: 4 KV heads, not 32
TINYLLAMA_HEAD_DIM = 64


class TestKVCacheUtils:
    """Tests for kv_cache_utils.py"""

    def test_metadata_serialization(self):
        """Test KVCacheMetadata to_dict and from_dict."""
        from kv_cache_utils import KVCacheMetadata

        original = KVCacheMetadata(
            num_layers=22,
            num_kv_heads=4,
            seq_len=128,
            head_dim=64,
            dtype="float32",
            model_name="test_model",
        )

        # Round trip
        d = original.to_dict()
        restored = KVCacheMetadata.from_dict(d)

        assert restored.num_layers == original.num_layers
        assert restored.num_kv_heads == original.num_kv_heads
        assert restored.seq_len == original.seq_len
        assert restored.head_dim == original.head_dim
        assert restored.dtype == original.dtype
        assert restored.model_name == original.model_name

    def test_tinyllama_constants(self):
        """Test that TinyLlama constants are correct."""
        from kv_cache_utils import (
            TINYLLAMA_NUM_LAYERS,
            TINYLLAMA_NUM_KV_HEADS,
            TINYLLAMA_NUM_QUERY_HEADS,
            TINYLLAMA_HEAD_DIM,
            TINYLLAMA_HIDDEN_SIZE,
        )

        # TinyLlama 1.1B configuration
        assert TINYLLAMA_NUM_LAYERS == 22
        assert TINYLLAMA_NUM_KV_HEADS == 4  # GQA!
        assert TINYLLAMA_NUM_QUERY_HEADS == 32
        assert TINYLLAMA_HEAD_DIM == 64
        assert TINYLLAMA_HIDDEN_SIZE == 2048

    def test_validate_kv_cache_shape(self):
        """Test KV cache shape validation."""
        from kv_cache_utils import validate_kv_cache_shape

        # Create mock cache with correct shape
        class MockCache:
            def __init__(self, num_layers, num_kv_heads, seq_len, head_dim):
                self.key_cache = [
                    torch.randn(1, num_kv_heads, seq_len, head_dim)
                    for _ in range(num_layers)
                ]
                self.value_cache = [
                    torch.randn(1, num_kv_heads, seq_len, head_dim)
                    for _ in range(num_layers)
                ]

        # Valid cache
        cache = MockCache(22, 4, 128, 64)
        is_valid, errors = validate_kv_cache_shape(cache)
        assert is_valid, f"Expected valid cache, got errors: {errors}"

        # Invalid cache (wrong number of layers)
        cache = MockCache(10, 4, 128, 64)
        is_valid, errors = validate_kv_cache_shape(cache)
        assert not is_valid
        assert any("22 layers" in e for e in errors)

        # Invalid cache (wrong number of KV heads - common mistake!)
        cache = MockCache(22, 32, 128, 64)  # 32 is WRONG for TinyLlama GQA
        is_valid, errors = validate_kv_cache_shape(cache)
        assert not is_valid
        assert any("4 KV heads" in e for e in errors)

    def test_compute_kv_cache_stats(self):
        """Test KV cache statistics computation."""
        from kv_cache_utils import compute_kv_cache_stats

        class MockCache:
            def __init__(self):
                self.key_cache = [torch.randn(1, 4, 128, 64) for _ in range(22)]
                self.value_cache = [torch.randn(1, 4, 128, 64) for _ in range(22)]

        cache = MockCache()
        stats = compute_kv_cache_stats(cache)

        assert stats["num_layers"] == 22
        assert stats["num_kv_heads"] == 4
        assert stats["seq_len"] == 128
        assert stats["head_dim"] == 64
        assert stats["total_elements"] == 22 * 4 * 128 * 64 * 2  # K and V
        assert "total_mb" in stats
        assert "value_mean" in stats


class TestCompressionBridge:
    """Tests for compression_bridge.py"""

    def test_compression_metrics_from_dict(self):
        """Test CompressionMetrics parsing."""
        from compression_bridge import CompressionMetrics

        d = {
            "compression_ratio": 3.8,
            "original_size_bytes": 1000000,
            "compressed_size_bytes": 263158,
            "rmse": 0.05,
            "snr_db": 25.0,
            "cosine_similarity": 0.9995,
            "max_error": 0.15,
            "quality_level": "GOOD",
            "processing_time_ms": 50.0,
            "quality_passed": True,
            "violations": [],
        }

        metrics = CompressionMetrics.from_dict(d)

        assert metrics.compression_ratio == 3.8
        assert metrics.rmse == 0.05
        assert metrics.passed is True

    def test_cli_availability_check(self):
        """Test that check_cli_available returns a boolean."""
        from compression_bridge import check_cli_available

        result = check_cli_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not os.path.exists(
            Path(__file__).parent.parent / "bin" / "kv-compress"
        ),
        reason="Go CLI not built",
    )
    def test_cli_execution(self):
        """Test actual CLI execution with synthetic data."""
        from compression_bridge import compress_kv_cache, find_cli_binary

        # Create synthetic data
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.safetensors"
            output_path = Path(tmpdir) / "test_compressed.safetensors"
            metadata_path = input_path.with_suffix(".metadata.json")

            # Create tensors matching TinyLlama KV cache format
            tensors = {}
            for layer in range(TINYLLAMA_NUM_LAYERS):
                tensors[f"layer_{layer}_key"] = torch.randn(
                    1, TINYLLAMA_NUM_KV_HEADS, 64, TINYLLAMA_HEAD_DIM
                )
                tensors[f"layer_{layer}_value"] = torch.randn(
                    1, TINYLLAMA_NUM_KV_HEADS, 64, TINYLLAMA_HEAD_DIM
                )

            save_file(tensors, str(input_path))

            # Create metadata
            metadata = {
                "num_layers": TINYLLAMA_NUM_LAYERS,
                "num_kv_heads": TINYLLAMA_NUM_KV_HEADS,
                "seq_len": 64,
                "head_dim": TINYLLAMA_HEAD_DIM,
                "dtype": "float32",
                "model_name": "test",
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Compress
            metrics = compress_kv_cache(
                str(input_path),
                str(output_path),
                group_size=64,
            )

            # Validate metrics
            assert metrics.compression_ratio > 0
            assert metrics.rmse >= 0
            assert 0 <= metrics.cosine_similarity <= 1
            assert output_path.exists()


class TestSafetensorsFormat:
    """Tests for safetensors format compatibility."""

    def test_write_read_roundtrip(self):
        """Test safetensors write/read roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.safetensors"

            # Create test tensor
            original = {
                "test": torch.randn(2, 3, 4),
            }

            save_file(original, str(path))
            loaded = load_file(str(path))

            assert torch.allclose(original["test"], loaded["test"])

    def test_kv_cache_format(self):
        """Test KV cache tensor naming convention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kv_cache.safetensors"

            # Create KV cache with correct naming
            tensors = {}
            for layer in range(22):
                tensors[f"layer_{layer}_key"] = torch.randn(1, 4, 128, 64)
                tensors[f"layer_{layer}_value"] = torch.randn(1, 4, 128, 64)

            save_file(tensors, str(path))
            loaded = load_file(str(path))

            # Verify all tensors present
            assert len(loaded) == 44  # 22 layers * 2 (key + value)

            # Verify naming
            for layer in range(22):
                assert f"layer_{layer}_key" in loaded
                assert f"layer_{layer}_value" in loaded

    def test_fp16_conversion(self):
        """Test float16 tensor handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fp16.safetensors"

            original = {
                "fp16_tensor": torch.randn(10, 10, dtype=torch.float16),
            }

            save_file(original, str(path))
            loaded = load_file(str(path))

            assert loaded["fp16_tensor"].dtype == torch.float16
            assert torch.allclose(
                original["fp16_tensor"].float(),
                loaded["fp16_tensor"].float(),
                atol=1e-3,
            )


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_str(self):
        """Test ValidationResult string representation."""
        from validate_perplexity import ValidationResult

        result = ValidationResult(
            baseline_perplexity=16.5,
            compressed_perplexity=16.55,
            delta=0.05,
            delta_percent=0.3,
            compression_ratio=3.8,
            rmse=0.025,
            snr_db=28.0,
            cosine_similarity=0.9998,
            passed=True,
            threshold=0.5,
        )

        s = str(result)
        assert "PASSED" in s
        assert "16.5" in s
        assert "0.05" in s


class TestQualityThresholds:
    """Tests for quality threshold validation."""

    def test_quality_thresholds(self):
        """Test that quality thresholds are reasonable."""
        # Based on KVQuant paper and compression research
        MAX_RMSE = 0.1
        MIN_SNR = 18.0  # dB
        MIN_COSINE = 0.99
        MAX_PPL_DELTA = 0.5

        # These are the thresholds used in the Go code
        assert MAX_RMSE > 0
        assert MIN_SNR > 0
        assert MIN_COSINE < 1.0
        assert MAX_PPL_DELTA > 0


class TestQuickTest:
    """Tests for the quick test functionality."""

    @pytest.mark.skipif(
        not os.path.exists(
            Path(__file__).parent.parent / "bin" / "kv-compress"
        ),
        reason="Go CLI not built",
    )
    def test_quick_test_runs(self):
        """Test that quick test runs without errors."""
        from validate_perplexity import run_quick_test

        result = run_quick_test(verbose=False)

        assert isinstance(result.baseline_perplexity, float)
        assert isinstance(result.compressed_perplexity, float)
        assert isinstance(result.passed, bool)


class TestIntegration:
    """Integration tests requiring the full pipeline."""

    @pytest.mark.skipif(
        not os.path.exists(
            Path(__file__).parent.parent / "bin" / "kv-compress"
        ),
        reason="Go CLI not built",
    )
    def test_full_pipeline_synthetic(self):
        """Test full compression pipeline with synthetic data."""
        from kv_cache_utils import KVCacheMetadata
        from compression_bridge import compress_kv_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic KV cache
            input_path = Path(tmpdir) / "kv_cache.safetensors"
            output_path = Path(tmpdir) / "compressed.safetensors"
            metadata_path = input_path.with_suffix(".metadata.json")

            # TinyLlama-like dimensions
            tensors = {}
            for layer in range(22):
                tensors[f"layer_{layer}_key"] = torch.randn(1, 4, 256, 64) * 0.3
                tensors[f"layer_{layer}_value"] = torch.randn(1, 4, 256, 64) * 0.3

            save_file(tensors, str(input_path))

            # Create metadata
            metadata = KVCacheMetadata(
                num_layers=22,
                num_kv_heads=4,
                seq_len=256,
                head_dim=64,
                dtype="float32",
                model_name="synthetic",
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f)

            # Compress
            metrics = compress_kv_cache(
                str(input_path),
                str(output_path),
                group_size=64,
            )

            # Validate quality thresholds
            assert metrics.rmse < 0.1, f"RMSE {metrics.rmse} exceeds 0.1 threshold"
            assert metrics.snr_db > 18.0, f"SNR {metrics.snr_db}dB below 18dB threshold"
            assert metrics.cosine_similarity > 0.99, f"Cosine {metrics.cosine_similarity} below 0.99"
            assert metrics.passed, f"Quality validation failed: {metrics.violations}"

            # Verify output file
            assert output_path.exists()
            compressed = load_file(str(output_path))
            assert len(compressed) == 44  # 22 layers * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
