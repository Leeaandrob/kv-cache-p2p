"""
Compression Bridge for Go CLI.

This module provides a Python interface to the Go compression CLI,
enabling seamless compression/decompression of KV cache files.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CompressionMetrics:
    """Metrics returned by the Go CLI."""
    compression_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int
    rmse: float
    snr_db: float
    cosine_similarity: float
    max_error: float
    quality_level: str
    processing_time_ms: float
    passed: bool
    violations: list

    @classmethod
    def from_dict(cls, d: dict) -> "CompressionMetrics":
        """Create from dictionary (JSON output from CLI)."""
        return cls(
            compression_ratio=d.get("compression_ratio", 0.0),
            original_size_bytes=d.get("original_size_bytes", 0),
            compressed_size_bytes=d.get("compressed_size_bytes", 0),
            rmse=d.get("rmse", 0.0),
            snr_db=d.get("snr_db", 0.0),
            cosine_similarity=d.get("cosine_similarity", 0.0),
            max_error=d.get("max_error", 0.0),
            quality_level=d.get("quality_level", "UNKNOWN"),
            processing_time_ms=d.get("processing_time_ms", 0.0),
            passed=d.get("quality_passed", False),
            violations=d.get("violations", []),
        )


def _run_cli_command(cmd: list, verbose: bool = False) -> CompressionMetrics:
    """
    Execute a CLI command and parse the JSON response.

    Args:
        cmd: Command list to execute
        verbose: Print command before execution

    Returns:
        CompressionMetrics parsed from JSON output

    Raises:
        subprocess.CalledProcessError: If command fails
        ValueError: If output is not valid JSON
    """
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    try:
        metrics_dict = json.loads(result.stdout)
        return CompressionMetrics.from_dict(metrics_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from CLI: {result.stdout}") from e


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
    # Check environment variable
    env_path = os.environ.get("KV_COMPRESS_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check relative paths
    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent / "bin" / "kv-compress",
        Path.cwd() / "bin" / "kv-compress",
        script_dir.parent / "bin" / "kv-compress.exe",  # Windows
        Path.cwd() / "bin" / "kv-compress.exe",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # Check PATH
    import shutil
    path_binary = shutil.which("kv-compress")
    if path_binary:
        return path_binary

    raise FileNotFoundError(
        "Could not find kv-compress binary. "
        "Build it with 'make build-compress' or set KV_COMPRESS_PATH environment variable."
    )


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
    """
    cli_path = find_cli_binary()

    cmd = [
        cli_path,
        "-input", str(input_path),
        "-output", str(output_path),
        "-group-size", str(group_size),
        "-sparsity", str(sparsity),
    ]

    if verbose:
        cmd.append("-verbose")

    return _run_cli_command(cmd, verbose)


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
    cli_path = find_cli_binary()

    cmd = [
        cli_path,
        "-decompress",
        "-input", str(input_path),
        "-output", str(output_path),
    ]

    if verbose:
        cmd.append("-verbose")

    return _run_cli_command(cmd, verbose)


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
    cli_path = find_cli_binary()

    cmd = [
        cli_path,
        "-input", str(input_path),
        "-metrics-only",
        "-group-size", str(group_size),
    ]

    if verbose:
        cmd.append("-verbose")

    return _run_cli_command(cmd, verbose)


def check_cli_available() -> bool:
    """
    Check if the Go CLI is available.

    Returns:
        True if CLI is found and executable
    """
    try:
        cli_path = find_cli_binary()
        result = subprocess.run(
            [cli_path, "-h"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 or "-input" in result.stderr
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def print_metrics_report(metrics: CompressionMetrics, title: str = "Compression Metrics"):
    """
    Print a formatted report of compression metrics.

    Args:
        metrics: CompressionMetrics to display
        title: Report title
    """
    status = "PASSED" if metrics.passed else "FAILED"
    status_symbol = "[OK]" if metrics.passed else "[FAIL]"

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Status: {status_symbol} {status}")
    print(f"  Quality Level: {metrics.quality_level}")
    print(f"{'-' * 60}")
    print(f"  Compression Ratio:  {metrics.compression_ratio:.2f}x")
    print(f"  Original Size:      {metrics.original_size_bytes:,} bytes")
    print(f"  Compressed Size:    {metrics.compressed_size_bytes:,} bytes")
    print(f"{'-' * 60}")
    print(f"  RMSE:               {metrics.rmse:.6f}")
    print(f"  SNR:                {metrics.snr_db:.2f} dB")
    print(f"  Cosine Similarity:  {metrics.cosine_similarity:.6f}")
    print(f"  Max Error:          {metrics.max_error:.6f}")
    print(f"{'-' * 60}")
    print(f"  Processing Time:    {metrics.processing_time_ms:.2f} ms")

    if metrics.violations:
        print(f"{'-' * 60}")
        print("  Violations:")
        for v in metrics.violations:
            print(f"    - {v}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print("Compression Bridge - Quick Test")
    print("================================")

    # Check CLI availability
    if check_cli_available():
        print("[OK] Go CLI found")
        try:
            cli_path = find_cli_binary()
            print(f"     Path: {cli_path}")
        except FileNotFoundError:
            pass
    else:
        print("[WARN] Go CLI not found")
        print("       Build with: cd .. && make build-compress")

    # Test metrics dataclass
    test_metrics = CompressionMetrics(
        compression_ratio=3.8,
        original_size_bytes=1000000,
        compressed_size_bytes=263158,
        rmse=0.05,
        snr_db=25.0,
        cosine_similarity=0.9995,
        max_error=0.15,
        quality_level="GOOD",
        processing_time_ms=50.0,
        passed=True,
        violations=[],
    )
    print("\n[OK] CompressionMetrics test:")
    print_metrics_report(test_metrics, "Test Metrics")
