"""FFT auto-benchmarking system for optimal backend selection.

This module provides automatic benchmarking of available FFT backends
to determine the fastest option for the user's hardware. Results are
cached for quick subsequent access.
"""

import json
import platform
import time
from datetime import datetime, timedelta
from typing import Any, TypedDict

import numpy as np


class BackendResult(TypedDict):
    """Results for a single backend benchmark."""

    time_ms: float | None
    speedup: float | None
    available: bool


class PlatformInfo(TypedDict):
    """Platform information."""

    system: str
    processor: str
    python_version: str


class BenchmarkResults(TypedDict):
    """Complete benchmark results structure."""

    version: str
    timestamp: str
    grid_size: list[int]
    platform: PlatformInfo
    results: dict[str, BackendResult]
    fastest: str | None


def _synchronize_backend(backend: Any, array: Any) -> None:
    """Ensure GPU operations complete before timing.

    Parameters
    ----------
    backend : Any
        Backend instance
    array : Any
        Array to synchronize
    """
    # backend parameter currently unused but kept for API consistency
    _ = backend
    if hasattr(array, "get"):  # CuPy
        array.get()
    elif hasattr(array, "queue"):  # OpenCL
        array.queue.finish()


def benchmark_backend(
    backend_name: str, grid_size: tuple[int, int], num_trials: int = 5
) -> float | None:
    """Benchmark FFT performance for a single backend.

    Parameters
    ----------
    backend_name : str
        Name of backend to test ("CPU", "CUPY", or "CL")
    grid_size : tuple[int, int]
        Grid dimensions (NX, NY) for testing
    num_trials : int
        Number of timing trials to run

    Returns
    -------
    float or None
        Median FFT time in milliseconds, or None if backend fails

    """
    try:
        # Import here to avoid circular dependency
        from . import get_backend

        backend = get_backend(backend_name)

        # Allocate test array (initialized to zeros)
        dtype = np.dtype(np.complex64)
        A = backend.allocate_field(grid_size, dtype)

        # Build FFT plan
        plans = backend.build_fft(grid_size, axes=(-2, -1), dtype=dtype)

        # Warmup runs
        for _ in range(3):
            backend.fft(A, plans)
            backend.ifft(A, plans)

        # Synchronize for GPU backends
        _synchronize_backend(backend, A)

        # Timed trials
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            backend.fft(A, plans)
            backend.ifft(A, plans)
            _synchronize_backend(backend, A)
            times.append((time.perf_counter() - t0) * 1000)

        return float(np.median(times))

    except Exception as e:
        print(f"Warning: Benchmarking {backend_name} failed: {e}")
        return None


def benchmark_all_backends(
    grid_size: tuple[int, int] = (2048, 2048),
) -> BenchmarkResults:
    """Benchmark all available FFT backends.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Grid dimensions for testing (default: 2048x2048)

    Returns
    -------
    BenchmarkResults
        Dictionary with benchmark results for each backend

    """
    from . import list_available_backends

    results: BenchmarkResults = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "grid_size": list(grid_size),
        "platform": {
            "system": platform.system(),
            "processor": platform.machine(),
            "python_version": platform.python_version(),
        },
        "results": {},
        "fastest": None,
    }

    available_backends = list_available_backends()

    # Benchmark each available backend
    for backend_name in available_backends:
        print(f"Benchmarking {backend_name} backend...")
        time_ms = benchmark_backend(backend_name, grid_size)

        if time_ms is not None:
            results["results"][backend_name] = BackendResult(
                time_ms=time_ms,
                speedup=None,  # Will be computed after
                available=True,
            )
        else:
            results["results"][backend_name] = BackendResult(
                time_ms=None, speedup=None, available=False
            )

    # Compute speedups relative to CPU
    if "CPU" in results["results"] and results["results"]["CPU"]["available"]:
        cpu_time = results["results"]["CPU"]["time_ms"]
        if cpu_time is not None:
            for backend_name in results["results"]:
                if results["results"][backend_name]["available"]:
                    backend_time = results["results"][backend_name]["time_ms"]
                    if backend_time is not None:
                        results["results"][backend_name]["speedup"] = (
                            cpu_time / backend_time
                        )

    # Determine fastest backend
    fastest: str | None = None
    fastest_time = float("inf")
    for backend_name, data in results["results"].items():
        if data["available"] and data["time_ms"] is not None:
            if data["time_ms"] < fastest_time:
                fastest_time = data["time_ms"]
                fastest = backend_name

    results["fastest"] = fastest

    return results


def load_benchmark_cache() -> BenchmarkResults | None:
    """Load cached benchmark results from disk.

    Returns
    -------
    BenchmarkResults or None
        Dictionary with cached results, or None if cache invalid/missing

    """
    from ..utils import get_benchmark_cache_path

    cache_path = get_benchmark_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cache = json.load(f)

        # Validate cache structure
        if "version" not in cache or "timestamp" not in cache:
            return None

        # Check if cache is stale (>30 days)
        cache_time = datetime.fromisoformat(cache["timestamp"])
        if datetime.now() - cache_time > timedelta(days=30):
            print("Benchmark cache is stale (>30 days), re-benchmarking...")
            return None

        return cache

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Warning: Invalid benchmark cache: {e}")
        return None


def save_benchmark_cache(results: BenchmarkResults) -> None:
    """Save benchmark results to cache file.

    Parameters
    ----------
    results : BenchmarkResults
        Dictionary with benchmark results
    """
    from ..utils import get_benchmark_cache_path

    cache_path = get_benchmark_cache_path()

    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results cached to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save benchmark cache: {e}")


def invalidate_cache() -> None:
    """Delete cached benchmark results, forcing re-benchmark on next call."""
    from ..utils import get_benchmark_cache_path

    cache_path = get_benchmark_cache_path()

    if cache_path.exists():
        cache_path.unlink()
        print(f"Benchmark cache invalidated: {cache_path}")
    else:
        print("No benchmark cache found")


def get_fastest_backend(
    grid_size: tuple[int, int] = (2048, 2048), force_benchmark: bool = False
) -> str:
    """Get the optimal backend for current hardware.

    Automatically benchmarks all available backends and returns
    the fastest one. Results are cached for future calls.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Typical grid size for benchmarking
    force_benchmark : bool
        Force re-benchmark even if cache exists

    Returns
    -------
    str
        Name of fastest backend ("CPU", "CUPY", or "CL")

    """
    # Check cache first (unless forced)
    if not force_benchmark:
        cache = load_benchmark_cache()
        if cache is not None:
            # Check if grid size matches
            cached_grid = tuple(cache.get("grid_size", []))
            if cached_grid == grid_size:
                fastest = cache.get("fastest")
                if fastest:
                    return fastest
            else:
                print(
                    f"Grid size mismatch (cached: {cached_grid}, requested: {grid_size}), re-benchmarking..."
                )

    # Run benchmarks
    print(f"Benchmarking FFT backends on {grid_size[0]}x{grid_size[1]} grid...")
    results = benchmark_all_backends(grid_size)

    # Save to cache
    save_benchmark_cache(results)

    # Return fastest backend (fallback to CPU if none found)
    fastest = results.get("fastest", "CPU")
    if fastest is None:
        print("Warning: No working backends found, falling back to CPU")
        fastest = "CPU"

    return fastest
