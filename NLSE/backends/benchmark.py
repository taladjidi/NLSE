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

from ..utils import machine_identity, say, write_json_atomically

# Bumped whenever a cached timing stops meaning what it meant, so an older
# cache is discarded rather than believed. 2: the CPU FFT moved from pyFFTW to
# scipy. 3: the CUPY timing waits on the stream instead of copying to the host,
# and the MLX one waits on the transform's result instead of its input. 4: one
# record per grid size in the one file, so a second grid does not evict the first.
CACHE_VERSION = "4"
# How long a record is believed: drivers and libraries change under it.
CACHE_MAX_AGE = timedelta(days=30)


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


class BenchmarkCache(TypedDict):
    """The cache file: one record per grid size, keyed by ``_grid_key``."""

    version: str
    grids: dict[str, BenchmarkResults]


def _grid_key(grid_size: tuple | list) -> str:
    """Return the key a grid's record is cached under, as ``"NXxNY"``.

    Parameters
    ----------
    grid_size : tuple or list
        Grid dimensions.

    Returns
    -------
    str
        The dimensions joined by ``x``: JSON keys are strings.
    """
    return "x".join(str(n) for n in grid_size)


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
        from . import _installed_backend

        backend = _installed_backend(backend_name)

        # Allocate test array (initialized to zeros)
        dtype = np.dtype(np.complex64)
        A = backend.allocate_field(grid_size, dtype)

        # Build FFT plan
        plans = backend.build_fft(grid_size, axes=(-2, -1), dtype=dtype)

        # Warmup runs. Each transform's result is kept: MLX's are lazy and
        # return a new array, so waiting on the input would time nothing.
        for _ in range(3):
            A = backend.fft(A, plans)
            A = backend.ifft(A, plans)

        # Synchronize for GPU backends
        backend.synchronize(A)

        # Timed trials
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            A = backend.fft(A, plans)
            A = backend.ifft(A, plans)
            backend.synchronize(A)
            times.append((time.perf_counter() - t0) * 1000)

        return float(np.median(times))

    except Exception as e:
        say(f"Warning: Benchmarking {backend_name} failed: {e}")
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
        "version": CACHE_VERSION,
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
        say(f"Benchmarking {backend_name} backend...")
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


def _is_grid_record(record: Any) -> bool:
    """Whether one grid's record has the fields read from it, typed as read.

    Parameters
    ----------
    record : Any
        One value of the cache file's ``grids``.

    Returns
    -------
    bool
        True if it is a record ``benchmark_all_backends`` could have written.
    """
    if not isinstance(record, dict):
        return False
    grid = record.get("grid_size")
    results = record.get("results")
    return (
        isinstance(record.get("version"), str)
        and isinstance(record.get("timestamp"), str)
        and isinstance(record.get("platform"), dict)
        and isinstance(grid, list)
        and all(isinstance(n, int) for n in grid)
        and isinstance(record.get("fastest"), (str, type(None)))
        and isinstance(results, dict)
        and all(
            isinstance(entry, dict)
            and isinstance(entry.get("time_ms"), (int, float, type(None)))
            for entry in results.values()
        )
    )


def _is_cache_record(cache: Any) -> bool:
    """Whether a decoded cache file is one this version writes.

    Parameters
    ----------
    cache : Any
        What ``json.load`` returned.

    Returns
    -------
    bool
        True if it is a ``BenchmarkCache`` of grid records.
    """
    if not isinstance(cache, dict):
        return False
    grids = cache.get("grids")
    return (
        isinstance(cache.get("version"), str)
        and isinstance(grids, dict)
        and all(_is_grid_record(record) for record in grids.values())
    )


def _read_cache_file(announce: bool) -> BenchmarkCache | None:
    """Read the cache file, if it is one of this version.

    Parameters
    ----------
    announce : bool
        Say why a file is ignored. False where the caller replaces it anyway.

    Returns
    -------
    BenchmarkCache or None
        The file's records; None if it is missing, unreadable, of another
        version or not of that form.
    """
    from ..utils import get_benchmark_cache_path

    # OSError: the cache directory cannot be made, or the file read.
    # ValueError: not JSON. RecursionError: JSON nested too deep.
    try:
        cache_path = get_benchmark_cache_path()
        if not cache_path.exists():
            return None
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError, RecursionError) as e:
        if announce:
            say(f"Warning: Invalid benchmark cache: {e}")
        return None

    if isinstance(cache, dict) and cache.get("version") != CACHE_VERSION:
        if announce:
            say(
                "Benchmark cache predates a change to what it timed, re-benchmarking..."
            )
        return None
    if not _is_cache_record(cache):
        if announce:
            say("Benchmark cache is not a record this version writes, ignoring it.")
        return None
    return cache


def load_benchmark_cache(grid_size: tuple | list) -> BenchmarkResults | None:
    """Load the cached benchmark results for one grid size.

    Parameters
    ----------
    grid_size : tuple or list
        Grid the results must have been measured at.

    Returns
    -------
    BenchmarkResults or None
        That grid's record, or None if there is none, or it is invalid, from
        another machine or stale.

    """
    cache = _read_cache_file(announce=True)
    if cache is None:
        return None
    record = cache["grids"].get(_grid_key(grid_size))
    if record is None:
        return None

    # Recorded from the start and, like the version, never compared. A
    # cache lives in the user's cache directory, which is inside a home
    # directory that a lab machine and a laptop may well share, and the
    # answer it holds is a statement about one machine's hardware.
    cached_platform = record["platform"]
    here = machine_identity()
    there = (cached_platform.get("system"), cached_platform.get("processor"))
    if there != here:
        say(f"Benchmark cache was written on {there}, not {here}, re-benchmarking...")
        return None

    # ValueError: a timestamp that is not a date. TypeError: a timestamp with
    # a time zone.
    try:
        age = datetime.now() - datetime.fromisoformat(record["timestamp"])
    except (ValueError, TypeError) as e:
        say(f"Warning: Invalid benchmark cache: {e}")
        return None
    if age > CACHE_MAX_AGE:
        say(
            f"Benchmark cache is stale (>{CACHE_MAX_AGE.days} days), re-benchmarking..."
        )
        return None

    return record


def save_benchmark_cache(results: BenchmarkResults) -> None:
    """Save one grid's benchmark results, keeping the other grids' records.

    Parameters
    ----------
    results : BenchmarkResults
        Dictionary with benchmark results
    """
    from ..utils import get_benchmark_cache_path

    try:
        cache_path = get_benchmark_cache_path()
    except OSError as e:
        say(f"Warning: Failed to save benchmark cache: {e}")
        return
    cache = _read_cache_file(announce=False)
    grids = {} if cache is None else cache["grids"]
    grids[_grid_key(results["grid_size"])] = results
    try:
        write_json_atomically(cache_path, {"version": CACHE_VERSION, "grids": grids})
        say(f"Benchmark results cached to {cache_path}")
    except Exception as e:
        say(f"Warning: Failed to save benchmark cache: {e}")


def invalidate_cache() -> None:
    """Delete cached benchmark results, forcing re-benchmark on next call."""
    from ..utils import get_benchmark_cache_path

    cache_path = get_benchmark_cache_path()

    if cache_path.exists():
        cache_path.unlink()
        say(f"Benchmark cache invalidated: {cache_path}")
    else:
        say("No benchmark cache found")


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
    from . import list_available_backends

    # Check cache first (unless forced)
    if not force_benchmark:
        cache = load_benchmark_cache(grid_size)
        available = list_available_backends()
        # No record for this grid, or one that timed other backends than the
        # ones installed here (a shared cache directory, a backend added
        # since): measure again.
        if cache is not None and set(cache["results"]) == set(available):
            fastest = cache.get("fastest")
            if fastest in available:
                return fastest

    # Run benchmarks
    say(f"Benchmarking FFT backends on {grid_size[0]}x{grid_size[1]} grid...")
    results = benchmark_all_backends(grid_size)

    # Save to cache
    save_benchmark_cache(results)

    # Return fastest backend (fallback to CPU if none found)
    fastest = results.get("fastest", "CPU")
    if fastest is None:
        say("Warning: No working backends found, falling back to CPU")
        fastest = "CPU"

    return fastest
