"""Tests for FFT auto-benchmarking system."""

import json
import time
from pathlib import Path

import pytest
from NLSE.backends import benchmark, list_available_backends
from NLSE.utils import get_benchmark_cache_path, get_cache_dir


class TestCacheDir:
    """Test that the cache directory lives outside the installed package."""

    def test_cache_dir_outside_package(self):
        """Cache dir must not be under the NLSE package directory.

        It used to be <package>/.cache, which fails on a read-only install
        and, on uninstall, leaves a directory behind that Python then
        imports as an empty namespace package.
        """
        import NLSE

        package_dir = Path(NLSE.__file__).resolve().parent
        cache_dir = get_cache_dir().resolve()
        assert cache_dir != package_dir
        assert package_dir not in cache_dir.parents

    def test_cache_dir_created(self):
        """get_cache_dir() should create the directory if needed."""
        cache_dir = get_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_benchmark_cache_in_cache_dir(self):
        """Benchmark cache file should be inside the cache dir."""
        cache_dir = get_cache_dir()
        bench_path = get_benchmark_cache_path()
        assert bench_path.parent == cache_dir
        assert bench_path.name == "fft_benchmark.json"

    def test_wisdom_in_cache_dir(self):
        """FFTW wisdom should be written inside the cache dir."""
        cache_dir = get_cache_dir()
        wisdom_path = cache_dir / "fft.wisdom"

        # Remove existing wisdom to test it gets recreated
        if wisdom_path.exists():
            wisdom_path.unlink()

        from NLSE.backends.cpu import CPUBackend

        backend = CPUBackend()
        shape = (64, 64)
        dtype = __import__("numpy").complex64
        backend.build_fft(shape, axes=(-2, -1), dtype=dtype)

        assert wisdom_path.exists()


class TestBenchmarkBackend:
    """Test benchmarking individual backends."""

    def test_benchmark_cpu(self):
        """CPU backend should always be available and benchmarkable."""
        time_ms = benchmark.benchmark_backend("CPU", (256, 256), num_trials=3)
        assert time_ms is not None
        assert time_ms > 0
        assert time_ms < 1000  # Sanity check - should be fast on small grid

    @pytest.mark.parametrize("backend_name", list_available_backends())
    def test_benchmark_each_available_backend(self, backend_name):
        """Test benchmarking each available backend."""
        time_ms = benchmark.benchmark_backend(backend_name, (256, 256), num_trials=3)
        assert time_ms is not None
        assert time_ms > 0

    def test_benchmark_invalid_backend(self):
        """Test that invalid backend returns None."""
        time_ms = benchmark.benchmark_backend("INVALID", (256, 256), num_trials=3)
        assert time_ms is None


class TestBenchmarkAll:
    """Test benchmarking all backends at once."""

    def test_benchmark_all_backends(self):
        """Test benchmarking all available backends."""
        results = benchmark.benchmark_all_backends((256, 256))

        # Check structure
        assert "version" in results
        assert "timestamp" in results
        assert "grid_size" in results
        assert "platform" in results
        assert "results" in results
        assert "fastest" in results

        # Check CPU is present and working
        assert "CPU" in results["results"]
        assert results["results"]["CPU"]["available"] is True
        assert results["results"]["CPU"]["time_ms"] > 0

        # Check fastest is valid
        assert results["fastest"] in list_available_backends()

    def test_benchmark_speedup_calculation(self):
        """Test that speedup is calculated correctly."""
        results = benchmark.benchmark_all_backends((256, 256))

        cpu_speedup = results["results"]["CPU"]["speedup"]

        # CPU should have speedup of 1.0
        assert cpu_speedup == 1.0

        # Other backends should have speedup relative to CPU
        for backend_name, data in results["results"].items():
            if data["available"] and backend_name != "CPU":
                assert data["speedup"] is not None
                assert data["speedup"] > 0


class TestGetFastestBackend:
    """Test automatic selection of fastest backend."""

    def test_get_fastest_backend(self):
        """Test fastest backend selection."""
        # Force benchmark to ensure clean state
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest in list_available_backends()

    def test_get_fastest_returns_valid_backend(self):
        """Test that fastest backend is always valid."""
        fastest = benchmark.get_fastest_backend((512, 512), force_benchmark=True)

        # Should be one of the available backends
        available = list_available_backends()
        assert fastest in available


class TestCachePersistence:
    """Test benchmark caching functionality."""

    def test_cache_creation(self):
        """Test that benchmark results are cached."""
        cache_path = get_benchmark_cache_path()

        # Clear cache first
        if cache_path.exists():
            cache_path.unlink()

        # Run benchmark
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        # Check cache was created
        assert cache_path.exists()

        # Verify cache content
        with open(cache_path) as f:
            cache = json.load(f)

        assert cache["fastest"] == fastest
        assert cache["grid_size"] == [256, 256]

    def test_cache_reuse(self):
        """Test that cached results are reused."""
        # First call (benchmarks and caches)
        t0 = time.perf_counter()
        fastest1 = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        time1 = time.perf_counter() - t0

        # Second call (uses cache)
        t0 = time.perf_counter()
        fastest2 = benchmark.get_fastest_backend((256, 256), force_benchmark=False)
        time2 = time.perf_counter() - t0

        # Should return same result
        assert fastest1 == fastest2

        # Cached call should be much faster (at least 10x)
        assert time2 < time1 / 10

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache_path = get_benchmark_cache_path()

        # Create cache
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert cache_path.exists()

        # Invalidate
        benchmark.invalidate_cache()
        assert not cache_path.exists()

    def test_cache_grid_size_mismatch(self):
        """Test that cache is invalidated on grid size mismatch."""
        # Create cache with one size
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        cache1 = benchmark.load_benchmark_cache()

        # Request different size (should re-benchmark)
        benchmark.get_fastest_backend((512, 512), force_benchmark=False)
        cache2 = benchmark.load_benchmark_cache()

        # Cache should have been updated
        assert cache1["grid_size"] != cache2["grid_size"]
        assert cache2["grid_size"] == [512, 512]


class TestCacheLoading:
    """Test cache loading and validation."""

    def test_load_nonexistent_cache(self):
        """Test loading cache when it doesn't exist."""
        cache_path = get_benchmark_cache_path()
        if cache_path.exists():
            cache_path.unlink()

        cache = benchmark.load_benchmark_cache()
        assert cache is None

    def test_load_valid_cache(self):
        """Test loading valid cache."""
        # Create cache
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        # Load it
        cache = benchmark.load_benchmark_cache()
        assert cache is not None
        assert "version" in cache
        assert "timestamp" in cache

    def test_load_corrupted_cache(self):
        """Test that corrupted cache is handled gracefully."""
        cache_path = get_benchmark_cache_path()

        # Write invalid JSON
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("invalid json {{{")

        # Should return None
        cache = benchmark.load_benchmark_cache()
        assert cache is None


class TestFallbackBehavior:
    """Test graceful fallback on benchmark failures."""

    def test_fallback_on_all_backends_fail(self):
        """Test fallback when all backends fail."""
        # This is hard to test without mocking, but we can verify
        # that get_fastest_backend always returns a valid backend
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest is not None
        assert fastest in list_available_backends()


class TestIntegration:
    """Integration tests for the benchmarking system."""

    def test_full_workflow(self):
        """Test complete workflow from benchmark to cache to reuse."""
        cache_path = get_benchmark_cache_path()

        # Step 1: Clear cache
        if cache_path.exists():
            cache_path.unlink()

        # Step 2: Get fastest backend (should benchmark)
        fastest1 = benchmark.get_fastest_backend((256, 256))
        assert cache_path.exists()

        # Step 3: Get again (should use cache)
        fastest2 = benchmark.get_fastest_backend((256, 256))
        assert fastest1 == fastest2

        # Step 4: Force re-benchmark
        fastest3 = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest3 in list_available_backends()  # Valid backend returned

        # Step 5: Invalidate
        benchmark.invalidate_cache()
        assert not cache_path.exists()

    def test_benchmark_results_structure(self):
        """Test that benchmark results have correct structure."""
        results = benchmark.benchmark_all_backends((256, 256))

        # Validate structure
        assert isinstance(results, dict)
        assert "version" in results
        assert "timestamp" in results
        assert "grid_size" in results
        assert "platform" in results
        assert "results" in results
        assert "fastest" in results

        # Validate platform info
        assert "system" in results["platform"]
        assert "processor" in results["platform"]
        assert "python_version" in results["platform"]

        # Validate results for each backend
        for _backend_name, data in results["results"].items():
            assert "time_ms" in data
            assert "speedup" in data
            assert "available" in data

            if data["available"]:
                assert data["time_ms"] is not None
                assert data["time_ms"] > 0
                assert data["speedup"] is not None
                assert data["speedup"] > 0
