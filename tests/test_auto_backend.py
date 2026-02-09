"""Integration tests for automatic backend selection."""

import time

from NLSE import NLSE
from NLSE.backends import benchmark, list_available_backends


class TestAutoBackendSelection:
    """Test NLSE initialization with auto backend."""

    def test_nlse_with_auto_backend(self):
        """Test NLSE initialization with auto backend."""
        simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )

        # Should select one of the available backends
        assert simu.backend in list_available_backends()

    def test_auto_backend_uses_cache(self):
        """Test that auto backend uses cached results quickly."""
        # Force benchmark first to populate cache
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        # Create solver (should use cache, be fast)
        t0 = time.perf_counter()
        _simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )
        elapsed = time.perf_counter() - t0

        # Should be very fast (<200ms) when using cache
        # Note: First import may take longer, so we're generous
        assert elapsed < 0.5

    def test_auto_backend_adapts_to_grid_size(self):
        """Test that auto backend considers actual grid size."""
        # Create solver with specific grid size
        simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=512,
            NY=512,
            backend="auto",
        )

        # Should have selected a valid backend
        assert simu.backend in list_available_backends()

        # Cache should contain results for 512x512
        cache = benchmark.load_benchmark_cache()
        assert cache is not None
        assert cache["grid_size"] == [512, 512]

    def test_manual_backend_override(self):
        """Test that manual backend selection still works."""
        simu_cpu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="CPU",
        )

        assert simu_cpu.backend == "CPU"

    def test_backend_property_with_auto(self):
        """Test setting backend property to 'auto'."""
        simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="CPU",
        )

        # Change to auto
        simu.backend = "auto"

        # Should have selected a valid backend
        assert simu.backend in list_available_backends()


class TestEnvironmentVariables:
    """Test environment variable control."""

    def test_env_backend_override(self, monkeypatch):
        """Test NLSE_BACKEND environment variable."""
        # Need to reload module for env var to take effect
        import importlib

        import NLSE.backends
        from NLSE import NLSE as NLSESolver

        monkeypatch.setenv("NLSE_BACKEND", "CPU")
        importlib.reload(NLSE.backends)

        # Even with "auto", should use CPU from env var
        simu = NLSESolver(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )

        assert simu.backend == "CPU"

        # Clean up - reload without env var
        monkeypatch.delenv("NLSE_BACKEND")
        importlib.reload(NLSE.backends)

    def test_env_force_benchmark(self, monkeypatch):
        """Test NLSE_FORCE_BENCHMARK environment variable."""
        import importlib

        import NLSE.backends
        from NLSE import NLSE as NLSESolver
        from NLSE.utils import get_benchmark_cache_path

        monkeypatch.setenv("NLSE_FORCE_BENCHMARK", "1")
        importlib.reload(NLSE.backends)

        # Should force re-benchmark even with cache
        cache_path = get_benchmark_cache_path()
        if cache_path.exists():
            mtime_before = cache_path.stat().st_mtime
        else:
            mtime_before = None

        _simu = NLSESolver(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )

        # Cache should have been updated
        if mtime_before is not None:
            mtime_after = cache_path.stat().st_mtime
            assert mtime_after > mtime_before

        # Clean up
        monkeypatch.delenv("NLSE_FORCE_BENCHMARK")
        importlib.reload(NLSE.backends)


class TestBackendConsistency:
    """Test that auto-selected backend works correctly."""

    def test_auto_backend_runs_simulation(self):
        """Test that auto-selected backend can actually run a simulation."""
        import numpy as np

        simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )

        # Create initial field
        E = np.ones((256, 256), dtype=np.complex64)
        E_out = simu.out_field(E, z=1e-3, verbose=False, plot=False)

        # Should complete without error
        assert E_out is not None

    def test_different_grid_sizes(self):
        """Test auto selection with various grid sizes."""
        grid_sizes = [(128, 128), (256, 256), (512, 512)]

        for nx, ny in grid_sizes:
            simu = NLSE(
                alpha=0,
                power=1,
                window=1e-3,
                n2=1e-20,
                V=None,
                L=1e-2,
                NX=nx,
                NY=ny,
                backend="auto",
            )

            # Should select valid backend for each size
            assert simu.backend in list_available_backends()


class TestCacheManagement:
    """Test cache management with NLSE integration."""

    def test_cache_invalidation_forces_rebenchmark(self):
        """Test that cache invalidation works in full workflow."""
        # Create solver with auto backend
        simu1 = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )
        backend1 = simu1.backend

        # Invalidate cache
        benchmark.invalidate_cache()

        # Create new solver (should re-benchmark)
        simu2 = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )
        backend2 = simu2.backend

        # Should get same result (hardware didn't change)
        assert backend1 == backend2

    def test_view_benchmark_results(self):
        """Test viewing benchmark results after auto selection."""
        # Create solver with auto backend
        simu = NLSE(
            alpha=0,
            power=1,
            window=1e-3,
            n2=1e-20,
            V=None,
            L=1e-2,
            NX=256,
            NY=256,
            backend="auto",
        )

        # Load and verify benchmark results
        results = benchmark.load_benchmark_cache()
        assert results is not None
        assert "fastest" in results
        assert results["fastest"] == simu.backend
