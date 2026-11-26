"""Integration tests for automatic backend selection."""

import importlib
from unittest.mock import Mock

import NLSE.backends as backends
from NLSE import NLSE
from NLSE.backends import benchmark, list_available_backends

# Every solver in this file is the same one; the tests differ in backend and,
# in two of them, grid size.
N = 256
alpha = 0
power = 1
window = 1e-3
n2 = 1e-20
L = 1e-2
# What a benchmark that should not have run answers with: get_backend raises
# on it, so the run cannot pass by benchmarking and happening to pick CPU.
NOT_A_BACKEND = "NO_SUCH_BACKEND"


def make_solver(backend="CPU", n=N, **overrides):
    """Return an NLSE with this module's parameters.

    Parameters
    ----------
    backend : str
        Backend name.
    n : int
        Grid size, square.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    NLSE
        The solver.
    """
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": None,
        "L": L,
        "NX": n,
        "NY": n,
        "backend": backend,
    }
    params.update(overrides)
    return NLSE(**params)


class TestAutoBackendSelection:
    """Test NLSE initialization with auto backend."""

    def test_nlse_with_auto_backend(self):
        """Test NLSE initialization with auto backend."""
        simu = make_solver("auto")

        # Should select one of the available backends
        assert simu.backend in list_available_backends()

    def test_auto_backend_adapts_to_grid_size(self):
        """Test that auto backend considers actual grid size."""
        # Create solver with specific grid size
        simu = make_solver("auto", n=512)

        # Should have selected a valid backend
        assert simu.backend in list_available_backends()

        # Cache should contain results for 512x512
        cache = benchmark.load_benchmark_cache()
        assert cache is not None
        assert cache["grid_size"] == [512, 512]

    def test_manual_backend_override(self):
        """Test that manual backend selection still works."""
        simu_cpu = make_solver()

        assert simu_cpu.backend == "CPU"

    def test_backend_property_with_auto(self):
        """Test setting backend property to 'auto'."""
        simu = make_solver()

        # Change to auto
        simu.backend = "auto"

        # Should have selected a valid backend
        assert simu.backend in list_available_backends()


class TestEnvironmentVariables:
    """Test environment variable control.

    Both variables are read when NLSE.backends is imported, so each test
    reloads it with the variable set, and again without it in a ``finally``,
    so a failed assertion does not leave the variable in force for later tests.
    """

    def test_env_backend_override(self, monkeypatch):
        """NLSE_BACKEND wins over "auto", which then benchmarks nothing."""
        monkeypatch.setenv("NLSE_BACKEND", "cpu")
        importlib.reload(backends)
        try:
            monkeypatch.setattr(
                backends,
                "get_optimal_backend",
                lambda *args, **kwargs: NOT_A_BACKEND,
            )
            assert backends.get_backend("auto").name == "CPU"
        finally:
            monkeypatch.delenv("NLSE_BACKEND")
            importlib.reload(backends)

    def test_env_force_benchmark(self, monkeypatch):
        """NLSE_FORCE_BENCHMARK=1 measures again although a cache fits."""
        benchmark.get_fastest_backend((N, N), force_benchmark=True)
        spy = Mock(wraps=benchmark.benchmark_all_backends)
        monkeypatch.setattr(benchmark, "benchmark_all_backends", spy)
        monkeypatch.setenv("NLSE_FORCE_BENCHMARK", "1")
        importlib.reload(backends)
        try:
            backends.get_backend("auto", (N, N))
        finally:
            monkeypatch.delenv("NLSE_FORCE_BENCHMARK")
            importlib.reload(backends)
        spy.assert_called_once_with((N, N))


class TestBackendConsistency:
    """Test that auto-selected backend works correctly."""

    def test_auto_backend_runs_simulation(self):
        """Test that auto-selected backend can actually run a simulation."""
        import numpy as np

        simu = make_solver("auto")

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
        from NLSE.utils import get_benchmark_cache_path

        # Create solver with auto backend to populate cache
        simu1 = make_solver("auto")
        assert simu1.backend in list_available_backends()

        # Invalidate cache
        benchmark.invalidate_cache()
        cache_path = get_benchmark_cache_path()
        assert not cache_path.exists(), "Cache should be invalidated"

        # Create new solver (should re-benchmark and recreate cache)
        simu2 = make_solver("auto")

        # Should select a valid backend and recreate cache
        assert simu2.backend in list_available_backends()
        assert cache_path.exists(), "Cache should be recreated after invalidation"

    def test_view_benchmark_results(self):
        """Test viewing benchmark results after auto selection."""
        # Create solver with auto backend
        simu = make_solver("auto")

        # Load and verify benchmark results
        results = benchmark.load_benchmark_cache()
        assert results is not None
        assert "fastest" in results
        assert results["fastest"] == simu.backend
