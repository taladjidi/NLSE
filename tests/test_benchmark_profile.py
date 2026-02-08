"""Benchmark tests for comparing solver performance across backends.

Run with: pytest tests/test_benchmark_profile.py -v -m benchmark
"""

import time

import numpy as np
import pytest
from scipy.constants import atomic_mass

from NLSE import CNLSE, CNLSE_1d, GPE, NLSE, NLSE_1d

# Small grid for quick benchmark runs
N = 64
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3

# GPE parameters
N_at = 1e6
g = 1e3 / (N_at / 1e-3**2)
m = 87 * atomic_mass
gpe_window = 1e-3

# Collect available backends
BACKENDS = ["CPU"]
if NLSE.__CUPY_AVAILABLE__:
    BACKENDS.append("CUPY")
if NLSE.__METAL_AVAILABLE__:
    BACKENDS.append("Metal")


def _run_timed(func):
    """Run a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


@pytest.mark.benchmark
class TestNLSEBenchmark:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d(self, backend):
        simu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )
        E = np.ones((N, N), dtype=np.complex64)
        A, elapsed = _run_timed(
            lambda: simu.out_field(E, L, verbose=False, plot=False, precision="single")
        )
        assert np.all(np.isfinite(A)), f"NLSE 2D {backend} produced non-finite values"
        print(f"\nNLSE 2D [{backend}]: {elapsed:.4f}s")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_1d(self, backend):
        simu = NLSE_1d(
            0, power, window, n2, None, L,
            NX=N, Isat=Isat, backend=backend,
        )
        E = np.ones(N, dtype=np.complex64)
        A, elapsed = _run_timed(
            lambda: simu.out_field(E, L, verbose=False, plot=False, precision="single")
        )
        assert np.all(np.isfinite(A)), f"NLSE 1D {backend} produced non-finite values"
        print(f"\nNLSE 1D [{backend}]: {elapsed:.4f}s")


@pytest.mark.benchmark
class TestCNLSEBenchmark:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_2d(self, backend):
        simu = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )
        E = np.ones((2, N, N), dtype=np.complex64)
        A, elapsed = _run_timed(
            lambda: simu.out_field(E, L, verbose=False, plot=False, precision="single")
        )
        assert np.all(np.isfinite(A)), f"CNLSE 2D {backend} produced non-finite values"
        print(f"\nCNLSE 2D [{backend}]: {elapsed:.4f}s")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_1d(self, backend):
        simu = CNLSE_1d(
            0, power, window, n2, n12, None, L,
            NX=N, Isat=Isat, backend=backend,
        )
        E = np.ones((2, N), dtype=np.complex64)
        A, elapsed = _run_timed(
            lambda: simu.out_field(E, L, verbose=False, plot=False, precision="single")
        )
        assert np.all(np.isfinite(A)), f"CNLSE 1D {backend} produced non-finite values"
        print(f"\nCNLSE 1D [{backend}]: {elapsed:.4f}s")


@pytest.mark.benchmark
class TestGPEBenchmark:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gpe(self, backend):
        simu = GPE(
            gamma=0, N=N_at, window=gpe_window, g=g, V=None, m=m,
            NX=N, NY=N, backend=backend,
        )
        E = np.exp(
            -(simu.XX**2 + simu.YY**2) / waist**2
        ).astype(np.complex64)
        A, elapsed = _run_timed(
            lambda: simu.out_field(
                E, 1e-4, verbose=False, plot=False, precision="single"
            )
        )
        assert np.all(np.isfinite(A)), f"GPE {backend} produced non-finite values"
        print(f"\nGPE [{backend}]: {elapsed:.4f}s")
