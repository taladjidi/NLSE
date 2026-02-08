"""Benchmark tests for comparing solver performance across backends.

Run with: pytest tests/test_benchmark_profile.py -v -m benchmark --benchmark-only
"""

import numpy as np
import pytest
from scipy.constants import atomic_mass

from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.kernels_cpu import (
    nl_prop,
    nl_prop_c,
    nl_prop_without_V,
    nl_prop_without_V_c,
    rabi_coupling,
    square_mod,
    vortex,
)

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

# Kernel benchmark parameters (matching test_kernels_crossbackend.py)
dz_k = 1e-5
alpha_k = 10.0
g_k = 1e-3
Isat_k = 1e4
omega_k = 1e3
g12_k = 5e-4
Isat2_k = 2e4

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Collect available backends
BACKENDS = ["CPU"]
if NLSE.__CUPY_AVAILABLE__:
    BACKENDS.append("CUPY")
if NLSE.__METAL_AVAILABLE__:
    BACKENDS.append("Metal")


# ---- Helpers for kernel benchmarks ----


def _random_field_2d(shape=(N, N)):
    """Generate a random complex field."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        PRECISION_COMPLEX
    )


def _random_real_2d(shape=(N, N)):
    """Generate a random real field (positive, for A_sq / potential)."""
    rng = np.random.default_rng(123)
    return np.abs(rng.standard_normal(shape)).astype(PRECISION_REAL) + 0.01


# ============================================================
# Kernel benchmarks (CPU only)
# ============================================================


@pytest.mark.benchmark
class TestKernelBenchmark:
    def test_square_mod(self, benchmark):
        A = _random_field_2d()
        A_sq = np.zeros((N, N), dtype=PRECISION_REAL)

        def run():
            A_sq_c = A_sq.copy()
            square_mod(A.copy(), A_sq_c)
            return A_sq_c

        result = benchmark(run)
        assert np.all(np.isfinite(result))

    def test_nl_prop(self, benchmark):
        A = _random_field_2d()
        A_sq = (A.real**2 + A.imag**2).astype(PRECISION_REAL)
        V = _random_real_2d()

        def run():
            A_c = A.copy()
            nl_prop(A_c, A_sq.copy(), dz_k, alpha_k, V, g_k, Isat_k)
            return A_c

        result = benchmark(run)
        assert np.all(np.isfinite(result))

    def test_nl_prop_without_V(self, benchmark):
        A = _random_field_2d()
        A_sq = (A.real**2 + A.imag**2).astype(PRECISION_REAL)

        def run():
            A_c = A.copy()
            nl_prop_without_V(A_c, A_sq.copy(), dz_k, alpha_k, g_k, Isat_k)
            return A_c

        result = benchmark(run)
        assert np.all(np.isfinite(result))

    def test_nl_prop_c(self, benchmark):
        A1 = _random_field_2d()
        A_sq_1 = (A1.real**2 + A1.imag**2).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        V = _random_real_2d()

        def run():
            A1_c = A1.copy()
            nl_prop_c(
                A1_c,
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz_k,
                alpha_k,
                V,
                g_k,
                g12_k,
                Isat_k,
                Isat2_k,
            )
            return A1_c

        result = benchmark(run)
        assert np.all(np.isfinite(result))

    def test_nl_prop_without_V_c(self, benchmark):
        A1 = _random_field_2d()
        A_sq_1 = (A1.real**2 + A1.imag**2).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()

        def run():
            A1_c = A1.copy()
            nl_prop_without_V_c(
                A1_c,
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz_k,
                alpha_k,
                g_k,
                g12_k,
                Isat_k,
                Isat2_k,
            )
            return A1_c

        result = benchmark(run)
        assert np.all(np.isfinite(result))

    def test_rabi_coupling(self, benchmark):
        A1 = _random_field_2d()
        rng = np.random.default_rng(99)
        A2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )

        def run():
            A1_c = A1.copy()
            A2_c = A2.copy()
            rabi_coupling(A1_c, A2_c, dz_k, omega_k)
            return A1_c, A2_c

        result = benchmark(run)
        assert np.all(np.isfinite(result[0]))
        assert np.all(np.isfinite(result[1]))

    def test_vortex(self, benchmark):
        ii, jj = np.meshgrid(
            np.arange(N, dtype=PRECISION_REAL),
            np.arange(N, dtype=PRECISION_REAL),
            indexing="ij",
        )
        i_pos, j_pos = N // 2, N // 2
        ll = 1

        def run():
            im = np.zeros((N, N), dtype=PRECISION_REAL)
            vortex(im, i_pos, j_pos, ii, jj, ll)
            return im

        result = benchmark(run)
        assert np.all(np.isfinite(result))


# ============================================================
# Single-component solver benchmarks
# ============================================================


@pytest.mark.benchmark
class TestSolverBenchmark:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d(self, benchmark, backend):
        simu = NLSE(
            0,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E = np.ones((N, N), dtype=np.complex64)
        A = benchmark(
            simu.out_field, E, L, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), f"NLSE 2D {backend} produced non-finite values"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_1d(self, benchmark, backend):
        simu = NLSE_1d(
            0,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E = np.ones(N, dtype=np.complex64)
        A = benchmark(
            simu.out_field, E, L, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), f"NLSE 1D {backend} produced non-finite values"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gpe(self, benchmark, backend):
        simu = GPE(
            gamma=0,
            N=N_at,
            window=gpe_window,
            g=g,
            V=None,
            m=m,
            NX=N,
            NY=N,
            backend=backend,
        )
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
        A = benchmark(
            simu.out_field, E, 1e-4, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), f"GPE {backend} produced non-finite values"


# ============================================================
# Coupled solver benchmarks
# ============================================================


@pytest.mark.benchmark
class TestCoupledSolverBenchmark:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_2d(self, benchmark, backend):
        simu = CNLSE(
            0,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E = np.ones((2, N, N), dtype=np.complex64)
        A = benchmark(
            simu.out_field, E, L, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), f"CNLSE 2D {backend} produced non-finite values"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_1d(self, benchmark, backend):
        simu = CNLSE_1d(
            0,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E = np.ones((2, N), dtype=np.complex64)
        A = benchmark(
            simu.out_field, E, L, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), f"CNLSE 1D {backend} produced non-finite values"

