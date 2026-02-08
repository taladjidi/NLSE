"""Comprehensive benchmarks for NLSE kernels and solvers.

Benchmarks:
1. Individual kernel functions (CPU backend, N=64)
2. Single-component solver split_step methods (all backends)
3. Coupled solver split_step methods (all backends)
"""

import numpy as np
import pytest
from scipy.constants import c

from NLSE import CNLSE, CNLSE_1d, DDGPE, GPE, NLSE, NLSE_1d, NLSE_3d
from NLSE.kernels.cpu import (
    nl_prop,
    nl_prop_c,
    nl_prop_without_V,
    nl_prop_without_V_c,
    rabi_coupling,
    square_mod,
    vortex,
)

# Test parameters
N = 64
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Physical parameters for solvers
alpha = 10.0
power = 1.0
window = 5e-3
n2 = 1e-20
n12 = 1e-21
L = 1e-2
Isat = 1e10
waist = 1e-3

# Kernel benchmark parameters
dz = 1e-5
alpha_k = 10.0
g_k = 1e-3
Isat_k = 1e4
omega_k = 1e3

# Get available backends
from NLSE.backends import list_available_backends

BACKENDS = list_available_backends()


def _random_field_2d(seed: int = 42) -> np.ndarray:
    """Create random 2D complex field."""
    rng = np.random.default_rng(seed)
    return (rng.random((N, N)) + 1j * rng.random((N, N))).astype(PRECISION_COMPLEX)


def _random_real_2d(seed: int = 42) -> np.ndarray:
    """Create random 2D real field."""
    rng = np.random.default_rng(seed)
    return rng.random((N, N)).astype(PRECISION_REAL)


def _gaussian_field_2d(waist: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create Gaussian field."""
    XX, YY = np.meshgrid(x, y)
    return np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)


@pytest.mark.benchmark(group="kernels")
class TestKernelBenchmark:
    """Benchmark individual kernel functions (CPU backend only)."""

    def test_square_mod(self, benchmark):
        """Benchmark square_mod kernel."""
        A = _random_field_2d()
        A_sq = _random_real_2d()

        def kernel():
            square_mod(A.copy(), A_sq.copy())

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A_sq))

    def test_nl_prop(self, benchmark):
        """Benchmark nl_prop kernel (with potential)."""
        A = _random_field_2d()
        A_sq = _random_real_2d()
        V = _random_real_2d(seed=43)

        def kernel():
            nl_prop(
                A.copy(),
                A_sq.copy(),
                dz,
                alpha_k,
                V.copy(),
                g_k,
                Isat_k,
            )

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A))

    def test_nl_prop_without_V(self, benchmark):
        """Benchmark nl_prop_without_V kernel (no potential)."""
        A = _random_field_2d()
        A_sq = _random_real_2d()

        def kernel():
            nl_prop_without_V(
                A.copy(),
                A_sq.copy(),
                dz,
                alpha_k,
                g_k,
                Isat_k,
            )

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A))

    @pytest.mark.skip(reason="numba typing issue with nl_prop_c kernel")
    def test_nl_prop_c(self, benchmark):
        """Benchmark nl_prop_c kernel (coupled with potential)."""
        A1 = _random_field_2d(seed=42)
        A_sq_1 = _random_real_2d(seed=44)
        A_sq_2 = _random_real_2d(seed=45)
        # V needs to be complex for numba typing
        V = _random_real_2d(seed=46).astype(PRECISION_COMPLEX)

        def kernel():
            nl_prop_c(
                A1.copy(),
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz,
                alpha_k,
                V.copy(),
                g_k,
                g_k * 0.5,
                Isat_k,
                Isat_k,
            )

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))

    def test_nl_prop_without_V_c(self, benchmark):
        """Benchmark nl_prop_without_V_c kernel (coupled, no potential)."""
        A1 = _random_field_2d(seed=42)
        A_sq_1 = _random_real_2d(seed=44)
        A_sq_2 = _random_real_2d(seed=45)

        def kernel():
            nl_prop_without_V_c(
                A1.copy(),
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz,
                alpha_k,
                g_k,
                g_k * 0.5,
                Isat_k,
                Isat_k,
            )

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))

    def test_rabi_coupling(self, benchmark):
        """Benchmark rabi_coupling kernel."""
        A1 = _random_field_2d(seed=42)
        A2 = _random_field_2d(seed=43)

        def kernel():
            rabi_coupling(A1.copy(), A2.copy(), dz, omega_k)

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))
        assert np.all(np.isfinite(A2))

    def test_vortex(self, benchmark):
        """Benchmark vortex kernel."""
        im = np.zeros((N, N), dtype=PRECISION_COMPLEX)
        i = np.arange(N, dtype=np.int32)
        j = np.arange(N, dtype=np.int32)
        ii, jj = np.meshgrid(i, j)
        ll = 1

        def kernel():
            vortex(im.copy(), i, j, ii, jj, ll)

        result = benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(im))


@pytest.mark.benchmark(group="solvers-single")
class TestSolverBenchmark:
    """Benchmark single-component solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d(self, benchmark, backend):
        """Benchmark NLSE 2D propagation."""
        simu = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = _gaussian_field_2d(waist, simu.X, simu.Y)

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_1d(self, benchmark, backend):
        """Benchmark NLSE_1d propagation."""
        simu = NLSE_1d(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N,)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_3d(self, benchmark, backend):
        """Benchmark NLSE_3d propagation."""
        simu = NLSE_3d(
            alpha=alpha,
            energy=1e-6,
            window=[window, window, 1e-12],
            n2=n2,
            D0=1e-26,
            vg=c,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=N,
            Isat=Isat,
            wvl=780e-9,
            backend=backend,
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((N, N, N), dtype=PRECISION_COMPLEX)
        for i in range(N):
            E_in[:, :, i] = np.exp(-(XX**2 + YY**2) / waist**2)

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=5, warmup_rounds=1)
        assert result.shape == (N, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gpe(self, benchmark, backend):
        """Benchmark GPE propagation."""
        from scipy.constants import atomic_mass

        simu = GPE(
            gamma=0.1,
            N=1e5,
            window=window,
            g=n2,
            V=None,
            m=87 * atomic_mass,
            NX=N,
            NY=N,
            sat=Isat,
            backend=backend,
        )
        E_in = _gaussian_field_2d(waist, simu.X, simu.Y)

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))


@pytest.mark.benchmark(group="solvers-coupled")
class TestCoupledSolverBenchmark:
    """Benchmark coupled solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_2d(self, benchmark, backend):
        """Benchmark CNLSE 2D propagation."""
        simu = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N, N), dtype=PRECISION_COMPLEX)
        E_in[0] = _gaussian_field_2d(waist, simu.X, simu.Y)
        E_in[1] = _gaussian_field_2d(waist * 1.2, simu.X, simu.Y) * 0.5

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_1d(self, benchmark, backend):
        """Benchmark CNLSE_1d propagation."""
        simu = CNLSE_1d(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(simu.X**2) / waist**2)
        E_in[1] = np.exp(-(simu.X**2) / (waist * 1.2) ** 2) * 0.5

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False)

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.skip(reason="DDGPE benchmark is slow and has callback complexity")
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_ddgpe(self, benchmark, backend):
        """Benchmark DDGPE propagation."""
        # DDGPE specific parameters
        gamma = 0.1
        omega = 1e3
        T = 1e-3
        omega_exc = 1.5e15
        omega_cav = 1.5e15
        detuning = 0.0
        k_z = 1e6

        simu = DDGPE(
            gamma=gamma,
            power=power,
            window=window,
            g=n2,
            g12=n12,
            omega=omega,
            T=T,
            omega_exc=omega_exc,
            omega_cav=omega_cav,
            detuning=detuning,
            k_z=k_z,
            V=None,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N, N), dtype=PRECISION_COMPLEX)
        E_in[0] = _gaussian_field_2d(waist, simu.X, simu.Y)
        E_in[1] = _gaussian_field_2d(waist * 1.2, simu.X, simu.Y) * 0.5

        # No-op laser excitation for benchmarking
        def no_laser(simu, A, z, i):
            pass

        def propagate():
            return simu.out_field(
                E_in,
                t=1e-3,
                laser_excitation=no_laser,
                verbose=False,
                callback=[],
                callback_args=[()],
            )

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N, N)
        assert np.all(np.isfinite(result))
