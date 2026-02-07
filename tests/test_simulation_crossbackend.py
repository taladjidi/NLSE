"""Cross-backend simulation correctness tests.

Run the same simulation on CPU and CL (and future Metal) backends,
comparing full out_field results to ensure all backends produce
equivalent output.
"""

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

from NLSE import NLSE, NLSE_1d, CNLSE, CNLSE_1d

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Use small grids for speed
N = 64
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3

# Collect non-CPU backends available
EXTRA_BACKENDS = []
if NLSE.__PYOPENCL_AVAILABLE__:
    EXTRA_BACKENDS.append("CL")
if NLSE.__METAL_AVAILABLE__:
    EXTRA_BACKENDS.append("Metal")

# Backends that support coupled NLSE (Metal doesn't yet)
EXTRA_BACKENDS_CNLSE = [b for b in EXTRA_BACKENDS if b != "Metal"]


def _skip_if_no_extra():
    if not EXTRA_BACKENDS:
        pytest.skip("No non-CPU backends available")


def _gaussian_2d(simu):
    """Create a Gaussian beam field."""
    return np.exp(
        -(simu.XX**2 + simu.YY**2) / waist**2
    ).astype(PRECISION_COMPLEX)


def _gaussian_1d(simu):
    """Create a 1D Gaussian beam field."""
    return np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)


# ============================================================
# NLSE 2D cross-backend tests
# ============================================================


class TestNLSECrossBackend:
    """Compare NLSE out_field between CPU and other backends."""

    def test_out_field_no_potential(self):
        _skip_if_no_extra()
        simu_cpu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        E_cpu = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        for backend in EXTRA_BACKENDS:
            simu = NLSE(
                0, power, window, n2, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            E = np.ones((N, N), dtype=PRECISION_COMPLEX)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"NLSE out_field CPU != {backend}"
            )

    def test_out_field_with_potential(self):
        _skip_if_no_extra()
        simu_cpu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        V = -1e-4 * np.exp(
            -(simu_cpu.XX**2 + simu_cpu.YY**2) / (70e-6) ** 2
        ).astype(PRECISION_REAL)
        simu_cpu.V = V.copy()
        E_cpu = _gaussian_2d(simu_cpu)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        for backend in EXTRA_BACKENDS:
            simu = NLSE(
                0, power, window, n2, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            simu.V = V.copy()
            E = _gaussian_2d(simu)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"NLSE out_field with V: CPU != {backend}"
            )

    def test_out_field_with_losses(self):
        _skip_if_no_extra()
        alpha = 20
        simu_cpu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        E_cpu = _gaussian_2d(simu_cpu)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        for backend in EXTRA_BACKENDS:
            simu = NLSE(
                alpha, power, window, n2, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            E = _gaussian_2d(simu)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"NLSE out_field with losses: CPU != {backend}"
            )

    def test_out_field_double_precision(self):
        _skip_if_no_extra()
        simu_cpu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        E_cpu = _gaussian_2d(simu_cpu)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="double"
        )
        for backend in EXTRA_BACKENDS:
            simu = NLSE(
                0, power, window, n2, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            E = _gaussian_2d(simu)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="double"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"NLSE out_field double: CPU != {backend}"
            )

    def test_norm_conservation_matches(self):
        """CPU and other backends should conserve norm equally well."""
        _skip_if_no_extra()
        simu_cpu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        E_cpu = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        norm_cpu = (
            np.sum(np.abs(A_cpu) ** 2 * simu_cpu.delta_X * simu_cpu.delta_Y)
            * c * epsilon_0 / 2
        )
        for backend in EXTRA_BACKENDS:
            simu = NLSE(
                0, power, window, n2, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            E = np.ones((N, N), dtype=PRECISION_COMPLEX)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            norm = (
                np.sum(np.abs(A) ** 2 * simu.delta_X * simu.delta_Y)
                * c * epsilon_0 / 2
            )
            assert np.allclose(norm, norm_cpu, rtol=1e-3), (
                f"Norm mismatch: CPU={norm_cpu}, {backend}={norm}"
            )


# ============================================================
# NLSE 1D cross-backend tests
# ============================================================


class TestNLSE1DCrossBackend:
    """1D classes: CL backend not yet supported (FFT plan mismatch).
    These tests run CPU as baseline, ready for Metal backend.
    """

    def test_out_field_cpu_baseline(self):
        """Establish CPU baseline for 1D. Future backends compare against this."""
        simu_cpu = NLSE_1d(
            0, power, window, n2, None, L,
            NX=N, Isat=Isat, backend="CPU",
        )
        E = np.ones(N, dtype=PRECISION_COMPLEX)
        A = simu_cpu.out_field(
            E, L, verbose=False, plot=False, precision="single"
        )
        norm = np.sum(np.abs(A) ** 2 * simu_cpu.delta_X**2) * c * epsilon_0 / 2
        assert np.allclose(norm, power, rtol=1e-3), (
            f"NLSE_1d CPU norm not conserved: {norm} vs {power}"
        )


# ============================================================
# CNLSE cross-backend tests
# ============================================================


class TestCNLSECrossBackend:
    def test_out_field_no_potential(self):
        if not EXTRA_BACKENDS_CNLSE:
            pytest.skip("No non-CPU backends available for CNLSE")
        simu_cpu = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        E_cpu = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        for backend in EXTRA_BACKENDS_CNLSE:
            simu = CNLSE(
                0, power, window, n2, n12, None, L,
                NX=N, NY=N, Isat=Isat, backend=backend,
            )
            E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"CNLSE out_field: CPU != {backend}"
            )

    def test_out_field_with_rabi(self):
        if not EXTRA_BACKENDS_CNLSE:
            pytest.skip("No non-CPU backends available for CNLSE")
        omega = 1e3
        simu_cpu = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, omega=omega, backend="CPU",
        )
        E_cpu = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        E_cpu[0] *= 2.0
        E_cpu[1] *= 0.5
        A_cpu = simu_cpu.out_field(
            E_cpu, L, verbose=False, plot=False, precision="single"
        )
        for backend in EXTRA_BACKENDS_CNLSE:
            simu = CNLSE(
                0, power, window, n2, n12, None, L,
                NX=N, NY=N, Isat=Isat, omega=omega, backend=backend,
            )
            E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
            E[0] *= 2.0
            E[1] *= 0.5
            A = simu.out_field(
                E, L, verbose=False, plot=False, precision="single"
            )
            assert np.allclose(A, A_cpu, rtol=1e-3, atol=1e-6), (
                f"CNLSE out_field with Rabi: CPU != {backend}"
            )


# ============================================================
# CNLSE 1D cross-backend tests
# ============================================================


class TestCNLSE1DCrossBackend:
    """1D coupled: CL backend not yet supported. CPU baseline for future Metal."""

    def test_out_field_cpu_baseline(self):
        simu_cpu = CNLSE_1d(
            0, power, window, n2, n12, None, L,
            NX=N, Isat=Isat, backend="CPU",
        )
        E = np.ones((2, N), dtype=PRECISION_COMPLEX)
        A = simu_cpu.out_field(
            E, L, verbose=False, plot=False, precision="single"
        )
        assert np.all(np.isfinite(A)), "CNLSE_1d CPU produced non-finite values"
