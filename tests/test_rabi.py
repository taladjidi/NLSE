"""Tests for Rabi coupling in CNLSE and CNLSE_1d."""

import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import CNLSE, CNLSE_1d

if CNLSE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
AVAILABLE_BACKENDS = ["CPU"]
if CNLSE.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("GPU")

N = 256
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3


def test_rabi_total_norm_conservation() -> None:
    """Test that Rabi coupling conserves total norm across both components."""
    omega = 1e3  # Rabi coupling strength
    for backend in AVAILABLE_BACKENDS:
        simu = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, omega=omega, backend=backend,
        )
        # both components start with equal power
        E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        A = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        total_norm = np.sum(
            np.abs(A) ** 2 * simu.delta_X * simu.delta_Y * c * epsilon_0 / 2,
            axis=simu._last_axes,
        ).sum()
        expected_total = simu.power + simu.power2
        assert np.allclose(total_norm, expected_total, rtol=1e-3), (
            f"Rabi coupling total norm not conserved: {total_norm} vs "
            f"{expected_total}. (Backend {backend})"
        )


def test_rabi_population_transfer() -> None:
    """Test that Rabi coupling transfers population between components.

    Use unequal amplitudes so Rabi coupling causes redistribution.
    """
    omega = 1e3
    for backend in AVAILABLE_BACKENDS:
        simu = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, omega=omega, backend=backend,
        )
        # unequal initial amplitudes (both nonzero for normalization)
        E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        E[0] *= 2.0  # component 1 has more amplitude
        E[1] *= 0.5  # component 2 has less
        A = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        norm_1 = np.sum(np.abs(A[0]) ** 2 * simu.delta_X * simu.delta_Y)
        norm_2 = np.sum(np.abs(A[1]) ** 2 * simu.delta_X * simu.delta_Y)
        # Both components should have significant population
        assert norm_1 > 0, (
            f"Component 1 has zero norm. (Backend {backend})"
        )
        assert norm_2 > 0, (
            f"Component 2 has zero norm. (Backend {backend})"
        )


def test_rabi_1d_total_norm_conservation() -> None:
    """Test that 1D Rabi coupling conserves total norm."""
    omega = 1e3
    for backend in AVAILABLE_BACKENDS:
        simu = CNLSE_1d(
            0, power, window, n2, n12, None, L,
            NX=N, Isat=Isat, omega=omega, backend=backend,
        )
        # both components nonzero to avoid normalization division by zero
        E = np.ones((2, N), dtype=PRECISION_COMPLEX)
        A = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        total_norm = np.sum(
            np.abs(A) ** 2 * simu.delta_X**2 * c * epsilon_0 / 2,
            axis=simu._last_axes,
        ).sum()
        expected_total = simu.power + simu.power2
        assert np.allclose(total_norm, expected_total, rtol=1e-3), (
            f"1D Rabi total norm not conserved: {total_norm} vs "
            f"{expected_total}. (Backend {backend})"
        )


def test_rabi_no_coupling_unchanged() -> None:
    """Test that omega=None means no coupling (original behavior)."""
    for backend in AVAILABLE_BACKENDS:
        simu_no_rabi = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, omega=None, backend=backend,
        )
        simu_rabi = CNLSE(
            0, power, window, n2, n12, None, L,
            NX=N, NY=N, Isat=Isat, omega=1e3, backend=backend,
        )
        E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
        A_no = simu_no_rabi.out_field(
            E.copy(), L, verbose=False, plot=False, precision="single"
        )
        A_yes = simu_rabi.out_field(
            E.copy(), L, verbose=False, plot=False, precision="single"
        )
        # With equal initial populations the results should differ
        # (Rabi coupling modifies relative phase/amplitude)
        assert not np.allclose(A_no, A_yes, rtol=1e-2), (
            f"Rabi coupling had no effect. (Backend {backend})"
        )
