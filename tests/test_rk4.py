"""Tests for the RK4 solver across NLSE variants."""

import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import NLSE, NLSE_1d

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
AVAILABLE_BACKENDS = ["CPU"]
if NLSE.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CUPY")

N_1d = 512
N_2d = 256
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3


def test_build_propagator_rk4() -> None:
    """Test that RK4 propagator is built correctly (no exp)."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            20, power, window, n2, None, L,
            NX=N_2d, NY=N_2d, Isat=Isat, backend=backend,
        )
        prop = simu._build_propagator(precision="RK4")
        expected = -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k
        assert np.allclose(prop, expected), (
            f"RK4 propagator is wrong. (Backend {backend})"
        )


def test_build_propagator_rk4_1d() -> None:
    """Test that 1D RK4 propagator is built correctly."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_1d(
            20, power, window, n2, None, L,
            NX=N_1d, Isat=Isat, backend=backend,
        )
        prop = simu._build_propagator(precision="RK4")
        expected = -1j * 0.5 * (simu.Kx**2) / simu.k
        assert np.allclose(prop, expected), (
            f"1D RK4 propagator is wrong. (Backend {backend})"
        )


def test_rk4_norm_conservation() -> None:
    """Test that RK4 solver conserves norm (no losses)."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            0, power, window, n2, None, L,
            NX=N_2d, NY=N_2d, Isat=Isat, backend=backend,
        )
        E = np.ones((N_2d, N_2d), dtype=PRECISION_COMPLEX)
        A = simu.out_field(
            E, L, verbose=False, plot=False, precision="RK4"
        )
        norm_out = np.sum(
            np.abs(A) ** 2 * simu.delta_X * simu.delta_Y
        ) * c * epsilon_0 / 2
        assert np.allclose(norm_out, power, rtol=1e-3), (
            f"RK4 norm not conserved: {norm_out} vs {power}. (Backend {backend})"
        )


def test_rk4_norm_conservation_1d() -> None:
    """Test that 1D RK4 solver conserves norm (no losses)."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_1d(
            0, power, window, n2, None, L,
            NX=N_1d, Isat=Isat, backend=backend,
        )
        E = np.ones(N_1d, dtype=PRECISION_COMPLEX)
        A = simu.out_field(
            E, L, verbose=False, plot=False, precision="RK4"
        )
        norm_out = np.sum(
            np.abs(A) ** 2 * simu.delta_X**2
        ) * c * epsilon_0 / 2
        assert np.allclose(norm_out, power, rtol=1e-3), (
            f"1D RK4 norm not conserved: {norm_out} vs {power}. (Backend {backend})"
        )


def test_rk4_split_step_single() -> None:
    """Test that a single RK4 step doesn't crash and produces valid output."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            20, power, window, n2, None, L,
            NX=N_2d, NY=N_2d, Isat=Isat, backend=backend,
        )
        simu.propagator = simu._build_propagator(precision="RK4")
        E = np.ones((N_2d, N_2d), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=True)
        simu.plans = simu._build_fft_plan(A)
        if (
            backend == "CUPY" and NLSE.__CUPY_AVAILABLE__
            or backend == "CL" and NLSE.__PYOPENCL_AVAILABLE__
        ):
            simu._send_arrays_to_gpu()
        simu.split_step_RK4(A, simu.V, simu.propagator, simu.plans)
        # output should contain finite values
        if backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
            A_np = A.get()
        else:
            A_np = np.asarray(A)
        assert np.all(np.isfinite(A_np)), (
            f"RK4 step produced non-finite values. (Backend {backend})"
        )


def test_rk4_with_potential() -> None:
    """Test RK4 solver with a potential field."""
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            0, power, window, n2, None, L,
            NX=N_2d, NY=N_2d, Isat=Isat, backend=backend,
        )
        V = -1e-4 * np.exp(
            -(simu.XX**2 + simu.YY**2) / (70e-6) ** 2
        ).astype(PRECISION_REAL)
        simu.V = V
        E = np.exp(
            -(simu.XX**2 + simu.YY**2) / waist**2
        ).astype(PRECISION_COMPLEX)
        A = simu.out_field(
            E, L, verbose=False, plot=False, precision="RK4"
        )
        assert np.all(np.isfinite(A)), (
            f"RK4 with potential produced non-finite values. (Backend {backend})"
        )
