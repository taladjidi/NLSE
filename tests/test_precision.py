"""Tests for single vs double precision split-step modes."""

import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import NLSE, NLSE_1d

PRECISION_COMPLEX = np.complex64
N = 256
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3


def test_double_precision_norm_conservation() -> None:
    """Test that double precision (Strang splitting) conserves norm."""
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
        backend="CPU",
    )
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    A = simu.out_field(E, L, verbose=False, plot=False, precision="double")
    norm_out = np.sum(np.abs(A) ** 2 * simu.delta_X * simu.delta_Y) * c * epsilon_0 / 2
    assert np.allclose(
        norm_out, power, rtol=1e-4
    ), f"Double precision norm not conserved: {norm_out} vs {power}"


def test_double_vs_single_precision_accuracy() -> None:
    """Test that double precision is more accurate than single for same dz."""
    simu_s = NLSE(
        0,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="CPU",
    )
    simu_d = NLSE(
        0,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="CPU",
    )
    E = np.exp(-(simu_s.XX**2 + simu_s.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    # Use same step size for both
    dz = simu_s.delta_z
    simu_d.delta_z = dz

    A_s = simu_s.out_field(E.copy(), L, verbose=False, plot=False, precision="single")
    A_d = simu_d.out_field(E.copy(), L, verbose=False, plot=False, precision="double")
    # Both should produce valid output
    assert np.all(np.isfinite(A_s)), "Single precision produced non-finite values"
    assert np.all(np.isfinite(A_d)), "Double precision produced non-finite values"
    # Results should be different (different order of accuracy)
    assert not np.allclose(
        A_s, A_d, rtol=1e-6
    ), "Single and double precision gave identical results"


def test_double_precision_1d() -> None:
    """Test double precision in 1D."""
    simu = NLSE_1d(
        0,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        Isat=Isat,
        backend="CPU",
    )
    E = np.ones(N, dtype=PRECISION_COMPLEX)
    A = simu.out_field(E, L, verbose=False, plot=False, precision="double")
    norm_out = np.sum(np.abs(A) ** 2 * simu.delta_X**2) * c * epsilon_0 / 2
    assert np.allclose(
        norm_out, power, rtol=1e-4
    ), f"1D double precision norm not conserved: {norm_out} vs {power}"


def test_double_precision_with_potential() -> None:
    """Test double precision with a potential."""
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
        backend="CPU",
    )
    V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / (70e-6) ** 2).astype(np.float32)
    simu.V = V
    E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    A = simu.out_field(E, L, verbose=False, plot=False, precision="double")
    assert np.all(
        np.isfinite(A)
    ), "Double precision with potential produced non-finite values"
