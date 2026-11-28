"""``adapt_delta_z`` against the nonlinear length, in each solver's own units."""

import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE
from NLSE.callbacks import adapt_delta_z
from scipy.constants import atomic_mass, c, epsilon_0

STEPS_PER_LENGTH = 12  # NLSE/callbacks.py, STEPS_PER_NONLINEAR_LENGTH
N_GRID = 32  # grid points per axis

MASS = 87 * atomic_mass  # kg, rubidium 87
GAS_WINDOW = 100e-6  # m
DENSITY = 1e14  # atoms/m^2, a typical 2D condensate
G_INTERACTION = 1e-11  # rad/s*m^2, so g*n = 1e3 rad/s

BEAM_WINDOW = 8.9e-3  # m
WAVELENGTH = 780e-9  # m, NLSE's default wvl
N2 = -1.6e-9  # m^2/W
INTENSITY = 1e5  # W/m^2
I_SAT = 1e5  # W/m^2, halves the index change at INTENSITY

# CNLSE with the cross coupling alone, both components at INTENSITY: one
# saturation 1 / (1 + 2 INTENSITY / I_SAT) for both, as the kernels apply it.
CROSS_SATURATION = 1 / (1 + 2 * INTENSITY / I_SAT)

# An asymmetric CNLSE, so each constant multiplies an intensity no other
# does: its own index per component, a cross index, a second wavenumber.
N22 = -2.4e-9  # m^2/W
N12 = -0.8e-9  # m^2/W
K2_OVER_K = 1.5
# Second intensity over the first: at 0.3 the second component turns faster
# (2.28 against 1.84, in k * 1e-9 * INTENSITY * sat), at 0.05 the first
# (1.64 against 1.38).
SECOND_INTENSITY_FRACTIONS = (0.3, 0.05)
I_SAT2 = 4e5  # W/m^2, so the second intensity saturates on its own scale

# Relative; float32 |A|^2 against the float64 closed form. Measured 3.8e-9
# (GPE) and 8.7e-9 (NLSE) on CPU and MLX.
STEP_REL_TOL = 1e-6


def step_for(simu, amplitude):
    """Return the step adapt_delta_z asks for on a uniform field.

    Parameters
    ----------
    simu : NLSE
        The solver, of any class.
    amplitude : float
        Field amplitude everywhere on the grid, in the solver's units.

    Returns
    -------
    float
        The step it returns.
    """
    A = np.full((N_GRID, N_GRID), amplitude, dtype=np.complex64)
    return adapt_delta_z(simu, simu._backend.from_numpy(A), 0.0, 0, 1, [])


def test_the_gpe_step_is_a_twelfth_of_one_over_g_n(backend):
    """``dt = 1 / (12 g n)`` s for a uniform gas, 8.3e-5 s here."""
    simu = GPE(
        gamma=0,
        N=DENSITY * GAS_WINDOW**2,
        window=GAS_WINDOW,
        g=G_INTERACTION,
        V=None,
        m=MASS,
        NX=N_GRID,
        NY=N_GRID,
        backend=backend,
    )
    expected = 1 / (STEPS_PER_LENGTH * G_INTERACTION * DENSITY)
    assert step_for(simu, np.sqrt(DENSITY)) == pytest.approx(expected, rel=STEP_REL_TOL)


def test_the_nlse_step_is_a_twelfth_of_the_nonlinear_length(backend):
    """``dz = 1 / (12 k |n2| I / (1 + I / Isat))`` m for a uniform beam."""
    simu = NLSE(
        alpha=0,
        power=1.0,
        window=BEAM_WINDOW,
        n2=N2,
        V=None,
        L=1e-2,
        NX=N_GRID,
        NY=N_GRID,
        Isat=I_SAT,
        wvl=WAVELENGTH,
        backend=backend,
    )
    k = 2 * np.pi / WAVELENGTH
    rate = k * abs(N2) * INTENSITY / (1 + INTENSITY / I_SAT)
    amplitude = np.sqrt(2 * INTENSITY / (c * epsilon_0))
    assert step_for(simu, amplitude) == pytest.approx(
        1 / (STEPS_PER_LENGTH * rate), rel=STEP_REL_TOL
    )


def test_the_cnlse_step_counts_the_cross_coupling(backend):
    """``dz = 1 / (12 k |n12| I sat)`` m with ``n2 = n22 = 0``: g12 alone turns."""
    simu = CNLSE(
        alpha=0,
        power=1.0,
        window=BEAM_WINDOW,
        n2=0,
        n12=N2,
        V=None,
        L=1e-2,
        NX=N_GRID,
        NY=N_GRID,
        Isat=I_SAT,
        wvl=WAVELENGTH,
        backend=backend,
    )
    simu.n22 = 0
    amplitude = np.sqrt(2 * INTENSITY / (c * epsilon_0))
    A = np.full((2, N_GRID, N_GRID), amplitude, dtype=np.complex64)
    got = adapt_delta_z(simu, simu._backend.from_numpy(A), 0.0, 0, 1, [])
    k = 2 * np.pi / WAVELENGTH
    rate = k * abs(N2) * INTENSITY * CROSS_SATURATION
    assert got == pytest.approx(1 / (STEPS_PER_LENGTH * rate), rel=STEP_REL_TOL), (
        f"CNLSE asked for {got} with only n12 set. (Backend {backend})"
    )


@pytest.mark.parametrize("fraction", SECOND_INTENSITY_FRACTIONS)
def test_the_cnlse_step_follows_the_faster_component(backend, fraction):
    """``dz = 1 / (12 max_i k_i (|n_ii| I_i + |n12| I_j) sat)`` m.

    ``sat = 1 / (1 + I1 / Isat + I2 / Isat2)``, and the rate the default step
    and RK4's bound read is the same one.

    Parameters
    ----------
    backend : str
        Backend to run on.
    fraction : float
        Second component's intensity over the first's.
    """
    simu = CNLSE(
        alpha=0,
        power=1.0,
        window=BEAM_WINDOW,
        n2=N2,
        n12=N12,
        V=None,
        L=1e-2,
        NX=N_GRID,
        NY=N_GRID,
        Isat=I_SAT,
        wvl=WAVELENGTH,
        backend=backend,
    )
    simu.n22 = N22
    simu.k2 = K2_OVER_K * simu.k
    simu.I_sat2 = I_SAT2
    I1, I2 = INTENSITY, fraction * INTENSITY
    A = np.empty((2, N_GRID, N_GRID), dtype=np.complex64)
    A[0] = np.sqrt(2 * I1 / (c * epsilon_0))
    A[1] = np.sqrt(2 * I2 / (c * epsilon_0))
    A = simu._backend.from_numpy(A)
    got = adapt_delta_z(simu, A, 0.0, 0, 1, [])
    k = 2 * np.pi / WAVELENGTH
    sat = 1 / (1 + I1 / I_SAT + I2 / I_SAT2)
    rate = sat * max(
        k * (abs(N2) * I1 + abs(N12) * I2),
        K2_OVER_K * k * (abs(N22) * I2 + abs(N12) * I1),
    )
    assert got == pytest.approx(1 / (STEPS_PER_LENGTH * rate), rel=STEP_REL_TOL), (
        f"CNLSE asked for {got} at I2 = {fraction} I1. (Backend {backend})"
    )
    # On a uniform field the weighted mean the default step reads is the peak.
    assert simu._energy_rates(A)["interaction"] == pytest.approx(rate, rel=STEP_REL_TOL)
