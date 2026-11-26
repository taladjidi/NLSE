"""``adapt_delta_z`` against the nonlinear length, in each solver's own units.

The callback returns a twelfth of ``1 / max(rate)``, with the rate the phase
the nonlinearity turns per unit of the propagation variable: ``k |n2| I /
(1 + I / Isat)`` per metre for NLSE, ``|g| n`` per second for GPE. A uniform
field makes the maximum the value everywhere, so the step has a closed form.
"""

import numpy as np
import pytest
from NLSE import GPE, NLSE
from NLSE.backends import list_available_backends
from NLSE.callbacks import adapt_delta_z
from scipy.constants import atomic_mass, c, epsilon_0

AVAILABLE_BACKENDS = list_available_backends()

STEPS_PER_LENGTH = 12  # steps per nonlinear length, NLSE/callbacks.py:173
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


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
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


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
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
