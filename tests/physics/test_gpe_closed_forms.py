"""GPE against closed forms, in the units it documents.

The equation, with ``V`` and ``g`` the energies divided by hbar (rad/s and
rad/s*m^2)::

    i hbar dpsi/dt = -hbar^2/(2m) grad^2 psi + hbar (V + g|psi|^2) psi

so a uniform condensate turns at ``-(V + g n)`` rad/s, ``V > 0`` repels,
``g > 0`` is repulsive, and ``gamma`` empties the gas as
``N(t) = N(0) exp(-gamma t)``. Each test isolates one term against that, on
every backend, for the three splittings and for RK4.
"""

import numpy as np
import pytest
from NLSE import GPE
from scipy.constants import atomic_mass, hbar

# (method, splitting) pairs out_field accepts; RK4 ignores the splitting.
SCHEMES = [
    ("split_step", "lie"),
    ("split_step", "strang"),
    ("split_step", "yoshida"),
    ("RK4", "lie"),
]
SCHEME_IDS = ["lie", "strang", "yoshida", "RK4"]

MASS = 87 * atomic_mass  # kg, rubidium 87
N_UNIFORM = 32  # grid points per axis for the uniform-gas tests
WINDOW_UNIFORM = 100e-6  # m
DENSITY = 1e14  # atoms/m^2, a typical 2D condensate
N_ATOMS = DENSITY * WINDOW_UNIFORM**2  # atoms filling the window at DENSITY
G_INTERACTION = 1e-11  # rad/s*m^2, so g*n = 1e3 rad/s
SAT_DENSITY = DENSITY  # atoms/m^2, halves the interaction at DENSITY
V_UNIFORM = 1e3  # rad/s
T_UNIFORM = 1e-3  # s, one radian at either rate
GAMMA = 200.0  # 1/s
T_LOSS = 5e-3  # s, one e-folding of the atom number at GAMMA
# Tolerances: float32 rounding over the run. The largest error measured over
# the four schemes on CPU and MLX is quoted after each.
UNIFORM_PHASE_TOL = 1e-4  # rad; measured 5.9e-6
DENSITY_REL_TOL = 2e-5  # relative; measured 4.6e-6
ATOMS_REL_TOL = 1e-4  # relative; measured 5.5e-6

TRAP_OMEGA = 2 * np.pi * 100  # rad/s, trap angular frequency
TRAP_LENGTH = np.sqrt(hbar / (MASS * TRAP_OMEGA))  # m, ground-state width
N_TRAP = 64  # grid points per axis for the trap
WINDOW_TRAP = 16 * TRAP_LENGTH  # m, the Gaussian is 1e-28 at the edge
TRAP_PERIODS = 3  # trap periods the ground state is followed over
# s. The default RK4 step gives NaN on this grid, before this fix too and for a
# free Gaussian as well: an open step-size issue. Below 2.83 / (16*pi^2*w), the
# grid's largest kinetic rate (dx = TRAP_LENGTH/4), RK4 is stable.
TRAP_STEP = 1e-5
TRAP_DENSITY_TOL = 1e-3  # density change relative to the peak; measured 2.7e-5
TRAP_OVERLAP_TOL = 1e-3  # |<psi0|psi(t)> / N - exp(-i w t)|; measured 2.2e-4

RATE_REL_TOL = 1e-5  # relative, float32 sums over the grid; measured 4.9e-7


def uniform_gas(backend, g=0.0, gamma=0.0, sat=np.inf):
    """Return a GPE on the uniform-gas grid, without a potential.

    Parameters
    ----------
    backend : str
        Backend name.
    g : float
        Interaction, rad/s*m^2.
    gamma : float
        Loss rate, 1/s.
    sat : float
        Saturation density, atoms/m^2.

    Returns
    -------
    GPE
        The solver.
    """
    return GPE(
        gamma=gamma,
        N=N_ATOMS,
        window=WINDOW_UNIFORM,
        g=g,
        V=None,
        m=MASS,
        NX=N_UNIFORM,
        NY=N_UNIFORM,
        sat=sat,
        backend=backend,
    )


def evolve(simu, psi, t, method, splitting, delta_z=None):
    """Return the field after ``t`` seconds, as a numpy array.

    Parameters
    ----------
    simu : GPE
        The solver.
    psi : np.ndarray
        Initial field; normalized to ``simu.N`` atoms by out_field.
    t : float
        Duration in s.
    method : str
        "split_step" or "RK4".
    splitting : str
        Splitting for split_step.
    delta_z : float or None
        Time step in s; None lets the solver choose.

    Returns
    -------
    np.ndarray
        The field at ``t``.
    """
    out = simu.out_field(
        psi,
        t,
        delta_z=delta_z,
        verbose=False,
        plot=False,
        method=method,
        splitting=splitting,
    )
    return np.asarray(simu._backend.to_numpy(out))


@pytest.mark.parametrize(("method", "splitting"), SCHEMES, ids=SCHEME_IDS)
def test_a_uniform_gas_turns_at_minus_g_n(backend, method, splitting):
    """``psi(t) = sqrt(n) exp(-i g n t)``: repulsive g lowers the phase."""
    simu = uniform_gas(backend, g=G_INTERACTION)
    psi = np.ones((N_UNIFORM, N_UNIFORM), dtype=np.complex64)
    out = evolve(simu, psi, T_UNIFORM, method, splitting)
    expected = -G_INTERACTION * DENSITY * T_UNIFORM
    phase = np.angle(out * np.exp(-1j * expected))
    assert np.max(np.abs(phase)) < UNIFORM_PHASE_TOL, (
        f"{backend} {method}/{splitting}: phase {np.angle(out[0, 0]):+.6f} rad "
        f"against -g*n*t = {expected:+.6f} rad"
    )
    np.testing.assert_allclose(np.abs(out) ** 2, DENSITY, rtol=DENSITY_REL_TOL)


@pytest.mark.parametrize(("method", "splitting"), SCHEMES, ids=SCHEME_IDS)
def test_saturation_divides_the_interaction_by_one_plus_n_over_sat(
    backend, method, splitting
):
    """``psi(t) = sqrt(n) exp(-i g n t / (1 + n / sat))``, -0.5 rad at sat = n."""
    simu = uniform_gas(backend, g=G_INTERACTION, sat=SAT_DENSITY)
    psi = np.ones((N_UNIFORM, N_UNIFORM), dtype=np.complex64)
    out = evolve(simu, psi, T_UNIFORM, method, splitting)
    expected = -G_INTERACTION * DENSITY / (1 + DENSITY / SAT_DENSITY) * T_UNIFORM
    phase = np.angle(out * np.exp(-1j * expected))
    assert np.max(np.abs(phase)) < UNIFORM_PHASE_TOL, (
        f"{backend} {method}/{splitting}: phase {np.angle(out[0, 0]):+.6f} rad "
        f"against -g*n*t/(1 + n/sat) = {expected:+.6f} rad"
    )
    np.testing.assert_allclose(np.abs(out) ** 2, DENSITY, rtol=DENSITY_REL_TOL)


@pytest.mark.parametrize(("method", "splitting"), SCHEMES, ids=SCHEME_IDS)
def test_a_uniform_potential_turns_the_phase_at_minus_v(backend, method, splitting):
    """``psi(t) = psi(0) exp(-i V0 t)`` with V0 in rad/s."""
    simu = uniform_gas(backend)
    simu.V = np.full((N_UNIFORM, N_UNIFORM), V_UNIFORM)
    psi = np.ones((N_UNIFORM, N_UNIFORM), dtype=np.complex64)
    out = evolve(simu, psi, T_UNIFORM, method, splitting)
    expected = -V_UNIFORM * T_UNIFORM
    phase = np.angle(out * np.exp(-1j * expected))
    assert np.max(np.abs(phase)) < UNIFORM_PHASE_TOL, (
        f"{backend} {method}/{splitting}: phase {np.angle(out[0, 0]):+.6f} rad "
        f"against -V0*t = {expected:+.6f} rad"
    )


@pytest.mark.parametrize(("method", "splitting"), SCHEMES, ids=SCHEME_IDS)
def test_gamma_empties_the_gas_exponentially(backend, method, splitting):
    """``N(t) = N0 exp(-gamma t)``, and the phase ``-g n0 (1 - exp(-gamma t))/gamma``.

    The phase is the interaction integrated over the decaying density, which
    is what the lossy real-space step is solved from.
    """
    simu = uniform_gas(backend, g=G_INTERACTION, gamma=GAMMA)
    psi = np.ones((N_UNIFORM, N_UNIFORM), dtype=np.complex64)
    out = evolve(simu, psi, T_LOSS, method, splitting)
    atoms = float(np.sum(np.abs(out) ** 2) * simu.delta_X * simu.delta_Y)
    assert atoms == pytest.approx(
        N_ATOMS * np.exp(-GAMMA * T_LOSS), rel=ATOMS_REL_TOL
    ), (
        f"{backend} {method}/{splitting}: {atoms:.6g} atoms left against "
        f"N0 exp(-gamma t) = {N_ATOMS * np.exp(-GAMMA * T_LOSS):.6g}"
    )
    expected = -G_INTERACTION * DENSITY * (1 - np.exp(-GAMMA * T_LOSS)) / GAMMA
    phase = np.angle(out * np.exp(-1j * expected))
    assert np.max(np.abs(phase)) < UNIFORM_PHASE_TOL, (
        f"{backend} {method}/{splitting}: phase {np.angle(out[0, 0]):+.6f} rad "
        f"against {expected:+.6f} rad"
    )


@pytest.mark.parametrize(("method", "splitting"), SCHEMES, ids=SCHEME_IDS)
def test_the_trap_ground_state_turns_at_the_zero_point_energy(
    backend, method, splitting
):
    """``V = m w^2 r^2 / (2 hbar)``: the ground state is stationary at ``-w t``.

    ``exp(-r^2 / (2 l^2))`` with ``l = sqrt(hbar / (m w))`` is the 2D ground
    state, of energy ``hbar w``: over several periods its density stays put
    and its overlap with the initial state turns as ``exp(-i w t)``.
    """
    simu = GPE(
        gamma=0,
        N=1.0,
        window=WINDOW_TRAP,
        g=0,
        V=None,
        m=MASS,
        NX=N_TRAP,
        NY=N_TRAP,
        backend=backend,
    )
    r2 = simu.XX**2 + simu.YY**2
    simu.V = MASS * TRAP_OMEGA**2 * r2 / (2 * hbar)
    psi = np.exp(-r2 / (2 * TRAP_LENGTH**2)).astype(np.complex64)
    t = TRAP_PERIODS * 2 * np.pi / TRAP_OMEGA
    out = evolve(simu, psi, t, method, splitting, TRAP_STEP)

    density_0 = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
    density_t = np.abs(out) ** 2 / np.sum(np.abs(out) ** 2)
    change = float(np.max(np.abs(density_t - density_0)) / np.max(density_0))
    assert change < TRAP_DENSITY_TOL, (
        f"{backend} {method}/{splitting}: the ground state's density moved by "
        f"{change:.2e} of its peak over {TRAP_PERIODS} trap periods"
    )
    overlap = np.sum(np.conj(psi) * out) / np.sqrt(
        np.sum(np.abs(psi) ** 2) * np.sum(np.abs(out) ** 2)
    )
    expected = np.exp(-1j * TRAP_OMEGA * t)
    assert abs(overlap - expected) < TRAP_OVERLAP_TOL, (
        f"{backend} {method}/{splitting}: overlap {overlap:.6f} against "
        f"exp(-i w t) = {expected:.6f}"
    )


def test_the_step_limits_see_the_rates_in_rad_per_s(backend):
    """The step size is built from ``g n`` and ``V``, not a converted copy.

    With the optical conversion left in, the potential came out 7e-26 of this
    and the interaction 2e-28.
    """
    simu = uniform_gas(backend, g=G_INTERACTION)
    simu.V = np.full((N_UNIFORM, N_UNIFORM), V_UNIFORM)
    psi = np.ones((N_UNIFORM, N_UNIFORM), dtype=np.complex64)
    A, _ = simu._prepare_output_array(psi, True)
    simu._precompute_step_constants(simu.V, np.complex64)
    rates = simu._energy_rates(A)
    assert rates["interaction"] == pytest.approx(
        G_INTERACTION * DENSITY, rel=RATE_REL_TOL
    )
    assert rates["potential"] == pytest.approx(V_UNIFORM, rel=RATE_REL_TOL)
