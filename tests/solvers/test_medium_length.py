"""The medium acts over exactly its length L, whatever the step."""

import math
import warnings

import numpy as np
import pytest
from NLSE import NLSE
from scipy.constants import c, epsilon_0

N = 16
WINDOW = 1e-2  # m; coarse enough that dispersion does not cap the steps below
WAVELENGTH = 780e-9  # m, the constructor's default
K = 2 * np.pi / WAVELENGTH  # 1/m
L = 1e-2  # m, length of the medium
Z = 2e-2  # m, twice the medium
N2 = 1e-9  # m^2/W, focusing
# Total nonlinear phase over the medium, rad. Small enough that RK4's own
# truncation error at 0.1 rad per step stays below the tolerance.
PHASE = 0.25
INTENSITY = PHASE / (K * N2 * L)  # W/m^2
AMPLITUDE = math.sqrt(2 * INTENSITY / (c * epsilon_0))  # V/m
# 1/m, intensity absorption. Its step ceiling, LOSS_PER_STEP_LIMIT / (2 * ALPHA)
# = 1.25e-2 m (NLSE/solvers/step_size.py, _loss_max_dz), is above 4e-3 m.
ALPHA = 2.0
GROWN_STEP = 5e-3  # m, what the callback below grows a 1e-3 m step to
GROW_AT = 7  # step index after which it grows: at z = 8e-3 m, 2e-3 m short of L
SCHEMES = {
    "lie": {"splitting": "lie"},
    "strang": {"splitting": "strang"},
    "yoshida": {"splitting": "yoshida"},
    "RK4": {"method": "RK4"},
}
# Relative error of the phase and of the intensity. Measured at most 9.1e-6
# (RK4, dz = 4e-3, its truncation error) and 2.9e-6 (yoshida, complex64).
PHASE_RTOL = 2e-5
BEER_RTOL = 1e-5


def propagate(backend, scheme, delta_z, n2=N2, alpha=0.0, **kwargs):
    """Propagate a uniform field over Z and return it with the solver.

    Parameters
    ----------
    backend : str
        Backend to run on.
    scheme : str
        Key of SCHEMES.
    delta_z : float
        Step, negative for a backward run.
    n2 : float
        Nonlinear index in m^2/W.
    alpha : float
        Intensity absorption in 1/m.
    **kwargs
        Passed on to out_field (callbacks).

    Returns
    -------
    tuple
        The output field on the host and the solver.
    """
    simu = NLSE(
        alpha=alpha,
        power=1.0,
        window=WINDOW,
        n2=n2,
        V=None,
        L=L,
        NX=N,
        NY=N,
        backend=backend,
    )
    double = simu._backend.supports_double_precision()
    field = np.full((N, N), AMPLITUDE, dtype=np.complex128 if double else np.complex64)
    # Yoshida warns on complex64 and with loss; neither is what is tested here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = simu.out_field(
            field,
            Z,
            delta_z=delta_z,
            verbose=False,
            normalize=False,
            **SCHEMES[scheme],
            **kwargs,
        )
    return np.asarray(simu._backend.to_numpy(out)), simu


def grow_once(simu, A, z, i, seen):
    """Record the position and grow the step once, a little short of L."""
    seen.append(z)
    return math.copysign(GROWN_STEP, simu._current_delta_z) if i == GROW_AT else None


def record_step(simu, A, z, i, steps):
    """Record the step the callbacks are shown."""
    steps.append(simu._current_delta_z)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("delta_z", [3e-3, 4e-3, 1e-3, -3e-3])
def test_the_medium_imprints_its_phase_over_l(backend, scheme, delta_z):
    """Phase k * n2 * I * L after a run to 2L, signed by the direction."""
    out, simu = propagate(backend, scheme, delta_z)
    assert simu._current_delta_z == delta_z, "the step was capped, not tested"
    got = float(np.angle(out[N // 2, N // 2]))
    expected = math.copysign(PHASE, delta_z)
    assert got == pytest.approx(expected, rel=PHASE_RTOL), (
        f"{scheme}, dz={delta_z:g}: phase {got:.5f} rad, but a medium {L:g} m "
        f"long imprints {expected:.5f} rad ({got / expected:.3f} times)"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("delta_z", [1e-3, -1e-3])
def test_a_step_grown_by_a_callback_is_cut_at_l(backend, scheme, delta_z):
    """A callback that grows the step must not carry the medium past L."""
    seen = []
    out, _ = propagate(
        backend, scheme, delta_z, callback=grow_once, callback_args=(seen,)
    )
    got = float(np.angle(out[N // 2, N // 2]))
    expected = math.copysign(PHASE, delta_z)
    assert got == pytest.approx(expected, rel=PHASE_RTOL), (
        f"{scheme}, dz={delta_z:g} grown to {GROWN_STEP:g}: phase {got:.5f} "
        f"rad, expected {expected:.5f} rad"
    )
    assert abs(seen[-1]) == pytest.approx(Z, rel=1e-12), (
        f"the run stopped at {seen[-1]:.6e} m instead of {Z:g} m"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("delta_z", [3e-3, -3e-3])
def test_callbacks_see_the_step_just_taken(backend, scheme, delta_z):
    """The steps the callbacks are shown add up to the distance run."""
    steps = []
    propagate(backend, scheme, delta_z, callback=record_step, callback_args=(steps,))
    assert sum(steps) == pytest.approx(math.copysign(Z, delta_z), rel=1e-12), (
        f"{scheme}, dz={delta_z:g}: the callbacks were shown steps adding up to "
        f"{sum(steps):.6e} m over a run of {Z:g} m"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("delta_z", [3e-3, 4e-3])
def test_absorption_stops_at_l(backend, scheme, delta_z):
    """Beer's law over the medium and nothing after it: I0 * exp(-alpha L)."""
    out, simu = propagate(backend, scheme, delta_z, n2=0.0, alpha=ALPHA)
    assert simu._current_delta_z == delta_z, "the step was capped, not tested"
    got = float(np.abs(out[N // 2, N // 2]) ** 2) / AMPLITUDE**2
    expected = math.exp(-ALPHA * L)
    assert got == pytest.approx(expected, rel=BEER_RTOL), (
        f"{scheme}, dz={delta_z:g}: transmission {got:.6f}, Beer's law over "
        f"{L:g} m gives {expected:.6f}"
    )
