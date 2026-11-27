"""A negative delta_z propagates backwards over the whole of z."""

import warnings

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.callbacks import adapt_delta_z, adapt_delta_z_to_error

N = 64
WINDOW = 1e-3  # m
WAIST = 1e-4  # m; narrow, so diffraction over the partial step is visible
L = 1e-2  # m, longer than any z below: the medium is not left
STEP = 1e-4  # m
SCHEMES = {
    "lie": {"splitting": "lie"},
    "strang": {"splitting": "strang"},
    "yoshida": {"splitting": "yoshida"},
    "RK4": {"method": "RK4"},
}
# Round-trip error, relative L2, by field precision. Measured at most 8.2e-12
# (RK4, its own truncation; the splittings 1.1e-14) and 1.7e-5 (yoshida in
# complex64); a partial step taken forward gives 1.76e-3.
ROUNDTRIP_TOL = {np.complex128: 1e-10, np.complex64: 1e-4}
# Focusing weakly enough that adapt_delta_z keeps the step near 1e-4 m.
ADAPTIVE_N2 = -1.6e-12  # m^2/W
# Arguments of adapt_delta_z_to_error: tolerance, update_every, bounds,
# safety, and a floor of 1e-5 m so that a step pinned to it ends in 100 steps.
ERROR_CONTROL_ARGS = (1e-4, 1, (0.5, 2.0), 0.9, 1e-5)
# Relative difference between the backward steps and the forward ones negated.
# Measured at most 2.0e-6 (complex64) and 5.2e-16 (complex128).
STEP_HISTORY_RTOL = 1e-5


def solver(backend, n2=0.0):
    """Return a lossless solver on a fine grid, linear unless n2 is given.

    Parameters
    ----------
    backend : str
        Backend to run on.
    n2 : float
        Nonlinear index in m^2/W.

    Returns
    -------
    NLSE
        The solver.
    """
    return NLSE(
        alpha=0.0,
        power=1.0,
        window=WINDOW,
        n2=n2,
        V=None,
        L=L,
        NX=N,
        NY=N,
        backend=backend,
    )


def beam(simu):
    """Return a narrow Gaussian, in the widest precision the backend has."""
    double = simu._backend.supports_double_precision()
    dtype = np.complex128 if double else np.complex64
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(dtype)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("z", [1.05e-3, 1e-3])
def test_a_backward_run_undoes_a_forward_one(backend, scheme, z):
    """Forward then backward over z returns the input, whatever z / dz is."""
    simu = solver(backend)
    field = beam(simu)
    kwargs = {"verbose": False, "normalize": False, **SCHEMES[scheme]}
    # Yoshida warns on complex64; the precision is not what is tested here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        there = simu.out_field(field.copy(), z, delta_z=STEP, **kwargs)
        back = simu.out_field(there, z, delta_z=-STEP, **kwargs)
    back = np.asarray(simu._backend.to_numpy(back))
    err = float(np.linalg.norm(back - field) / np.linalg.norm(field))
    assert err < ROUNDTRIP_TOL[field.dtype.type], (
        f"{scheme}, z={z:g}, dz={STEP:g}: the round trip is off by {err:.2e}; "
        f"the backward run's partial step went the wrong way"
    )


def test_a_negative_distance_is_refused(backend):
    """A negative z is not a backward run; it used to return the input."""
    simu = solver(backend)
    with pytest.raises(ValueError, match="negative"):
        simu.out_field(beam(simu), -1e-3, delta_z=STEP, verbose=False)


def test_a_backward_step_is_capped_like_a_forward_one(backend):
    """A step past the accuracy limit is lowered by magnitude, keeping its sign."""
    simu = solver(backend, n2=-1.6e-7)
    # complex64, where the cap is the only warning a Lie run gives.
    field = beam(simu).astype(np.complex64)
    with pytest.warns(UserWarning, match="exceeds"):
        simu.out_field(field, 1e-3, delta_z=-1.0, verbose=False)
    used = simu._current_delta_z
    assert -1.0 < used < 0, f"a backward step of -1 m came out as {used:g} m"


def record_step(simu, A, z, i, seen):
    """Record the position and the step the callbacks are shown."""
    seen.append((z, simu._current_delta_z))


def step_history(backend, delta_z, callback, args):
    """Run with an adaptive callback and return the (z, step) it went through.

    Parameters
    ----------
    backend : str
        Backend to run on.
    delta_z : float
        Starting step, negative for a backward run.
    callback : callable
        The adaptive callback.
    args : tuple
        Its arguments.

    Returns
    -------
    list of tuple
        Position and step after every step.
    """
    simu = solver(backend, n2=ADAPTIVE_N2)
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simu.out_field(
            beam(simu),
            1e-3,
            delta_z=delta_z,
            verbose=False,
            callback=[callback, record_step],
            callback_args=[args, (seen,)],
        )
    return seen


def assert_mirrors_the_forward_run(backend, callback, args):
    """Check that a backward run takes the forward run's steps, negated."""
    forward = step_history(backend, STEP, callback, args)
    backward = step_history(backend, -STEP, callback, args)
    name = callback.__name__
    assert backward[-1][0] == pytest.approx(-1e-3, rel=1e-12), (
        f"{name}: the backward run ended at z={backward[-1][0]:.3e} m, not -1e-3 m"
    )
    assert len(backward) == len(forward), (
        f"{name}: {len(backward)} backward steps against {len(forward)} forward"
    )
    np.testing.assert_allclose(
        [step for _, step in backward],
        [-step for _, step in forward],
        rtol=STEP_HISTORY_RTOL,
        err_msg=f"{name}: the backward steps are not the forward ones negated",
    )


def test_adapt_delta_z_keeps_a_backward_run_backward(backend):
    """adapt_delta_z returns a size; the run keeps its direction."""
    assert_mirrors_the_forward_run(backend, adapt_delta_z, (1, []))


def test_adapt_delta_z_to_error_keeps_a_backward_run_backward(backend):
    """The error controller sizes a backward step like a forward one."""
    assert_mirrors_the_forward_run(backend, adapt_delta_z_to_error, ERROR_CONTROL_ARGS)
