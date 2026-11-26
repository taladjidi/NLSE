"""What out_field accepts for its step and its callbacks.

A complex delta_z (imaginary time) was documented and crashed deep in the step
limits, the propagator cache or the kernels; it is now refused up front. A list
of callbacks with callback_args left at its default raised from zip(strict=True)
although a single callback runs with it. A zero step with a callback never
returned, and a NaN one returned the input; both are refused, from out_field
and from a callback.
"""

import math
from unittest import mock

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 32
WINDOW = 1e-2  # m
WAIST = 2e-3  # m
L = 1e-3  # m
STEP = 2e-4  # m, divides L into 5
RUNAWAY = 100  # callback calls; the longest run here takes 10
# (z, delta_z, steps) in m: distances on a step boundary to round-off.
# 0.3e-3 / 0.1e-3 is 2.9999999999999996, and ten steps of 1e-4 m end 3e-16 m
# short of the second distance; the last two are less than a step, and none.
STEP_COUNTS = [
    (0.3e-3, 0.1e-3, 3),
    (L + 3e-16, 1e-4, 10),
    (0.5e-4, 1e-4, 1),
    (0.0, 1e-4, 0),
]


def solver(backend):
    """Return a small lossless solver.

    Parameters
    ----------
    backend : str
        Backend to run on.

    Returns
    -------
    NLSE
        The solver.
    """
    return NLSE(
        alpha=0.0,
        power=1.0,
        window=WINDOW,
        n2=-1e-9,
        V=None,
        L=L,
        NX=N,
        NY=N,
        backend=backend,
    )


def beam(simu):
    """Return a Gaussian field."""
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)


def record_position(simu, A, z, i, seen):
    """Append the position the callback is handed."""
    seen.append(z)


def fail_if_endless(simu, A, z, i):
    """Raise once a run has taken more steps than any here needs.

    A zero step never ends the run, and a test that can only hang is not one.
    """
    if i > RUNAWAY:
        raise RuntimeError(f"the run is still going after {i} steps")


def return_step(simu, A, z, i, step):
    """Ask the loop for this step."""
    return step


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("delta_z", [1e-4j, complex(1e-4, 0), np.complex64(1e-4)])
def test_a_complex_step_is_refused(backend, delta_z):
    """Imaginary time is not supported, and says so rather than crashing."""
    simu = solver(backend)
    with pytest.raises(TypeError, match="imaginary-time"):
        simu.out_field(beam(simu), L, delta_z=delta_z, verbose=False)


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_list_of_callbacks_runs_with_default_arguments(backend):
    """Each callback of a list is called at every step, at the field's position."""
    simu = solver(backend)
    first, second = [], []
    simu.out_field(
        beam(simu),
        L,
        delta_z=STEP,
        verbose=False,
        callback=[
            lambda s, A, z, i: first.append(z),
            lambda s, A, z, i: second.append(z),
        ],
    )
    expected = [STEP * (k + 1) for k in range(round(L / STEP))]
    assert first == pytest.approx(expected), f"first callback saw {first}"
    assert second == pytest.approx(expected), f"second callback saw {second}"


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_list_of_callbacks_with_too_few_arguments_is_refused(backend):
    """Arguments given for a list must still come one tuple per callback."""
    simu = solver(backend)
    with pytest.raises(ValueError):
        simu.out_field(
            beam(simu),
            L,
            delta_z=STEP,
            verbose=False,
            callback=[record_position, record_position],
            callback_args=[([],)],
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("delta_z", [0.0, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("callback", [None, fail_if_endless])
def test_a_step_that_is_zero_or_not_finite_is_refused(backend, delta_z, callback):
    """A zero step with a callback never returned; a NaN one returned the input."""
    simu = solver(backend)
    with pytest.raises(ValueError, match="not a step"):
        simu.out_field(beam(simu), L, delta_z=delta_z, verbose=False, callback=callback)


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("z", [math.nan, math.inf])
def test_a_distance_that_is_not_finite_is_refused(backend, z):
    """A NaN distance returned the input; an infinite one never returned."""
    simu = solver(backend)
    with pytest.raises(ValueError, match="finite"):
        simu.out_field(
            beam(simu), z, delta_z=STEP, verbose=False, callback=fail_if_endless
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize(
    "returned, error",
    [(0.0, ValueError), (math.nan, ValueError), (1e-4j, TypeError)],
)
def test_a_callback_cannot_return_what_out_field_refuses(backend, returned, error):
    """A step a callback returns is checked like the one out_field is given."""
    simu = solver(backend)
    with pytest.raises(error, match="delta_z"):
        simu.out_field(
            beam(simu),
            L,
            delta_z=STEP,
            verbose=False,
            callback=[return_step, fail_if_endless],
            callback_args=[(returned,), ()],
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("z, delta_z, count", STEP_COUNTS)
@pytest.mark.parametrize("direction", [1, -1])
def test_round_off_neither_drops_a_step_nor_adds_a_sliver(
    backend, z, delta_z, count, direction
):
    """A run takes the steps that fit, and ends where it was asked to."""
    simu = solver(backend)
    seen = []
    simu.out_field(
        beam(simu),
        z,
        delta_z=direction * delta_z,
        verbose=False,
        callback=record_position,
        callback_args=(seen,),
    )
    assert len(seen) == count, f"{len(seen)} steps over {z:g} m at {delta_z:g} m"
    end = seen[-1] if seen else 0.0
    assert end == pytest.approx(direction * z, rel=1e-12), f"ended at {end!r} m"


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("callback", [None, fail_if_endless])
def test_a_distance_of_whole_steps_takes_no_cut_one(backend, callback):
    """Three steps cover 0.3e-3 m, though the quotient is 2.9999999999999996."""
    simu = solver(backend)
    z, delta_z, _ = STEP_COUNTS[0]
    with mock.patch.object(
        simu, "_take_partial_step", wraps=simu._take_partial_step
    ) as cut:
        simu.out_field(beam(simu), z, delta_z=delta_z, verbose=False, callback=callback)
    assert cut.call_count == 0, f"{cut.call_count} cut steps in a run of 3 whole"
