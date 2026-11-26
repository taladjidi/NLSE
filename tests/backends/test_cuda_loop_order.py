"""The captured CuPy loop runs after the work queued before it.

``execute_loop`` replays its steps on a non-blocking stream, and such a stream
does not wait for the null stream. The half step a merged Strang run applies
before the loop was still running when the loop's first step read the field.
The loop with callbacks takes the same steps one at a time on one stream, so
it is the reference here.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import list_available_backends

pytestmark = pytest.mark.skipif(
    "CUPY" not in list_available_backends(), reason="needs CUDA"
)

N = 1024  # points per side: a half step long enough to outlast the loop's start
WINDOW = 1e-2  # m
WAIST = 2e-3  # m
L = 1e-3  # m
STEP = 5e-5  # m, under the 7.0e-5 m this grid caps a step at
Z = 1.65e-4  # m: three whole steps, the fewest the loop captures, and a cut one
TRIALS = 3
# Largest difference between the two loops, over the largest amplitude.
# Measured 1.36e-6 in twelve runs of twelve on an RTX 3050; without the wait
# the captured loop returned NaN in all twelve.
RTOL = 2e-5


def no_callback(simu, A, z, i):
    """Do nothing: a callback selects the loop that steps one at a time."""


def propagate(callback):
    """Return the field after Z, through the loop the callback selects.

    Parameters
    ----------
    callback : callable or None
        None for the captured loop.

    Returns
    -------
    np.ndarray
        The output field on the host.
    """
    simu = NLSE(
        alpha=0.0,
        power=1.0,
        window=WINDOW,
        n2=-1e-9,
        V=None,
        L=L,
        NX=N,
        NY=N,
        backend="CUPY",
    )
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    out = simu.out_field(
        field, Z, delta_z=STEP, verbose=False, splitting="strang", callback=callback
    )
    assert simu._current_delta_z == STEP, "the step was capped, not tested"
    return np.asarray(simu._backend.to_numpy(out))


@pytest.mark.parametrize("trial", range(TRIALS))
def test_the_captured_loop_agrees_with_the_stepped_one(trial):
    """Both loops take the same steps, so they return the same field."""
    stepped = propagate(no_callback)
    captured = propagate(None)
    difference = np.max(np.abs(captured - stepped)) / np.max(np.abs(stepped))
    assert difference < RTOL, (
        f"trial {trial}: the captured loop is {difference:.3e} away from the "
        f"stepped one"
    )
