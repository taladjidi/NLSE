"""The default step follows the potential of the run that takes it.

The step estimate keeps the potential's grids where the field is, under a key
that names the grid and not the potential. A second run with another ``V``
took the step the first one's allowed: 2.4e-6 m where a solver that had never
seen the first potential took 1.7e-4 m.
"""

import numpy as np
import pytest
from NLSE import NLSE

N = 32
WINDOW = 4e-3  # m
WAIST = 1e-3  # m
L = 1e-2  # m, length of the medium
# Over ten default steps under SHALLOW: a shorter run takes Z / 10 instead,
# the floor of DEFAULT_MIN_STEPS, whatever the potential.
Z = 2e-3  # m
# Uniform index offsets. The default steps they give were measured: 2.4e-6 m
# under DEEP and 1.7e-4 m under SHALLOW.
DEEP = 1e-2
SHALLOW = 1e-5
# Least ratio of the two steps for the test to tell them apart; measured 69.
STEP_RATIO = 10


def solver(backend):
    """Return a small lossless NLSE.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    NLSE
        The solver.
    """
    return NLSE(
        alpha=0,
        power=1.0,
        window=WINDOW,
        n2=-1e-9,
        V=None,
        L=L,
        NX=N,
        NY=N,
        Isat=1e5,
        backend=backend,
    )


def record_step(simu, A, z, i, steps):
    """Append the step of this call to ``steps``.

    Parameters
    ----------
    simu : NLSE
        The solver.
    A : Any
        The field, where the backend keeps it.
    z : float
        Distance reached, in m.
    i : int
        Step index.
    steps : list
        Where the step goes.
    """
    steps.append(simu._current_delta_z)


def first_step(simu, V):
    """Run under ``V`` with the default step and return the first one.

    Parameters
    ----------
    simu : NLSE
        The solver.
    V : np.ndarray
        Potential of this run.

    Returns
    -------
    float
        The first step, in m.
    """
    simu.V = V
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    steps = []
    simu.out_field(
        field,
        Z,
        verbose=False,
        plot=False,
        callback=record_step,
        callback_args=(steps,),
    )
    return steps[0]


@pytest.mark.parametrize("in_place", [False, True])
def test_a_second_run_takes_the_step_of_its_own_potential(backend, in_place):
    """A solver that ran under ``DEEP`` steps under ``SHALLOW`` as a new one.

    The second potential is another array, or the first one overwritten.
    """
    fresh = first_step(solver(backend), np.full((N, N), SHALLOW, np.float32))
    simu = solver(backend)
    V = np.full((N, N), DEEP, np.float32)
    assert first_step(simu, V) < fresh / STEP_RATIO
    if in_place:
        V[:] = SHALLOW
    else:
        V = np.full((N, N), SHALLOW, np.float32)
    assert first_step(simu, V) == fresh
