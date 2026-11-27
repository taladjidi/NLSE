"""Switching a solver's backend keeps nothing the old one built.

The setter used to swap the backend alone. The propagators and dispersion
operators a run caches are arrays of the backend that built them, so MLX to
CPU raised in the CPU kernels, and CPU to MLX propagated with a propagator
built on the host instead of the one MLX builds. The RK4 scratch buffers
outlived a switch the same way, and CNLSE's scaled second potential outlived
the run itself.
"""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import CNLSE, NLSE

# Every ordered pair of distinct backends; empty, so skipped, with only one.
SWITCHES = [(a, b) for a in AVAILABLE_BACKENDS for b in AVAILABLE_BACKENDS if a != b]

N = 64
WAIST = 2.23e-3  # m
L = 1e-2  # m, length of the medium
Z = 2e-3  # m, distance each run propagates
DELTA_Z = 1e-4  # m


def solver(backend, cls=NLSE, **kwargs):
    """Return a small self-defocusing solver, with any argument overridden.

    Parameters
    ----------
    backend : str
        Backend name.
    cls : type
        NLSE or CNLSE.
    **kwargs
        Constructor arguments overriding the defaults.

    Returns
    -------
    NLSE
        The solver.
    """
    base = {
        "alpha": 0,
        "power": 1.05,
        "window": 4 * WAIST,
        "n2": -1.6e-9,
        "V": None,
        "L": L,
        "NX": N,
        "NY": N,
        "Isat": 1e5,
        "backend": backend,
    }
    return cls(**{**base, **kwargs})


def gaussian(simu):
    """Return a Gaussian beam on the solver's grid.

    Parameters
    ----------
    simu : NLSE
        The solver.

    Returns
    -------
    np.ndarray
        The field, complex64, on the host.
    """
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)


def propagate(simu, method="split_step"):
    """Return a Gaussian beam propagated over ``Z``.

    Parameters
    ----------
    simu : NLSE
        The solver.
    method : str
        ``out_field``'s method.

    Returns
    -------
    np.ndarray
        The field, on the host.
    """
    return np.asarray(
        simu.out_field(
            gaussian(simu), Z, delta_z=DELTA_Z, method=method, verbose=False, plot=False
        )
    )


def rk4_step(simu):
    """Return a Gaussian beam after one ``split_step_RK4`` taken by hand.

    Parameters
    ----------
    simu : NLSE
        The solver.

    Returns
    -------
    np.ndarray
        The field, on the host.
    """
    backend = simu._backend
    A = backend.from_numpy(gaussian(simu))
    propagator = backend.from_numpy(simu._build_propagator_rk4(np.complex64))
    out = simu.split_step_RK4(A, None, propagator, simu._build_fft_plan(A), DELTA_Z)
    return np.asarray(backend.to_numpy(out))


@pytest.mark.parametrize("first,second", SWITCHES)
def test_a_switched_solver_matches_one_built_on_the_new_backend(first, second):
    """After a run on ``first``, a run on ``second`` is a fresh solver's run."""
    switched = solver(first)
    propagate(switched)
    switched.backend = second
    np.testing.assert_array_equal(propagate(switched), propagate(solver(second)))


@pytest.mark.parametrize("first,second", SWITCHES)
def test_a_switched_solver_takes_a_fresh_solvers_rk4_step(first, second):
    """After an RK4 run on ``first``, an RK4 step on ``second`` is a fresh one."""
    switched = solver(first)
    propagate(switched, method="RK4")
    switched.backend = second
    np.testing.assert_array_equal(rk4_step(switched), rk4_step(solver(second)))


def test_a_coupled_run_leaves_no_second_potential(backend):
    """CNLSE's scaled second potential ends with the run, and a switch keeps it so."""
    simu = solver(backend, cls=CNLSE, n12=0, V=np.zeros((N, N), dtype=np.float32))
    field = np.stack([gaussian(simu)] * 2)
    simu.out_field(field, Z, delta_z=DELTA_Z, verbose=False, plot=False)
    assert simu._V2_scaled is None, f"{backend}: the run left _V2_scaled set"
    simu.backend = "CPU"
    assert simu._V2_scaled is None, f"{backend} to CPU: _V2_scaled came back"
