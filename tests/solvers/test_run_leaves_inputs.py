"""A run gives back the arrays it was given, and refuses a bad field plainly."""

import numpy as np
import pytest
from NLSE import NLSE

N = 32
WINDOW = 4e-3  # m
Z = 1e-4  # m, distance each run propagates
POTENTIAL = 1e-6  # largest value of the potential, small enough to change nothing
BATCH_N2 = (-1e-9, -2e-9)  # m^2/W, one per simulation of a batch


def solver(backend, **kwargs):
    """Return a small linear NLSE, with any constructor argument overridden.

    Parameters
    ----------
    backend : str
        Backend name.
    **kwargs
        Constructor arguments overriding the defaults.

    Returns
    -------
    NLSE
        The solver.
    """
    base = {
        "alpha": 0,
        "power": 1,
        "window": WINDOW,
        "n2": 0,
        "V": None,
        "L": 1e-2,
        "NX": N,
        "NY": N,
        "Isat": 1e30,
        "backend": backend,
    }
    return NLSE(**{**base, **kwargs})


def run(simu, field):
    """Propagate ``field`` over ``Z`` without normalizing it.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        The field.
    """
    simu.out_field(field, Z, verbose=False, plot=False, normalize=False)


def test_the_potential_comes_back_as_the_array_given(backend):
    """``simu.V`` is the caller's float64 array after the run, not a copy."""
    simu = solver(backend)
    potential = np.linspace(0, POTENTIAL, N * N).reshape(N, N)
    simu.V = potential
    run(simu, np.ones((N, N), dtype=np.complex64))
    assert simu.V is potential, f"{backend}: V came back as {simu.V.dtype}"


def test_a_batched_parameter_comes_back_as_the_array_given(backend):
    """A batched ``n2`` is the caller's array after the run, not a copy."""
    n2 = np.array(BATCH_N2).reshape(len(BATCH_N2), 1, 1)
    simu = solver(backend, n2=n2)
    run(simu, np.ones((len(BATCH_N2), N, N), dtype=np.complex64))
    assert simu.n2 is n2, f"{backend}: n2 came back as {simu.n2.dtype}"


def test_a_real_field_is_refused_with_a_type_error(backend):
    """A float32 field raises TypeError, which survives ``python -O``."""
    with pytest.raises(TypeError, match="complex64 or complex128"):
        run(solver(backend), np.ones((N, N), dtype=np.float32))


def test_a_field_off_the_grid_is_refused_with_a_value_error(backend):
    """A field whose last axes are not the grid's raises ValueError."""
    with pytest.raises(ValueError, match="does not end in the grid's"):
        run(solver(backend), np.ones((N, N + 2), dtype=np.complex64))
