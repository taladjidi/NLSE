"""Backend-agnostic helpers for the solver tests.

The solver tests run against every available backend. Assertions therefore
have to avoid numpy-only APIs and hand-rolled ``backend == "..."`` chains,
which is what previously left the MLX paths uncovered.
"""

import numpy as np

PRECISION_COMPLEX = np.complex64


def as_numpy(simu, array):
    """Return a backend array as numpy so assertions can be backend-agnostic.

    Parameters
    ----------
    simu : NLSE
        Solver owning the backend the array lives on.
    array : Any
        Array on the host or on a device.

    Returns
    -------
    np.ndarray
        The array on the host.
    """
    if array is None or isinstance(array, np.ndarray):
        return array
    return simu._backend.to_numpy(array)


def assert_c_contiguous(array, message):
    """Assert C-contiguity for array types that expose the concept.

    MLX arrays have no ``.flags``; they are contiguous by construction, so
    there is nothing to check.

    Parameters
    ----------
    array : Any
        Array to check.
    message : str
        Assertion message.
    """
    if hasattr(array, "flags"):
        assert array.flags.c_contiguous, message


def random_field(shape, dtype=PRECISION_COMPLEX):
    """Return a random complex field as numpy.

    Solvers accept numpy input on every backend and convert internally, so
    tests do not need to build backend-specific arrays.

    Parameters
    ----------
    shape : tuple
        Shape of the field.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    np.ndarray
        Random complex field.
    """
    rng = np.random.default_rng(0)
    return (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
