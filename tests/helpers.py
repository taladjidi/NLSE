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


# ── Building a solver ────────────────────────────────────────────────────────
#
# Eleven modules used to carry their own factory around the same nine
# parameters: the same function, differing in the class it built and whether it
# passed NY. The physics is stated once here, and a module states only what
# makes it different.
#
# Not every solver shares it. NLSE_3d propagates a pulse and takes an energy, a
# dispersion and a group velocity over a window that is part length and part
# time; GPE takes a chemical coupling and an atomic mass; DDGPE a pump and a
# detuning. Those keep their own parameters, because theirs really are
# different rather than restated.

WAIST = 2.23e-3
WAIST2 = 70e-6  # the second component's, deliberately not the first's
WINDOW = 4 * WAIST

PHYSICS = {
    "alpha": 20,
    "power": 1.05,
    "window": WINDOW,
    "n2": -1.6e-9,
    "n12": -1e-10,
    "V": None,
    "L": 1e-3,
    "Isat": 10e4,
}

# Component 2's parameters, as multiples of component 1's. The constructor
# makes them equal -- alpha2 = alpha, n22 = n2, I_sat2 = I_sat, k2 = k -- so a
# test that leaves them alone cannot see a kernel reading the wrong one of the
# pair. Five such mutations survived the suite before this.
ASYMMETRY = {"alpha2": 0.5, "n22": 0.4, "I_sat2": 0.25}
WAVELENGTH2 = 795e-9


def make(cls, backend="CPU", n=None, symmetric=True, **overrides):
    """Return a solver of this class with the shared optical parameters.

    Only the arguments the class's own signature accepts are passed, so one
    call serves the one- and two-dimensional solvers and the coupled ones.

    Parameters
    ----------
    cls : type
        Solver class.
    backend : str
        Backend name.
    n : int, optional
        Grid points along every spatial axis.
    symmetric : bool
        Whether the two components of a coupled solver share one set of
        parameters, as the constructor leaves them. Pass False to give them
        their own, which is what makes a kernel reading the wrong one of a
        pair visible; the default matches the constructor so that converting
        a test to this factory does not change what it runs.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    NLSE
        The solver.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    params = {k: v for k, v in PHYSICS.items() if k in accepted}
    if n is not None:
        for axis in ("NX", "NY"):
            if axis in accepted:
                params[axis] = n
    params["backend"] = backend
    params.update(overrides)
    solver = cls(**params)
    if hasattr(solver, "n22") and not symmetric:
        for name, factor in ASYMMETRY.items():
            setattr(solver, name, factor * getattr(solver, name))
        solver.k2 = 2 * np.pi / WAVELENGTH2
    return solver


def gaussian(shape, waists=(WAIST, WAIST2), window=WINDOW, dtype=PRECISION_COMPLEX):
    """Return a smooth field of this shape, one Gaussian per component.

    A beam rather than noise: a nonlinear run amplifies a last-bit difference
    into a visible one, so an adversarial field measures the chaos rather than
    whatever is under test. The components are given different widths, or
    ``|A1|**2`` equals ``|A2|**2`` and a kernel reading the wrong one of the
    pair produces no difference at all.

    Parameters
    ----------
    shape : tuple
        Field shape, with a leading 2 for a coupled field.
    waists : tuple
        Beam waist per component.
    window : float
        Extent of the grid.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    np.ndarray
        The field.
    """
    coupled = len(shape) > 1 and shape[0] == 2
    grid = shape[1:] if coupled else shape
    axes = [np.linspace(-window / 2, window / 2, n) for n in grid]
    r2 = (
        sum(a**2 for a in np.meshgrid(*axes, indexing="ij"))
        if len(axes) > 1
        else axes[0] ** 2
    )
    if not coupled:
        return np.exp(-r2 / waists[0] ** 2).astype(dtype)
    return np.stack([np.exp(-r2 / w**2) for w in waists[:2]]).astype(dtype)
