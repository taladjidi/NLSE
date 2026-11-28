"""Backend-agnostic helpers for the solver tests."""

import numpy as np
from NLSE import CNLSE_1d, NLSE_1d
from NLSE.backends import list_available_backends

PRECISION_COMPLEX = np.complex64
AVAILABLE_BACKENDS = list_available_backends()

# Convergence order of each splitting and of RK4: the slope
# tests/solvers/test_rabi_coupling.py measures and the README's table states.
ORDERS = {"lie": 1, "strang": 2, "yoshida": 4, "RK4": 4}


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
# The physics is stated once here; a module states only what differs.
# NLSE_3d, GPE and DDGPE keep their own parameters, which really differ.

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
# Solvers on a line, whose power is per unit length.
LINE_SOLVERS = (NLSE_1d, CNLSE_1d)
# W/m, 375.7: a Gaussian of WAIST on a line peaks at sqrt(2/pi) P / w, so this
# gives it the peak intensity of the 2D beam, 2 P / (pi w^2), at PHYSICS power.
LINE_POWER = PHYSICS["power"] * np.sqrt(2 / np.pi) / WAIST


def make(cls, backend="CPU", n=None, symmetric=True, **overrides):
    """Return a solver of this class with the shared optical parameters.

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
        Any constructor argument, by keyword. A 1D solver's ``power``
        defaults to `LINE_POWER`, in W/m.

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
    if issubclass(cls, LINE_SOLVERS):
        params["power"] = LINE_POWER
    params.update(overrides)
    solver = cls(**params)
    if hasattr(solver, "n22") and not symmetric:
        for name, factor in ASYMMETRY.items():
            setattr(solver, name, factor * getattr(solver, name))
        solver.k2 = 2 * np.pi / WAVELENGTH2
    return solver


def gaussian_beam(r_sq, waist, k, z, dims):
    """Return the paraxial Gaussian beam, the closed form of free diffraction.

    Parameters
    ----------
    r_sq : np.ndarray
        Squared distance to the axis, in m^2.
    waist : float
        1/e radius of the field at ``z = 0``, in m.
    k : float
        Wavenumber, in 1/m.
    z : float
        Distance from the waist, in m.
    dims : int
        Number of transverse dimensions, 1 or 2.

    Returns
    -------
    np.ndarray
        The complex field, unit amplitude on axis at the waist.
    """
    q = 1 + 2j * z / (k * waist**2)
    return q ** (-dims / 2) * np.exp(-r_sq / (waist**2 * q))


def gaussian(shape, waists=(WAIST, WAIST2), window=WINDOW, dtype=PRECISION_COMPLEX):
    """Return a smooth field of this shape, one Gaussian per component.

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
