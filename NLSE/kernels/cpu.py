import numba
import numpy as np


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def nl_prop(
    A: np.ndarray,
    A_sq: np.ndarray,
    dz: float,
    alpha: float,
    V: np.ndarray,
    g: float,
    Isat: float,
) -> None:
    """Apply real space terms with compiled parallel implementation.

    Parameters
    ----------
    A : np.ndarray
        The field to propagate
    A_sq : np.ndarray
        The field modulus squared
    dz : float
        Propagation step in m
    alpha : float
        Losses
    V : np.ndarray
        Potential
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A = A.ravel()
    A_sq = A_sq.ravel()
    V = V.ravel()
    for i in numba.prange(A.size):
        # saturation
        sat = 1 / (1 + A_sq[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq[i] * sat + 1j * V[i]
        A[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def nl_prop_without_V(
    A: np.ndarray,
    A_sq: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Apply real space terms without potential.

    Parameters
    ----------
    A : cp.ndarray
        The field to propagate
    A_sq : cp.ndarray
        The field modulus squared
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A = A.ravel()
    A_sq = A_sq.ravel()
    for i in numba.prange(A.size):
        # saturation
        sat = 1 / (1 + A_sq[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq[i] * sat
        A[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def nl_prop_c(
    A1: np.ndarray,
    A_sq_1: np.ndarray,
    A_sq_2: np.ndarray,
    dz: float,
    alpha: float,
    V: np.ndarray,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """Apply coupled real space terms with potential.

    Parameters
    ----------
    A1 : cp.ndarray
        The field to propagate (1st component)
    A_sq_1 : cp.ndarray
        The field modulus squared (1st component)
    A_sq_2 : cp.ndarray
        The field modulus squared (2nd component)
    dz : float
        Propagation step in m
    alpha : float
        Losses
    V : cp.ndarray
        Potential
    g11 : float
        Intra-component interactions
    g12 : float
        Inter-component interactions
    Isat1 : float
        Saturation parameter of first component
    Isat2 : float
        Saturation parameter of second component.
    """
    A1 = A1.ravel()
    A_sq_1 = A_sq_1.ravel()
    A_sq_2 = A_sq_2.ravel()
    V = V.ravel()
    for i in numba.prange(A1.size):
        # Saturation parameter
        sat = 1 / (1 + A_sq_1[i] * 1 / Isat1 + A_sq_2[i] * 1 / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += 1j * (g11 * A_sq_1[i] * sat + g12 * A_sq_2[i] * sat)
        # Potential
        arg += 1j * V[i]
        A1[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def nl_prop_without_V_c(
    A1: np.ndarray,
    A_sq_1: np.ndarray,
    A_sq_2: np.ndarray,
    dz: float,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """Apply coupled real space terms without potential.

    Parameters
    ----------
    A1 : cp.ndarray
        The field to propagate (1st component)
    A_sq_1 : cp.ndarray
        The field modulus squared (1st component)
    A_sq_2 : cp.ndarray
        The field modulus squared (2nd component)
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g11 : float
        Intra-component interactions
    g12 : float
        Inter-component interactions
    Isat1 : float
        Saturation parameter of first component
    Isat2 : float
        Saturation parameter of second component.
    """
    A1 = A1.ravel()
    A_sq_1 = A_sq_1.ravel()
    A_sq_2 = A_sq_2.ravel()
    for i in numba.prange(A1.size):
        # Saturation parameter
        sat = 1 / (1 + A_sq_1[i] * 1 / Isat1 + A_sq_2[i] * 1 / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += 1j * (g11 * A_sq_1[i] * sat + g12 * A_sq_2[i] * sat)
        A1[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rabi_coupling(A1: np.ndarray, A2: np.ndarray, dz: float, omega: float) -> None:
    """Apply Rabi coupling term.

    Implement the Rabi hopping term, exchanging density between components.

    Parameters
    ----------
    A1 : np.ndarray
        First field / component
    A2 : np.ndarray
        Second field / component
    dz : float
        Solver step
    omega : float
        Rabi coupling strength
    """
    A1 = A1.ravel()
    A2 = A2.ravel()
    cos_val = np.cos(omega * dz)
    sin_val = np.sin(omega * dz)
    for i in numba.prange(A1.size):
        a1 = A1[i]
        A1[i] = cos_val * a1 - 1j * sin_val * A2[i]
        A2[i] = cos_val * A2[i] - 1j * sin_val * a1


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def vortex(
    im: np.ndarray, i: int, j: int, ii: np.ndarray, jj: np.ndarray, ll: int
) -> None:
    """Generate a vortex of charge l at position (i,j) on the image im.

    Parameters
    ----------
    im : np.ndarray
        Image
    i : int
        position row of the vortex
    j : int
        position column of the vortex
    ii : int
        meshgrid position row (coordinates of the image)
    jj : int
        meshgrid position column (coordinates of the image)
    ll : int
        vortex charge

    Returns
    -------
    None

    """
    for i in numba.prange(im.shape[0]):
        for j in numba.prange(im.shape[1]):
            im[i, j] += np.angle(((ii[i, j] - i) + 1j * (jj[i, j] - j)) ** ll)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod(A: np.ndarray, A_sq: np.ndarray) -> None:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : np.ndarray
        The field
    A_sq : np.ndarray
        The modulus squared of the field

    Returns
    -------
    None

    """
    A = A.ravel()
    A_sq = A_sq.ravel()
    for i in numba.prange(A.size):
        A_sq[i] = A[i].real * A[i].real + A[i].imag * A[i].imag


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_nl_prop(
    A: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Compute |A|^2 inline and apply nonlinear propagation in a single pass.

    Parameters
    ----------
    A : np.ndarray
        The field to propagate (modified in-place)
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A = A.ravel()
    for i in numba.prange(A.size):
        A_sq_val = A[i].real * A[i].real + A[i].imag * A[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        arg = -alpha * sat + 1j * g * A_sq_val * sat
        A[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_nl_prop_v(
    A: np.ndarray,
    V: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Compute |A|^2 inline and apply nonlinear propagation with potential.

    Parameters
    ----------
    A : np.ndarray
        The field to propagate (modified in-place)
    V : np.ndarray
        Potential
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A = A.ravel()
    V = V.ravel()
    for i in numba.prange(A.size):
        A_sq_val = A[i].real * A[i].real + A[i].imag * A[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        arg = -alpha * sat + 1j * g * A_sq_val * sat + 1j * V[i]
        A[i] *= np.exp(dz * arg)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def apply_propagator(A: np.ndarray, propagator: np.ndarray) -> None:
    """Multiply A by propagator in-place, avoiding numpy temporaries.

    Parameters
    ----------
    A : np.ndarray
        The field array (modified in-place)
    propagator : np.ndarray
        The propagator array
    """
    A = A.ravel()
    propagator = propagator.ravel()
    for i in numba.prange(A.size):
        a = A[i]
        p = propagator[i]
        A[i] = (a.real * p.real - a.imag * p.imag) + 1j * (
            a.real * p.imag + a.imag * p.real
        )
