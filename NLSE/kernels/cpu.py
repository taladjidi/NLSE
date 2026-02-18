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
) -> np.ndarray:
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
    A_flat = A.ravel()
    A_sq_flat = A_sq.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A_flat.size):
        # saturation
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq_flat[i] * sat + 1j * V_flat[i]
        A_flat[i] *= np.exp(dz * arg)
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def nl_prop_without_V(
    A: np.ndarray,
    A_sq: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Apply real space terms without potential.

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
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_flat = A.ravel()
    A_sq_flat = A_sq.ravel()
    for i in numba.prange(A_flat.size):
        # saturation
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq_flat[i] * sat
        A_flat[i] *= np.exp(dz * arg)
    return A


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
) -> np.ndarray:
    """Apply coupled real space terms with potential.

    Parameters
    ----------
    A1 : np.ndarray
        The field to propagate (1st component)
    A_sq_1 : np.ndarray
        The field modulus squared (1st component)
    A_sq_2 : np.ndarray
        The field modulus squared (2nd component)
    dz : float
        Propagation step in m
    alpha : float
        Losses
    V : np.ndarray
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
    A1_flat = A1.ravel()
    A_sq_1_flat = A_sq_1.ravel()
    A_sq_2_flat = A_sq_2.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A1_flat.size):
        # Saturation parameter
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += 1j * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
        # Potential
        arg += 1j * V_flat[i]
        A1_flat[i] *= np.exp(dz * arg)
    return A1


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
) -> np.ndarray:
    """Apply coupled real space terms without potential.

    Parameters
    ----------
    A1 : np.ndarray
        The field to propagate (1st component)
    A_sq_1 : np.ndarray
        The field modulus squared (1st component)
    A_sq_2 : np.ndarray
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
    A1_flat = A1.ravel()
    A_sq_1_flat = A_sq_1.ravel()
    A_sq_2_flat = A_sq_2.ravel()
    for i in numba.prange(A1_flat.size):
        # Saturation parameter
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += 1j * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
        A1_flat[i] *= np.exp(dz * arg)
    return A1


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rabi_coupling(A1: np.ndarray, A2: np.ndarray, dz: float, omega: float) -> tuple:
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
    A1_flat = A1.ravel()
    A2_flat = A2.ravel()
    cos_val = np.cos(omega * dz)
    sin_val = np.sin(omega * dz)
    for i in numba.prange(A1_flat.size):
        a1 = A1_flat[i]
        A1_flat[i] = cos_val * a1 - 1j * sin_val * A2_flat[i]
        A2_flat[i] = cos_val * A2_flat[i] - 1j * sin_val * a1
    return A1, A2


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def vortex(
    im: np.ndarray, i0: int, j0: int, ii: np.ndarray, jj: np.ndarray, ll: int
) -> None:
    """Generate a vortex of charge l at position (i0, j0) on the image im.

    Parameters
    ----------
    im : np.ndarray
        Image.
    i0 : int
        Row position of the vortex.
    j0 : int
        Column position of the vortex.
    ii : np.ndarray
        Meshgrid row coordinates of the image.
    jj : np.ndarray
        Meshgrid column coordinates of the image.
    ll : int
        Vortex charge.
    """
    for i in numba.prange(im.shape[0]):
        for j in numba.prange(im.shape[1]):
            im[i, j] += np.angle(((ii[i, j] - i0) + 1j * (jj[i, j] - j0)) ** ll)


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod(A: np.ndarray, A_sq: np.ndarray) -> np.ndarray:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : np.ndarray
        The field.
    A_sq : np.ndarray
        The modulus squared of the field (output buffer).

    Returns
    -------
    np.ndarray
        The modulus squared of the field.
    """
    A_flat = A.ravel()
    A_sq_flat = A_sq.ravel()
    for i in numba.prange(A_flat.size):
        A_sq_flat[i] = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
    return A_sq


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_nl_prop(
    A: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
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
    A_flat = A.ravel()
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        arg = -alpha * sat + 1j * g * A_sq_val * sat
        A_flat[i] *= np.exp(dz * arg)
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_nl_prop_v(
    A: np.ndarray,
    V: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
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
    A_flat = A.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        arg = -alpha * sat + 1j * g * A_sq_val * sat + 1j * V_flat[i]
        A_flat[i] *= np.exp(dz * arg)
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def apply_propagator(A: np.ndarray, propagator: np.ndarray) -> np.ndarray:
    """Multiply A by propagator in-place, avoiding numpy temporaries.

    Parameters
    ----------
    A : np.ndarray
        The field array (modified in-place)
    propagator : np.ndarray
        The propagator array
    """
    A_flat = A.ravel()
    prop_flat = propagator.ravel()
    for i in numba.prange(A_flat.size):
        a = A_flat[i]
        p = prop_flat[i]
        A_flat[i] = (a.real * p.real - a.imag * p.imag) + 1j * (
            a.real * p.imag + a.imag * p.real
        )
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_axpy(
    out: np.ndarray,
    A: np.ndarray,
    c: float,
    k: np.ndarray,
) -> np.ndarray:
    """Compute out = A + c * k element-wise for RK4 stage arguments.

    Parameters
    ----------
    out : np.ndarray
        Output array (modified in-place)
    A : np.ndarray
        Base field
    c : float
        Scalar coefficient
    k : np.ndarray
        RK4 slope array
    """
    out_flat = out.ravel()
    A_flat = A.ravel()
    k_flat = k.ravel()
    for i in numba.prange(A_flat.size):
        out_flat[i] = A_flat[i] + c * k_flat[i]
    return out


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_accumulate(
    acc: np.ndarray,
    w: float,
    k: np.ndarray,
) -> np.ndarray:
    """Compute acc += w * k element-wise for RK4 weighted accumulation.

    Parameters
    ----------
    acc : np.ndarray
        Accumulator array (modified in-place)
    w : float
        Weight coefficient
    k : np.ndarray
        RK4 slope array
    """
    acc_flat = acc.ravel()
    k_flat = k.ravel()
    for i in numba.prange(acc_flat.size):
        acc_flat[i] += w * k_flat[i]
    return acc


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_nl_rhs(
    A_prop: np.ndarray,
    A: np.ndarray,
    A_sq: np.ndarray,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A : np.ndarray
        Original field
    A_sq : np.ndarray
        Field modulus squared
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A.ravel()
    A_sq_flat = A_sq.ravel()
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (1j * g * A_sq_flat[i] * sat - alpha * sat) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_nl_rhs_v(
    A_prop: np.ndarray,
    A: np.ndarray,
    A_sq: np.ndarray,
    V: np.ndarray,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A : np.ndarray
        Original field
    A_sq : np.ndarray
        Field modulus squared
    V : np.ndarray
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A.ravel()
    A_sq_flat = A_sq.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (
            1j * g * A_sq_flat[i] * sat - alpha * sat + 1j * V_flat[i]
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_rk4_nl_rhs(
    A_prop: np.ndarray,
    A: np.ndarray,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A : np.ndarray
        Original field
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A.ravel()
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        A_prop_flat[i] += (1j * g * A_sq_val * sat - alpha * sat) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def square_mod_rk4_nl_rhs_v(
    A_prop: np.ndarray,
    A: np.ndarray,
    V: np.ndarray,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A : np.ndarray
        Original field
    V : np.ndarray
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        A_prop_flat[i] += (
            1j * g * A_sq_val * sat - alpha * sat + 1j * V_flat[i]
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_nl_rhs_c(
    A_prop: np.ndarray,
    A_orig: np.ndarray,
    A_sq_1: np.ndarray,
    A_sq_2: np.ndarray,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> np.ndarray:
    """Accumulate coupled nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A_orig : np.ndarray
        Original field (this component)
    A_sq_1 : np.ndarray
        Modulus squared of first component
    A_sq_2 : np.ndarray
        Modulus squared of second component
    alpha : float
        Losses
    g11 : float
        Intra-component interactions
    g12 : float
        Inter-component interactions
    Isat1 : float
        Saturation parameter of first component
    Isat2 : float
        Saturation parameter of second component
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A_orig.ravel()
    A_sq_1_flat = A_sq_1.ravel()
    A_sq_2_flat = A_sq_2.ravel()
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            1j * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat - alpha * sat
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def rk4_nl_rhs_c_v(
    A_prop: np.ndarray,
    A_orig: np.ndarray,
    A_sq_1: np.ndarray,
    A_sq_2: np.ndarray,
    V: np.ndarray,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> np.ndarray:
    """Accumulate coupled nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : np.ndarray
        Linearly propagated field (modified in-place)
    A_orig : np.ndarray
        Original field (this component)
    A_sq_1 : np.ndarray
        Modulus squared of first component
    A_sq_2 : np.ndarray
        Modulus squared of second component
    V : np.ndarray
        Potential (pre-scaled)
    alpha : float
        Losses
    g11 : float
        Intra-component interactions
    g12 : float
        Inter-component interactions
    Isat1 : float
        Saturation parameter of first component
    Isat2 : float
        Saturation parameter of second component
    """
    A_prop_flat = A_prop.ravel()
    A_flat = A_orig.ravel()
    A_sq_1_flat = A_sq_1.ravel()
    A_sq_2_flat = A_sq_2.ravel()
    V_flat = V.ravel()
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            1j * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat
            - alpha * sat
            + 1j * V_flat[i]
        ) * A_flat[i]
    return A_prop
