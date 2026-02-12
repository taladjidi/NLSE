import cupy as cp


@cp.fuse(kernel_name="nl_prop")
def _nl_prop_fused(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g: float,
    Isat: float,
) -> None:
    """Fused implementation of nl_prop."""
    # saturation
    sat = 1 / (1 + A_sq / Isat)
    # Interactions
    arg = 1j * g * A_sq * sat
    # Losses
    arg += -alpha * sat
    # Potential
    arg += 1j * V
    arg *= dz
    cp.exp(arg, out=arg)
    A *= arg


def nl_prop(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g: float,
    Isat: float,
) -> cp.ndarray:
    """Apply real space terms with potential.

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
    V : cp.ndarray
        Potential
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _nl_prop_fused(A, A_sq, dz, alpha, V, g, Isat)
    return A


@cp.fuse(kernel_name="nl_prop_without_V")
def _nl_prop_without_V_fused(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Fused implementation of nl_prop_without_V."""
    # saturation
    sat = 1 / (1 + A_sq / Isat)
    # Interactions
    arg = 1j * g * A_sq * sat
    # Losses
    arg += -alpha * sat
    arg *= dz
    cp.exp(arg, out=arg)
    A *= arg


def nl_prop_without_V(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> cp.ndarray:
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

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _nl_prop_without_V_fused(A, A_sq, dz, alpha, g, Isat)
    return A


@cp.fuse(kernel_name="nl_prop_c")
def _nl_prop_c_fused(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """Fused implementation of nl_prop_c."""
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 * 1 / Isat1 + A_sq_2 * 1 / Isat2)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    # Potential
    arg += 1j * V
    arg *= dz
    cp.exp(arg, out=arg)
    A1 *= arg


def nl_prop_c(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> cp.ndarray:
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

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _nl_prop_c_fused(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2)
    return A1


@cp.fuse(kernel_name="nl_prop_without_V_c")
def _nl_prop_without_V_c_fused(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """Fused implementation of nl_prop_without_V_c."""
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 * 1 / Isat1 + A_sq_2 * 1 / Isat2)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    arg *= dz
    cp.exp(arg, out=arg)
    A1 *= arg


def nl_prop_without_V_c(
    A1: cp.ndarray,
    A_sq_1: cp.ndarray,
    A_sq_2: cp.ndarray,
    dz: float,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> cp.ndarray:
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

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _nl_prop_without_V_c_fused(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2)
    return A1


@cp.fuse(kernel_name="rabi_coupling")
def _rabi_coupling_fused(A1: cp.array, A2: cp.array, dz: float, omega: float) -> None:
    """Fused implementation of rabi_coupling."""
    A1_old = A1.copy()
    A1[:] = cp.cos(omega * dz) * A1 - 1j * cp.sin(omega * dz) * A2
    A2[:] = cp.cos(omega * dz) * A2 - 1j * cp.sin(omega * dz) * A1_old


def rabi_coupling(A1: cp.array, A2: cp.array, dz: float, omega: float) -> tuple:
    """Apply Rabi coupling term.

    Implement the Rabi hopping term, exchanging density between components.

    Parameters
    ----------
    A1 : cp.ndarray
        First field / component
    A2 : cp.ndarray
        Second field / component
    dz : float
        Solver step
    omega : float
        Rabi coupling strength

    Returns
    -------
    tuple
        The modified fields (A1, A2).
    """
    _rabi_coupling_fused(A1, A2, dz, omega)
    return A1, A2


@cp.fuse(kernel_name="vortex_cp")
def vortex_cp(
    im: cp.ndarray, i: int, j: int, ii: cp.ndarray, jj: cp.ndarray, ll: int
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
    im += cp.angle(((ii - i) + 1j * (jj - j)) ** ll)


def apply_propagator(A: cp.ndarray, propagator: cp.ndarray) -> cp.ndarray:
    """Apply the linear propagator in Fourier space.

    Parameters
    ----------
    A : cp.ndarray
        The field in Fourier space.
    propagator : cp.ndarray
        The propagator matrix.

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    A *= propagator
    return A


@cp.fuse(kernel_name="square_mod_cp")
def _square_mod_fused(A: cp.ndarray, A_sq: cp.ndarray) -> None:
    """Fused implementation of square_mod."""
    A_sq[:] = (A * A.conj()).real


def square_mod(A: cp.ndarray, A_sq: cp.ndarray) -> cp.ndarray:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : cp.ndarray
        The field
    A_sq : cp.ndarray
        The modulus squared of the field

    Returns
    -------
    cp.ndarray
        The modulus squared of the field.
    """
    _square_mod_fused(A, A_sq)
    return A_sq


# Fused kernels (call separate operations for CUPY backend)
@cp.fuse(kernel_name="square_mod_nl_prop")
def _square_mod_nl_prop_fused(A, dz, alpha, g, Isat):
    """Fused implementation of square_mod_nl_prop."""
    A_sq = (A * A.conj()).real
    _nl_prop_without_V_fused(A, A_sq, dz, alpha, g, Isat)


def square_mod_nl_prop(A, dz, alpha, g, Isat):
    """Fuse square_mod + nl_prop_without_V.

    Parameters
    ----------
    A : cp.ndarray
        The field to propagate.
    dz : float
        Propagation step in m.
    alpha : float
        Losses.
    g : float
        Interactions.
    Isat : float
        Saturation.

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _square_mod_nl_prop_fused(A, dz, alpha, g, Isat)
    return A


@cp.fuse(kernel_name="square_mod_nl_prop_v")
def _square_mod_nl_prop_v_fused(A, V, dz, alpha, g, Isat):
    """Fused implementation of square_mod_nl_prop_v."""
    A_sq = (A * A.conj()).real
    _nl_prop_fused(A, A_sq, dz, alpha, V, g, Isat)


def square_mod_nl_prop_v(A, V, dz, alpha, g, Isat):
    """Fuse square_mod + nl_prop.

    Parameters
    ----------
    A : cp.ndarray
        The field to propagate.
    V : cp.ndarray
        Potential.
    dz : float
        Propagation step in m.
    alpha : float
        Losses.
    g : float
        Interactions.
    Isat : float
        Saturation.

    Returns
    -------
    cp.ndarray
        The modified field.
    """
    _square_mod_nl_prop_v_fused(A, V, dz, alpha, g, Isat)
    return A
