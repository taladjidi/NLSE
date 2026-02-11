import cupy as cp


@cp.fuse(kernel_name="nl_prop")
def nl_prop(
    A: cp.ndarray,
    A_sq: cp.ndarray,
    dz: float,
    alpha: float,
    V: cp.ndarray,
    g: float,
    Isat: float,
) -> None:
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
    """
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


@cp.fuse(kernel_name="nl_prop_without_V")
def nl_prop_without_V(
    A: cp.ndarray,
    A_sq: cp.ndarray,
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
    # saturation
    sat = 1 / (1 + A_sq / Isat)
    # Interactions
    arg = 1j * g * A_sq * sat
    # Losses
    arg += -alpha * sat
    arg *= dz
    cp.exp(arg, out=arg)
    A *= arg


@cp.fuse(kernel_name="nl_prop_c")
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


@cp.fuse(kernel_name="nl_prop_without_V_c")
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
    # Saturation parameter
    sat = 1 / (1 + A_sq_1 * 1 / Isat1 + A_sq_2 * 1 / Isat2)
    # Interactions
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    # Losses
    arg += -alpha * sat
    arg *= dz
    cp.exp(arg, out=arg)
    A1 *= arg


@cp.fuse(kernel_name="rabi_coupling")
def rabi_coupling(A1: cp.array, A2: cp.array, dz: float, omega: float) -> None:
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
    """
    A1_old = A1.copy()
    A1[:] = cp.cos(omega * dz) * A1 - 1j * cp.sin(omega * dz) * A2
    A2[:] = cp.cos(omega * dz) * A2 - 1j * cp.sin(omega * dz) * A1_old


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


@cp.fuse(kernel_name="square_mod_cp")
def square_mod(A: cp.ndarray, A_sq: cp.ndarray) -> None:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : cp.ndarray
        The field
    A_sq : cp.ndarray
        The modulus squared of the field

    Returns
    -------
    None
    """
    A_sq[:] = (A * A.conj()).real


# Fused kernels (call separate operations for CUPY backend)
@cp.fuse(kernel_name="square_mod_nl_prop")
def square_mod_nl_prop(A, dz, alpha, g, Isat):
    """Fused square_mod + nl_prop_without_V."""
    A_sq = (A * A.conj()).real
    nl_prop_without_V(A, A_sq, dz, alpha, g, Isat)


@cp.fuse(kernel_name="square_mod_nl_prop_v")
def square_mod_nl_prop_v(A, V, dz, alpha, g, Isat):
    """Fused square_mod + nl_prop."""
    A_sq = (A * A.conj()).real
    nl_prop(A, A_sq, dz, alpha, V, g, Isat)
