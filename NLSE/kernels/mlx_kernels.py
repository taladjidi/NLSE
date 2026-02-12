"""MLX kernel implementations for Apple Silicon GPU acceleration.

All kernel functions modify arrays in-place and return None,
matching the convention of CPU/CuPy/CL backends.
"""

import mlx.core as mx


def nl_prop(
    A: mx.array,
    A_sq: mx.array,
    dz: float,
    alpha: float,
    V: mx.array,
    g: float,
    Isat: float,
) -> None:
    """Apply real space terms with potential.

    Parameters
    ----------
    A : mx.array
        The field to propagate
    A_sq : mx.array
        The field modulus squared
    dz : float
        Propagation step in m
    alpha : float
        Losses
    V : mx.array
        Potential
    g : float
        Interactions
    Isat : float
        Saturation
    """
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat + 1j * V
    A *= mx.exp(dz * arg)


def nl_prop_without_V(
    A: mx.array,
    A_sq: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Apply real space terms without potential.

    Parameters
    ----------
    A : mx.array
        The field to propagate
    A_sq : mx.array
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
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat
    A *= mx.exp(dz * arg)


def nl_prop_c(
    A1: mx.array,
    A_sq_1: mx.array,
    A_sq_2: mx.array,
    dz: float,
    alpha: float,
    V: mx.array,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> None:
    """Apply coupled real space terms with potential.

    Parameters
    ----------
    A1 : mx.array
        The field to propagate (1st component)
    A_sq_1 : mx.array
        The field modulus squared (1st component)
    A_sq_2 : mx.array
        The field modulus squared (2nd component)
    dz : float
        Propagation step in m
    alpha : float
        Losses
    V : mx.array
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
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat) - alpha * sat + 1j * V
    A1 *= mx.exp(dz * arg)


def nl_prop_without_V_c(
    A1: mx.array,
    A_sq_1: mx.array,
    A_sq_2: mx.array,
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
    A1 : mx.array
        The field to propagate (1st component)
    A_sq_1 : mx.array
        The field modulus squared (1st component)
    A_sq_2 : mx.array
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
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat) - alpha * sat
    A1 *= mx.exp(dz * arg)


def square_mod(A: mx.array, A_sq: mx.array) -> None:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : mx.array
        The field
    A_sq : mx.array
        The modulus squared of the field
    """
    A_sq[:] = (A * mx.conj(A)).real


def square_mod_nl_prop(
    A: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Compute |A|^2 inline and apply nonlinear propagation without potential.

    Parameters
    ----------
    A : mx.array
        The field to propagate
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat
    A *= mx.exp(dz * arg)


def square_mod_nl_prop_v(
    A: mx.array,
    V: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Compute |A|^2 inline and apply nonlinear propagation with potential.

    Parameters
    ----------
    A : mx.array
        The field to propagate
    V : mx.array
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
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat + 1j * V
    A *= mx.exp(dz * arg)


def apply_propagator(A: mx.array, propagator: mx.array) -> None:
    """Apply the linear propagator in Fourier space.

    Parameters
    ----------
    A : mx.array
        The field in Fourier space.
    propagator : mx.array
        The propagator matrix.
    """
    A *= propagator


def rabi_coupling(A1: mx.array, A2: mx.array, dz: float, omega: float) -> None:
    """Apply Rabi coupling term.

    Implement the Rabi hopping term, exchanging density between components.

    Parameters
    ----------
    A1 : mx.array
        First field / component
    A2 : mx.array
        Second field / component
    dz : float
        Solver step
    omega : float
        Rabi coupling strength
    """
    cos_val = mx.cos(omega * dz)
    sin_val = mx.sin(omega * dz)
    new_A1 = cos_val * A1 - 1j * sin_val * A2
    new_A2 = cos_val * A2 - 1j * sin_val * A1
    A1[:] = new_A1
    A2[:] = new_A2
