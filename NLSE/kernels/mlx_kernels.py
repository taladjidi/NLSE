"""MLX kernel implementations using mx.compile for fused Metal kernels.

All kernel functions return the modified array (new allocation via donation).
Scalar arguments are converted to 0-dim mx.array so that mx.compile traces
them as proper inputs whose values can change between calls.
"""

import mlx.core as mx


def _to_mx(val):
    """Convert a Python scalar to a 0-dim mx.array for compiled functions."""
    if isinstance(val, mx.array):
        return val
    return mx.array(float(val), dtype=mx.float32)


# ── Pure implementations (no side effects, return new arrays) ───────────────


def _nl_prop_pure(A, A_sq, dz, alpha, V, g, Isat):
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat + 1j * V
    return A * mx.exp(dz * arg)


def _nl_prop_without_V_pure(A, A_sq, dz, alpha, g, Isat):
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat
    return A * mx.exp(dz * arg)


def _nl_prop_c_pure(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2):
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat) - alpha * sat + 1j * V
    return A1 * mx.exp(dz * arg)


def _nl_prop_without_V_c_pure(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2):
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat) - alpha * sat
    return A1 * mx.exp(dz * arg)


def _square_mod_pure(A, _A_sq):
    return (A * mx.conj(A)).real


def _square_mod_nl_prop_pure(A, dz, alpha, g, Isat):
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat
    return A * mx.exp(dz * arg)


def _square_mod_nl_prop_v_pure(A, V, dz, alpha, g, Isat):
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat - alpha * sat + 1j * V
    return A * mx.exp(dz * arg)


def _apply_propagator_pure(A, propagator):
    return A * propagator


def _rabi_coupling_pure(A1, A2, cos_val, sin_val):
    new_A1 = cos_val * A1 - 1j * sin_val * A2
    new_A2 = cos_val * A2 - 1j * sin_val * A1
    return new_A1, new_A2


# ── Compiled versions ───────────────────────────────────────────────────────

_c_nl_prop = mx.compile(_nl_prop_pure)
_c_nl_prop_without_V = mx.compile(_nl_prop_without_V_pure)
_c_nl_prop_c = mx.compile(_nl_prop_c_pure)
_c_nl_prop_without_V_c = mx.compile(_nl_prop_without_V_c_pure)
_c_square_mod = mx.compile(_square_mod_pure)
_c_square_mod_nl_prop = mx.compile(_square_mod_nl_prop_pure)
_c_square_mod_nl_prop_v = mx.compile(_square_mod_nl_prop_v_pure)
_c_apply_propagator = mx.compile(_apply_propagator_pure)
_c_rabi_coupling = mx.compile(_rabi_coupling_pure)


# ── Public API (same signatures as other backends, but return new arrays) ───


def nl_prop(
    A: mx.array,
    A_sq: mx.array,
    dz: float,
    alpha: float,
    V: mx.array,
    g: float,
    Isat: float,
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_nl_prop(A, A_sq, _to_mx(dz), _to_mx(alpha), V, _to_mx(g), _to_mx(Isat))


def nl_prop_without_V(
    A: mx.array,
    A_sq: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_nl_prop_without_V(
        A, A_sq, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat)
    )


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
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_nl_prop_c(
        A1,
        A_sq_1,
        A_sq_2,
        _to_mx(dz),
        _to_mx(alpha),
        V,
        _to_mx(g11),
        _to_mx(g12),
        _to_mx(Isat1),
        _to_mx(Isat2),
    )


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
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_nl_prop_without_V_c(
        A1,
        A_sq_1,
        A_sq_2,
        _to_mx(dz),
        _to_mx(alpha),
        _to_mx(g11),
        _to_mx(g12),
        _to_mx(Isat1),
        _to_mx(Isat2),
    )


def square_mod(A: mx.array, A_sq: mx.array) -> mx.array:
    """Compute the square modulus of the field.

    Parameters
    ----------
    A : mx.array
        The field
    A_sq : mx.array
        The modulus squared of the field (unused, kept for API compatibility)

    Returns
    -------
    mx.array
        The modulus squared.
    """
    return _c_square_mod(A, A_sq)


def square_mod_nl_prop(
    A: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_square_mod_nl_prop(A, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))


def square_mod_nl_prop_v(
    A: mx.array,
    V: mx.array,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
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

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_square_mod_nl_prop_v(
        A, V, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat)
    )


def apply_propagator(A: mx.array, propagator: mx.array) -> mx.array:
    """Apply the linear propagator in Fourier space.

    Parameters
    ----------
    A : mx.array
        The field in Fourier space.
    propagator : mx.array
        The propagator matrix.

    Returns
    -------
    mx.array
        The propagated field.
    """
    return _c_apply_propagator(A, propagator)


def rabi_coupling(
    A1: mx.array, A2: mx.array, dz: float, omega: float
) -> tuple[mx.array, mx.array]:
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

    Returns
    -------
    tuple[mx.array, mx.array]
        The coupled fields (A1, A2).
    """
    cos_val = _to_mx(float(mx.cos(_to_mx(omega * dz))))
    sin_val = _to_mx(float(mx.sin(_to_mx(omega * dz))))
    return _c_rabi_coupling(A1, A2, cos_val, sin_val)
