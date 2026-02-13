"""MLX kernel implementations using mx.compile for fused Metal kernels.

All kernel functions return the modified array (new allocation via donation).
Scalar arguments are converted to 0-dim mx.array so that mx.compile traces
them as proper inputs whose values can change between calls.
"""

import mlx.core as mx

_SCALAR_CACHE: dict[float, mx.array] = {}


def _to_mx(val):
    """Convert a Python scalar to a 0-dim mx.array for compiled functions."""
    if isinstance(val, mx.array):
        return val
    fval = float(val)
    cached = _SCALAR_CACHE.get(fval)
    if cached is not None:
        return cached
    arr = mx.array(fval, dtype=mx.float32)
    _SCALAR_CACHE[fval] = arr
    return arr


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


# ── Fused linear step (FFT + propagator + IFFT) ──────────────────────────────


def _make_linear_step(axes):
    def _linear_step_pure(A, propagator):
        A = mx.fft.fftn(A, axes=axes)
        A = A * propagator
        A = mx.fft.ifftn(A, axes=axes)
        return A

    return mx.compile(_linear_step_pure)


_LINEAR_STEP_CACHE: dict[tuple, object] = {}


def linear_step(A: mx.array, propagator: mx.array, axes: tuple) -> mx.array:
    """Apply fused linear propagation step (FFT + propagator + IFFT).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The propagator matrix.
    axes : tuple
        FFT axes.

    Returns
    -------
    mx.array
        The propagated field.
    """
    if axes not in _LINEAR_STEP_CACHE:
        _LINEAR_STEP_CACHE[axes] = _make_linear_step(axes)
    return _LINEAR_STEP_CACHE[axes](A, propagator)


# ── Fused split step (nl_length == 0 only) ───────────────────────────────────


def _make_split_step(precision, has_V, axes):
    if precision == "single" and not has_V:

        def _pure(A, propagator, dz, alpha, g, Isat):
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            return A * mx.exp(dz * (1j * g * A_sq * sat - alpha * sat))

    elif precision == "single" and has_V:

        def _pure(A, propagator, V, dz, alpha, g, k_half, Isat):
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            return A * mx.exp(
                dz * (1j * g * A_sq * sat - alpha * sat + 1j * k_half * V)
            )

    elif precision == "double" and not has_V:

        def _pure(A, propagator, dz_half, alpha, g, Isat):
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            A = A * mx.exp(dz_half * (1j * g * A_sq * sat - alpha * sat))
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            return A * mx.exp(dz_half * (1j * g * A_sq * sat - alpha * sat))

    else:  # double, has_V

        def _pure(A, propagator, V, dz_half, alpha, g, k_half, Isat):
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            A = A * mx.exp(
                dz_half * (1j * g * A_sq * sat - alpha * sat + 1j * k_half * V)
            )
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            A_sq = (A * mx.conj(A)).real
            sat = 1 / (1 + A_sq / Isat)
            return A * mx.exp(
                dz_half * (1j * g * A_sq * sat - alpha * sat + 1j * k_half * V)
            )

    return mx.compile(_pure)


_SPLIT_STEP_CACHE: dict[tuple, object] = {}


def split_step(
    A: mx.array,
    propagator: mx.array,
    V: mx.array | None,
    dz: float,
    alpha: float,
    g: float,
    k_half: float,
    Isat: float,
    precision: str,
    axes: tuple,
) -> mx.array:
    """Execute fused split step for MLX (nl_length == 0 only).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The propagator matrix.
    V : mx.array or None
        Potential field (unscaled).
    dz : float
        Propagation step (full for single, half for double precision).
    alpha : float
        Loss coefficient (half of total).
    g : float
        Nonlinear interaction strength.
    k_half : float
        Half the wavenumber (k/2) for V scaling.
    Isat : float
        Saturation intensity (converted units).
    precision : str
        "single" or "double" split step precision.
    axes : tuple
        FFT axes.

    Returns
    -------
    mx.array
        The propagated field.
    """
    key = (precision, V is not None, axes)
    if key not in _SPLIT_STEP_CACHE:
        _SPLIT_STEP_CACHE[key] = _make_split_step(precision, V is not None, axes)
    fn = _SPLIT_STEP_CACHE[key]
    if V is not None:
        return fn(
            A,
            propagator,
            V,
            _to_mx(dz),
            _to_mx(alpha),
            _to_mx(g),
            _to_mx(k_half),
            _to_mx(Isat),
        )
    return fn(A, propagator, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))


# ── RK4 nonlinear RHS kernels ────────────────────────────────────────────────


def _rk4_nl_rhs_pure(A_prop, A, A_sq, alpha, g, Isat):
    sat = 1 / (1 + A_sq / Isat)
    return A_prop + (1j * g * A_sq * sat - alpha * sat) * A


def _rk4_nl_rhs_v_pure(A_prop, A, A_sq, V, alpha, g, Isat):
    sat = 1 / (1 + A_sq / Isat)
    return A_prop + (1j * g * A_sq * sat - alpha * sat + 1j * V) * A


def _square_mod_rk4_nl_rhs_pure(A_prop, A, alpha, g, Isat):
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    return A_prop + (1j * g * A_sq * sat - alpha * sat) * A


def _square_mod_rk4_nl_rhs_v_pure(A_prop, A, V, alpha, g, Isat):
    A_sq = (A * mx.conj(A)).real
    sat = 1 / (1 + A_sq / Isat)
    return A_prop + (1j * g * A_sq * sat - alpha * sat + 1j * V) * A


def _rk4_nl_rhs_c_pure(A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2):
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    return A_prop + 1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat - alpha * sat * A_orig


def _rk4_nl_rhs_c_v_pure(
    A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
):
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    return (
        A_prop
        + 1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat
        + (-alpha * sat + 1j * V) * A_orig
    )


_c_rk4_nl_rhs = mx.compile(_rk4_nl_rhs_pure)
_c_rk4_nl_rhs_v = mx.compile(_rk4_nl_rhs_v_pure)
_c_square_mod_rk4_nl_rhs = mx.compile(_square_mod_rk4_nl_rhs_pure)
_c_square_mod_rk4_nl_rhs_v = mx.compile(_square_mod_rk4_nl_rhs_v_pure)
_c_rk4_nl_rhs_c = mx.compile(_rk4_nl_rhs_c_pure)
_c_rk4_nl_rhs_c_v = mx.compile(_rk4_nl_rhs_c_v_pure)


def rk4_nl_rhs(
    A_prop: mx.array,
    A: mx.array,
    A_sq: mx.array,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
    """Accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A : mx.array
        Original field
    A_sq : mx.array
        Field modulus squared
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_rk4_nl_rhs(A_prop, A, A_sq, _to_mx(alpha), _to_mx(g), _to_mx(Isat))


def rk4_nl_rhs_v(
    A_prop: mx.array,
    A: mx.array,
    A_sq: mx.array,
    V: mx.array,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
    """Accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A : mx.array
        Original field
    A_sq : mx.array
        Field modulus squared
    V : mx.array
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_rk4_nl_rhs_v(A_prop, A, A_sq, V, _to_mx(alpha), _to_mx(g), _to_mx(Isat))


def square_mod_rk4_nl_rhs(
    A_prop: mx.array,
    A: mx.array,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A : mx.array
        Original field
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_square_mod_rk4_nl_rhs(A_prop, A, _to_mx(alpha), _to_mx(g), _to_mx(Isat))


def square_mod_rk4_nl_rhs_v(
    A_prop: mx.array,
    A: mx.array,
    V: mx.array,
    alpha: float,
    g: float,
    Isat: float,
) -> mx.array:
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A : mx.array
        Original field
    V : mx.array
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_square_mod_rk4_nl_rhs_v(
        A_prop, A, V, _to_mx(alpha), _to_mx(g), _to_mx(Isat)
    )


def rk4_nl_rhs_c(
    A_prop: mx.array,
    A_orig: mx.array,
    A_sq_1: mx.array,
    A_sq_2: mx.array,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> mx.array:
    """Accumulate coupled nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A_orig : mx.array
        Original field (this component)
    A_sq_1 : mx.array
        Modulus squared of first component
    A_sq_2 : mx.array
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

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_rk4_nl_rhs_c(
        A_prop,
        A_orig,
        A_sq_1,
        A_sq_2,
        _to_mx(alpha),
        _to_mx(g11),
        _to_mx(g12),
        _to_mx(Isat1),
        _to_mx(Isat2),
    )


def rk4_nl_rhs_c_v(
    A_prop: mx.array,
    A_orig: mx.array,
    A_sq_1: mx.array,
    A_sq_2: mx.array,
    V: mx.array,
    alpha: float,
    g11: float,
    g12: float,
    Isat1: float,
    Isat2: float,
) -> mx.array:
    """Accumulate coupled nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : mx.array
        Linearly propagated field
    A_orig : mx.array
        Original field (this component)
    A_sq_1 : mx.array
        Modulus squared of first component
    A_sq_2 : mx.array
        Modulus squared of second component
    V : mx.array
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

    Returns
    -------
    mx.array
        The updated field.
    """
    return _c_rk4_nl_rhs_c_v(
        A_prop,
        A_orig,
        A_sq_1,
        A_sq_2,
        V,
        _to_mx(alpha),
        _to_mx(g11),
        _to_mx(g12),
        _to_mx(Isat1),
        _to_mx(Isat2),
    )


# ── Fused RK4 split step (nl_length == 0, uncoupled only) ────────────────────


def _make_split_step_rk4(has_V, axes):
    if not has_V:

        def _pure(A, propagator, dz, alpha, g, Isat):
            def rhs(A_in):
                A_prop = mx.fft.fftn(A_in, axes=axes)
                A_prop = A_prop * propagator
                A_prop = mx.fft.ifftn(A_prop, axes=axes)
                A_sq = (A_in * mx.conj(A_in)).real
                sat = 1 / (1 + A_sq / Isat)
                return A_prop + (1j * g * A_sq * sat - alpha * sat) * A_in

            k1 = rhs(A)
            k2 = rhs(A + k1 * dz / 3)
            k3 = rhs(A + (-k1 / 3 + k2) * dz)
            k4 = rhs(A + (k1 - k2 + k3) * dz)
            return A + dz / 8 * (k1 + 3 * k2 + 3 * k3 + k4)

    else:

        def _pure(A, propagator, V, dz, alpha, g, k_half, Isat):
            def rhs(A_in):
                A_prop = mx.fft.fftn(A_in, axes=axes)
                A_prop = A_prop * propagator
                A_prop = mx.fft.ifftn(A_prop, axes=axes)
                A_sq = (A_in * mx.conj(A_in)).real
                sat = 1 / (1 + A_sq / Isat)
                return (
                    A_prop
                    + (1j * g * A_sq * sat - alpha * sat + 1j * k_half * V) * A_in
                )

            k1 = rhs(A)
            k2 = rhs(A + k1 * dz / 3)
            k3 = rhs(A + (-k1 / 3 + k2) * dz)
            k4 = rhs(A + (k1 - k2 + k3) * dz)
            return A + dz / 8 * (k1 + 3 * k2 + 3 * k3 + k4)

    return mx.compile(_pure)


_SPLIT_STEP_RK4_CACHE: dict[tuple, object] = {}


def split_step_rk4(
    A: mx.array,
    propagator: mx.array,
    V: mx.array | None,
    dz: float,
    alpha: float,
    g: float,
    k_half: float,
    Isat: float,
    axes: tuple,
) -> mx.array:
    """Execute fused RK4 split step for MLX (nl_length == 0 only).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The RK4 propagator (dispersion operator, not exponentiated).
    V : mx.array or None
        Potential field (unscaled).
    dz : float
        Propagation step.
    alpha : float
        Loss coefficient (half of total).
    g : float
        Nonlinear interaction strength.
    k_half : float
        Half the wavenumber (k/2) for V scaling.
    Isat : float
        Saturation intensity (converted units).
    axes : tuple
        FFT axes.

    Returns
    -------
    mx.array
        The propagated field.
    """
    key = (V is not None, axes)
    if key not in _SPLIT_STEP_RK4_CACHE:
        _SPLIT_STEP_RK4_CACHE[key] = _make_split_step_rk4(V is not None, axes)
    fn = _SPLIT_STEP_RK4_CACHE[key]
    if V is not None:
        return fn(
            A,
            propagator,
            V,
            _to_mx(dz),
            _to_mx(alpha),
            _to_mx(g),
            _to_mx(k_half),
            _to_mx(Isat),
        )
    return fn(A, propagator, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))
