"""MLX kernel implementations using mx.compile for fused Metal kernels.

All kernel functions return the modified array (new allocation via donation).
Scalar arguments are converted to 0-dim mx.array so that mx.compile traces
them as proper inputs whose values can change between calls.
"""

import math

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


# Largest |u| = |2*alpha*dz| the solved step below is used at. See
# _LOSS_SOLVED_LIMIT in kernels/cpu.py, the same number for the same reason.
_LOSS_SOLVED_LIMIT = 0.1


# ── Pure implementations (no side effects, return new arrays) ───────────────


def _nl_factor(A_sq, dz, alpha, g, Isat, V=None):
    """Return what a real-space step multiplies the field by.

    Every scalar real-space step in this file goes through here, because the
    lossy part of it is not the exponential it looks like. ``_loss_factor`` in
    kernels/cpu.py carries the derivation: freezing ``|A|^2`` across a step is
    exact only while the step preserves it, which a pure rotation does and loss
    does not, and the frozen form costs Lie, Strang and Yoshida alike their
    order. With ``u = 2*alpha*dz`` the step applies ``g*|A|^2*P*dz`` for the
    phase and ``sqrt(1 - P*u)`` for the amplitude.

    Branch-free *per element*, since ``alpha`` may carry a batch axis and a
    branch on a device value would mean a synchronization. At ``u = 0`` the
    iteration returns ``sat`` and the amplitude factor is exactly 1, so a
    lossless step is unchanged -- but it is not free, because both arms of
    every ``mx.where`` are evaluated. ``_nl_factor_lossless`` is what a
    lossless run actually takes; see ``_is_lossless``.

    Parameters
    ----------
    A_sq : mx.array
        ``|A|^2`` entering the step.
    dz : mx.array
        Step length.
    alpha : mx.array
        Loss coefficient.
    g : mx.array
        Interaction strength.
    Isat : mx.array
        Saturation intensity.
    V : mx.array or None
        Scaled potential, if there is one.

    Returns
    -------
    mx.array
        The factor to multiply the field by.
    """
    sat = 1 / (1 + A_sq / Isat)
    u = 2 * alpha * dz
    P_solved = sat
    for _ in range(3):
        Pu = P_solved * u
        P_solved = sat * (1 - Pu * P_solved * (0.5 + Pu / 3 + Pu * Pu / 4))
    # Above _LOSS_SOLVED_LIMIT the iteration is out of its range and the step
    # is frozen instead, as it was before. Chosen per element, since alpha may
    # carry a batch axis.
    solved = mx.abs(u) <= _LOSS_SOLVED_LIMIT
    P = mx.where(solved, P_solved, sat)
    decay = mx.where(
        solved,
        mx.sqrt(mx.maximum(1 - P_solved * u, 0)),
        mx.exp(-alpha * sat * dz),
    )
    arg = 1j * g * A_sq * P
    if V is not None:
        arg = arg + 1j * V
    return decay * mx.exp(dz * arg)


def _nl_factor_lossless(A_sq, dz, g, Isat, V=None):
    """Return the same factor for ``alpha = 0``, without the loss arithmetic.

    A pure rotation preserves ``|A|^2``, so freezing it is exact and there is
    nothing to solve. This is what ``_nl_factor`` reduces to at ``u = 0``,
    term by term: the iteration returns ``sat`` and the decay is exactly 1.

    Parameters
    ----------
    A_sq : mx.array
        ``|A|^2`` entering the step.
    dz : mx.array
        Step length.
    g : mx.array
        Interaction strength.
    Isat : mx.array
        Saturation intensity.
    V : mx.array or None
        Scaled potential, if there is one.

    Returns
    -------
    mx.array
        The factor to multiply the field by.
    """
    sat = 1 / (1 + A_sq / Isat)
    arg = 1j * g * A_sq * sat
    if V is not None:
        arg = arg + 1j * V
    return mx.exp(dz * arg)


# The solved step as one Metal kernel, which is where its cost went. Written
# out of registers rather than out of MLX ops: mx.compile does not fuse the
# iteration, so each of its ~9 elementwise ops materialized a full array and
# went to memory. Measured at 512x512, chained, against the frozen step this
# replaces: 3.9x for the graph form, 1.4x for this one; over a whole split
# step, 1.62x against 1.18x. The `sincos` is one instruction here and two
# passes over memory there.
_LOSSY_BODY = """
    uint i = thread_position_in_grid.x;
    float ar = A[i].real;
    float ai = A[i].imag;
    float A_sq = ar * ar + ai * ai;
    float sat = 1.0f / (1.0f + A_sq / Isat);
    float u = 2.0f * alpha * dz;

    // The fixed point of _loss_factor in kernels/cpu.py, three passes, in
    // Horner form. The caller has already checked |u| is inside its range.
    float P = sat;
    for (int k = 0; k < 3; ++k) {
        float Pu = P * u;
        P = sat * (1.0f - Pu * P * (0.5f + Pu * (1.0f / 3.0f + Pu * 0.25f)));
    }
    float decay = sqrt(fmax(1.0f - P * u, 0.0f));
    float phase = dz * (g * A_sq * P%s);
    float c;
    float s = sincos(phase, c);
    out[i].real = decay * (ar * c - ai * s);
    out[i].imag = decay * (ar * s + ai * c);
"""

_LOSSY_KERNELS: dict[bool, object] = {}


def _lossy_kernel(has_V):
    """Return the Metal kernel for the solved step, with or without ``V``.

    Parameters
    ----------
    has_V : bool
        Whether a real potential is added to the phase.

    Returns
    -------
    object
        The compiled ``mx.fast.metal_kernel``.
    """
    if has_V not in _LOSSY_KERNELS:
        _LOSSY_KERNELS[has_V] = mx.fast.metal_kernel(
            name=f"nlse_lossy_{'V' if has_V else 'noV'}",
            input_names=["A", "dz", "alpha", "g", "Isat"] + (["V"] if has_V else []),
            output_names=["out"],
            source=_LOSSY_BODY % (" + V[i]" if has_V else ""),
        )
    return _LOSSY_KERNELS[has_V]


def _apply_lossy(A, dz, alpha, g, Isat, V=None):
    """Apply the solved real-space step to ``A`` in one kernel.

    Returns ``A`` times what ``_nl_factor`` returns, computed without
    materializing any of the intermediates.

    Parameters
    ----------
    A : mx.array
        The field, contiguous and complex64.
    dz : mx.array
        Step length.
    alpha : mx.array
        Loss coefficient.
    g : mx.array
        Interaction strength.
    Isat : mx.array
        Saturation intensity.
    V : mx.array or None
        Real scaled potential, if there is one.

    Returns
    -------
    mx.array
        The propagated field.
    """
    inputs = [A, dz, alpha, g, Isat] + ([V] if V is not None else [])
    return _lossy_kernel(V is not None)(
        inputs=inputs,
        grid=(A.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[A.shape],
        output_dtypes=[A.dtype],
    )[0]


def _host_scalar(value):
    """Return ``value`` as a float, or None if it is not a host number.

    A device scalar cannot be read without synchronizing, and a batched
    parameter is an array of its own; either way the decision this feeds
    cannot be made on the host.

    Parameters
    ----------
    value : Any
        The scalar as the caller passed it.

    Returns
    -------
    float or None
        The value, or None if it has to stay on the device.
    """
    if isinstance(value, mx.array):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loss_mode(alpha, dz, g, Isat, A=None, V=None):
    """Say which of the three real-space graphs this call should take.

    Only ``"solved"`` is restrictive: it is a Metal kernel, so every scalar
    has to be a real scalar and every array has to be indexable by one flat
    id. Everything it turns down falls back to the graph, which is correct
    for all of it -- just slower.

    Parameters
    ----------
    alpha : float or mx.array
        Loss coefficient.
    dz : float or mx.array
        Step length, which sets ``u`` with it.
    g : float or mx.array
        Interaction strength; an array where ``n2`` carries a batch axis.
    Isat : float or mx.array
        Saturation intensity, batched on the same terms.
    A : mx.array or None
        The field, needed only to compare shapes with ``V``.
    V : mx.array or None
        Scaled potential, if there is one.

    Returns
    -------
    str
        ``"lossless"``, ``"solved"`` or ``"general"``.
    """
    a, d = _host_scalar(alpha), _host_scalar(dz)
    if a is None or d is None:
        return "general"
    if a == 0.0:
        return "lossless"
    # A batched parameter reaches a kernel as a pointer, not a number, and
    # the Metal source multiplies by it. The graph broadcasts it instead.
    if _host_scalar(g) is None or _host_scalar(Isat) is None:
        return "general"
    if V is not None:
        # The kernel writes a real phase, so an absorbing potential -- whose
        # imaginary part is a second decay channel -- goes back to the graph.
        if mx.issubdtype(V.dtype, mx.complexfloating):
            return "general"
        # One flat id indexes both, so a potential shared across a batch
        # would be read past its end. That is the bug the broadcasting tests
        # exist for; the graph broadcasts it correctly.
        if A is not None and V.shape != A.shape:
            return "general"
    # Outside the iteration's range the frozen step applies, and only the
    # graph has it. LOSS_PER_STEP_LIMIT keeps a propagation well inside.
    return "solved" if abs(2 * a * d) <= _LOSS_SOLVED_LIMIT else "general"


def _is_lossless(alpha):
    """Say whether ``alpha`` is a host zero, which needs no device read.

    A device scalar cannot be tested without synchronizing, so an
    ``mx.array`` is never taken as lossless -- it only means the general
    kernel runs, which is correct at any ``alpha``.

    Emitting only the arm a *lossy* host scalar takes was tried too, since
    ``u = 2*alpha*dz`` is known there as well. It saves two selects and an
    exponential and measured nothing (1.54x against 1.51x, on 5-7% noise):
    what the solved step costs MLX is the iteration, not the arm it drops.

    Parameters
    ----------
    alpha : float or mx.array
        Loss coefficient as the caller passed it.

    Returns
    -------
    bool
        True when the lossless kernel may be used.
    """
    if isinstance(alpha, mx.array):
        return False
    try:
        return float(alpha) == 0.0
    except (TypeError, ValueError):  # batched: an array of its own
        return False


def _nl_prop_pure(A, A_sq, dz, alpha, V, g, Isat):
    return A * _nl_factor(A_sq, dz, alpha, g, Isat, V)


def _nl_prop_lossless_pure(A, A_sq, dz, V, g, Isat):
    return A * _nl_factor_lossless(A_sq, dz, g, Isat, V)


def _nl_prop_without_V_pure(A, A_sq, dz, alpha, g, Isat):
    return A * _nl_factor(A_sq, dz, alpha, g, Isat)


def _nl_prop_without_V_lossless_pure(A, A_sq, dz, g, Isat):
    return A * _nl_factor_lossless(A_sq, dz, g, Isat)


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
    return A * _nl_factor(A_sq, dz, alpha, g, Isat)


def _square_mod_nl_prop_lossless_pure(A, dz, g, Isat):
    A_sq = (A * mx.conj(A)).real
    return A * _nl_factor_lossless(A_sq, dz, g, Isat)


def _square_mod_nl_prop_v_pure(A, V, dz, alpha, g, Isat):
    A_sq = (A * mx.conj(A)).real
    return A * _nl_factor(A_sq, dz, alpha, g, Isat, V)


def _square_mod_nl_prop_v_lossless_pure(A, V, dz, g, Isat):
    A_sq = (A * mx.conj(A)).real
    return A * _nl_factor_lossless(A_sq, dz, g, Isat, V)


def _apply_propagator_pure(A, propagator):
    return A * propagator


def _rabi_coupling_pure(A1, A2, cos_val, sin_val):
    new_A1 = cos_val * A1 - 1j * sin_val * A2
    new_A2 = cos_val * A2 - 1j * sin_val * A1
    return new_A1, new_A2


# ── Compiled versions ───────────────────────────────────────────────────────

_c_nl_prop = mx.compile(_nl_prop_pure)
_c_nl_prop_lossless = mx.compile(_nl_prop_lossless_pure)
_c_nl_prop_without_V = mx.compile(_nl_prop_without_V_pure)
_c_nl_prop_without_V_lossless = mx.compile(_nl_prop_without_V_lossless_pure)
_c_nl_prop_c = mx.compile(_nl_prop_c_pure)
_c_nl_prop_without_V_c = mx.compile(_nl_prop_without_V_c_pure)
_c_square_mod = mx.compile(_square_mod_pure)
_c_square_mod_nl_prop = mx.compile(_square_mod_nl_prop_pure)
_c_square_mod_nl_prop_lossless = mx.compile(_square_mod_nl_prop_lossless_pure)
_c_square_mod_nl_prop_v = mx.compile(_square_mod_nl_prop_v_pure)
_c_square_mod_nl_prop_v_lossless = mx.compile(_square_mod_nl_prop_v_lossless_pure)
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
    if _is_lossless(alpha):
        return _c_nl_prop_lossless(A, A_sq, _to_mx(dz), V, _to_mx(g), _to_mx(Isat))
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
    if _is_lossless(alpha):
        return _c_nl_prop_without_V_lossless(
            A, A_sq, _to_mx(dz), _to_mx(g), _to_mx(Isat)
        )
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
    mode = _loss_mode(alpha, dz, g, Isat, A)
    if mode == "lossless":
        return _c_square_mod_nl_prop_lossless(A, _to_mx(dz), _to_mx(g), _to_mx(Isat))
    if mode == "solved":
        return _apply_lossy(A, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))
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
    mode = _loss_mode(alpha, dz, g, Isat, A, V)
    if mode == "lossless":
        return _c_square_mod_nl_prop_v_lossless(
            A, V, _to_mx(dz), _to_mx(g), _to_mx(Isat)
        )
    if mode == "solved":
        return _apply_lossy(A, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat), V)
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
    # On the host: mx.cos here would evaluate on the GPU and drag a single
    # float back, stalling the queue twice per call for 1.6x on the kernel.
    cos_val = _to_mx(math.cos(omega * dz))
    sin_val = _to_mx(math.sin(omega * dz))
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


def linear_step(
    A: mx.array, propagator: mx.array, plan: tuple, unnorm_ifft: bool = False
) -> mx.array:
    """Apply fused linear propagation step (FFT + propagator + IFFT).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The propagator matrix.
    plan : tuple
        FFT axes (MLX has no plan objects).
    unnorm_ifft : bool
        Accepted for signature compatibility with the other fused
        backends and ignored: MLX always normalizes its inverse FFT.
        MLX does not declare supports_unnormalized_ifft, so the solvers
        only ever pass False.

    Returns
    -------
    mx.array
        The propagated field.
    """
    if plan not in _LINEAR_STEP_CACHE:
        _LINEAR_STEP_CACHE[plan] = _make_linear_step(plan)
    return _LINEAR_STEP_CACHE[plan](A, propagator)


# ── Fused split step (nl_length == 0 only) ───────────────────────────────────


def _make_split_step(splitting, has_V, axes, mode="general"):
    # The real-space step is chosen when the graph is compiled, not per
    # element, because alpha and dz are read on the host. A lossless run never
    # traces the iteration at all -- both arms of an mx.where are evaluated,
    # which is what made a lossless step cost 1.45x -- and a solved one goes
    # to a Metal kernel rather than to ops MLX will not fuse.
    if mode == "solved":

        def apply_nl(A, dz, alpha, g, Isat, V=None):
            return _apply_lossy(A, dz, alpha, g, Isat, V)
    elif mode == "lossless":

        def apply_nl(A, dz, alpha, g, Isat, V=None):
            A_sq = (A * mx.conj(A)).real
            return A * _nl_factor_lossless(A_sq, dz, g, Isat, V)
    else:

        def apply_nl(A, dz, alpha, g, Isat, V=None):
            A_sq = (A * mx.conj(A)).real
            return A * _nl_factor(A_sq, dz, alpha, g, Isat, V)

    if splitting == "lie" and not has_V:

        def _pure(A, propagator, dz, alpha, g, Isat):
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            return apply_nl(A, dz, alpha, g, Isat)

    elif splitting == "lie" and has_V:

        def _pure(A, propagator, V_scaled, dz, alpha, g, Isat):
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            return apply_nl(A, dz, alpha, g, Isat, V_scaled)

    elif splitting == "strang" and not has_V:

        def _pure(A, propagator, dz_half, alpha, g, Isat):
            A = apply_nl(A, dz_half, alpha, g, Isat)
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            return apply_nl(A, dz_half, alpha, g, Isat)

    else:  # strang, has_V

        def _pure(A, propagator, V_scaled, dz_half, alpha, g, Isat):
            A = apply_nl(A, dz_half, alpha, g, Isat, V_scaled)
            A = mx.fft.fftn(A, axes=axes)
            A = A * propagator
            A = mx.fft.ifftn(A, axes=axes)
            return apply_nl(A, dz_half, alpha, g, Isat, V_scaled)

    return mx.compile(_pure)


_SPLIT_STEP_CACHE: dict[tuple, object] = {}


def split_step_fused(
    A: mx.array,
    propagator: mx.array,
    V_scaled: mx.array | None,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
    splitting: str,
    plan: tuple,
    unnorm_ifft: bool = False,
) -> mx.array:
    """Execute fused split step for MLX (nl_length == 0 only).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The propagator matrix.
    V_scaled : mx.array or None
        Pre-scaled potential (V * k/2), or None.
    dz : float
        Propagation step (full for single, half for double precision).
    alpha : float
        Loss coefficient (half of total).
    g : float
        Nonlinear interaction strength.
    Isat : float
        Saturation intensity (converted units).
    splitting : str
        "lie" or "strang" split step splitting.
    plan : tuple
        FFT axes (MLX has no plan objects).
    unnorm_ifft : bool
        Accepted for signature compatibility with the other fused
        backends and ignored: MLX always normalizes its inverse FFT.

    Returns
    -------
    mx.array
        The propagated field.
    """
    axes = plan
    # Keyed on the mode as well: each of the three compiles to its own graph,
    # and the step length decides it as much as alpha does.
    mode = _loss_mode(alpha, dz, g, Isat, A, V_scaled)
    key = (splitting, V_scaled is not None, axes, mode)
    if key not in _SPLIT_STEP_CACHE:
        _SPLIT_STEP_CACHE[key] = _make_split_step(
            splitting, V_scaled is not None, axes, mode
        )
    fn = _SPLIT_STEP_CACHE[key]
    if V_scaled is not None:
        return fn(
            A,
            propagator,
            V_scaled,
            _to_mx(dz),
            _to_mx(alpha),
            _to_mx(g),
            _to_mx(Isat),
        )
    return fn(A, propagator, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))


# ── RK4 utility kernels (stage building and accumulation) ────────────────────


def _rk4_axpy_pure(out, A, c, k):
    return A + c * k


def _rk4_accumulate_pure(acc, w, k):
    return acc + w * k


_c_rk4_axpy = mx.compile(_rk4_axpy_pure)
_c_rk4_accumulate = mx.compile(_rk4_accumulate_pure)


def rk4_axpy(
    out: mx.array,
    A: mx.array,
    c: float,
    k: mx.array,
) -> mx.array:
    """Compute out = A + c * k element-wise for RK4 stage arguments.

    Parameters
    ----------
    out : mx.array
        Output array (unused, kept for API compatibility)
    A : mx.array
        Base field
    c : float
        Scalar coefficient
    k : mx.array
        RK4 slope array

    Returns
    -------
    mx.array
        The result A + c * k.
    """
    return _c_rk4_axpy(out, A, _to_mx(c), k)


def rk4_accumulate(
    acc: mx.array,
    w: float,
    k: mx.array,
) -> mx.array:
    """Compute acc + w * k element-wise for RK4 weighted accumulation.

    Parameters
    ----------
    acc : mx.array
        Accumulator array
    w : float
        Weight coefficient
    k : mx.array
        RK4 slope array

    Returns
    -------
    mx.array
        The result acc + w * k.
    """
    return _c_rk4_accumulate(acc, _to_mx(w), k)


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
    return A_prop + (1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat - alpha * sat) * A_orig


def _rk4_nl_rhs_c_v_pure(
    A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
):
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    return (
        A_prop
        + (1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat - alpha * sat + 1j * V) * A_orig
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
            k2 = rhs(A + dz / 2 * k1)
            k3 = rhs(A + dz / 2 * k2)
            k4 = rhs(A + dz * k3)
            return A + dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    else:

        def _pure(A, propagator, V_scaled, dz, alpha, g, Isat):
            def rhs(A_in):
                A_prop = mx.fft.fftn(A_in, axes=axes)
                A_prop = A_prop * propagator
                A_prop = mx.fft.ifftn(A_prop, axes=axes)
                A_sq = (A_in * mx.conj(A_in)).real
                sat = 1 / (1 + A_sq / Isat)
                return (
                    A_prop + (1j * g * A_sq * sat - alpha * sat + 1j * V_scaled) * A_in
                )

            k1 = rhs(A)
            k2 = rhs(A + dz / 2 * k1)
            k3 = rhs(A + dz / 2 * k2)
            k4 = rhs(A + dz * k3)
            return A + dz / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return mx.compile(_pure)


_SPLIT_STEP_RK4_CACHE: dict[tuple, object] = {}


def split_step_rk4_fused(
    A: mx.array,
    propagator: mx.array,
    V_scaled: mx.array | None,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
    plan: tuple,
) -> mx.array:
    """Execute a whole fused RK4 step for MLX (nl_length == 0 only).

    Parameters
    ----------
    A : mx.array
        The field to propagate.
    propagator : mx.array
        The RK4 propagator (dispersion operator, not exponentiated).
    V_scaled : mx.array or None
        Pre-scaled potential (V * k/2), or None.
    dz : float
        Propagation step.
    alpha : float
        Loss coefficient (half of total).
    g : float
        Nonlinear interaction strength.
    Isat : float
        Saturation intensity (converted units).
    plan : tuple
        FFT axes (MLX has no plan objects).

    Returns
    -------
    mx.array
        The propagated field.
    """
    axes = plan
    key = (V_scaled is not None, axes)
    if key not in _SPLIT_STEP_RK4_CACHE:
        _SPLIT_STEP_RK4_CACHE[key] = _make_split_step_rk4(V_scaled is not None, axes)
    fn = _SPLIT_STEP_RK4_CACHE[key]
    if V_scaled is not None:
        return fn(
            A,
            propagator,
            V_scaled,
            _to_mx(dz),
            _to_mx(alpha),
            _to_mx(g),
            _to_mx(Isat),
        )
    return fn(A, propagator, _to_mx(dz), _to_mx(alpha), _to_mx(g), _to_mx(Isat))


# ── Fused coupled split step (nl_length == 0) ─────────────────────────────


def _make_split_step_coupled(splitting, has_V, has_omega, axes):
    """Build the fused coupled step for one shape of the problem.

    Six variants over three choices: a potential or not, Rabi coupling or not,
    and a whole step (single) against a Strang pair of halves (double). They
    were written out as six bodies, so the interaction, saturation and loss
    terms appeared six times and could drift apart between them -- which is
    the failure this shape invites, because a run only takes one of the six.
    Here the arithmetic appears once and is traced into whichever variant asks
    for it.

    What still varies is the argument list, and it has to: mx.compile traces
    the arguments it is handed, so a potential that is absent cannot be passed
    as None. Hence six signatures over one set of terms rather than six of
    each. ``nonlinear`` takes V1/V2 as None for the variants without one,
    which is resolved while tracing rather than per step.
    """

    def nonlinear(A1, A2, dz, alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2, V1, V2):
        """Apply the real-space terms to both components."""
        sq1 = (A1 * mx.conj(A1)).real
        sq2 = (A2 * mx.conj(A2)).real
        sat = 1 / (1 + sq1 / Isat1 + sq2 / Isat2)
        arg1 = 1j * (g11 * sq1 + g12 * sq2) * sat - alpha1 * sat
        arg2 = 1j * (g22 * sq2 + g21 * sq1) * sat - alpha2 * sat
        if V1 is not None:
            arg1 = arg1 + 1j * V1
            arg2 = arg2 + 1j * V2
        return A1 * mx.exp(dz * arg1), A2 * mx.exp(dz * arg2)

    def linear(A, propagator):
        """Apply the dispersion, exactly, in Fourier space."""
        return mx.fft.ifftn(mx.fft.fftn(A, axes=axes) * propagator, axes=axes)

    def rabi(A1, A2, cos_val, sin_val):
        """Rotate the two components into each other."""
        return cos_val * A1 - 1j * sin_val * A2, cos_val * A2 - 1j * sin_val * A1

    if splitting == "lie" and not has_V and not has_omega:

        def _pure(A, propagator, dz, alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2):
            A = linear(A, propagator)
            A1, A2 = nonlinear(
                A[0],
                A[1],
                dz,
                alpha1,
                alpha2,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                None,
                None,
            )
            return mx.stack([A1, A2])

    elif splitting == "lie" and not has_V and has_omega:

        def _pure(
            A,
            propagator,
            dz,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
            cos_val,
            sin_val,
        ):
            A = linear(A, propagator)
            A1, A2 = nonlinear(
                A[0],
                A[1],
                dz,
                alpha1,
                alpha2,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                None,
                None,
            )
            return mx.stack(list(rabi(A1, A2, cos_val, sin_val)))

    elif splitting == "lie" and has_V and not has_omega:

        def _pure(
            A,
            propagator,
            V1_scaled,
            V2_scaled,
            dz,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
        ):
            A = linear(A, propagator)
            A1, A2 = nonlinear(
                A[0],
                A[1],
                dz,
                alpha1,
                alpha2,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                V1_scaled,
                V2_scaled,
            )
            return mx.stack([A1, A2])

    elif splitting == "lie" and has_V and has_omega:

        def _pure(
            A,
            propagator,
            V1_scaled,
            V2_scaled,
            dz,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
            cos_val,
            sin_val,
        ):
            A = linear(A, propagator)
            A1, A2 = nonlinear(
                A[0],
                A[1],
                dz,
                alpha1,
                alpha2,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                V1_scaled,
                V2_scaled,
            )
            return mx.stack(list(rabi(A1, A2, cos_val, sin_val)))

    elif splitting == "strang" and not has_V:

        def _pure(
            A, propagator, dz_half, alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2
        ):
            args = (alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2, None, None)
            A1, A2 = nonlinear(A[0], A[1], dz_half, *args)
            A = linear(mx.stack([A1, A2]), propagator)
            A1, A2 = nonlinear(A[0], A[1], dz_half, *args)
            return mx.stack([A1, A2])

    else:  # double, has_V

        def _pure(
            A,
            propagator,
            V1_scaled,
            V2_scaled,
            dz_half,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
        ):
            args = (
                alpha1,
                alpha2,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                V1_scaled,
                V2_scaled,
            )
            A1, A2 = nonlinear(A[0], A[1], dz_half, *args)
            A = linear(mx.stack([A1, A2]), propagator)
            A1, A2 = nonlinear(A[0], A[1], dz_half, *args)
            return mx.stack([A1, A2])

    return mx.compile(_pure)


_SPLIT_STEP_COUPLED_CACHE: dict[tuple, object] = {}


def split_step_coupled_fused(
    A: mx.array,
    propagator: mx.array,
    V1_scaled: mx.array | None,
    V2_scaled: mx.array | None,
    dz: float,
    alpha1: float,
    alpha2: float,
    g11: float,
    g12: float,
    g21: float,
    g22: float,
    Isat1: float,
    Isat2: float,
    splitting: str,
    plan: tuple,
    omega: float | None = None,
    unnorm_ifft: bool = False,
) -> mx.array:
    """Execute fused coupled split step for MLX (nl_length == 0 only).

    Parameters
    ----------
    A : mx.array
        The coupled field (2, ...) to propagate.
    propagator : mx.array
        The propagator matrix.
    V1_scaled : mx.array or None
        Pre-scaled potential for component 1 (V * k/2), or None.
    V2_scaled : mx.array or None
        Pre-scaled potential for component 2 (V * k2/2), or None.
    dz : float
        Step size (full for single, half for double).
    alpha1 : float
        Half-loss, component 1.
    alpha2 : float
        Half-loss, component 2.
    g11 : float
        Intra-component 1 interaction.
    g12 : float
        Cross-component interaction in component 1's equation.
    g21 : float
        Cross-component interaction in component 2's equation.
    g22 : float
        Intra-component 2 interaction.
    Isat1 : float
        Saturation, component 1.
    Isat2 : float
        Saturation, component 2.
    splitting : str
        "lie" or "strang".
    plan : tuple
        FFT axes (MLX has no plan objects).
    omega : float or None
        Rabi coupling (half). None to skip.
    unnorm_ifft : bool
        Accepted for signature compatibility with the other fused
        backends and ignored: MLX always normalizes its inverse FFT.

    Returns
    -------
    mx.array
        The propagated field.
    """
    axes = plan
    has_V = V1_scaled is not None
    has_omega = omega is not None and splitting == "lie"
    key = (splitting, has_V, has_omega, axes)
    if key not in _SPLIT_STEP_COUPLED_CACHE:
        _SPLIT_STEP_COUPLED_CACHE[key] = _make_split_step_coupled(
            splitting, has_V, has_omega, axes
        )
    fn = _SPLIT_STEP_COUPLED_CACHE[key]
    dz_mx = _to_mx(dz)
    a1_mx = _to_mx(alpha1)
    a2_mx = _to_mx(alpha2)
    g11_mx = _to_mx(g11)
    g12_mx = _to_mx(g12)
    g21_mx = _to_mx(g21)
    g22_mx = _to_mx(g22)
    Isat1_mx = _to_mx(Isat1)
    Isat2_mx = _to_mx(Isat2)

    if has_V and has_omega:
        cos_val = _to_mx(math.cos(omega * dz))
        sin_val = _to_mx(math.sin(omega * dz))
        return fn(
            A,
            propagator,
            V1_scaled,
            V2_scaled,
            dz_mx,
            a1_mx,
            a2_mx,
            g11_mx,
            g12_mx,
            g21_mx,
            g22_mx,
            Isat1_mx,
            Isat2_mx,
            cos_val,
            sin_val,
        )
    if has_V:
        return fn(
            A,
            propagator,
            V1_scaled,
            V2_scaled,
            dz_mx,
            a1_mx,
            a2_mx,
            g11_mx,
            g12_mx,
            g21_mx,
            g22_mx,
            Isat1_mx,
            Isat2_mx,
        )
    if has_omega:
        cos_val = _to_mx(math.cos(omega * dz))
        sin_val = _to_mx(math.sin(omega * dz))
        return fn(
            A,
            propagator,
            dz_mx,
            a1_mx,
            a2_mx,
            g11_mx,
            g12_mx,
            g21_mx,
            g22_mx,
            Isat1_mx,
            Isat2_mx,
            cos_val,
            sin_val,
        )
    return fn(
        A,
        propagator,
        dz_mx,
        a1_mx,
        a2_mx,
        g11_mx,
        g12_mx,
        g21_mx,
        g22_mx,
        Isat1_mx,
        Isat2_mx,
    )


# ── Fused coupled RK4 RHS (nl_length == 0) ────────────────────────────────


def _make_rk4_rhs_coupled(has_V, axes):
    if not has_V:

        def _pure(
            A_in,
            propagator,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
        ):
            k = mx.fft.fftn(A_in, axes=axes)
            k = k * propagator
            k = mx.fft.ifftn(k, axes=axes)
            a1 = A_in[0]
            a2 = A_in[1]
            sq1 = (a1 * mx.conj(a1)).real
            sq2 = (a2 * mx.conj(a2)).real
            sat = 1 / (1 + sq1 / Isat1 + sq2 / Isat2)
            interact1 = 1j * (g11 * sq1 + g12 * sq2) * sat
            interact2 = 1j * (g22 * sq2 + g21 * sq1) * sat
            k1 = k[0] + (interact1 - alpha1 * sat) * a1
            k2 = k[1] + (interact2 - alpha2 * sat) * a2
            return mx.stack([k1, k2])

    else:

        def _pure(
            A_in,
            propagator,
            V1,
            V2,
            alpha1,
            alpha2,
            g11,
            g12,
            g21,
            g22,
            Isat1,
            Isat2,
        ):
            k = mx.fft.fftn(A_in, axes=axes)
            k = k * propagator
            k = mx.fft.ifftn(k, axes=axes)
            a1 = A_in[0]
            a2 = A_in[1]
            sq1 = (a1 * mx.conj(a1)).real
            sq2 = (a2 * mx.conj(a2)).real
            sat = 1 / (1 + sq1 / Isat1 + sq2 / Isat2)
            interact1 = 1j * (g11 * sq1 + g12 * sq2) * sat
            interact2 = 1j * (g22 * sq2 + g21 * sq1) * sat
            k1 = k[0] + (interact1 - alpha1 * sat + 1j * V1) * a1
            k2 = k[1] + (interact2 - alpha2 * sat + 1j * V2) * a2
            return mx.stack([k1, k2])

    return mx.compile(_pure)


_RK4_RHS_COUPLED_CACHE: dict[tuple, object] = {}


def rk4_rhs_coupled_fused(
    A_in: mx.array,
    k: mx.array,
    V1_scaled: mx.array | None,
    V2_scaled: mx.array | None,
    propagator: mx.array,
    plan: tuple,
    alpha1: float,
    alpha2: float,
    g11: float,
    g12: float,
    g21: float,
    g22: float,
    Isat1: float,
    Isat2: float,
    unnorm_ifft: bool = False,
) -> mx.array:
    """Execute fused coupled RK4 RHS for MLX (nl_length == 0 only).

    Parameters
    ----------
    A_in : mx.array
        Input field (2, ...), not modified.
    k : mx.array
        Output buffer. Accepted for signature compatibility with the
        other fused backends and unused: MLX is functional, so the
        result is returned as a new array.
    V1_scaled : mx.array or None
        Pre-scaled potential, component 1.
    V2_scaled : mx.array or None
        Pre-scaled potential, component 2.
    propagator : mx.array
        RK4 propagator (dispersion operator).
    plan : tuple
        FFT axes (MLX has no plan objects).
    alpha1 : float
        Half-loss, component 1.
    alpha2 : float
        Half-loss, component 2.
    g11 : float
        Intra-component 1 interaction.
    g12 : float
        Cross-component interaction in component 1's equation.
    g21 : float
        Cross-component interaction in component 2's equation.
    g22 : float
        Intra-component 2 interaction.
    Isat1 : float
        Saturation, component 1.
    Isat2 : float
        Saturation, component 2.
    unnorm_ifft : bool
        Accepted for signature compatibility and ignored: MLX always
        normalizes its inverse FFT.

    Returns
    -------
    mx.array
        The RHS result.
    """
    axes = plan
    has_V = V1_scaled is not None
    key = (has_V, axes)
    if key not in _RK4_RHS_COUPLED_CACHE:
        _RK4_RHS_COUPLED_CACHE[key] = _make_rk4_rhs_coupled(has_V, axes)
    fn = _RK4_RHS_COUPLED_CACHE[key]
    a1_mx = _to_mx(alpha1)
    a2_mx = _to_mx(alpha2)
    g11_mx = _to_mx(g11)
    g12_mx = _to_mx(g12)
    g21_mx = _to_mx(g21)
    g22_mx = _to_mx(g22)
    Isat1_mx = _to_mx(Isat1)
    Isat2_mx = _to_mx(Isat2)
    if has_V:
        return fn(
            A_in,
            propagator,
            V1_scaled,
            V2_scaled,
            a1_mx,
            a2_mx,
            g11_mx,
            g12_mx,
            g21_mx,
            g22_mx,
            Isat1_mx,
            Isat2_mx,
        )
    return fn(
        A_in,
        propagator,
        a1_mx,
        a2_mx,
        g11_mx,
        g12_mx,
        g21_mx,
        g22_mx,
        Isat1_mx,
        Isat2_mx,
    )
