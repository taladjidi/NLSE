"""Broadcasting kernels for the CuPy backend, built with ``cp.fuse``.

``cupy_kernels.py`` holds the raw CUDA C kernels the backend uses by default.
Those take scalar parameters and index every array with one flat id, so they
cannot serve a batched parameter or a grid shared across a batch;
``CUDAKernels._needs_broadcast`` detects both and calls in here instead, where
CuPy's own broadcasting applies.

The two must agree on every kernel they both provide. What checks that is
``tests/integration/test_broadcasting.py``: on CUPY the batched run takes this
path and the individual runs take the raw one.
"""

import cupy as cp

# Largest |u| = |2*alpha*dz| the iteration below is used at. See
# _LOSS_SOLVED_LIMIT in kernels/cpu.py, which is the same number for the same
# reason.
_LOSS_SOLVED_LIMIT = 0.1


def _loss_factors(sat, u, alpha, dz):
    """Return the saturation and the decay a lossy step should apply.

    ``_loss_factor`` in kernels/cpu.py carries the derivation. Two differences
    here, both because ``alpha`` may be an array -- which is what this module
    exists for. The choice between the solved step and the frozen one is made
    per element rather than per step, and there is no early return for the
    lossless case: at ``u = 0`` the iteration returns ``sat`` untouched and the
    decay is exactly 1, so a lossless run computes what it computed before.

    Parameters
    ----------
    sat : cp.ndarray
        Saturation at the ``|A|^2`` entering the step.
    u : cp.ndarray or float
        ``2*alpha*dz``: the fraction of intensity the step takes out.
    alpha : cp.ndarray or float
        Loss coefficient, for the frozen fallback.
    dz : float
        Step length, for the frozen fallback.

    Returns
    -------
    tuple
        ``(P, decay)`` -- the saturation to evaluate the interaction at, and
        the factor to scale the amplitude by.
    """
    P = sat
    for _ in range(3):
        Pu = P * u
        P = sat * (1 - Pu * P * (0.5 + Pu / 3 + Pu * Pu / 4))
    # cp.abs, not abs: under cp.fuse the operands are proxies.
    solved = cp.abs(u) <= _LOSS_SOLVED_LIMIT
    return (
        cp.where(solved, P, sat),
        cp.where(
            solved,
            cp.sqrt(cp.maximum(1 - P * u, 0)),
            cp.exp(-alpha * sat * dz),
        ),
    )


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
    # Losses: solved across the step rather than frozen at its start, which is
    # what leaves them out of the exponential below. See _loss_factors.
    u = 2 * alpha * dz
    P, decay = _loss_factors(sat, u, alpha, dz)
    # Interactions
    arg = 1j * g * A_sq * P
    # Potential, whose own absorption stays in the exponential
    arg += 1j * V
    arg *= dz
    cp.exp(arg, out=arg)
    A *= decay * arg


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
    # Losses: see _nl_prop_fused above, and _loss_factors.
    u = 2 * alpha * dz
    P, decay = _loss_factors(sat, u, alpha, dz)
    # Interactions
    arg = 1j * g * A_sq * P
    arg *= dz
    cp.exp(arg, out=arg)
    A *= decay * arg


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
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
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
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
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


# Fused kernels (call separate operations for CUPY backend)
@cp.fuse(kernel_name="square_mod_nl_prop")
def _square_mod_nl_prop_fused(A, dz, alpha, g, Isat):
    """Fused implementation of square_mod_nl_prop."""
    A_sq = cp.real(A * cp.conj(A))
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
    A_sq = cp.real(A * cp.conj(A))
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


# RK4 nonlinear RHS kernels (additive, no exp)


@cp.fuse(kernel_name="rk4_nl_rhs")
def _rk4_nl_rhs_fused(A_prop, A, A_sq, alpha, g, Isat):
    """Fused RK4 NL RHS (no potential)."""
    sat = 1 / (1 + A_sq / Isat)
    A_prop += (1j * g * A_sq * sat - alpha * sat) * A


def rk4_nl_rhs(A_prop, A, A_sq, alpha, g, Isat):
    """Accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A : cp.ndarray
        Original field
    A_sq : cp.ndarray
        Field modulus squared
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    cp.ndarray
        The modified A_prop.
    """
    _rk4_nl_rhs_fused(A_prop, A, A_sq, alpha, g, Isat)
    return A_prop


@cp.fuse(kernel_name="rk4_nl_rhs_v")
def _rk4_nl_rhs_v_fused(A_prop, A, A_sq, V, alpha, g, Isat):
    """Fused RK4 NL RHS (with potential)."""
    sat = 1 / (1 + A_sq / Isat)
    A_prop += (1j * g * A_sq * sat - alpha * sat + 1j * V) * A


def rk4_nl_rhs_v(A_prop, A, A_sq, V, alpha, g, Isat):
    """Accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A : cp.ndarray
        Original field
    A_sq : cp.ndarray
        Field modulus squared
    V : cp.ndarray
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    cp.ndarray
        The modified A_prop.
    """
    _rk4_nl_rhs_v_fused(A_prop, A, A_sq, V, alpha, g, Isat)
    return A_prop


@cp.fuse(kernel_name="square_mod_rk4_nl_rhs")
def _square_mod_rk4_nl_rhs_fused(A_prop, A, alpha, g, Isat):
    """Fused |A|^2 + RK4 NL RHS (no potential)."""
    A_sq = cp.real(A * cp.conj(A))
    _rk4_nl_rhs_fused(A_prop, A, A_sq, alpha, g, Isat)


def square_mod_rk4_nl_rhs(A_prop, A, alpha, g, Isat):
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A : cp.ndarray
        Original field
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    cp.ndarray
        The modified A_prop.
    """
    _square_mod_rk4_nl_rhs_fused(A_prop, A, alpha, g, Isat)
    return A_prop


@cp.fuse(kernel_name="square_mod_rk4_nl_rhs_v")
def _square_mod_rk4_nl_rhs_v_fused(A_prop, A, V, alpha, g, Isat):
    """Fused |A|^2 + RK4 NL RHS (with potential)."""
    A_sq = cp.real(A * cp.conj(A))
    _rk4_nl_rhs_v_fused(A_prop, A, A_sq, V, alpha, g, Isat)


def square_mod_rk4_nl_rhs_v(A_prop, A, V, alpha, g, Isat):
    """Compute |A|^2 inline and accumulate nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A : cp.ndarray
        Original field
    V : cp.ndarray
        Potential (pre-scaled)
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation

    Returns
    -------
    cp.ndarray
        The modified A_prop.
    """
    _square_mod_rk4_nl_rhs_v_fused(A_prop, A, V, alpha, g, Isat)
    return A_prop


@cp.fuse(kernel_name="rk4_nl_rhs_c")
def _rk4_nl_rhs_c_fused(A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2):
    """Fused coupled RK4 NL RHS (no potential)."""
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    A_prop += (1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat - alpha * sat) * A_orig


def rk4_nl_rhs_c(A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2):
    """Accumulate coupled nonlinear RHS for RK4 (no potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A_orig : cp.ndarray
        Original field (this component)
    A_sq_1 : cp.ndarray
        Modulus squared of first component
    A_sq_2 : cp.ndarray
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
    cp.ndarray
        The modified A_prop.
    """
    _rk4_nl_rhs_c_fused(A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2)
    return A_prop


@cp.fuse(kernel_name="rk4_nl_rhs_c_v")
def _rk4_nl_rhs_c_v_fused(
    A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
):
    """Fused coupled RK4 NL RHS (with potential)."""
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    A_prop += (1j * (g11 * A_sq_1 + g12 * A_sq_2) * sat - alpha * sat + 1j * V) * A_orig


def rk4_nl_rhs_c_v(A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2):
    """Accumulate coupled nonlinear RHS for RK4 (with potential).

    Parameters
    ----------
    A_prop : cp.ndarray
        Linearly propagated field (modified in-place)
    A_orig : cp.ndarray
        Original field (this component)
    A_sq_1 : cp.ndarray
        Modulus squared of first component
    A_sq_2 : cp.ndarray
        Modulus squared of second component
    V : cp.ndarray
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
    cp.ndarray
        The modified A_prop.
    """
    _rk4_nl_rhs_c_v_fused(
        A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
    )
    return A_prop
