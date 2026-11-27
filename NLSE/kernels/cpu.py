import functools
import sys
from collections.abc import Callable

import numba
import numba.np.ufunc.parallel
import numpy as np
from numba.core import types
from numba.extending import overload

# Claim the OpenMP runtime first: a vendored copy (pyfftw's wheel) initialized
# before numba's segfaults at the first prange. If pyfftw is already loaded the
# window is gone, so use the workqueue pool, which owns no OpenMP runtime.
if "pyfftw" in sys.modules and not numba.np.ufunc.parallel._is_initialized:
    # numba.config builds its attributes at import, so a type checker sees none of them.
    numba.config.THREADING_LAYER = "workqueue"  # type: ignore[attr-defined]
numba.get_num_threads()


# sin/cos as arithmetic (Cody-Waite reduction, fdlibm polynomials) so LLVM
# vectorizes the loop: ~1.4x at 1.5 ulp (docs/optimization-log.md). No `reassoc`:
# it merges the split pi/2 and loses the precision the split exists for. Callers
# of _rotate/_mul_cexp carry this set: numba inlines them under their flags.
_SINCOS_FASTMATH = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}

# Exact in float32, so one constant serves both widths. A Python literal would
# not: numba types it float64 or int64 and widens the float32 next to it, which
# turns a complex64 kernel into double arithmetic on half as many lanes.
_ONE = np.float32(1.0)
_TWO = np.float32(2.0)
_THREE = np.float32(3.0)
_HALF = np.float32(0.5)
_QUARTER = np.float32(0.25)
_IMAG = np.complex64(1j)

# The quadrant of an angle is k modulo 4, read off the two low bits of k.
_QUADRANT_MASK = 3

# IEEE 754 binary32: the exponent's bias, and the bit its field starts at.
_F32_EXPONENT_BIAS = 127
_F32_EXPONENT_SHIFT = 23

# pi/2 split into pieces of 24 significant bits, so that k * piece is exact
# while |k| < 2**29 and the subtractions above cancel without error. Four of
# them carry 96 bits of pi/2, leaving ~60 after the worst cancellation.
_TWO_OVER_PI = 6.36619772367581382433e-01
_PIO2_1 = 1.5707963705062866
_PIO2_2 = -4.371138828673793e-08
_PIO2_3 = -1.7151245100058819e-15
_PIO2_4 = 1.056299898220315e-23

# fdlibm __kernel_sin / __kernel_cos, minimax on |r| <= pi/4.
_S1 = -1.66666666666666324348e-01
_S2 = 8.33333333332248946124e-03
_S3 = -1.98412698298579493134e-04
_S4 = 2.75573137070700676789e-06
_S5 = -2.50507602534068634195e-08
_S6 = 1.58969099521155010221e-10
_C1 = 4.16666666666666019037e-02
_C2 = -1.38888888888741095749e-03
_C3 = 2.48015872894767294178e-05
_C4 = -2.75573143513906633035e-07
_C5 = 2.08757232129817482790e-09
_C6 = -1.13596475577881948265e-11

# The float32 set, from cephes sinf.c:
# https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/single/sinf.c#L108-L122
# pi/2 in three pieces (twice its pi/4 ones), the first two of at most 11
# significant bits so that k * piece is exact while |k| < 2**13.
_TWO_OVER_PI_32 = np.float32(6.36619772367581382433e-01)
_PIO2_1_32 = np.float32(1.5703125)
_PIO2_2_32 = np.float32(4.837512969970703125e-4)
_PIO2_3_32 = np.float32(7.54978995489188216e-8)
_S1_32 = np.float32(-1.6666654611e-1)
_S2_32 = np.float32(8.3321608736e-3)
_S3_32 = np.float32(-1.9515295891e-4)
_C1_32 = np.float32(4.166664568298827e-2)
_C2_32 = np.float32(-1.388731625493765e-3)
_C3_32 = np.float32(2.443315711809948e-5)


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _sincos(x):
    """Return ``(sin x, cos x)`` without a call the vectorizer cannot cross.

    Accurate to 1.5 ulp for ``|x| < 2**29``, which is far past the point where
    the argument's own last bit is worth more than the result: by then a 1 ulp
    change in ``x`` moves the answer by more than 1e-7. Double precision only:
    see ``_rotate32`` for what a float32 angle gets instead.

    Parameters
    ----------
    x : float
        Angle in radians.

    Returns
    -------
    tuple of float
        The sine and cosine of ``x``.
    """
    k = np.rint(x * _TWO_OVER_PI)
    r = (((x - k * _PIO2_1) - k * _PIO2_2) - k * _PIO2_3) - k * _PIO2_4
    j = np.int64(k) & _QUADRANT_MASK
    z = r * r
    sin_r = r + (z * r) * (
        _S1 + z * (_S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6))))
    )
    cos_r = (
        _ONE
        - _HALF * z
        + (z * z) * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    )
    # Quadrant: sin and cos swap on odd j, and each changes sign on its own
    # half of the circle. Written as arithmetic so the loop stays branch-free.
    swap = np.float64(j & 1)
    sin_x = sin_r + swap * (cos_r - sin_r)
    cos_x = cos_r + swap * (sin_r - cos_r)
    return (
        sin_x * (_ONE - _TWO * np.float64((j >> 1) & 1)),
        cos_x * (_ONE - _TWO * np.float64(((j + 1) >> 1) & 1)),
    )


def _rotate32(a, theta):
    """Return ``a * exp(i theta)`` in float32, for ``|theta| < 8192``."""
    k = np.rint(theta * _TWO_OVER_PI_32)
    r = ((theta - k * _PIO2_1_32) - k * _PIO2_2_32) - k * _PIO2_3_32
    z = r * r
    sin_r = r + (z * r) * (_S1_32 + z * (_S2_32 + z * _S3_32))
    cosm1_r = (z * z) * (_C1_32 + z * (_C2_32 + z * _C3_32)) - _HALF * z
    # a + a*(cos r - 1 + i sin r) rather than a*(cos r + i sin r): the increment
    # is small, so is its rounding, and a is rounded once. A cosine rounded next
    # to 1 instead shrinks |a|^2 by 0.04 * 2**-24 a step on average.
    re = a.real + (a.real * cosm1_r - a.imag * sin_r)
    im = a.imag + (a.real * sin_r + a.imag * cosm1_r)
    # Times i**k: a swap and two signs, exact, and written branch-free.
    j = np.int32(k) & _QUADRANT_MASK
    swap = np.float32(j & 1)
    keep = _ONE - swap
    sign = _ONE - _TWO * np.float32((j >> 1) & 1)
    return complex(sign * (keep * re - swap * im), sign * (keep * im + swap * re))


def _rotate64(a, theta):
    """Return ``a * exp(i theta)`` in float64."""
    s, c = _sincos(theta)
    return complex(a.real * c - a.imag * s, a.real * s + a.imag * c)


def _rotate(a, theta):
    """Return ``a * exp(i theta)``, computed at the width of ``theta``.

    Parameters
    ----------
    a : complex
        The field value.
    theta : float
        Angle in radians.

    Returns
    -------
    complex
        ``a`` turned by ``theta``.
    """
    if isinstance(theta, np.float32):
        return _rotate32(a, theta)
    return _rotate64(a, theta)


@overload(_rotate, inline="always")
def _rotate_typed(a, theta):
    """Give numba the ``_rotate`` body for the width it typed ``theta`` at."""
    return _rotate32 if theta == types.float32 else _rotate64


# fdlibm __ieee754_exp, on |r| <= ln2/2 after x = k*ln2 + r.
_INV_LN2 = 1.44269504088896338700e00
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_E1 = 1.66666666666666019037e-01
_E2 = -2.77777777770155933842e-03
_E3 = 6.61375632143793436117e-05
_E4 = -1.65339022054652515390e-06
_E5 = 4.13813679705723846039e-08

# exp saturates outside this, so clamping the argument costs nothing and keeps
# k inside the exponent field. Clamping k instead would leave the reduced r
# large and hand the polynomial an argument it is not valid on.
_EXP_HI = 710.0
_EXP_LO = -746.0

# The float32 set, from cephes expf.c (peak relative error 1.7e-7 there):
# https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/single/expf.c#L70-L122
_INV_LN2_32 = np.float32(1.44269504088896341)
_LN2_HI_32 = np.float32(0.693359375)
_LN2_LO_32 = np.float32(-2.12194440e-4)
_EXP32_C0 = np.float32(5.0000001201e-1)
_EXP32_C1 = np.float32(1.6666665459e-1)
_EXP32_C2 = np.float32(4.1665795894e-2)
_EXP32_C3 = np.float32(8.3334519073e-3)
_EXP32_C4 = np.float32(1.3981999507e-3)
_EXP32_C5 = np.float32(1.9875691500e-4)

# exp(89) is past the largest float32 and exp(-105) rounds to zero.
_EXP_HI_32 = np.float32(89.0)
_EXP_LO_32 = np.float32(-105.0)


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _two_pow(k):
    """Return ``2.0**k`` by writing the exponent field, for ``|k| < 1023``."""
    return np.int64((k + 1023) << 52).view(np.float64)


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _two_pow32(k):
    """Return float32 ``2.0**k`` by writing the exponent field, for ``|k| < 127``."""
    return np.int32((k + _F32_EXPONENT_BIAS) << _F32_EXPONENT_SHIFT).view(np.float32)


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _exp(x):
    """Return ``exp(x)`` without a call the vectorizer cannot cross.

    Accurate to 1 ulp. ``math.ldexp`` would be the obvious way to apply the
    ``2**k``, and it is also a call: using it costs more than the polynomial
    saves (0.85x against libm, where writing the exponent field gives 1.5x).
    Double precision only: ``_exp32`` is what a float32 exponent gets.

    Parameters
    ----------
    x : float
        The exponent.

    Returns
    -------
    float
        ``e**x``, saturating to ``inf`` and ``0`` outside the double range.
    """
    x = min(max(x, _EXP_LO), _EXP_HI)
    k = np.rint(x * _INV_LN2)
    hi = x - k * _LN2_HI
    lo = k * _LN2_LO
    r = hi - lo
    z = r * r
    c = r - z * (_E1 + z * (_E2 + z * (_E3 + z * (_E4 + z * _E5))))
    y = _ONE - ((lo - (r * c) / (_TWO - c)) - hi)
    # In two halves: a single (k + 1023) << 52 overflows the field, and the
    # wrap turns exp(-1e4) into inf rather than 0 -- the opposite answer,
    # which then spreads through the field as NaN.
    ki = np.int64(k)
    half = ki >> 1
    return y * _two_pow(half) * _two_pow(ki - half)


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _exp32(x):
    """Return ``exp(x)`` of a float32, saturating to ``inf`` and ``0``."""
    x = min(max(x, _EXP_LO_32), _EXP_HI_32)
    k = np.rint(x * _INV_LN2_32)
    r = (x - k * _LN2_HI_32) - k * _LN2_LO_32
    poly = _EXP32_C0 + r * (
        _EXP32_C1 + r * (_EXP32_C2 + r * (_EXP32_C3 + r * (_EXP32_C4 + r * _EXP32_C5)))
    )
    y = (r * r) * poly + r + _ONE
    # In two halves, for the reason given in _exp.
    ki = np.int32(k)
    half = ki >> 1
    return y * _two_pow32(half) * _two_pow32(ki - half)


def _mul_cexp32(a, scale, w):
    """Return ``a * scale * exp(w)`` in float32, the rotation first."""
    return (scale * _exp32(w.real)) * _rotate(a, w.imag)


def _mul_cexp64(a, scale, w):
    """Return ``a * scale * exp(w)`` in float64."""
    decay = _exp(w.real)
    s, c = _sincos(w.imag)
    return a * (scale * complex(decay * c, decay * s))


def _mul_cexp(a, scale, w):
    """Return ``a * scale * exp(w)``, computed at the width of ``w``.

    Parameters
    ----------
    a : complex
        The field value.
    scale : float
        A real factor, the part of the decay that is not in the exponent.
    w : complex
        The exponent.

    Returns
    -------
    complex
        ``a * scale * e**w``.
    """
    if isinstance(w, np.complex64):
        return _mul_cexp32(a, scale, w)
    return _mul_cexp64(a, scale, w)


@overload(_mul_cexp, inline="always")
def _mul_cexp_typed(a, scale, w):
    """Give numba the ``_mul_cexp`` body for the width it typed ``w`` at."""
    return _mul_cexp32 if w == types.complex64 else _mul_cexp64


# Passes of the iteration in _loss_factor: each raises the order of the
# composition by one, so three gives fourth order, all Yoshida can use.
_LOSS_PASSES = 3

# Largest |u| = |2*alpha*dz| the iteration is used at: it contracts only while
# a step takes out a small fraction of the intensity. Above it the kernels use
# the frozen step; a propagation never gets here (LOSS_PER_STEP_LIMIT is half).
_LOSS_SOLVED_LIMIT = 0.1

# Elements per call of the serial body in the parallel nonlinear kernels: the
# call is lost against 4096 elements at ~1.5 ns each, and a 256x256 field still
# splits into 16 spans. Same speed as one loop, to the bit (M3 Max, measured).
_SPAN = 4096


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _loss_factor(sat, u):
    """Return the saturation the interaction should see over a lossy step.

    The real-space step is exact only where ``|A|^2`` is constant through it.
    That is true of a pure rotation and false the moment there is loss: the
    amplitude decays *inside* the step while the interaction goes on turning
    the phase at the rate the step began with. Frozen like that, the sub-step
    carries a local error of O(dz^2), and every composition built on it comes
    out first order -- Lie, Strang and Yoshida alike, which is measurable and
    was measured.

    Two exact facts say what it should apply instead. With ``y = |A|^2``,
    ``s(y) = 1/(1 + y/Isat)`` and ``u = 2*alpha*dz``::

        dy/dz = -2*alpha*s*y,  dphi/dz = g*y*s   =>   dphi/dy = -g/(2*alpha)

    so the phase over the step is exactly ``(g/(2*alpha))*(y0 - y_end)``
    whatever the saturation does in between, and ``y_end`` is fixed by
    ``ln y + y/Isat = ln y0 + y0/Isat - u``.

    Returned is ``P = (1 - y_end/y0)/u``, which is that phase written as a
    saturation: the step applies ``g*y0*P*dz`` where it applied
    ``g*y0*sat*dz``, and scales the amplitude by ``sqrt(1 - P*u)`` where it
    multiplied by ``exp(-alpha*sat*dz)``.

    P rather than ``y_end`` because P is what makes the lossless case free:
    at ``u = 0`` the iteration returns ``sat`` untouched and both expressions
    collapse to the ones they replace, bit for bit.

    The iteration is the fixed point of
    ``P = sat*(1 - P^2*u*(1/2 + P*u/3 + P^2*u^2/4))``, whose bracket is the
    series of ``(-log1p(-P*u) - P*u)/(P*u)^2``. It contracts while the step
    loses well under half the intensity, which ``_split_step_max_dz`` is what
    keeps it to.

    Parameters
    ----------
    sat : float
        Saturation at the ``|A|^2`` entering the step.
    u : float
        ``2*alpha*dz``: the fraction of intensity the step takes out.

    Returns
    -------
    float
        The saturation to evaluate the interaction at.
    """
    P = sat
    for _ in range(_LOSS_PASSES):
        Pu = P * u
        P = sat * (_ONE - Pu * P * (_HALF + Pu / _THREE + Pu * Pu * _QUARTER))
    return P


@numba.njit(parallel=True, fastmath=_SINCOS_FASTMATH, cache=True, boundscheck=False)
def _nl_prop(
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
    ft = A_flat.real.dtype.type
    dz, alpha, g, Isat = ft(dz), ft(alpha), ft(g), ft(Isat)
    u = _TWO * alpha * dz
    if abs(u) > _LOSS_SOLVED_LIMIT:
        # Either lossless, where freezing |A|^2 is exact, or a step so lossy
        # that solving it is out of reach. See _LOSS_SOLVED_LIMIT.
        for i in numba.prange(A_flat.size):
            # saturation
            sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
            # Losses and interactions
            arg = -alpha * sat + _IMAG * g * A_sq_flat[i] * sat + _IMAG * V_flat[i]
            A_flat[i] = _mul_cexp(A_flat[i], _ONE, dz * arg)
        return A
    # Lossy: the interaction at the saturation the step averages out to (see
    # _loss_factor). A complex potential's absorption stays in the exponential:
    # _loss_factor's identity holds for one decay channel only.
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
        P = _loss_factor(sat, u)
        arg = _IMAG * g * A_sq_flat[i] * P + _IMAG * V_flat[i]
        A_flat[i] = _mul_cexp(A_flat[i], np.sqrt(_ONE - P * u), dz * arg)
    return A


@numba.njit(parallel=True, fastmath=_SINCOS_FASTMATH, cache=True, boundscheck=False)
def _nl_prop_without_V(
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
    ft = A_flat.real.dtype.type
    dz, alpha, g, Isat = ft(dz), ft(alpha), ft(g), ft(Isat)
    if alpha == 0:
        # Lossless: the exponent is purely imaginary, so the step is a
        # rotation. exp(0) is exactly 1, so this is the same result computed
        # with sin and cos instead of a complex exponential -- and _rotate
        # keeps the loop free of the libm call that would stop it vectorizing.
        for i in numba.prange(A_flat.size):
            sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
            theta = dz * g * A_sq_flat[i] * sat
            A_flat[i] = _rotate(A_flat[i], theta)
        return A
    u = _TWO * alpha * dz
    if abs(u) > _LOSS_SOLVED_LIMIT:
        # Too lossy a step to solve; the frozen one, as before. See
        # _LOSS_SOLVED_LIMIT.
        for i in numba.prange(A_flat.size):
            sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
            arg = -alpha * sat + _IMAG * g * A_sq_flat[i] * sat
            A_flat[i] = _mul_cexp(A_flat[i], _ONE, dz * arg)
        return A
    # Lossy: see _loss_factor. sincos rather than a complex exponential,
    # because what is left in the exponent is again purely imaginary once the
    # decay is a factor of its own.
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
        P = _loss_factor(sat, u)
        decay = np.sqrt(_ONE - P * u)
        A_flat[i] = decay * _rotate(A_flat[i], dz * g * A_sq_flat[i] * P)
    return A


@numba.njit(parallel=True, fastmath=_SINCOS_FASTMATH, cache=True, boundscheck=False)
def _nl_prop_c(
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
    ft = A1_flat.real.dtype.type
    dz, alpha, g11, g12, Isat1, Isat2 = (
        ft(dz),
        ft(alpha),
        ft(g11),
        ft(g12),
        ft(Isat1),
        ft(Isat2),
    )
    for i in numba.prange(A1_flat.size):
        # Saturation parameter
        sat = _ONE / (_ONE + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += _IMAG * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
        # Potential
        arg += _IMAG * V_flat[i]
        A1_flat[i] = _mul_cexp(A1_flat[i], _ONE, dz * arg)
    return A1


@numba.njit(parallel=True, fastmath=_SINCOS_FASTMATH, cache=True, boundscheck=False)
def _nl_prop_without_V_c(
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
    ft = A1_flat.real.dtype.type
    dz, alpha, g11, g12, Isat1, Isat2 = (
        ft(dz),
        ft(alpha),
        ft(g11),
        ft(g12),
        ft(Isat1),
        ft(Isat2),
    )
    if alpha == 0:
        # See _nl_prop_without_V: lossless is a rotation, not an exponential.
        for i in numba.prange(A1_flat.size):
            sat = _ONE / (_ONE + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
            theta = dz * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
            A1_flat[i] = _rotate(A1_flat[i], theta)
        return A1
    for i in numba.prange(A1_flat.size):
        # Saturation parameter
        sat = _ONE / (_ONE + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        # Losses
        arg = -alpha * sat
        # Interactions
        arg += _IMAG * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
        A1_flat[i] = _mul_cexp(A1_flat[i], _ONE, dz * arg)
    return A1


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _rabi_coupling(A1: np.ndarray, A2: np.ndarray, dz: float, omega: float) -> tuple:
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
    # Double arithmetic at either width, on purpose: rounded to float32, cos and
    # sin leave the unit circle by up to 2**-24, and every step then moves the
    # total power the same way. It costs nothing measurable; the loop is copies.
    cos_val = np.cos(omega * dz)
    sin_val = np.sin(omega * dz)
    for i in numba.prange(A1_flat.size):
        a1 = A1_flat[i]
        A1_flat[i] = cos_val * a1 - 1j * sin_val * A2_flat[i]
        A2_flat[i] = cos_val * A2_flat[i] - 1j * sin_val * a1
    return A1, A2


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


@numba.njit(
    nogil=True,
    error_model="numpy",
    fastmath=_SINCOS_FASTMATH,
    cache=True,
    boundscheck=False,
)
def _square_mod_nl_prop_span(
    A_flat: np.ndarray,
    start: int,
    stop: int,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Apply the nonlinear step to ``A_flat[start:stop]``, on the calling thread.

    The body ``_square_mod_nl_prop`` runs on each thread's share, and the one
    the blocked step runs on each block. ``error_model="numpy"`` drops the
    division check that stops a serial loop vectorizing (15 against 2.5 ns a
    point on a Ryzen 3 4100; ``prange`` drops it by itself).

    Parameters
    ----------
    A_flat : np.ndarray
        The field, flattened, modified in place.
    start : int
        First element to step.
    stop : int
        One past the last element to step.
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    # A slice, not range(start, stop): an index that may start anywhere keeps
    # numba's negative-index wraparound in the loop, and that stops LLVM
    # vectorizing it (4.1 against 1.6 ns a point, complex64, M3 Max).
    part = A_flat[start:stop]
    ft = part.real.dtype.type
    dz, alpha, g, Isat = ft(dz), ft(alpha), ft(g), ft(Isat)
    if alpha == 0:
        # See _nl_prop_without_V: lossless is a rotation, not an exponential.
        for i in range(part.size):
            a = part[i]
            A_sq_val = a.real * a.real + a.imag * a.imag
            sat = _ONE / (_ONE + A_sq_val / Isat)
            theta = dz * g * A_sq_val * sat
            part[i] = _rotate(a, theta)
        return
    u = _TWO * alpha * dz
    if abs(u) > _LOSS_SOLVED_LIMIT:
        # See _LOSS_SOLVED_LIMIT: the frozen step, as before.
        for i in range(part.size):
            a = part[i]
            A_sq_val = a.real * a.real + a.imag * a.imag
            sat = _ONE / (_ONE + A_sq_val / Isat)
            arg = -alpha * sat + _IMAG * g * A_sq_val * sat
            part[i] = _mul_cexp(part[i], _ONE, dz * arg)
        return
    # Lossy: see _loss_factor, and _nl_prop_without_V for why sincos.
    for i in range(part.size):
        a = part[i]
        A_sq_val = a.real * a.real + a.imag * a.imag
        sat = _ONE / (_ONE + A_sq_val / Isat)
        P = _loss_factor(sat, u)
        decay = np.sqrt(_ONE - P * u)
        part[i] = decay * _rotate(a, dz * g * A_sq_val * P)


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _square_mod_nl_prop(
    A: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Compute |A|^2 inline and apply nonlinear propagation in a single pass.

    One share of the field per thread, each stepped by
    ``_square_mod_nl_prop_span``.

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
    for span in numba.prange((A_flat.size + _SPAN - 1) // _SPAN):
        _square_mod_nl_prop_span(
            A_flat,
            span * _SPAN,
            min((span + 1) * _SPAN, A_flat.size),
            dz,
            alpha,
            g,
            Isat,
        )
    return A


@numba.njit(
    nogil=True,
    error_model="numpy",
    fastmath=_SINCOS_FASTMATH,
    cache=True,
    boundscheck=False,
)
def _square_mod_nl_prop_v_span(
    A_flat: np.ndarray,
    V_flat: np.ndarray,
    start: int,
    stop: int,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> None:
    """Apply the nonlinear step with potential to ``A_flat[start:stop]``.

    The body of ``_square_mod_nl_prop_v``, as ``_square_mod_nl_prop_span`` is
    of ``_square_mod_nl_prop``.

    Parameters
    ----------
    A_flat : np.ndarray
        The field, flattened, modified in place.
    V_flat : np.ndarray
        Potential, flattened, indexed as ``A_flat``.
    start : int
        First element to step.
    stop : int
        One past the last element to step.
    dz : float
        Propagation step in m
    alpha : float
        Losses
    g : float
        Interactions
    Isat : float
        Saturation
    """
    # Sliced for the reason given in _square_mod_nl_prop_span.
    part = A_flat[start:stop]
    V_part = V_flat[start:stop]
    ft = part.real.dtype.type
    dz, alpha, g, Isat = ft(dz), ft(alpha), ft(g), ft(Isat)
    u = _TWO * alpha * dz
    if abs(u) > _LOSS_SOLVED_LIMIT:
        # Lossless, or too lossy to solve. See _LOSS_SOLVED_LIMIT.
        for i in range(part.size):
            A_sq_val = part[i].real * part[i].real + part[i].imag * part[i].imag
            sat = _ONE / (_ONE + A_sq_val / Isat)
            arg = -alpha * sat + _IMAG * g * A_sq_val * sat + _IMAG * V_part[i]
            part[i] = _mul_cexp(part[i], _ONE, dz * arg)
        return
    # Lossy: see _loss_factor, and _nl_prop for what stays in the exponential.
    for i in range(part.size):
        A_sq_val = part[i].real * part[i].real + part[i].imag * part[i].imag
        sat = _ONE / (_ONE + A_sq_val / Isat)
        P = _loss_factor(sat, u)
        arg = _IMAG * g * A_sq_val * P + _IMAG * V_part[i]
        part[i] = _mul_cexp(part[i], np.sqrt(_ONE - P * u), dz * arg)


@numba.njit(parallel=True, cache=True, boundscheck=False)
def _square_mod_nl_prop_v(
    A: np.ndarray,
    V: np.ndarray,
    dz: float,
    alpha: float,
    g: float,
    Isat: float,
) -> np.ndarray:
    """Compute |A|^2 inline and apply nonlinear propagation with potential.

    One share of the field per thread, each stepped by
    ``_square_mod_nl_prop_v_span``.

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
    for span in numba.prange((A_flat.size + _SPAN - 1) // _SPAN):
        _square_mod_nl_prop_v_span(
            A_flat,
            V_flat,
            span * _SPAN,
            min((span + 1) * _SPAN, A_flat.size),
            dz,
            alpha,
            g,
            Isat,
        )
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _apply_propagator(A: np.ndarray, propagator: np.ndarray) -> np.ndarray:
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
        A_flat[i] = (a.real * p.real - a.imag * p.imag) + _IMAG * (
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
    ft = out_flat.real.dtype.type
    c = ft(c)
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
    ft = acc_flat.real.dtype.type
    w = ft(w)
    for i in numba.prange(acc_flat.size):
        acc_flat[i] += w * k_flat[i]
    return acc


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _rk4_nl_rhs(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g, Isat = ft(alpha), ft(g), ft(Isat)
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (_IMAG * g * A_sq_flat[i] * sat - alpha * sat) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _rk4_nl_rhs_v(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g, Isat = ft(alpha), ft(g), ft(Isat)
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (
            _IMAG * g * A_sq_flat[i] * sat - alpha * sat + _IMAG * V_flat[i]
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _square_mod_rk4_nl_rhs(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g, Isat = ft(alpha), ft(g), ft(Isat)
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = _ONE / (_ONE + A_sq_val / Isat)
        A_prop_flat[i] += (_IMAG * g * A_sq_val * sat - alpha * sat) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _square_mod_rk4_nl_rhs_v(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g, Isat = ft(alpha), ft(g), ft(Isat)
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = _ONE / (_ONE + A_sq_val / Isat)
        A_prop_flat[i] += (
            _IMAG * g * A_sq_val * sat - alpha * sat + _IMAG * V_flat[i]
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _rk4_nl_rhs_c(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g11, g12, Isat1, Isat2 = ft(alpha), ft(g11), ft(g12), ft(Isat1), ft(Isat2)
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            _IMAG * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat - alpha * sat
        ) * A_flat[i]
    return A_prop


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _rk4_nl_rhs_c_v(
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
    ft = A_prop_flat.real.dtype.type
    alpha, g11, g12, Isat1, Isat2 = ft(alpha), ft(g11), ft(g12), ft(Isat1), ft(Isat2)
    for i in numba.prange(A_flat.size):
        sat = _ONE / (_ONE + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            _IMAG * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat
            - alpha * sat
            + _IMAG * V_flat[i]
        ) * A_flat[i]
    return A_prop


# ── Broadcasting over a batch of simulations ────────────────────────────────
# numba cannot type a parameter with a batch axis, so loop over the batch and
# call the kernel once per slice: a few extra dispatches per step. A batched
# field against a shared propagator reuses the propagator, not index past it.


def _collapse_singletons(args, scalar_positions):
    """Hand each one-value array parameter over as that value.

    One value applies to every point whatever its shape, so the kernel takes
    the field in one call. Left an array, ``np.array([0.1])`` sent a (256, 256)
    field through the loop one row at a time: 29 ms a call instead of 0.17.
    Numba cannot type an array where a scalar goes, so nothing is lost.
    """
    return [
        a.reshape(-1)[0]
        if i in scalar_positions and isinstance(a, np.ndarray) and a.size == 1
        else a
        for i, a in enumerate(args)
    ]


def _batch_len(args, scalar_positions):
    """Return the length of the field's leading axis if a parameter is an array.

    Any array-valued parameter left by ``_collapse_singletons`` holds several
    values, so the loop is needed. Numba cannot type one at a scalar position:
    ``No implementation of function imul found for signature
    (complex64, array(complex128, 3d, C))``.

    The length is the field's, not the parameter's: under numpy's rules a
    parameter may hold one value along an axis the field is batched over.
    """
    field = args[0]
    n = 0
    for i in scalar_positions:
        value = args[i]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            n = field.shape[0]
            if value.ndim >= field.ndim and value.shape[0] not in (1, n):
                raise ValueError(
                    f"a parameter of shape {value.shape} does not broadcast "
                    f"against a field of shape {field.shape}"
                )
    return n


def _pick_scalar(value, b, batched_ndim):
    """Take simulation b's share of a possibly-broadcast parameter.

    Peels the leading axis of a parameter that reaches the field's. One value
    left is handed back as a scalar; what is still an array varies over a
    further batch axis.
    """
    if not isinstance(value, np.ndarray):
        return value
    if 0 < value.ndim >= batched_ndim:
        value = value[b if value.shape[0] > 1 else 0]
    return value.reshape(-1)[0] if value.size == 1 else value


def _shared_grid_batch_len(args, scalar_positions):
    """Return the batch size implied by the field alone, or 0.

    A batch does not need a batched parameter: running several initial
    conditions through identical physics leaves every parameter scalar and
    only the field carries the extra axis. The kernels index the field and
    any shared grid (a potential, a nonlinear profile) with the same flat
    index, so that case has to be looped over too.
    """
    field = args[0]
    if not isinstance(field, np.ndarray):
        return 0
    for i, value in enumerate(args):
        if i in scalar_positions or not isinstance(value, np.ndarray):
            continue
        if 0 < value.ndim < field.ndim:
            return field.shape[0]
    return 0


def _pick_field(value, b, batched_ndim):
    """Take simulation b's slice of a field, if the field is batched.

    Compares against the primary field's ndim rather than the leading
    dimension, so a shared grid is never sliced just because its first axis
    happens to equal the batch size.
    """
    if isinstance(value, np.ndarray) and value.ndim == batched_ndim:
        return value[b]
    return value


def _require_contiguous(name, arrays):
    """Refuse an array these kernels would write to and then discard.

    Every kernel here starts by flattening its arguments, and ``ravel`` views
    a contiguous array but *copies* a strided one. The loop then writes into
    that copy, the kernel returns the argument it was handed, and the work is
    gone -- no error, and an answer that looks like the physics with a term
    left out. A batched coupled component is exactly that shape, (B, NX) out
    of (B, 2, NX), and it cost every batched coupled CPU run its whole
    real-space step until the solver was made to hand over contiguous
    components.

    Only the arrays written through are checked. The read-only ones may be
    strided freely: ravel copying them changes nothing.

    Parameters
    ----------
    name : str
        Kernel name, for the message.
    arrays : sequence
        The arguments the kernel writes into.
    """
    for i, array in enumerate(arrays):
        if isinstance(array, np.ndarray) and not array.flags.c_contiguous:
            raise ValueError(
                f"{name} would write to a non-contiguous array (argument "
                f"{i}, shape {array.shape}, strides {array.strides}) and the "
                f"write would be silently dropped: these kernels flatten with "
                f"ravel, which copies rather than views when the input is "
                f"strided. Pass np.ascontiguousarray(...) and write the "
                f"result back."
            )


def _pick_member(args, b, scalar_positions):
    """Return the arguments of simulation b along the field's leading axis."""
    ndim = args[0].ndim
    return [
        _pick_scalar(a, b, ndim) if i in scalar_positions else _pick_field(a, b, ndim)
        for i, a in enumerate(args)
    ]


def _broadcast_batch(*scalar_positions, n_outputs=1):
    """Wrap an njit kernel so broadcast scalar parameters are looped over."""
    positions = frozenset(scalar_positions)

    def decorator(kernel):
        @functools.wraps(kernel)
        def wrapper(*args):
            _require_contiguous(kernel.__name__, args[:n_outputs])
            args = _collapse_singletons(args, positions)
            n = _batch_len(args, positions) or _shared_grid_batch_len(args, positions)
            if n == 0:
                return kernel(*args)
            # A member that is itself a batch (a second batch axis) goes
            # through the wrapper again; every member has the first one's shape.
            first = _pick_member(args, 0, positions)
            nested = _batch_len(first, positions) or _shared_grid_batch_len(
                first, positions
            )
            run = wrapper if nested else kernel
            for b in range(n):
                run(*_pick_member(args, b, positions))
            return args[0] if n_outputs == 1 else tuple(args[:n_outputs])

        return wrapper

    return decorator


def apply_propagator(A: np.ndarray, propagator: np.ndarray) -> np.ndarray:
    """Multiply A by the propagator in-place, broadcasting over a batch.

    Parameters
    ----------
    A : np.ndarray
        The field array, modified in-place.
    propagator : np.ndarray
        The propagator, either matching A or shared across the batch.

    Returns
    -------
    np.ndarray
        The modified field array A.
    """
    # numba checks no shapes: a propagator of another shape or width reads past the
    # end and segfaults inside the kernel, with nothing naming the argument.
    if A.ndim > propagator.ndim:
        if A.shape[-propagator.ndim :] != propagator.shape:
            raise ValueError(
                f"a batch of {A.shape} fields cannot share a {propagator.shape} "
                f"propagator; the grid axes have to match"
            )
        if A.dtype != propagator.dtype:
            raise ValueError(
                f"field is {A.dtype} and propagator is {propagator.dtype}; "
                f"the kernel reads both as the same width"
            )
        _require_contiguous("apply_propagator", (A,))
        for field in A.reshape(-1, *propagator.shape):
            _apply_propagator(field, propagator)
        return A
    if A.shape != propagator.shape:
        raise ValueError(
            f"field is {A.shape} and propagator is {propagator.shape}; "
            f"they have to match"
        )
    if A.dtype != propagator.dtype:
        raise ValueError(
            f"field is {A.dtype} and propagator is {propagator.dtype}; "
            f"the kernel reads both as the same width"
        )
    _require_contiguous("apply_propagator", (A,))
    return _apply_propagator(A, propagator)


def _takes_blocked_step(A, shape, V_scaled, scalars) -> bool:
    """Whether the blocked step can take this call, all of it as given.

    One C-contiguous 2D field of the plan's grid, writable, with scalar
    constants, and no potential or one of the field's shape. A batch, or
    constants per member, is left to the step in two.

    Parameters
    ----------
    A : np.ndarray
        Field.
    shape : tuple
        The grid the plan, and so its step, was built for.
    V_scaled : np.ndarray or None
        Pre-scaled potential.
    scalars : tuple
        ``(alpha, g, Isat)``.

    Returns
    -------
    bool
        True if ``BlockedStep`` computes what was asked.
    """
    # The step's kernels skip bounds checks: a field of another grid than the
    # step's is declined here, and with it a field of other than two axes.
    return (
        A.shape == tuple(shape)
        and A.flags.c_contiguous
        and A.flags.writeable
        and all(np.ndim(value) == 0 for value in scalars)
        and (
            V_scaled is None
            or (V_scaled.shape == A.shape and V_scaled.flags.c_contiguous)
        )
    )


def linear_nonlinear_step(
    A: np.ndarray,
    propagator: np.ndarray,
    plan,
    V_scaled: np.ndarray | None,
    dz: float,
    alpha,
    g,
    Isat,
    factors: Callable | None = None,
) -> bool:
    """Take the linear step then the nonlinear step, in three blocked passes.

    The hook ``Backend.has_linear_nonlinear_step`` documents. Taken where the
    grid was measured to gain from it (``plan.blocked_step()``); declined,
    with A untouched, everywhere else.

    Parameters
    ----------
    A : np.ndarray
        Field, modified in place when the step is taken.
    propagator : np.ndarray
        The twin carrying 1/N. Unused: the step reads ``factors``.
    plan : CPUPlan
        The plan the solver built for A.
    V_scaled : np.ndarray or None
        Pre-scaled potential, or None.
    dz : float
        Step of the nonlinear part.
    alpha, g, Isat : float
        The solver's constants.
    factors : callable or None
        Returns ``(rows, columns)`` of the propagator, the rows carrying 1/N,
        or None. Called last: building them costs a pass over the dispersion.

    Returns
    -------
    bool
        True if the step was taken.
    """
    if factors is None or not _takes_blocked_step(
        A, plan.shape, V_scaled, (alpha, g, Isat)
    ):
        return False
    step = plan.blocked_step()
    if step is None:
        return False
    built = factors()
    # The kernels skip bounds checks here too: one factor per row and column.
    if built is None or (built[0].shape, built[1].shape) != (A.shape[:1], A.shape[1:]):
        return False
    nonlinear = (float(dz), float(alpha), float(g), float(Isat))
    step(A, built[0], built[1], V_scaled, nonlinear)
    return True


nl_prop = _broadcast_batch(2, 3, 5, 6)(_nl_prop)
nl_prop_without_V = _broadcast_batch(2, 3, 4, 5)(_nl_prop_without_V)
nl_prop_c = _broadcast_batch(3, 4, 6, 7, 8, 9)(_nl_prop_c)
nl_prop_without_V_c = _broadcast_batch(3, 4, 5, 6, 7, 8)(_nl_prop_without_V_c)
square_mod_nl_prop = _broadcast_batch(1, 2, 3, 4)(_square_mod_nl_prop)
square_mod_nl_prop_v = _broadcast_batch(2, 3, 4, 5)(_square_mod_nl_prop_v)
rk4_nl_rhs = _broadcast_batch(3, 4, 5)(_rk4_nl_rhs)
rk4_nl_rhs_v = _broadcast_batch(4, 5, 6)(_rk4_nl_rhs_v)
square_mod_rk4_nl_rhs = _broadcast_batch(2, 3, 4)(_square_mod_rk4_nl_rhs)
square_mod_rk4_nl_rhs_v = _broadcast_batch(3, 4, 5)(_square_mod_rk4_nl_rhs_v)
rk4_nl_rhs_c = _broadcast_batch(4, 5, 6, 7, 8)(_rk4_nl_rhs_c)
rk4_nl_rhs_c_v = _broadcast_batch(5, 6, 7, 8, 9)(_rk4_nl_rhs_c_v)
rabi_coupling = _broadcast_batch(2, 3, n_outputs=2)(_rabi_coupling)
