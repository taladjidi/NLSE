import functools
import sys

import numba
import numba.np.ufunc.parallel
import numpy as np

# Claim the OpenMP runtime before anything else in the process can.
#
# A library that vendors its own copy rather than linking the environment's --
# pyfftw's PyPI wheel is the one that bit us, under pyfftw/.dylibs -- can
# coexist with numba's, but only if numba's initialized first. The other order
# segfaults at the first prange, wherever in the process that happens to be,
# so it surfaces as a crash in whichever kernel ran first. This package no
# longer imports pyfftw, but it cannot stop a caller from importing one above
# it, and an environment that had NLSE before this still has pyfftw installed.
#
# Initializing here is enough whenever NLSE is imported first. When it is not,
# the window is already gone and the only safe pool is the one that owns no
# OpenMP runtime; it measures ~6% slower on these kernels, against not running.
if "pyfftw" in sys.modules and not numba.np.ufunc.parallel._is_initialized:
    # numba.config builds its attributes at import, so mypy sees none of them.
    numba.config.THREADING_LAYER = "workqueue"  # type: ignore[attr-defined]
numba.get_num_threads()


# A sine and cosine the vectorizer can get through.
#
# The lossless kernels below rotate each element by an angle, and `np.sin` and
# `np.cos` lower to a call to ___sincos_stret once per element. An opaque call
# in the loop body stops LLVM vectorizing it at all, which is why the nonlinear
# kernels sit near 15% of this machine's bandwidth while every other kernel
# here reaches 70-100%. Written as arithmetic instead -- Cody-Waite range
# reduction into [-pi/4, pi/4], then fdlibm's minimax polynomials -- the loop
# vectorizes and the kernel runs ~1.4x faster, at 1.5 ulp.
#
# `reassoc` is deliberately absent from the fast-math set. It is algebraically
# valid and numerically fatal here: it rewrites `(x - k*P1) - k*P2` into
# `x - k*(P1+P2)`, collapsing the split into a single rounded pi/2 and
# reintroducing exactly the cancellation the split exists to avoid. With it on,
# the error grows with the argument -- 4e-11 by |x| = 1e6 instead of staying at
# 3e-16. Every kernel that calls _sincos must therefore also carry this set
# rather than fastmath=True, since numba inlines the body into its caller and
# the caller's flags are what reach LLVM.
_SINCOS_FASTMATH = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}

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


@numba.njit(inline="always", fastmath=_SINCOS_FASTMATH, cache=True)
def _sincos(x):
    """Return ``(sin x, cos x)`` without a call the vectorizer cannot cross.

    Accurate to 1.5 ulp for ``|x| < 2**29``, which is far past the point where
    the argument's own last bit is worth more than the result: by then a 1 ulp
    change in ``x`` moves the answer by more than 1e-7.

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
    j = np.int64(k) & 3
    z = r * r
    sin_r = r + (z * r) * (
        _S1 + z * (_S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6))))
    )
    cos_r = (
        1.0
        - 0.5 * z
        + (z * z) * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    )
    # Quadrant: sin and cos swap on odd j, and each changes sign on its own
    # half of the circle. Written as arithmetic so the loop stays branch-free.
    swap = np.float64(j & 1)
    sin_x = sin_r + swap * (cos_r - sin_r)
    cos_x = cos_r + swap * (sin_r - cos_r)
    return (
        sin_x * (1.0 - 2.0 * np.float64((j >> 1) & 1)),
        cos_x * (1.0 - 2.0 * np.float64(((j + 1) >> 1) & 1)),
    )


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
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
    for i in numba.prange(A_flat.size):
        # saturation
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq_flat[i] * sat + 1j * V_flat[i]
        A_flat[i] *= np.exp(dz * arg)
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
    if alpha == 0:
        # Lossless: the exponent is purely imaginary, so the step is a
        # rotation. exp(0) is exactly 1, so this is the same result computed
        # with sin and cos instead of a complex exponential -- and _sincos
        # keeps the loop free of the libm call that would stop it vectorizing.
        for i in numba.prange(A_flat.size):
            sat = 1 / (1 + A_sq_flat[i] / Isat)
            theta = dz * g * A_sq_flat[i] * sat
            s, c = _sincos(theta)
            a = A_flat[i]
            A_flat[i] = complex(a.real * c - a.imag * s, a.real * s + a.imag * c)
        return A
    for i in numba.prange(A_flat.size):
        # saturation
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        # Losses and interactions
        arg = -alpha * sat + 1j * g * A_sq_flat[i] * sat
        A_flat[i] *= np.exp(dz * arg)
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
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
    if alpha == 0:
        # See _nl_prop_without_V: lossless is a rotation, not an exponential.
        for i in numba.prange(A1_flat.size):
            sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
            theta = dz * (g11 * A_sq_1_flat[i] * sat + g12 * A_sq_2_flat[i] * sat)
            s, c = _sincos(theta)
            a = A1_flat[i]
            A1_flat[i] = complex(a.real * c - a.imag * s, a.real * s + a.imag * c)
        return A1
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


@numba.njit(parallel=True, fastmath=_SINCOS_FASTMATH, cache=True, boundscheck=False)
def _square_mod_nl_prop(
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
    if alpha == 0:
        # See _nl_prop_without_V: lossless is a rotation, not an exponential.
        for i in numba.prange(A_flat.size):
            a = A_flat[i]
            A_sq_val = a.real * a.real + a.imag * a.imag
            sat = 1 / (1 + A_sq_val / Isat)
            theta = dz * g * A_sq_val * sat
            s, c = _sincos(theta)
            A_flat[i] = complex(a.real * c - a.imag * s, a.real * s + a.imag * c)
        return A
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        arg = -alpha * sat + 1j * g * A_sq_val * sat
        A_flat[i] *= np.exp(dz * arg)
    return A


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _square_mod_nl_prop_v(
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
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (1j * g * A_sq_flat[i] * sat - alpha * sat) * A_flat[i]
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
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_flat[i] / Isat)
        A_prop_flat[i] += (
            1j * g * A_sq_flat[i] * sat - alpha * sat + 1j * V_flat[i]
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
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        A_prop_flat[i] += (1j * g * A_sq_val * sat - alpha * sat) * A_flat[i]
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
    for i in numba.prange(A_flat.size):
        A_sq_val = A_flat[i].real * A_flat[i].real + A_flat[i].imag * A_flat[i].imag
        sat = 1 / (1 + A_sq_val / Isat)
        A_prop_flat[i] += (
            1j * g * A_sq_val * sat - alpha * sat + 1j * V_flat[i]
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
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            1j * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat - alpha * sat
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
    for i in numba.prange(A_flat.size):
        sat = 1 / (1 + A_sq_1_flat[i] / Isat1 + A_sq_2_flat[i] / Isat2)
        A_prop_flat[i] += (
            1j * (g11 * A_sq_1_flat[i] + g12 * A_sq_2_flat[i]) * sat
            - alpha * sat
            + 1j * V_flat[i]
        ) * A_flat[i]
    return A_prop


# ── Broadcasting over a batch of simulations ────────────────────────────────
# The numba kernels take scalar physical parameters. Running a batch of
# simulations at once broadcasts a parameter into an array with a leading
# batch axis, which numba cannot type: it fails with "No implementation of
# function imul found for signature (complex64, array(complex128, 3d, C))".
#
# Loop over the batch and call the compiled kernel once per slice. The loop
# is over simulations (a handful), and each call still processes a whole
# grid with the njit kernel, so this costs a few extra dispatches per step
# rather than giving up numba. It also keeps the fields and the parameters
# consistently sliced, which matters for apply_propagator: a batched field
# against a shared propagator must reuse the propagator, not index past it.


def _batch_len(args, scalar_positions):
    """Return the batch size implied by the scalar parameters, or 0.

    Any array-valued parameter means the loop is needed, including a batch of
    exactly one. Requiring more than one element let a ``(1, 1, 1)`` parameter
    through as a raw array, which is the very thing numba cannot type:
    ``No implementation of function imul found for signature
    (complex64, array(complex128, 3d, C))``.
    """
    n = 0
    for i in scalar_positions:
        value = args[i]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            n = max(n, value.shape[0])
    return n


def _pick_scalar(value, b):
    """Take simulation b's value from a possibly-broadcast parameter."""
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.size > 1:
        return value.reshape(value.shape[0], -1)[b, 0]
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


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


def _broadcast_batch(*scalar_positions, n_outputs=1):
    """Wrap an njit kernel so broadcast scalar parameters are looped over."""
    positions = frozenset(scalar_positions)

    def decorator(kernel):
        @functools.wraps(kernel)
        def wrapper(*args):
            _require_contiguous(kernel.__name__, args[:n_outputs])
            n = _batch_len(args, positions)
            if n == 0:
                n = _shared_grid_batch_len(args, positions)
            if n == 0:
                return kernel(*args)
            ndim = args[0].ndim
            for b in range(n):
                kernel(
                    *[
                        _pick_scalar(a, b)
                        if i in positions
                        else _pick_field(a, b, ndim)
                        for i, a in enumerate(args)
                    ]
                )
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
    # numba indexes both with the same loop and checks nothing. Handed a
    # propagator of another shape or width it reads and writes past the end of
    # one of them, and what surfaces is a segmentation fault inside the
    # compiled kernel with no indication of which argument was wrong -- not an
    # exception anything can catch or a traceback that names the caller.
    if A.ndim > propagator.ndim:
        if A.shape[1:] != propagator.shape:
            raise ValueError(
                f"a batch of {A.shape} fields cannot share a {propagator.shape} "
                f"propagator; the grid axes have to match"
            )
        if A.dtype != propagator.dtype:
            raise ValueError(
                f"field is {A.dtype} and propagator is {propagator.dtype}; "
                f"the kernel reads both as the same width"
            )
        for b in range(A.shape[0]):
            _apply_propagator(A[b], propagator)
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
