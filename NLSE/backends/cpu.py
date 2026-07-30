"""CPU backend implementation."""

import multiprocessing
import pickle
import re
import time
import warnings
from typing import Any

import numpy as np
import pyfftw
from scipy import fft as scipy_fft, signal

from ..kernels import cpu as kernels_cpu
from ..utils import get_cache_dir
from .backend import Backend

# The one place pyfftw is configured. It used to be set here and again in
# solvers/nlse.py with a different planner effort, so which one applied
# depended on import order. Measured on an M3 Max at 2048x2048, PATIENT was
# 42x slower to execute than MEASURE, so the difference was not academic.
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()

# How much slower than scipy's pocketfft a pyfftw roundtrip may be before it
# is worth telling someone. Two different FFTW builds of the same pyfftw
# version measured 35 ms and 9 ms for the same transform pair on one machine,
# so the spread between builds is far wider than this.
_FFT_SLOWDOWN_FACTOR = 2.0

# Below these the comparison cannot tell a slow library from the cost of
# calling one. A 32x32 pair is microseconds of arithmetic under a millisecond
# of dispatch, so the ratio is noise: it reported "2.2x slower" from 0.1 ms
# against 0.1 ms. The gap this looks for only bites at large grids anyway --
# 1.5 ms at 512, 35 at 2048 -- which is the same reason it is safe to skip.
_FFT_MIN_ELEMENTS = 128 * 128
_FFT_MIN_SECONDS = 1e-3

# Timings this short are one scheduling hiccup away from doubling, so the
# reference is the best of a few rather than a single run.
_FFT_REPEATS = 3

_warned_about_fft = False


def measurable(array: np.ndarray, seconds: float) -> bool:
    """Whether a timing on this array can mean anything.

    Below these the comparison is mostly the cost of dispatching a transform
    rather than performing one, and the ratio between two such is noise. It
    mattered: a spurious verdict here does not merely warn, it discards the
    wisdom cache and replans.

    Parameters
    ----------
    array : np.ndarray
        The array transformed.
    seconds : float
        What the transform pair took.

    Returns
    -------
    bool
        True if the measurement is worth comparing.
    """
    return array.size >= _FFT_MIN_ELEMENTS and seconds >= _FFT_MIN_SECONDS


def scipy_roundtrip_seconds(array: np.ndarray, axes: tuple) -> float:
    """Return what scipy takes for the same transform pair.

    Parameters
    ----------
    array : np.ndarray
        Array to transform. Not modified.
    axes : tuple
        Axes to transform over.

    Returns
    -------
    float
        Seconds for a forward and inverse transform.
    """
    reference = array.copy()
    best = float("inf")
    for _ in range(_FFT_REPEATS):
        start = time.perf_counter()
        scipy_fft.ifftn(
            scipy_fft.fftn(reference, axes=axes, workers=-1), axes=axes, workers=-1
        )
        best = min(best, time.perf_counter() - start)
    return best


# FFTW names every codelet it plans, and the vector ones carry the instruction
# set in the name: fftwf_codelet_n2fv_64_avx against fftwf_codelet_n1_8. That
# is the fact we actually want, and wisdom states it outright.
_SIMD_CODELET = re.compile(
    rb"codelet_[a-z0-9_]+_(sse2|avx512|avx2|avx|neon|vsx|altivec|generic_simd\d*)"
)


def fftw_lacks_simd() -> bool | None:
    """Whether this FFTW plans without vector codelets, or None if unknown.

    Which FFTW a pyfftw wheel is linked against decides most of a CPU step at
    large grid sizes: the PyPI arm64 wheel bundles a build with no NEON
    codelets at all and is four times slower than the conda-forge one.

    ``simd_alignment`` does not report it -- it read 4 for both of those. But
    FFTW records the codelets it planned, and names them for the instruction
    set they use, so exporting wisdom answers the question directly. That
    beats timing pyfftw against scipy and reading the ratio, which is what
    this did before: at small sizes both are mostly the cost of being called,
    and a 32x32 pair reported "2.2x slower" from 0.1 ms against 0.1 ms.

    Returns
    -------
    bool or None
        True if wisdom names codelets and none of them is vectorized, False
        if any is, and None if nothing has been planned yet -- which is no
        evidence either way, and is not reported as either.
    """
    try:
        wisdom = b"".join(pyfftw.export_wisdom())
    except Exception:  # a pyfftw that cannot export tells us nothing
        return None
    if b"codelet" not in wisdom:
        return None
    return not _SIMD_CODELET.search(wisdom)


def warn_if_fftw_lacks_simd() -> bool:
    """Warn once if this FFTW was built without vector instructions.

    Returns
    -------
    bool
        True if a warning was issued or had already been issued.
    """
    global _warned_about_fft
    if _warned_about_fft:
        return True
    if fftw_lacks_simd() is not True:
        return False
    _warned_about_fft = True
    warnings.warn(
        "pyFFTW planned this transform with no vector codelets, which means "
        "it is linked against an FFTW built without vector instructions. The "
        "transform is most of a CPU step at large grid sizes, and such a "
        "build measures four times slower on Apple silicon. Installing pyfftw "
        "from conda-forge rather than PyPI fixes it.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True


class CPUBackend(Backend):
    """CPU backend using NumPy and pyFFTW.

    Provides no fused entry points: pyFFTW plans are driven from Python and
    the numba kernels are already single-pass, so there is no launch
    overhead to amortize.

    It does skip the inverse transform's normalization. FFTW's backward
    transform is unnormalized and pyfftw divides by N in a pass of its own
    afterwards, which costs a fifth of the transform -- an inverse measures
    25% more than a forward for the same work. The factor goes into the
    propagator once instead.
    """

    supports_unnormalized_ifft = True

    @property
    def name(self) -> str:
        return "CPU"

    def allocate_field(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Allocate aligned array for FFTW."""
        return pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)

    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Allocate aligned real array."""
        return pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)

    @property
    def convolution(self):
        """Return scipy's overlap-add convolution."""
        return signal.oaconvolve

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        """Already numpy, return as-is."""
        return array

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        """Convert to contiguous array."""
        return np.ascontiguousarray(array)

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build pyFFTW plans with the actual propagation array.

        Parameters
        ----------
        shape : tuple
            Array shape
        axes : tuple
            FFT axes
        dtype : np.dtype
            Array dtype
        array : np.ndarray or None
            The actual array to transform (required for optimal performance)

        Returns
        -------
        list
            List of [forward_plan, inverse_plan]

        """
        # Use provided array or create temporary aligned array
        A = (
            array
            if array is not None
            else pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)
        )

        # Load FFTW wisdom if available
        wisdom_path = get_cache_dir() / "fft.wisdom"
        try:
            with open(wisdom_path, "rb") as file:
                wisdom = pickle.load(file)
                pyfftw.import_wisdom(wisdom)
        except (FileNotFoundError, Exception):
            pass

        # FFTW planning (FFTW_MEASURE) overwrites the input/output arrays.
        # Save and restore array contents to preserve field data.
        saved = A.copy()

        plan_fft = pyfftw.FFTW(
            A,
            A,
            direction="FFTW_FORWARD",
            threads=multiprocessing.cpu_count(),
            axes=axes,
        )
        plan_ifft = pyfftw.FFTW(
            A,
            A,
            direction="FFTW_BACKWARD",
            threads=multiprocessing.cpu_count(),
            axes=axes,
        )

        # Wisdom is cached on disk and outlives the FFTW it was recorded
        # against. Swapping one build for another leaves plans that are valid
        # but slow: on this machine a 2048x2048 pair took 34 ms on wisdom from
        # the old library against 8.8 ms once it was discarded, a whole CPU
        # step going from 40 ms to 12 ms.
        #
        # An absolute threshold cannot see that. The old one allowed 400 ms at
        # this size, ten times the bad plan's cost, so it never fired. What
        # separates a good plan from a stale one is how it compares to scipy
        # on the same array -- but only where that comparison means something,
        # hence `measurable`: at 32x32 the ratio is dispatch overhead, and
        # acting on it discards the whole wisdom cache to replan.
        #
        # Whether the *library* has vector codelets is a different question,
        # and wisdom answers it outright rather than by timing.
        plan_fft(A, A)
        plan_ifft(A, A)
        t0 = time.perf_counter()
        plan_fft(A, A)
        plan_ifft(A, A)
        t_roundtrip = time.perf_counter() - t0

        if measurable(A, t_roundtrip) and t_roundtrip > (
            _FFT_SLOWDOWN_FACTOR * scipy_roundtrip_seconds(A, axes)
        ):
            pyfftw.forget_wisdom()
            plan_fft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_FORWARD",
                threads=multiprocessing.cpu_count(),
                axes=axes,
            )
            plan_ifft = pyfftw.FFTW(
                A,
                A,
                direction="FFTW_BACKWARD",
                threads=multiprocessing.cpu_count(),
                axes=axes,
            )
            plan_fft(A, A)
            plan_ifft(A, A)

        # Asked whatever the timing said, and of the library rather than of
        # the plan: replanning cannot give an FFTW vector codelets it was
        # built without.
        warn_if_fftw_lacks_simd()

        # Restore array contents after planning
        A[:] = saved

        # Save FFTW wisdom
        with open(wisdom_path, "wb") as file:
            wisdom = pyfftw.export_wisdom()
            pickle.dump(wisdom, file)

        return [plan_fft, plan_ifft]

    def fft(self, array: np.ndarray, plan: list) -> np.ndarray:
        """Perform forward FFT in-place."""
        plan[0](array, array)
        return array

    def ifft(self, array: np.ndarray, plan: list, normalize: bool = True) -> np.ndarray:
        """Perform inverse FFT in-place.

        FFTW's backward transform is unnormalized; the 1/N is a separate pass
        pyfftw makes afterwards, and it costs a fifth of the transform. When
        the caller has folded the factor into the propagator, that pass is
        skipped.
        """
        plan[1](array, array, normalise_idft=normalize)
        return array

    @property
    def kernels(self) -> Any:
        """Return CPU kernels module."""
        return kernels_cpu

    def supports_double_precision(self) -> bool:
        """CPU always supports double precision."""
        return True
