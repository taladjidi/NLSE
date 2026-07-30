"""CPU backend implementation."""

import multiprocessing
import pickle
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

_warned_about_fft = False


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
    start = time.perf_counter()
    scipy_fft.ifftn(
        scipy_fft.fftn(reference, axes=axes, workers=-1), axes=axes, workers=-1
    )
    return time.perf_counter() - start


def warn_if_fft_is_slow(array: np.ndarray, seconds: float, axes: tuple) -> bool:
    """Warn once if pyfftw is far slower than scipy on the same transform.

    Which FFTW a pyfftw wheel is linked against decides most of a CPU step at
    large grid sizes, and nothing in the library reports it. ``simd_alignment``
    does not: it read 4 both for the PyPI arm64 wheel, whose bundled FFTW has
    no NEON codelets at all, and for the conda-forge build that is four times
    faster. So this measures instead of asking.

    Parameters
    ----------
    array : np.ndarray
        The array that was transformed, used to time the comparison.
    seconds : float
        What the pyfftw roundtrip took.
    axes : tuple
        Axes the transform was over.

    Returns
    -------
    bool
        True if a warning was issued or had already been issued.
    """
    global _warned_about_fft
    if _warned_about_fft:
        return True
    scipy_seconds = scipy_roundtrip_seconds(array, axes)
    if seconds <= _FFT_SLOWDOWN_FACTOR * scipy_seconds:
        return False
    _warned_about_fft = True
    warnings.warn(
        f"pyFFTW is {seconds / scipy_seconds:.1f}x slower than scipy.fft on this "
        f"machine ({seconds * 1e3:.1f} ms against {scipy_seconds * 1e3:.1f} ms for "
        f"a {array.shape} transform pair), which usually means it is linked "
        f"against an FFTW built without vector instructions. The transform is "
        f"most of a CPU step at large grid sizes. Installing pyfftw from "
        f"conda-forge rather than PyPI has measured 4x faster on Apple silicon.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True


class CPUBackend(Backend):
    """CPU backend using NumPy and pyFFTW.

    Provides no fused entry points: pyFFTW plans are driven from Python and
    the numba kernels are already single-pass, so there is no launch
    overhead to amortize.
    """

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
        # separates a good plan from a stale one is the same measurement that
        # separates a good FFTW build from a scalar one: how it compares to
        # scipy on the same array.
        plan_fft(A, A)
        plan_ifft(A, A)
        t0 = time.perf_counter()
        plan_fft(A, A)
        plan_ifft(A, A)
        t_roundtrip = time.perf_counter() - t0

        if t_roundtrip > _FFT_SLOWDOWN_FACTOR * scipy_roundtrip_seconds(A, axes):
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
            t0 = time.perf_counter()
            plan_fft(A, A)
            plan_ifft(A, A)
            # Still slow with fresh wisdom means the library itself is slow,
            # which no amount of replanning fixes.
            warn_if_fft_is_slow(A, time.perf_counter() - t0, axes)

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

    def ifft(self, array: np.ndarray, plan: list) -> np.ndarray:
        """Perform inverse FFT in-place."""
        plan[1](array, array)
        return array

    @property
    def kernels(self) -> Any:
        """Return CPU kernels module."""
        return kernels_cpu

    def supports_double_precision(self) -> bool:
        """CPU always supports double precision."""
        return True
