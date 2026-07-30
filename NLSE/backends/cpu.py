"""CPU backend implementation."""

import multiprocessing
import pickle
import re
import sys
import time
import warnings
from typing import Any

import numba
import numba.np.ufunc.parallel
import numpy as np
import numpy.typing as npt

# Start numba's thread pool before FFTW's, and never the other way round.
#
# The two ship separate copies of the same OpenMP runtime -- pyfftw's PyPI
# wheel bundles libomp.dylib under pyfftw/.dylibs, numba's omppool links the
# one in the environment -- and a process can hold both, but only in that
# order. Let FFTW's initialize first and the *first* prange in the process
# segfaults, wherever it happens to be, which reads as a crash in whichever
# kernel ran first rather than as the import conflict it is.
#
# Launching the pool here is enough, because this line runs before pyfftw is
# loaded: this module is the only importer of it, and the package __init__
# runs before any submodule. The window is gone if the caller imported pyfftw
# already, and then the only safe layer is the one that owns no OpenMP
# runtime. It measures ~6% slower on the kernels, which beats crashing.
if "pyfftw" in sys.modules and not numba.np.ufunc.parallel._is_initialized:
    # numba.config builds its attributes at import, so mypy sees none of them.
    numba.config.THREADING_LAYER = "workqueue"  # type: ignore[attr-defined]
numba.get_num_threads()

import pyfftw  # noqa: E402
from scipy import fft as scipy_fft, signal  # noqa: E402

from ..kernels import cpu as kernels_cpu  # noqa: E402
from ..utils import get_cache_dir  # noqa: E402
from .backend import Backend  # noqa: E402

# The one place pyfftw is configured. It used to be set here and again in
# solvers/nlse.py with a different planner effort, so which one applied
# depended on import order. Measured on an M3 Max at 2048x2048, PATIENT was
# 42x slower to execute than MEASURE, so the difference was not academic.
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

# pyfftw.interfaces.cache.enable() used to be here. It caches the objects
# pyfftw.interfaces.numpy_fft and friends build, and this backend calls none
# of them -- it holds pyfftw.FFTW plans of its own, which is the whole of
# _build_fft. So it bought nothing, and what it cost was a background
# keepalive thread holding FFTW objects: a thread that segfaulted on macOS
# during a run that also calls pyfftw.forget_wisdom() to discard stale plans.
# That path is only reached where FFTW is slow enough to look stale, which is
# why it showed on one machine and not another.

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
    large grid sizes, and on arm64 neither prebuilt build is vectorized --
    not the PyPI wheel and not the conda-forge one. So this is worth reporting
    but not worth prescribing a fix for: there is no install that answers it,
    only an FFTW built from source with NEON on.

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
        f"pyFFTW planned this transform with no vector codelets, so it is "
        f"linked against an FFTW built without vector instructions. The "
        f"transform is most of a CPU step at large grid sizes, and on an M3 "
        f"Max this build measures 37 ms against scipy's 6.5 at 2048x2048, "
        f"with FFTW on all threads.\n"
        f"  On arm64 no prebuilt pyfftw answers this -- neither the PyPI "
        f"wheel nor the conda-forge build is vectorized -- so swapping "
        f"between them will not help, and only an FFTW built from source "
        f"with NEON on will. On x86 the wheels are vectorized and seeing "
        f"this means something else supplied the library.\n"
        f"  If you do change it, discard the plans recorded against the old "
        f"one, which outlive it:\n"
        f"    rm {get_cache_dir() / 'fft.wisdom'}\n"
        f"  To check what you have, plan any transform and look at what FFTW "
        f"says it used:\n"
        f'    python -c "import pyfftw,numpy as np;'
        f"A=pyfftw.empty_aligned((64,64),dtype='complex64');"
        f"pyfftw.FFTW(A,A,axes=(-2,-1));"
        f"print(b''.join(pyfftw.export_wisdom()))\"\n"
        f"  A vectorized build names the instruction set in the codelet -- "
        f"fftwf_codelet_n1fv_64_neon -- where a scalar one says "
        f"fftwf_codelet_n1_64.",
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

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate aligned array for FFTW."""
        return pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
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

    def exp(self, array: Any) -> Any:
        """Exponentiate without leaving this backend."""
        return np.exp(array)

    def sum(self, array: Any) -> float:
        """Reduce without leaving this backend."""
        return float(np.sum(array))

    @property
    def kernels(self) -> Any:
        """Return CPU kernels module."""
        return kernels_cpu

    def supports_double_precision(self) -> bool:
        """CPU always supports double precision."""
        return True
