"""CPU backend implementation."""

import multiprocessing
import pickle
from typing import Any

import numpy as np
import pyfftw
from scipy import signal

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

        # Validate plans: a 1024x1024 FFT should take <100ms.
        # Stale wisdom can cause 100x slowdowns.
        import time

        plan_fft(A, A)
        plan_ifft(A, A)
        t0 = time.perf_counter()
        plan_fft(A, A)
        plan_ifft(A, A)
        t_roundtrip = time.perf_counter() - t0

        # Heuristic: >200ms for a roundtrip means bad wisdom
        n_elements = 1
        for s in shape:
            n_elements *= s
        expected_max = max(0.2, n_elements / 1024**2 * 0.1)
        if t_roundtrip > expected_max:
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
