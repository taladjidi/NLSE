"""CPU backend implementation."""

import multiprocessing
import pickle
from typing import Any

import numpy as np
import pyfftw

from ..kernels import cpu as kernels_cpu
from .backend import Backend

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_PATIENT"
pyfftw.interfaces.cache.enable()


class CPUBackend(Backend):
    """CPU backend using NumPy and pyFFTW."""

    @property
    def name(self) -> str:
        return "CPU"

    def allocate_field(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Allocate aligned array for FFTW."""
        return pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)

    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Allocate aligned real array."""
        return pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        """Already numpy, return as-is."""
        return array

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        """Convert to contiguous array."""
        return np.ascontiguousarray(array)

    def build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build pyFFTW plans with the actual propagation array.

        Args:
            shape: Array shape
            axes: FFT axes
            dtype: Array dtype
            array: The actual array to transform (required for optimal performance)

        Returns:
            List of [forward_plan, inverse_plan]
        """
        # Use provided array or create temporary aligned array
        A = (
            array
            if array is not None
            else pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)
        )

        # Load FFTW wisdom if available
        try:
            with open("fft.wisdom", "rb") as file:
                wisdom = pickle.load(file)
                pyfftw.import_wisdom(wisdom)
        except (FileNotFoundError, Exception):
            pass

        # Build FFT plans with actual array
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

        # Save FFTW wisdom
        with open("fft.wisdom", "wb") as file:
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
