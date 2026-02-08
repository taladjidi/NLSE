"""CPU backend implementation."""

from typing import Any

import numpy as np
import pyfftw

from ..kernels import cpu as kernels_cpu
from .backend import Backend


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

    def build_fft(self, shape: tuple, axes: tuple, dtype: np.dtype) -> list:
        """Build pyFFTW plans.

        Returns:
            List of [forward_plan, inverse_plan]
        """
        A = pyfftw.zeros_aligned(shape, dtype=dtype, n=pyfftw.simd_alignment)
        fft_forward = pyfftw.FFTW(
            A,
            A,
            axes=axes,
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
            threads=pyfftw.config.NUM_THREADS,
        )
        fft_backward = pyfftw.FFTW(
            A,
            A,
            axes=axes,
            direction="FFTW_BACKWARD",
            flags=("FFTW_MEASURE",),
            threads=pyfftw.config.NUM_THREADS,
        )
        return [fft_forward, fft_backward]

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
