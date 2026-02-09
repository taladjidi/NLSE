"""CPU backend implementation."""

import os
import platform
from typing import Any

import numpy as np
import pyfftw

from ..kernels import cpu as kernels_cpu
from .backend import Backend

# Configure pyFFTW for optimal performance
pyfftw.config.NUM_THREADS = os.cpu_count() or 1  # Use all available cores
pyfftw.interfaces.cache.enable()  # Enable plan caching for faster repeated FFTs

# Platform detection for FFT optimization
_CPU_VENDOR = platform.processor().lower()
_IS_INTEL = "intel" in _CPU_VENDOR or "genuine" in _CPU_VENDOR
_IS_APPLE = "apple" in _CPU_VENDOR or "arm" in _CPU_VENDOR
_IS_AMD = "amd" in _CPU_VENDOR or "authent" in _CPU_VENDOR

# TODO: Platform-specific FFT optimization
# - Intel CPUs: Intel MKL FFT is 10-30% faster than FFTW
#   Install: conda install mkl mkl-service
#   Usage: import mkl_fft; mkl_fft.fftn(array)
#
# - Apple Silicon (M1/M2/M3): Accelerate framework is 2-3x faster than FFTW
#   Already available on macOS via scipy.fft (automatically uses vDSP)
#   Usage: from scipy import fft; fft.fftn(array)
#
# - AMD CPUs: FFTW is already optimal, PATIENT planning gives best SIMD selection


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
        self, shape: tuple, axes: tuple, dtype: np.dtype, array: np.ndarray | None = None
    ) -> list:
        """Build pyFFTW plans.

        IMPORTANT: Plans should be built with the actual array that will be used
        during propagation for optimal performance. If no array is provided,
        creates a temporary aligned array (slower).

        Platform-specific optimization notes:
        - Intel CPUs: Consider Intel MKL (10-30% faster)
        - Apple Silicon: Consider Accelerate/vDSP (2-3x faster)
        - AMD CPUs: FFTW is optimal (already using best option)

        Args:
            shape: Array shape
            axes: FFT axes
            dtype: Array dtype
            array: The actual array to transform (for in-place optimization)

        Returns:
            List of [forward_plan, inverse_plan]
        """
        import pickle

        # Load FFTW wisdom for faster planning
        try:
            with open("fft.wisdom", "rb") as file:
                wisdom = pickle.load(file)
                pyfftw.import_wisdom(wisdom)
        except FileNotFoundError:
            pass  # Wisdom will be saved after planning

        # Use provided array or create temporary aligned array
        A = array if array is not None else pyfftw.zeros_aligned(
            shape, dtype=dtype, n=pyfftw.simd_alignment
        )

        # FFTW_MEASURE: Fast planning with good performance
        planning_mode = "FFTW_MEASURE"

        # Platform detection (for future optimization)
        if _IS_APPLE:
            # TODO: Switch to scipy.fft which uses vDSP on macOS (2-3x faster)
            pass
        elif _IS_INTEL:
            # TODO: Try Intel MKL if available (10-30% faster)
            pass
        # AMD and others: FFTW is already optimal

        fft_forward = pyfftw.FFTW(
            A,
            A,
            axes=axes,
            direction="FFTW_FORWARD",
            flags=(planning_mode,),
            threads=pyfftw.config.NUM_THREADS,
        )
        fft_backward = pyfftw.FFTW(
            A,
            A,
            axes=axes,
            direction="FFTW_BACKWARD",
            flags=(planning_mode,),
            threads=pyfftw.config.NUM_THREADS,
        )

        # Save FFTW wisdom for future use
        with open("fft.wisdom", "wb") as file:
            wisdom = pyfftw.export_wisdom()
            pickle.dump(wisdom, file)

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
