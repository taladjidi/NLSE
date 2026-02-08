"""CUPY backend implementation."""

from typing import Any

import numpy as np

from ..utils import __CUPY_AVAILABLE__
from .backend import Backend

if not __CUPY_AVAILABLE__:
    raise ImportError("CuPy is not available - cannot import CUPYBackend")

import cupy as cp
from pyvkfft.cuda import VkFFTApp


class CUPYBackend(Backend):
    """CUPY backend using CuPy and VkFFT."""

    def __init__(self):
        self._vkfft_apps = {}

    @property
    def name(self) -> str:
        return "CUPY"

    def allocate_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate array on GPU."""
        return cp.zeros(shape, dtype=dtype)

    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate real array on GPU."""
        return cp.zeros(shape, dtype=dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from GPU to CPU."""
        return cp.asnumpy(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to GPU."""
        return cp.asarray(array, dtype=array.dtype)

    def build_fft(self, shape: tuple, axes: tuple, dtype: np.dtype) -> list:
        """Build VkFFT app for CUDA.

        Returns:
            List containing VkFFTApp instance (for consistency with CPU backend)
        """
        A = cp.zeros(shape, dtype=dtype)
        app = VkFFTApp(A.shape, A.dtype, axes=axes, ndim=len(axes))
        return [app]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        return plan[0].fft(array, array)

    def ifft(self, array: Any, plan: list) -> Any:
        """Perform inverse FFT."""
        return plan[0].ifft(array, array)

    @property
    def kernels(self) -> Any:
        """Return CUPY kernels module."""
        from ..kernels import cupy as kernels_cupy

        return kernels_cupy

    def supports_double_precision(self) -> bool:
        """CUDA GPUs typically support double precision."""
        return True
