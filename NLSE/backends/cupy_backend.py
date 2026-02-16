"""CUPY backend implementation."""

from typing import Any

import numpy as np

from ..utils import __CUPY_AVAILABLE__
from .backend import Backend

if not __CUPY_AVAILABLE__:
    raise ImportError("CuPy is not available - cannot import CUPYBackend")

import cupy as cp
import cupyx.scipy.fft as _cufft
from cupyx.scipy.fftpack import get_fft_plan


class _CuFFTPlan:
    """cuFFT plan wrapper with .fft()/.ifft() API matching VkFFTApp.

    This allows CUDAKernels.linear_step to call plan.fft(A, A) / plan.ifft(A, A)
    without knowing which FFT library is behind it.
    """

    __slots__ = ("_axes", "_plan")

    def __init__(self, a: Any, axes: tuple) -> None:
        self._plan = get_fft_plan(a, axes=axes, value_type="C2C")
        self._axes = axes

    def fft(self, a: Any, out: Any) -> Any:
        """Forward FFT, in-place when out is a."""
        return _cufft.fftn(a, axes=self._axes, overwrite_x=(out is a), plan=self._plan)

    def ifft(self, a: Any, out: Any) -> Any:
        """Inverse FFT (normalized by 1/N), in-place when out is a."""
        return _cufft.ifftn(
            a, axes=self._axes, overwrite_x=(out is a), plan=self._plan
        )

    def ifft_unnorm(self, a: Any, out: Any) -> Any:
        """Inverse FFT without 1/N normalization (raw cuFFT).

        Used by linear_step where 1/N is absorbed into the propagator.
        """
        self._plan.fft(a, out, cp.cuda.cufft.CUFFT_INVERSE)
        return out


class CUPYBackend(Backend):
    """CUPY backend using CuPy and cuFFT."""

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

    def build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build cuFFT plan for CUDA.

        Returns
        -------
        list
            List containing _CuFFTPlan instance (for consistency with CPU backend)

        """
        A = cp.zeros(shape, dtype=dtype)
        plan = _CuFFTPlan(A, axes=axes)
        return [plan]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        return plan[0].fft(array, array)

    def ifft(self, array: Any, plan: list) -> Any:
        """Perform inverse FFT."""
        return plan[0].ifft(array, array)

    @property
    def kernels(self) -> Any:
        """Return CUDA C kernels (--use_fast_math, with broadcasting fallback)."""
        if not hasattr(self, "_cuda_kernels"):
            from ..kernels.cupy_kernels import CUDAKernels

            self._cuda_kernels = CUDAKernels()
        return self._cuda_kernels

    def supports_double_precision(self) -> bool:
        """CUDA GPUs typically support double precision."""
        return True
