"""MLX backend implementation for Apple Silicon GPU acceleration."""

from typing import Any

import numpy as np

from ..utils import __MLX_AVAILABLE__
from .backend import Backend

if not __MLX_AVAILABLE__:
    raise ImportError("MLX is not available - cannot import MLXBackend")

import mlx.core as mx

# MLX dtype mapping from numpy dtypes
_NUMPY_TO_MLX_DTYPE = {
    np.dtype(np.float32): mx.float32,
    np.dtype(np.float64): mx.float32,  # downcast
    np.dtype(np.complex64): mx.complex64,
    np.dtype(np.complex128): mx.complex64,  # downcast
}


class MLXBackend(Backend):
    """MLX backend for Apple Silicon GPU acceleration."""

    @property
    def name(self) -> str:
        return "MLX"

    def allocate_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate complex array on MLX device."""
        mx_dtype = _NUMPY_TO_MLX_DTYPE.get(np.dtype(dtype), mx.complex64)
        return mx.zeros(shape, dtype=mx_dtype)

    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate real array on MLX device."""
        mx_dtype = _NUMPY_TO_MLX_DTYPE.get(np.dtype(dtype), mx.float32)
        return mx.zeros(shape, dtype=mx_dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from MLX device to CPU."""
        mx.eval(array)
        return np.array(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to MLX device.

        Downcasts float64/complex128 to float32/complex64 since MLX
        does not support double precision.
        """
        if array.dtype == np.complex128:
            array = array.astype(np.complex64)
        elif array.dtype == np.float64:
            array = array.astype(np.float32)
        return mx.array(array)

    def build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build FFT plan for MLX (stores axes tuple).

        MLX FFT does not use plan objects. We store the axes for use
        in fft/ifft calls.

        Returns
        -------
        list
            List containing the axes tuple.

        """
        return [axes]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        axes = plan[0]
        result = mx.fft.fftn(array, axes=axes)
        mx.eval(result)
        return result

    def ifft(self, array: Any, plan: list) -> Any:
        """Perform inverse FFT."""
        axes = plan[0]
        result = mx.fft.ifftn(array, axes=axes)
        mx.eval(result)
        return result

    @property
    def kernels(self) -> Any:
        """Return MLX kernels module."""
        from ..kernels import mlx_kernels

        return mlx_kernels

    def supports_double_precision(self) -> bool:
        """MLX does not support double precision."""
        return False
