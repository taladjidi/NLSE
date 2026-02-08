"""OpenCL backend implementation."""

from typing import Any

import numpy as np

from ..utils import __PYOPENCL_AVAILABLE__, __PYOPENCL_DOUBLE_SUPPORT__
from .backend import Backend

if __PYOPENCL_AVAILABLE__:
    import pyopencl as cl
    from pyopencl import array as cla
    from pyvkfft.opencl import VkFFTApp


class OpenCLBackend(Backend):
    """OpenCL backend using PyOpenCL and VkFFT."""

    def __init__(self):
        if not __PYOPENCL_AVAILABLE__:
            raise ImportError("PyOpenCL is not available")
        self._context = cl.create_some_context(interactive=False)
        self._queue = cl.CommandQueue(self._context)
        self._vkfft_apps = {}

    @property
    def name(self) -> str:
        return "CL"

    @property
    def queue(self) -> Any:
        """Get OpenCL command queue."""
        return self._queue

    @property
    def context(self) -> Any:
        """Get OpenCL context."""
        return self._context

    def allocate_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate array on OpenCL device."""
        return cla.zeros(self._queue, shape, dtype)

    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate real array on OpenCL device."""
        return cla.zeros(self._queue, shape, dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from OpenCL device to CPU."""
        return array.get()

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to OpenCL device."""
        return cla.to_device(self._queue, array)

    def build_fft(self, shape: tuple, axes: tuple, dtype: np.dtype) -> list:
        """Build VkFFT app for OpenCL.

        Returns:
            List containing VkFFTApp instance (for consistency with CPU backend)
        """
        A = cla.zeros(self._queue, shape, dtype)
        app = VkFFTApp(A.shape, A.dtype, queue=self._queue, axes=axes, ndim=len(axes))
        return [app]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        return plan[0].fft(array, array)

    def ifft(self, array: Any, plan: list) -> Any:
        """Perform inverse FFT."""
        return plan[0].ifft(array, array)

    @property
    def kernels(self) -> Any:
        """Return OpenCL kernels module."""
        from ..kernels import cl as kernels_cl
        return kernels_cl

    def supports_double_precision(self) -> bool:
        """Check OpenCL device double precision support."""
        return __PYOPENCL_DOUBLE_SUPPORT__
