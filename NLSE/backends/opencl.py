"""OpenCL backend implementation."""

from typing import Any

import numpy as np

from ..utils import __PYOPENCL_AVAILABLE__
from .backend import Backend

if not __PYOPENCL_AVAILABLE__:
    raise ImportError("PyOpenCL is not available - cannot import OpenCLBackend")

import pyopencl as cl
from pyopencl import array as cla
from pyvkfft.opencl import VkFFTApp


class _VkFFTPlan:
    """VkFFT plan wrapper with .fft()/.ifft()/.ifft_unnorm() API.

    Matches _CuFFTPlan so that OpenCLKernels.linear_step can call
    plan.fft(A, A) / plan.ifft(A, A) / plan.ifft_unnorm(A, A)
    without knowing which FFT library is behind it.

    The three apps are built on first use. Each compiles its own kernels,
    and a given run needs at most two: the unnormalized one only where the
    1/N is folded into the propagator, the out-of-place one only for RK4.
    """

    __slots__ = ("_app", "_app_oop", "_app_unnorm", "_spec")

    def __init__(self, shape, dtype, queue, axes, ndim):
        self._spec = {
            "shape": shape,
            "dtype": dtype,
            "queue": queue,
            "axes": axes,
            "ndim": ndim,
        }
        self._app = None
        self._app_unnorm = None
        self._app_oop = None

    def _make(self, **extra):
        """Build a VkFFTApp for this plan's transform."""
        return VkFFTApp(**self._spec, **extra)

    @property
    def app(self):
        """Return the in-place, normalized app."""
        if self._app is None:
            self._app = self._make()
        return self._app

    def fft(self, a, out):
        """Forward FFT (in-place when a is out)."""
        return self.app.fft(a, out)

    def fft_oop(self, src, dest):
        """Out-of-place forward FFT (src is not modified)."""
        if self._app_oop is None:
            self._app_oop = self._make(inplace=False)
        return self._app_oop.fft(src, dest)

    def ifft(self, a, out):
        """Inverse FFT (normalized by 1/N)."""
        return self.app.ifft(a, out)

    def ifft_unnorm(self, a, out):
        """Inverse FFT without 1/N normalization."""
        if self._app_unnorm is None:
            self._app_unnorm = self._make(norm=0)
        return self._app_unnorm.ifft(a, out)


class OpenCLBackend(Backend):
    """OpenCL backend using PyOpenCL and VkFFT.

    Fuses aggressively: OpenCL has no equivalent of CUDA graph capture, so
    each avoided dispatch is a real saving.
    """

    has_linear_step = True
    supports_unnormalized_ifft = True
    has_fused_split_step = True
    has_fused_rk4_rhs = True
    has_fused_rk4_stage_update = True
    has_fused_coupled_split_step = True
    has_fused_coupled_rk4_rhs = True

    def __init__(self):
        self._context = cl.create_some_context(interactive=False)
        self._queue = cl.CommandQueue(self._context)
        self._kernels = None

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

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build VkFFT app for OpenCL.

        Returns
        -------
        list
            List containing VkFFTApp instance (for consistency with CPU backend)

        """
        plan = _VkFFTPlan(shape, dtype, queue=self._queue, axes=axes, ndim=len(axes))
        return [plan]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        return plan[0].fft(array, array)

    def ifft(self, array: Any, plan: list) -> Any:
        """Perform inverse FFT."""
        return plan[0].ifft(array, array)

    @property
    def kernels(self) -> Any:
        """Return OpenCL kernels."""
        if self._kernels is None:
            from ..kernels.cl import OpenCLKernels

            self._kernels = OpenCLKernels(self._context, self._queue)
        return self._kernels

    def supports_double_precision(self) -> bool:
        """Check OpenCL device double precision support.

        Asked of this backend's own device, so importing NLSE does not need
        to create a context to find out.
        """
        return bool(self._context.devices[0].double_fp_config)
