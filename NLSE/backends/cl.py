"""OpenCL backend using PyOpenCL + pyvkfft."""

import numpy as np
import pyopencl as cl
from pyopencl import array as cla
from pyopencl import clmath
from pyvkfft.opencl import VkFFTApp as VkFFTApp_cl

from . import Backend, FFTPlan


class CLFFTPlan(FFTPlan):
    """FFT plan wrapping VkFFT OpenCL plan."""

    def __init__(self, vkfft_app):
        self._plan = vkfft_app

    def fft(self, A):
        self._plan.fft(A, A)

    def ifft(self, A):
        self._plan.ifft(A, A)


class CLBackend(Backend):
    def __init__(self):
        self._cl_queue = cl.CommandQueue(cl.create_some_context(interactive=False))

    @property
    def cl_queue(self):
        return self._cl_queue

    @property
    def name(self) -> str:
        return "CL"

    @property
    def kernels(self):
        from ..kernels import cl as k

        return k

    def allocate_pair(self, shape, complex_dtype):
        real_dtype = np.zeros(1, dtype=complex_dtype).real.dtype
        A = cla.zeros(self._cl_queue, shape, complex_dtype)
        A_sq = cla.zeros(self._cl_queue, shape, real_dtype)
        return A, A_sq

    def to_device(self, arr):
        if isinstance(arr, np.ndarray):
            return cla.to_device(self._cl_queue, arr)
        return arr

    def to_host(self, arr):
        if isinstance(arr, cla.Array):
            return arr.get()
        return arr

    def is_device_array(self, arr):
        return isinstance(arr, cla.Array)

    def build_fft_plan(self, A, axes):
        plan = VkFFTApp_cl(
            A.shape,
            A.dtype,
            ndim=len(axes),
            queue=self._cl_queue,
            inplace=True,
            norm=1,
            tune=True,
        )
        return CLFFTPlan(plan)

    def fft(self, plan, A):
        plan.fft(A)

    def ifft(self, plan, A):
        plan.ifft(A)

    def sum(self, arr, axis=None):
        if isinstance(arr, cla.Array):
            return cla.sum(arr, dtype=arr.dtype, queue=self._cl_queue)
        return np.sum(arr, axis=axis)

    def sqrt(self, x):
        if isinstance(x, cla.Array):
            return clmath.sqrt(x)
        return np.sqrt(x)
