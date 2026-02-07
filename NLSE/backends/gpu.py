"""GPU backend using CuPy + pyvkfft."""

import cupy as cp
import cupyx.scipy.signal as signal_cp
import numpy as np
from pyvkfft.cuda import VkFFTApp as VkFFTApp_cuda

from . import Backend, FFTPlan


class GPUFFTPlan(FFTPlan):
    """FFT plan wrapping VkFFT CUDA plan."""

    def __init__(self, vkfft_app):
        self._plan = vkfft_app

    def fft(self, A):
        self._plan.fft(A, A)

    def ifft(self, A):
        self._plan.ifft(A, A)


class GPUBackend(Backend):

    @property
    def name(self) -> str:
        return "GPU"

    @property
    def kernels(self):
        from ..kernels import gpu as k

        return k

    def allocate_pair(self, shape, complex_dtype):
        A = cp.zeros(shape, dtype=complex_dtype)
        real_dtype = np.zeros(1, dtype=complex_dtype).real.dtype
        A_sq = cp.zeros(shape, dtype=real_dtype)
        return A, A_sq

    def to_device(self, arr):
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr

    def to_host(self, arr):
        if isinstance(arr, cp.ndarray):
            return arr.get()
        return arr

    def is_device_array(self, arr):
        return isinstance(arr, cp.ndarray)

    def build_fft_plan(self, A, axes):
        stream = cp.cuda.get_current_stream()
        plan = VkFFTApp_cuda(
            A.shape,
            A.dtype,
            ndim=len(axes),
            stream=stream,
            inplace=True,
            norm=1,
            tune=True,
        )
        return GPUFFTPlan(plan)

    def fft(self, plan, A):
        plan.fft(A)

    def ifft(self, plan, A):
        plan.ifft(A)

    def sum(self, arr, axis=None):
        return cp.sum(arr, axis=axis)

    def sqrt(self, x):
        return cp.sqrt(x)

    def convolution(self, A, kernel, mode, axes):
        return signal_cp.oaconvolve(A, kernel, mode=mode, axes=axes)
