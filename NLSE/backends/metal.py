"""Metal backend for Apple Silicon."""

import numpy as np
from scipy import signal

from . import Backend, FFTPlan


class MetalFFTPlanWrapper(FFTPlan):
    """FFT plan wrapping Metal's numpy-based FFT."""

    def __init__(self, metal_plan):
        self._plan = metal_plan

    def fft(self, A):
        self._plan.fft(A, A)

    def ifft(self, A):
        self._plan.ifft(A, A)


class MetalBackend(Backend):

    @property
    def name(self) -> str:
        return "Metal"

    @property
    def kernels(self):
        from ..kernels import metal as k

        return k

    def allocate_pair(self, shape, complex_dtype):
        from ..kernels.metal import MetalArray

        A_np = np.zeros(shape, dtype=complex_dtype)
        A_sq_np = np.zeros(shape, dtype=np.float32)
        A = MetalArray.from_numpy(A_np)
        A_sq = MetalArray.from_numpy(A_sq_np)
        return A, A_sq

    def to_device(self, arr):
        from ..kernels.metal import MetalArray

        if isinstance(arr, np.ndarray):
            # Cast to complex64 for Metal compute shaders
            return MetalArray.from_numpy(
                np.ascontiguousarray(arr).astype(np.complex64)
            )
        return arr

    def to_host(self, arr):
        from ..kernels.metal import MetalArray

        if isinstance(arr, MetalArray):
            return arr.get()
        return arr

    def is_device_array(self, arr):
        from ..kernels.metal import MetalArray

        return isinstance(arr, MetalArray)

    def build_fft_plan(self, A, axes):
        from ..kernels.metal import MetalFFTPlan

        plan = MetalFFTPlan(A.shape, ndim=len(axes))
        return MetalFFTPlanWrapper(plan)

    def fft(self, plan, A):
        plan.fft(A)

    def ifft(self, plan, A):
        plan.ifft(A)

    def sum(self, arr, axis=None):
        from ..kernels.metal import MetalArray

        if isinstance(arr, MetalArray):
            return arr.get().sum(axis=axis)
        return np.sum(arr, axis=axis)

    def sqrt(self, x):
        return np.sqrt(x)

    def convolution(self, A, kernel, mode, axes):
        from ..kernels.metal import MetalArray

        a = A.get() if isinstance(A, MetalArray) else A
        k = kernel.get() if isinstance(kernel, MetalArray) else kernel
        return signal.oaconvolve(a, k, mode=mode, axes=axes)
