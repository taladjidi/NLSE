"""CPU backend using pyfftw + numba."""

import multiprocessing
import pickle

import numpy as np
import pyfftw
from scipy import signal

from . import Backend, FFTPlan


class CPUFFTPlan(FFTPlan):
    """FFT plan wrapping pyfftw FFTW objects."""

    def __init__(self, plan_fft, plan_ifft):
        self._fft = plan_fft
        self._ifft = plan_ifft

    def fft(self, A):
        self._fft(input_array=A, output_array=A)

    def ifft(self, A):
        self._ifft(input_array=A, output_array=A, normalise_idft=True)


class CPUBackend(Backend):
    @property
    def name(self) -> str:
        return "CPU"

    @property
    def kernels(self):
        from ..kernels import cpu as k

        return k

    def allocate_pair(self, shape, complex_dtype):
        A = pyfftw.zeros_aligned(shape, dtype=complex_dtype, n=pyfftw.simd_alignment)
        real_dtype = np.zeros(1, dtype=complex_dtype).real.dtype
        A_sq = np.zeros(shape, dtype=real_dtype)
        return A, A_sq

    def to_device(self, arr):
        return arr

    def to_host(self, arr):
        return arr

    def is_device_array(self, arr):
        return False

    def build_fft_plan(self, A, axes):
        try:
            with open("fft.wisdom", "rb") as file:
                wisdom = pickle.load(file)
                pyfftw.import_wisdom(wisdom)
        except FileNotFoundError:
            print("No FFT wisdom found, starting over ...")
        plan_fft = pyfftw.FFTW(
            A,
            A,
            direction="FFTW_FORWARD",
            threads=multiprocessing.cpu_count(),
            axes=axes,
        )
        plan_ifft = pyfftw.FFTW(
            A,
            A,
            direction="FFTW_BACKWARD",
            threads=multiprocessing.cpu_count(),
            axes=axes,
        )
        with open("fft.wisdom", "wb") as file:
            wisdom = pyfftw.export_wisdom()
            pickle.dump(wisdom, file)
        return CPUFFTPlan(plan_fft, plan_ifft)

    def fft(self, plan, A):
        plan.fft(A)

    def ifft(self, plan, A):
        plan.ifft(A)

    def sum(self, arr, axis=None):
        return np.sum(arr, axis=axis)

    def sqrt(self, x):
        return np.sqrt(x)

    def convolution(self, A, kernel, mode, axes):
        return signal.oaconvolve(A, kernel, mode=mode, axes=axes)
