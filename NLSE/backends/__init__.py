"""Backend abstraction layer for NLSE solver.

Provides a unified interface across CPU, GPU (CuPy), OpenCL, and Metal backends.
"""

from abc import ABC, abstractmethod

from scipy import signal


class FFTPlan:
    """Unified FFT plan interface wrapping backend-specific plans."""

    def __init__(self, fft_obj, ifft_obj=None):
        self._fft = fft_obj
        self._ifft = ifft_obj

    def fft(self, A):
        raise NotImplementedError

    def ifft(self, A):
        raise NotImplementedError


class Backend(ABC):
    """Unified interface for all compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name ('CPU', 'CUPY', 'CL', 'Metal')."""

    @property
    @abstractmethod
    def kernels(self):
        """Return the kernel module (square_mod, nl_prop, etc.)."""

    @abstractmethod
    def allocate_pair(self, shape, complex_dtype):
        """Allocate (A, A_sq) arrays: complex field + real modulus squared.

        Returns device-ready arrays (pyfftw-aligned for CPU, cupy for GPU, etc.).
        """

    @abstractmethod
    def to_device(self, arr):
        """Transfer numpy array to device memory. No-op for CPU."""

    @abstractmethod
    def to_host(self, arr):
        """Transfer device array to numpy. No-op for CPU."""

    @abstractmethod
    def is_device_array(self, arr):
        """Check if arr is a device array (non-numpy)."""

    @abstractmethod
    def build_fft_plan(self, A, axes):
        """Build FFT plan. Returns a plan object with .fft() and .ifft() methods."""

    @abstractmethod
    def fft(self, plan, A):
        """In-place forward FFT."""

    @abstractmethod
    def ifft(self, plan, A):
        """In-place inverse FFT."""

    @abstractmethod
    def sum(self, arr, axis=None):
        """Reduce-sum (needed for normalization)."""

    @abstractmethod
    def sqrt(self, x):
        """Scalar or element-wise sqrt."""

    def convolution(self, A, kernel, mode, axes):
        """Spatial convolution. Default uses scipy."""
        return signal.oaconvolve(A, kernel, mode=mode, axes=axes)


def get_backend(name: str) -> Backend:
    """Factory: 'CPU' | 'CUPY' | 'CL' | 'Metal' -> Backend instance."""
    match name:
        case "CPU":
            from .cpu import CPUBackend

            return CPUBackend()
        case "CUPY":
            from .cupy import CupyBackend

            return CupyBackend()
        case "CL":
            from .cl import CLBackend

            return CLBackend()
        case "Metal":
            from .metal import MetalBackend

            return MetalBackend()
        case _:
            raise ValueError(f"Unknown backend: {name}")
