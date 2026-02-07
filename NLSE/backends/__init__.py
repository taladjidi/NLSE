"""Backend abstraction layer for NLSE solver.

Provides a unified interface across CPU, GPU (CuPy), OpenCL, and Metal backends.
"""

from __future__ import annotations

import types
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import signal


class FFTPlan:
    """Unified FFT plan interface wrapping backend-specific plans."""

    def __init__(self, fft_obj: Any, ifft_obj: Any = None) -> None:
        self._fft = fft_obj
        self._ifft = ifft_obj

    def fft(self, A: Any) -> None:
        raise NotImplementedError

    def ifft(self, A: Any) -> None:
        raise NotImplementedError


class Backend(ABC):
    """Unified interface for all compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (``'CPU'``, ``'CUPY'``, ``'CL'``, ``'Metal'``)."""

    @property
    @abstractmethod
    def kernels(self) -> types.ModuleType:
        """Return the kernel module (``square_mod``, ``nl_prop``, etc.)."""

    @abstractmethod
    def allocate_pair(
        self, shape: tuple[int, ...], complex_dtype: np.dtype[Any]
    ) -> tuple[Any, Any]:
        """Allocate ``(A, A_sq)`` arrays: complex field + real modulus squared.

        Returns device-ready arrays (pyfftw-aligned for CPU, cupy for GPU, etc.).
        """

    @abstractmethod
    def to_device(self, arr: np.ndarray) -> Any:
        """Transfer numpy array to device memory. No-op for CPU."""

    @abstractmethod
    def to_host(self, arr: Any) -> np.ndarray:
        """Transfer device array to numpy. No-op for CPU."""

    @abstractmethod
    def is_device_array(self, arr: Any) -> bool:
        """Check if *arr* is a device array (non-numpy)."""

    @abstractmethod
    def build_fft_plan(self, A: Any, axes: tuple[int, ...]) -> FFTPlan:
        """Build FFT plan.

        Returns a plan object with ``.fft()`` and ``.ifft()`` methods.
        """

    @abstractmethod
    def fft(self, plan: FFTPlan, A: Any) -> None:
        """In-place forward FFT."""

    @abstractmethod
    def ifft(self, plan: FFTPlan, A: Any) -> None:
        """In-place inverse FFT."""

    @abstractmethod
    def sum(self, arr: Any, axis: tuple[int, ...] | int | None = None) -> Any:
        """Reduce-sum (needed for normalization)."""

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """Scalar or element-wise sqrt."""

    def convolution(
        self,
        A: Any,
        kernel: Any,
        mode: str,
        axes: tuple[int, ...],
    ) -> Any:
        """Spatial convolution. Default uses scipy."""
        return signal.oaconvolve(A, kernel, mode=mode, axes=axes)  # type: ignore[call-overload]


def get_backend(name: str) -> Backend:
    """Factory: ``'CPU'`` | ``'CUPY'`` | ``'CL'`` | ``'Metal'`` -> Backend instance."""
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
