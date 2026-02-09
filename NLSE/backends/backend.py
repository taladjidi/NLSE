"""Abstract base class for NLSE backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Abstract base class for compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @abstractmethod
    def allocate_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate a field array on this backend.

        Args:
            shape: Shape of the array
            dtype: Data type

        Returns:
            Array allocated on the appropriate device
        """
        pass

    @abstractmethod
    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate a real-valued field array.

        Args:
            shape: Shape of the array
            dtype: Data type (should be real)

        Returns:
            Array allocated on the appropriate device
        """
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert array to numpy on CPU.

        Args:
            array: Array on device

        Returns:
            Numpy array on CPU
        """
        pass

    @abstractmethod
    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert numpy array to device array.

        Args:
            array: Numpy array

        Returns:
            Array on device
        """
        pass

    @abstractmethod
    def build_fft(
        self, shape: tuple, axes: tuple, dtype: np.dtype, array: np.ndarray | None = None
    ) -> Any:
        """Build FFT plan for this backend.

        Args:
            shape: Array shape
            axes: Axes to transform
            dtype: Data type

        Returns:
            FFT plan object
        """
        pass

    @abstractmethod
    def fft(self, array: Any, plan: Any) -> Any:
        """Perform forward FFT.

        Args:
            array: Input array
            plan: FFT plan

        Returns:
            Transformed array
        """
        pass

    @abstractmethod
    def ifft(self, array: Any, plan: Any) -> Any:
        """Perform inverse FFT.

        Args:
            array: Input array
            plan: FFT plan

        Returns:
            Transformed array
        """
        pass

    @property
    @abstractmethod
    def kernels(self) -> Any:
        """Get kernel module for this backend."""
        pass

    @abstractmethod
    def supports_double_precision(self) -> bool:
        """Check if backend supports double precision.

        Returns:
            True if double precision is supported
        """
        pass
