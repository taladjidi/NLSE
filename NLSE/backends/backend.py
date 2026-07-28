"""Abstract base class for NLSE backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Backend(ABC):
    """Abstract base class for compute backends.

    Besides the array/FFT interface, a backend declares which optional
    kernel entry points its ``kernels`` object provides. The solvers branch
    on these flags to pick a fused fast path. Declaring them here keeps the
    solver/kernel contract in one place: previously the solvers probed for
    the methods with ``hasattr``, so the contract existed only implicitly
    and differed silently between backends.

    Every flag defaults to False, meaning "take the generic path". A
    backend that sets a flag to True must provide the matching method on
    its ``kernels`` object with the signature documented below.
    """

    # kernels.linear_step(A, propagator, plan, unnorm_ifft=False)
    # Fused FFT -> propagator multiply -> IFFT.
    has_linear_step = False

    # The inverse FFT can skip its 1/N normalization, so the factor can be
    # folded into the propagator once instead of costing a pass per step.
    # Requires kernels.linear_step to honour unnorm_ifft.
    supports_unnormalized_ifft = False

    # kernels.split_step_fused(
    #     A, propagator, V_scaled, dz, alpha, g, Isat, precision, plan,
    #     unnorm_ifft=False)
    # A whole single-component split step without returning to Python.
    has_fused_split_step = False

    # kernels.rk4_rhs_fused(
    #     A_in, k, V_scaled, propagator, plan, alpha, g, Isat,
    #     unnorm_ifft=False)
    has_fused_rk4_rhs = False

    # kernels.split_step_rk4_fused(
    #     A, propagator, V_scaled, dz, alpha, g, Isat, plan)
    # A whole RK4 step (all four stages) in one call.
    has_fused_rk4_step = False

    # kernels.rk4_set_and_axpy / kernels.rk4_acc_and_axpy
    # Combine the accumulate and axpy of an RK4 stage into one launch.
    has_fused_rk4_stage_update = False

    # kernels.split_step_coupled_fused(
    #     A, propagator, V1_scaled, V2_scaled, dz, alpha1, alpha2,
    #     g11, g12, g22, Isat1, Isat2, precision, plan, omega=None,
    #     unnorm_ifft=False)
    has_fused_coupled_split_step = False

    # kernels.rk4_rhs_coupled_fused(
    #     A_in, k, V1_scaled, V2_scaled, propagator, plan, alpha1, alpha2,
    #     g11, g12, g22, Isat1, Isat2, unnorm_ifft=False)
    has_fused_coupled_rk4_rhs = False

    # A batched physical parameter can be handed to the kernels as a device
    # array and broadcast inside them (CUPY's cp.fuse kernels, MLX's traced
    # graphs). Backends that leave this False take one simulation's scalar
    # value per launch instead, so their batched parameters must stay on the
    # host: CPU loops in _broadcast_batch, CL loops with global_offset.
    broadcasts_parameters_natively = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""
        pass

    @property
    def is_device_backend(self) -> bool:
        """Whether this backend runs on a device (GPU/accelerator)."""
        return self.name in ("CUPY", "CL", "MLX")

    @abstractmethod
    def allocate_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate a field array on this backend.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : np.dtype
            Data type

        Returns
        -------
        Any
            Array allocated on the appropriate device

        """
        pass

    @abstractmethod
    def allocate_real_field(self, shape: tuple, dtype: np.dtype) -> Any:
        """Allocate a real-valued field array.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : np.dtype
            Data type (should be real)

        Returns
        -------
        Any
            Array allocated on the appropriate device

        """
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert array to numpy on CPU.

        Parameters
        ----------
        array : Any
            Array on device

        Returns
        -------
        np.ndarray
            Numpy array on CPU

        """
        pass

    @abstractmethod
    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert numpy array to device array.

        Parameters
        ----------
        array : np.ndarray
            Numpy array

        Returns
        -------
        Any
            Array on device

        """
        pass

    @abstractmethod
    def build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> Any:
        """Build FFT plan for this backend.

        Parameters
        ----------
        shape : tuple
            Array shape
        axes : tuple
            Axes to transform
        dtype : np.dtype
            Data type

        Returns
        -------
        Any
            FFT plan object

        """
        pass

    @abstractmethod
    def fft(self, array: Any, plan: Any) -> Any:
        """Perform forward FFT.

        Parameters
        ----------
        array : Any
            Input array
        plan : Any
            FFT plan

        Returns
        -------
        Any
            Transformed array

        """
        pass

    @abstractmethod
    def ifft(self, array: Any, plan: Any) -> Any:
        """Perform inverse FFT.

        Parameters
        ----------
        array : Any
            Input array
        plan : Any
            FFT plan

        Returns
        -------
        Any
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

        Returns
        -------
        bool
            True if double precision is supported

        """
        pass

    def execute_loop(self, step_fn: Any, n_iters: int) -> None:
        """Execute step_fn n_iters times, optimally for this backend.

        Backends may override this to use hardware-specific acceleration
        (e.g. CUDA graph capture/replay).

        Parameters
        ----------
        step_fn : callable
            Function that performs one propagation step (in-place on
            pre-allocated arrays).
        n_iters : int
            Number of iterations to execute.

        """
        for _ in range(n_iters):
            step_fn()
