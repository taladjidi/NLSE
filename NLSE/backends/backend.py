"""Abstract base class for NLSE backends."""

import contextlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Timing:
    """How long a propagation took.

    ``device`` is the time the device itself reports, where it can: a wall
    clock measures a queue being filled rather than the work being done.
    """

    wall: float = 0.0
    device: float | None = None

    def __str__(self) -> str:
        """Return the line ``out_field`` prints when verbose."""
        if self.device is None:
            return f"Time spent to solve : {self.wall} s (CPU)"
        return f"Time spent to solve : {self.device} s (GPU) / {self.wall} s (CPU)"


def _dtype_key(dtype: Any) -> str:
    """Return a stable cache key for a dtype.

    Normalised through numpy where possible, so ``np.complex64`` and
    ``np.dtype("complex64")`` map to one entry. MLX passes its own dtype
    objects, which numpy cannot interpret, so those fall back to their
    string form.

    Parameters
    ----------
    dtype : Any
        A numpy dtype, a numpy scalar type, or a backend's own dtype.

    Returns
    -------
    str
        Key identifying the dtype.
    """
    try:
        return np.dtype(dtype).str
    except TypeError:
        return str(dtype)


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

    # The field is normalized on the host: the reduction is done in numpy and
    # the result sent back. For backends whose array type cannot do the
    # reduction in place, or does it more slowly than the round trip costs.
    normalizes_on_host = False

    # Operations return a freshly allocated array rather than writing into a
    # destination, so staging through a pre-allocated scratch buffer costs a
    # copy and saves nothing.
    #
    # Not a preference: the two routes are not interchangeable either way. A
    # backend leaving this False may have its FFT plan bound to a particular
    # buffer, as pyfftw's is, and returns NaN if handed another array.
    operations_allocate_output = False

    @property
    def convolution(self) -> Callable | None:
        """Return this backend's overlap-add convolution, or None.

        Non-local interaction is a convolution between the field's intensity
        and the non-local kernel. A backend without one cannot run a non-local
        simulation, which is the whole of what ``nl_length > 0`` requires, so
        this doubles as the capability check.
        """
        return None

    def synchronize(self, array: Any = None) -> None:
        """Block until work already submitted has finished.

        A no-op where submission is execution. Elsewhere it is what makes a
        wall-clock measurement mean anything, and MLX needs the array itself
        since its graph is lazy.

        Parameters
        ----------
        array : Any, optional
            The array whose value is needed.
        """

    @contextlib.contextmanager
    def timed(self) -> Iterator[Timing]:
        """Time the enclosed region, filling in the Timing it yields.

        Synchronize before leaving the block, or the wall time is the time
        taken to queue the work.

        Yields
        ------
        Timing
            Filled in on exit.
        """
        timing = Timing()
        start = time.perf_counter()
        try:
            yield timing
        finally:
            timing.wall = time.perf_counter() - start

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
    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> Any:
        """Construct an FFT plan. Called once per distinct transform.

        Parameters
        ----------
        shape : tuple
            Array shape
        axes : tuple
            Axes to transform
        dtype : np.dtype
            Data type
        array : np.ndarray or None
            An array of that shape and dtype, when the backend can plan
            better with a concrete one.

        Returns
        -------
        Any
            FFT plan object

        """
        pass

    def build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> Any:
        """Return the FFT plan for this transform, building it once.

        A plan depends only on the shape, the axes and the dtype, so it is
        built once and reused. Planning is expensive: the CPU backend
        reloads FFTW wisdom from disk and times a roundtrip to validate it,
        and VkFFT compiles its own kernels.

        The cache lives on the backend, and backends are shared per name, so
        a parameter sweep that builds a solver per point plans once rather
        than once per point.

        Parameters
        ----------
        shape : tuple
            Array shape
        axes : tuple
            Axes to transform
        dtype : np.dtype
            Data type
        array : np.ndarray or None
            An array of that shape and dtype, used only when the plan has to
            be built.

        Returns
        -------
        Any
            FFT plan object

        """
        key = (tuple(shape), tuple(axes), _dtype_key(dtype))
        if key not in self._fft_plan_cache:
            self._fft_plan_cache[key] = self._build_fft(shape, axes, dtype, array)
        return self._fft_plan_cache[key]

    @property
    def _fft_plan_cache(self) -> dict:
        """Return this backend's plan cache, creating it on first use.

        A property rather than an __init__ assignment because the backends
        do not share a constructor.
        """
        if not hasattr(self, "_fft_plans"):
            self._fft_plans: dict = {}
        return self._fft_plans

    def clear_fft_plans(self) -> None:
        """Drop every cached FFT plan.

        Only useful to a test, or to reclaim the device memory the plans
        hold when a long-lived process changes grid size for good.
        """
        self._fft_plan_cache.clear()

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
