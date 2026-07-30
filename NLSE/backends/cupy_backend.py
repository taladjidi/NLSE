"""CUPY backend implementation."""

import contextlib
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from ..utils import __CUPY_AVAILABLE__
from .backend import Backend, Timing

if not __CUPY_AVAILABLE__:
    raise ImportError("CuPy is not available - cannot import CUPYBackend")

import cupy as cp
import cupyx.scipy.fft as _cufft
import cupyx.scipy.signal as _cusignal
from cupyx.scipy.fftpack import get_fft_plan


class _CuFFTPlan:
    """cuFFT plan wrapper with .fft()/.ifft() API matching VkFFTApp.

    This allows CUDAKernels.linear_step to call plan.fft(A, A) / plan.ifft(A, A)
    without knowing which FFT library is behind it.
    """

    __slots__ = ("_axes", "_plan")

    def __init__(self, a: Any, axes: tuple) -> None:
        self._plan = get_fft_plan(a, axes=axes, value_type="C2C")
        self._axes = axes

    def fft(self, a: Any, out: Any) -> Any:
        """Forward FFT (unnormalized, raw cuFFT). Graph-capture safe."""
        self._plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
        return out

    def ifft(self, a: Any, out: Any) -> Any:
        """Inverse FFT (normalized by 1/N), in-place when out is a."""
        return _cufft.ifftn(a, axes=self._axes, overwrite_x=(out is a), plan=self._plan)

    def ifft_unnorm(self, a: Any, out: Any) -> Any:
        """Inverse FFT without 1/N normalization (raw cuFFT).

        Used by linear_step where 1/N is absorbed into the propagator.
        """
        self._plan.fft(a, out, cp.cuda.cufft.CUFFT_INVERSE)
        return out


class CUPYBackend(Backend):
    """CUPY backend using CuPy and cuFFT.

    Exposes no fused *split* step: execute_loop captures the whole
    propagation step into a CUDA graph, which removes the launch overhead
    that fusion exists to amortize on the other GPU backends.

    The coupled and RK4 fusions below are declared all the same, because
    what they save is not launches but traffic. Reaching a one-component
    kernel with a two-component field means copying each component out and
    the result back, and starting an RK4 stage in place means copying the
    field into the stage buffer first. A CUDA graph replays those copies as
    faithfully as it replays the arithmetic.
    """

    has_linear_step = True
    supports_unnormalized_ifft = True
    broadcasts_parameters_natively = True
    has_fused_rk4_rhs = True
    has_fused_rk4_stage_update = True
    has_fused_rk4_final_update = True
    has_fused_rk4_stage = True
    has_fused_coupled_split_step = True
    has_fused_coupled_rk4_rhs = True

    @property
    def name(self) -> str:
        return "CUPY"

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate array on GPU."""
        return cp.zeros(shape, dtype=dtype)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate real array on GPU."""
        return cp.zeros(shape, dtype=dtype)

    @property
    def convolution(self):
        """Return CuPy's overlap-add convolution."""
        return _cusignal.oaconvolve

    def synchronize(self, array=None) -> None:
        """Wait for the null stream, which is where the kernels are queued."""
        cp.cuda.Stream.null.synchronize()

    @contextlib.contextmanager
    def timed(self):
        """Time with CUDA events as well as a wall clock.

        A wall clock times the queueing; the events time the work.

        Yields
        ------
        Timing
            Filled in on exit, ``device`` included.
        """
        timing = Timing()
        start_gpu, end_gpu = cp.cuda.Event(), cp.cuda.Event()
        start_gpu.record()
        start = time.perf_counter()
        try:
            yield timing
        finally:
            timing.wall = time.perf_counter() - start
            end_gpu.record()
            end_gpu.synchronize()
            timing.device = cp.cuda.get_elapsed_time(start_gpu, end_gpu) * 1e-3

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from GPU to CPU."""
        return cp.asnumpy(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to GPU."""
        return cp.asarray(array, dtype=array.dtype)

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build cuFFT plan for CUDA.

        Returns
        -------
        list
            List containing _CuFFTPlan instance (for consistency with CPU backend)

        """
        A = cp.zeros(shape, dtype=dtype)
        plan = _CuFFTPlan(A, axes=axes)
        return [plan]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        return plan[0].fft(array, array)

    def ifft(self, array: Any, plan: list, normalize: bool = True) -> Any:
        """Perform inverse FFT."""
        if normalize:
            return plan[0].ifft(array, array)
        return plan[0].ifft_unnorm(array, array)

    def norm(self, array: Any) -> float:
        """Reduce on the GPU; only the scalar comes back."""
        return float(cp.linalg.norm(array))

    def exp(self, array: Any) -> Any:
        """Exponentiate without leaving this backend."""
        return cp.exp(array)

    def sum(self, array: Any) -> float:
        """Reduce without leaving this backend."""
        return float(cp.sum(array))

    @property
    def kernels(self) -> Any:
        """Return CUDA C kernels (--use_fast_math, with broadcasting fallback)."""
        if not hasattr(self, "_cuda_kernels"):
            from ..kernels.cupy_kernels import CUDAKernels

            self._cuda_kernels = CUDAKernels()
        return self._cuda_kernels

    def supports_double_precision(self) -> bool:
        """CUDA GPUs typically support double precision."""
        return True

    def execute_loop(self, step_fn: Any, n_iters: int) -> None:
        """Execute step_fn using CUDA graph capture/replay.

        One warmup iteration runs normally to prime cuFFT plans and
        JIT-compiled kernels. Then one iteration is captured into a
        CUDA graph, and the remaining iterations replay that graph.

        Parameters
        ----------
        step_fn : callable
            One propagation step (in-place on pre-allocated GPU arrays).
        n_iters : int
            Total number of iterations.

        """
        if n_iters < 3:
            for _ in range(n_iters):
                step_fn()
            return

        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            # Warmup: execute one step to prime cuFFT / lazy kernels
            step_fn()
            # Capture: record one step (operations are NOT executed)
            stream.begin_capture()
            step_fn()
            graph = stream.end_capture()
            # Replay the captured graph for the remaining iterations
            for _ in range(n_iters - 1):
                graph.launch(stream)
        stream.synchronize()
