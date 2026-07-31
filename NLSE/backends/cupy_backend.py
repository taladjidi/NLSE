"""CUPY backend implementation."""

import contextlib
import os
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from ..utils import __CUPY_AVAILABLE__, say
from .backend import Backend, Timing

if not __CUPY_AVAILABLE__:
    raise ImportError("CuPy is not available - cannot import CUPYBackend")

import cupy as cp
import cupyx.scipy.fft as _cufft
import cupyx.scipy.signal as _cusignal
from cupyx.scipy.fftpack import get_fft_plan

# Rewrites the block that the store callback reads the propagator pointer
# through. A kernel and not a host-to-device copy, because a step that changes
# propagator part way -- Yoshida's three sub-steps -- would otherwise be
# copying from pageable host memory during a CUDA graph capture, which is not
# allowed. A launch is recorded into the graph and replayed with it.
_WRITE_PROPAGATOR_INFO = cp.RawKernel(
    r"""
extern "C" __global__ void nlse_write_propagator_info(
    unsigned long long* info,
    const unsigned long long values,
    const unsigned long long size
) {
    info[0] = values;
    info[1] = size;
}
""",
    "nlse_write_propagator_info",
)


class _CuFFTPlan:
    """cuFFT plan wrapper with .fft()/.ifft() API matching VkFFTApp.

    This allows CUDAKernels.linear_step to call plan.fft(A, A) / plan.ifft(A, A)
    without knowing which FFT library is behind it.

    Carries a second plan for the forward transform, one whose store callback
    applies the propagator as it writes (see ``fft_propagate`` and
    fft_callbacks.cu). Second rather than only, because a callback fires in
    both directions: a plan that multiplies on the way out would multiply
    again on the way back.

    The second plan is built on first use and only if it can be: the callback
    machinery needs nvrtc and a cuFFT new enough to link LTO-IR, and CuPy
    reaches it for multi-dimensional plans only. Everything here therefore
    reports whether it worked rather than assuming it did, and a run that
    cannot fuse loses the pass and nothing else.

    ``NLSE_FUSE_PROPAGATOR=0`` asks for the unfused path, and is read when the
    plan is built rather than per step, which is the difference between one
    dictionary lookup per run and one per transform. Plans are cached on the
    backend, so a session that changes the variable has to call
    ``backend.clear_fft_plans()`` for it to mean anything.
    """

    __slots__ = (
        "_axes",
        "_bound",
        "_dtype",
        "_fused",
        "_info",
        "_off",
        "_plan",
        "_shape",
    )

    def __init__(self, a: Any, axes: tuple) -> None:
        self._plan = get_fft_plan(a, axes=axes, value_type="C2C")
        self._axes = axes
        self._shape = a.shape
        self._dtype = a.dtype
        self._fused: dict = {}  # batched -> plan carrying that callback
        # Fusion ruled out for this plan, for good: asked against, or tried
        # once and found wanting.
        self._off = os.environ.get("NLSE_FUSE_PROPAGATOR", "1") == "0"
        self._info: Any = None  # block the callback reads the pointer through
        self._bound: tuple | None = None  # (pointer, size) that block holds

    def fft(self, a: Any, out: Any) -> Any:
        """Forward FFT (unnormalized, raw cuFFT). Graph-capture safe."""
        self._plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
        return out

    def fft_propagate(self, a: Any, out: Any, propagator: Any) -> bool:
        """Forward FFT with the propagator multiply folded into its store.

        Parameters
        ----------
        a : cp.ndarray
            Field to transform.
        out : cp.ndarray
            Where the transform lands, ``a`` itself for an in-place step.
        propagator : cp.ndarray
            Propagator to apply, of the field's shape or of the block a batch
            of fields shares.

        Returns
        -------
        bool
            Whether it happened. ``False`` leaves ``out`` untouched, and the
            caller owes both the transform and the multiply.

        """
        batched = self._callback_kind(a, propagator)
        if batched is None:
            return False
        plan = self._fused_plan(batched)
        if plan is None:
            return False
        self._bind(propagator)
        plan.fft(a, out, cp.cuda.cufft.CUFFT_FORWARD)
        return True

    def _callback_kind(self, a: Any, propagator: Any):
        """Return which callback these two arrays need, or None for neither.

        Parameters
        ----------
        a : cp.ndarray
            Field about to be transformed.
        propagator : cp.ndarray
            Propagator to apply to it.

        Returns
        -------
        bool or None
            ``False`` for the callback that indexes the propagator directly,
            ``True`` for the one that wraps it around a batch, ``None`` when
            no callback describes the pair and the multiply has to stay a
            kernel.

        """
        if a.shape != self._shape or a.dtype != self._dtype:
            return None
        if propagator.dtype != a.dtype:
            return None
        # Both are read by flat offset, which says nothing about either unless
        # they are contiguous.
        if not (a.flags.c_contiguous and propagator.flags.c_contiguous):
            return None
        if propagator.shape == a.shape:
            return False
        # A batch sharing one propagator. It has to be exactly the trailing
        # block of the field, since a flat offset reduced by its size is only
        # the index into it if the strides agree.
        trailing = a.shape[a.ndim - propagator.ndim :]
        if propagator.ndim and propagator.shape == trailing:
            return True
        return None

    def _fused_plan(self, batched: bool):
        """Return a plan carrying this callback, building it once, or None.

        Parameters
        ----------
        batched : bool
            Which callback the plan should carry.

        Returns
        -------
        cupy.cuda.cufft.PlanNd or None
            The plan, or None if this one cannot fuse.

        """
        if self._off:
            return None
        if batched in self._fused:
            return self._fused[batched]
        # A callback reads its output by flat offset, so it describes the
        # transform only when the transformed axes are the fastest ones.
        if self._axes != tuple(range(-len(self._axes), 0)):
            self._off = True
            return None
        try:
            self._fused[batched] = self._build_fused_plan(batched)
        except Exception as exc:
            self._off = True
            say(
                f"NLSE: cuFFT cannot fold the propagator into the transform "
                f"here ({type(exc).__name__}: {exc}); leaving it a kernel of "
                f"its own."
            )
            return None
        return self._fused[batched]

    def _build_fused_plan(self, batched: bool):
        """Build the plan whose store callback applies the propagator.

        Parameters
        ----------
        batched : bool
            Which callback to compile into it.

        Returns
        -------
        cupy.cuda.cufft.PlanNd
            A plan for this plan's shape, carrying the callback.

        """
        from ..kernels.templating import propagator_store_callback

        source, symbol = propagator_store_callback(self._dtype, batched)
        if self._info is None:
            self._info = cp.zeros(2, dtype=cp.uint64)
        # get_fft_plan measures a plan against an array; this one is a stand-in
        # for the fields the plan will see, and is freed on the way out.
        template = cp.empty(self._shape, dtype=self._dtype)
        with cp.fft.config.set_cufft_callbacks(
            cb_store=source,
            cb_store_name=symbol,
            cb_store_data=self._info.data,
            cb_ver="jit",
        ):
            return get_fft_plan(template, axes=self._axes, value_type="C2C")

    def _bind(self, propagator: Any) -> None:
        """Point the callback at this propagator, if it is not already.

        Parameters
        ----------
        propagator : cp.ndarray
            The propagator the next transform should apply.

        """
        bound = (propagator.data.ptr, propagator.size)
        if bound == self._bound:
            return
        _WRITE_PROPAGATOR_INFO(
            (1,),
            (1,),
            (self._info, np.uint64(bound[0]), np.uint64(bound[1])),
        )
        self._bound = bound

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
