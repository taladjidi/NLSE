"""CUPY backend implementation."""

import contextlib
import os
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from ..kernels.templating import (
    INFO_WORDS,
    NONLINEAR_CALLBACK,
    SEPARABLE_CALLBACK,
    STORE_CALLBACKS,
    WRITE_INFO_KERNEL,
    WRITE_INFO_SOURCE,
    nonlinear_words,
    store_callbacks_source,
    v_suffix,
)
from ..utils import __CUPY_AVAILABLE__, say
from .backend import Backend, Timing

if not __CUPY_AVAILABLE__:
    raise ImportError("CuPy is not available - cannot import CUPYBackend")

import cupy as cp
import cupyx.scipy.fft as _cufft
from cupyx.scipy.fftpack import get_fft_plan

# Rewrites a store callback's block. A launch rather than a host-to-device
# copy, which a CUDA graph capture forbids from pageable memory: Yoshida's
# sub-steps change the arguments within one captured step.
_WRITE_INFO = cp.RawKernel(WRITE_INFO_SOURCE, WRITE_INFO_KERNEL)


class _CuFFTPlan:
    """cuFFT plan wrapper with .fft()/.ifft() API matching VkFFTApp.

    This allows CUDAKernels.linear_step to call plan.fft(A, A) / plan.ifft(A, A)
    without knowing which FFT library is behind it.

    Carries further plans, one per store callback it has been asked for: the
    propagator on the forward transform (see ``fft_propagate``) and the
    nonlinear step on the inverse one (see ``ifft_nonlinear``), both in
    fft_callbacks.cu. One per callback, because a callback fires in both
    directions: a plan that multiplies on the way out would multiply again on
    the way back.

    Each is built on first use and only if it can be: the callback machinery
    needs nvrtc and a cuFFT new enough to link LTO-IR, and CuPy reaches it for
    multi-dimensional plans only. Everything here therefore reports whether it
    worked rather than assuming it did, and a run that cannot fuse loses the
    pass and nothing else.

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
        "_off",
        "_plan",
        "_refused",
        "_shape",
    )

    def __init__(self, a: Any, axes: tuple) -> None:
        self._plan = get_fft_plan(a, axes=axes, value_type="C2C")
        self._axes = axes
        self._shape = a.shape
        self._dtype = a.dtype
        # Callback symbol -> (plan carrying it, block it reads its arguments
        # through), and -> the words that block holds.
        self._fused: dict = {}
        self._bound: dict = {}
        # Every callback ruled out for this plan, for good: asked against, or
        # its axes are not the ones a flat offset describes.
        self._off = os.environ.get("NLSE_FUSE_PROPAGATOR", "1") == "0"
        # Callbacks tried once and found wanting. Per symbol, so that one cuFFT
        # refuses costs its own pass and leaves the others fused.
        self._refused: set = set()

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
        symbol = STORE_CALLBACKS[batched]
        if not self._callback_plan(symbol):
            return False
        words = (propagator.data.ptr, propagator.size)
        self._transform(symbol, words, a, out, cp.cuda.cufft.CUFFT_FORWARD)
        return True

    def fft_propagate_separable(
        self, a: Any, out: Any, rows: Any, columns: Any
    ) -> bool:
        """Forward FFT with a propagator of one factor per axis in its store.

        Parameters
        ----------
        a : cp.ndarray
            Field to transform, of ``rows.size`` rows of ``columns.size``.
        out : cp.ndarray
            Where the transform lands, ``a`` itself for an in-place step.
        rows : cp.ndarray
            Factor of each row, 1D, over every row of a batch.
        columns : cp.ndarray
            Factor of each column, 1D.

        Returns
        -------
        bool
            Whether it happened. ``False`` leaves ``out`` untouched, and the
            caller owes the transform and the multiply.

        """
        if a.shape != self._shape or a.dtype != self._dtype:
            return False
        if not a.flags.c_contiguous:
            return False
        # Read by index from their first element, so contiguous or not at all.
        for factor in (rows, columns):
            if factor.dtype != a.dtype or factor.ndim != 1:
                return False
            if not factor.flags.c_contiguous:
                return False
        if columns.size != a.shape[-1] or rows.size * columns.size != a.size:
            return False
        if not self._callback_plan(SEPARABLE_CALLBACK):
            return False
        words = (rows.data.ptr, columns.data.ptr, columns.size)
        self._transform(SEPARABLE_CALLBACK, words, a, out, cp.cuda.cufft.CUFFT_FORWARD)
        return True

    def ifft_nonlinear_ready(self, a: Any, V: Any) -> bool:
        """Whether ``ifft_nonlinear`` can take this field and potential.

        Builds the plan it needs on the first call that can.

        Parameters
        ----------
        a : cp.ndarray
            Field about to be transformed.
        V : cp.ndarray or None
            Pre-scaled potential, at the field's width.

        Returns
        -------
        bool
            True if ``ifft_nonlinear`` may be called with them.

        """
        if a.shape != self._shape or a.dtype != self._dtype:
            return False
        # The callback reads the field and the potential by one flat offset.
        if not a.flags.c_contiguous:
            return False
        if V is None:
            return self._callback_plan(NONLINEAR_CALLBACK)
        if V.shape != a.shape or not V.flags.c_contiguous:
            return False
        # The callback reads the potential at the field's width.
        if V.dtype not in (self._dtype, np.finfo(self._dtype).dtype):
            return False
        return self._callback_plan(NONLINEAR_CALLBACK + v_suffix(V))

    def ifft_nonlinear(
        self, a: Any, out: Any, V: Any, dz: Any, alpha: Any, g: Any, Isat: Any
    ) -> Any:
        """Unnormalized inverse FFT with the nonlinear step folded into its store.

        Call ``ifft_nonlinear_ready`` with the same field and potential first:
        this raises KeyError if the plan it needs was never built.

        Parameters
        ----------
        a : cp.ndarray
            Field to transform.
        out : cp.ndarray
            Where the result lands, ``a`` itself for an in-place step.
        V : cp.ndarray or None
            Pre-scaled potential, at the field's width.
        dz : float
            Distance the nonlinear step advances by.
        alpha : float
            Loss coefficient.
        g : float
            Nonlinear interaction strength.
        Isat : float
            Saturation intensity.

        Returns
        -------
        cp.ndarray
            ``out``.

        """
        pointer = 0 if V is None else V.data.ptr
        words = nonlinear_words(
            pointer, self._dtype, dz=dz, alpha=alpha, g=g, Isat=Isat
        )
        symbol = NONLINEAR_CALLBACK + v_suffix(V)
        self._transform(symbol, words, a, out, cp.cuda.cufft.CUFFT_INVERSE)
        return out

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

    def _callback_plan(self, symbol: str) -> bool:
        """Build the plan carrying this callback, once, and say if there is one.

        Parameters
        ----------
        symbol : str
            Name of the callback in fft_callbacks.cu.

        Returns
        -------
        bool
            Whether ``self._fused`` holds a plan for it.

        """
        if self._off or symbol in self._refused:
            return False
        if symbol in self._fused:
            return True
        # A callback reads its output by flat offset, so it describes the
        # transform only when the transformed axes are the fastest ones.
        if self._axes != tuple(range(-len(self._axes), 0)):
            self._off = True
            return False
        try:
            self._fused[symbol] = self._build_callback_plan(symbol)
        except Exception as exc:
            self._refused.add(symbol)
            say(
                f"NLSE: cuFFT cannot run the {symbol} store callback here "
                f"({type(exc).__name__}: {exc}); leaving what it would fold in "
                f"to a pass of its own."
            )
            return False
        return True

    def _build_callback_plan(self, symbol: str) -> tuple:
        """Build a plan whose store callback is ``symbol``, with its block.

        Parameters
        ----------
        symbol : str
            Name of the callback in fft_callbacks.cu.

        Returns
        -------
        tuple
            ``(plan, info)``: a cupy.cuda.cufft.PlanNd for this plan's shape,
            carrying the callback, and the INFO_WORDS-word block it reads.

        """
        info = cp.zeros(INFO_WORDS, dtype=cp.uint64)
        # get_fft_plan measures a plan against an array; this one is a stand-in
        # for the fields the plan will see, and is freed on the way out.
        template = cp.empty(self._shape, dtype=self._dtype)
        with cp.fft.config.set_cufft_callbacks(
            cb_store=store_callbacks_source(self._dtype),
            cb_store_name=symbol,
            cb_store_data=info.data,
            cb_ver="jit",
        ):
            return get_fft_plan(template, axes=self._axes, value_type="C2C"), info

    def _transform(
        self, symbol: str, words: tuple, a: Any, out: Any, direction: int
    ) -> None:
        """Run the plan carrying ``symbol``, its block holding ``words``.

        The block is rewritten only when the words change, so a run whose
        arguments stay put records no write into its CUDA graph.

        Parameters
        ----------
        symbol : str
            Callback whose plan to run; ``_callback_plan`` has built it.
        words : tuple
            Up to INFO_WORDS integers for the block, zero-padded.
        a : cp.ndarray
            Field to transform.
        out : cp.ndarray
            Where the transform lands.
        direction : int
            ``CUFFT_FORWARD`` or ``CUFFT_INVERSE``.

        """
        plan, info = self._fused[symbol]
        words = words + (0,) * (INFO_WORDS - len(words))
        if self._bound.get(symbol) != words:
            _WRITE_INFO((1,), (1,), (info, *(np.uint64(w) for w in words)))
            self._bound[symbol] = words
        plan.fft(a, out, direction)

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
    has_linear_nonlinear_step = True
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
        """Return CUDA C kernels (with broadcasting fallback)."""
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
        # A non-blocking stream does not wait for the null stream: without this
        # the first step raced the work queued before the loop, such as a
        # Strang half-step or the partial step that ends a segment.
        stream.wait_event(cp.cuda.get_current_stream().record())
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
