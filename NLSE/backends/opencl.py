"""OpenCL backend implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..utils import __PYOPENCL_AVAILABLE__
from .backend import Backend

if not __PYOPENCL_AVAILABLE__:
    raise ImportError("PyOpenCL is not available - cannot import OpenCLBackend")

import pyopencl as cl
import pyopencl.clmath as clmath
from pyopencl import array as cla
from pyvkfft.opencl import VkFFTApp


class _VkFFTPlan:
    """VkFFT plan wrapper with .fft()/.ifft()/.ifft_unnorm() API.

    Matches _CuFFTPlan so that OpenCLKernels.linear_step can call
    plan.fft(A, A) / plan.ifft(A, A) / plan.ifft_unnorm(A, A)
    without knowing which FFT library is behind it.

    The three apps are built on first use. Each compiles its own kernels,
    and a given run needs at most two: the unnormalized one only where the
    1/N is folded into the propagator, the out-of-place one only for RK4.
    """

    __slots__ = ("_app", "_app_oop", "_app_unnorm", "_spec")

    def __init__(self, shape, dtype, queue, axes, ndim):
        self._spec = {
            "shape": shape,
            "dtype": dtype,
            "queue": queue,
            "axes": axes,
            "ndim": ndim,
        }
        self._app = None
        self._app_unnorm = None
        self._app_oop = None

    def _make(self, **extra):
        """Build a VkFFTApp for this plan's transform."""
        return VkFFTApp(**self._spec, **extra)

    @property
    def app(self):
        """Return the in-place, normalized app."""
        if self._app is None:
            self._app = self._make()
        return self._app

    def fft(self, a, out):
        """Forward FFT (in-place when a is out)."""
        return self.app.fft(a, out)

    def fft_oop(self, src, dest):
        """Out-of-place forward FFT (src is not modified)."""
        if self._app_oop is None:
            self._app_oop = self._make(inplace=False)
        return self._app_oop.fft(src, dest)

    def ifft(self, a, out):
        """Inverse FFT (normalized by 1/N)."""
        return self.app.ifft(a, out)

    def ifft_unnorm(self, a, out):
        """Inverse FFT without 1/N normalization."""
        if self._app_unnorm is None:
            self._app_unnorm = self._make(norm=0)
        return self._app_unnorm.ifft(a, out)


_VKFFT_RADICES = (2, 3, 5, 7, 11, 13)


def _good_size(n: int) -> int:
    """Return the smallest size >= n that VkFFT transforms efficiently.

    A linear convolution needs at least ``n1 + n2 - 1`` points, but that is
    usually an awkward number -- for a 1024 grid convolved with a 1024 kernel
    it is 2047, which is 23 x 89 -- and VkFFT is fastest on products of small
    radices. Rounding up costs a few points and buys the good transform.
    """
    while True:
        rest = n
        for radix in _VKFFT_RADICES:
            while rest % radix == 0:
                rest //= radix
        if rest == 1:
            return n
        n += 1


def _rect_copy(queue, dst, src, shape, itemsize) -> None:
    """Copy a contiguous block into the corner of a larger array.

    PyOpenCL refuses both to copy and to assign into a non-contiguous slice,
    which is what zero-padding an array into a bigger one needs. OpenCL has a
    rectangular buffer copy for exactly this, so the padding goes through
    ``clEnqueueCopyBufferRect`` rather than through a kernel of our own.

    Parameters
    ----------
    queue : pyopencl.CommandQueue
        Queue to enqueue the copy on.
    dst, src : pyopencl.array.Array
        Destination and source, both contiguous and of the same dtype.
    shape : tuple of int
        Extent of the block to copy, at most three-dimensional.
    itemsize : int
        Bytes per element.
    """
    shape = (1,) * (3 - len(shape)) + tuple(shape)
    d0, d1, d2 = shape
    ds = (1,) * (3 - dst.ndim) + tuple(dst.shape)
    ss = (1,) * (3 - src.ndim) + tuple(src.shape)
    cl.enqueue_copy(
        queue,
        dst.base_data,
        src.base_data,
        src_origin=(0, 0, 0),
        dst_origin=(0, 0, 0),
        region=(d2 * itemsize, d1, d0),
        src_pitches=(ss[2] * itemsize, ss[2] * ss[1] * itemsize),
        dst_pitches=(ds[2] * itemsize, ds[2] * ds[1] * itemsize),
    )


def _rect_crop(queue, dst, src, offset, itemsize) -> None:
    """Copy a sub-block out of a larger array into a contiguous one.

    The mirror of :func:`_rect_copy`, used to take the "same" region out of
    the full convolution support.

    Parameters
    ----------
    queue : pyopencl.CommandQueue
        Queue to enqueue the copy on.
    dst, src : pyopencl.array.Array
        Destination (the block) and source (the larger array).
    offset : tuple of int
        Index of the block's first element in ``src``.
    itemsize : int
        Bytes per element.
    """
    shape = (1,) * (3 - dst.ndim) + tuple(dst.shape)
    ds = shape
    ss = (1,) * (3 - src.ndim) + tuple(src.shape)
    off = (0,) * (3 - len(offset)) + tuple(offset)
    cl.enqueue_copy(
        queue,
        dst.base_data,
        src.base_data,
        src_origin=(off[2] * itemsize, off[1], off[0]),
        dst_origin=(0, 0, 0),
        region=(ds[2] * itemsize, ds[1], ds[0]),
        src_pitches=(ss[2] * itemsize, ss[2] * ss[1] * itemsize),
        dst_pitches=(ds[2] * itemsize, ds[2] * ds[1] * itemsize),
    )


class OpenCLBackend(Backend):
    """OpenCL backend using PyOpenCL and VkFFT.

    Fuses aggressively: OpenCL has no equivalent of CUDA graph capture, so
    each avoided dispatch is a real saving.
    """

    has_linear_step = True
    normalizes_on_host = True
    supports_unnormalized_ifft = True
    has_fused_split_step = True
    has_fused_rk4_rhs = True
    has_fused_rk4_stage_update = True
    has_fused_coupled_split_step = True
    has_fused_coupled_rk4_rhs = True

    def __init__(self):
        self._context = cl.create_some_context(interactive=False)
        self._queue = cl.CommandQueue(self._context)
        self._kernels = None

    @property
    def name(self) -> str:
        return "CL"

    @property
    def queue(self) -> Any:
        """Get OpenCL command queue."""
        return self._queue

    @property
    def context(self) -> Any:
        """Get OpenCL context."""
        return self._context

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate array on OpenCL device."""
        return cla.zeros(self._queue, shape, dtype)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate real array on OpenCL device."""
        return cla.zeros(self._queue, shape, dtype)

    def synchronize(self, array=None) -> None:
        """Drain the command queue."""
        self._queue.finish()

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from OpenCL device to CPU.

        Accepts an array that is already on the host, as ``cp.asnumpy`` and
        MLX's ``np.array`` both do; not every path through a solver leaves the
        field on the device.
        """
        return array.get() if hasattr(array, "get") else np.asarray(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to OpenCL device."""
        return cla.to_device(self._queue, array)

    @property
    def convolution(self):
        """Return an FFT convolution with scipy's ``oaconvolve`` signature.

        PyOpenCL has no convolution, which is what kept the non-local
        interaction off this backend: ``nl_length > 0`` is gated on this
        property being non-None. VkFFT is already here for the propagation, so
        the convolution is a padded transform pair rather than anything new --
        the work is in the padding, since PyOpenCL will not write into a
        non-contiguous slice and OpenCL's rectangular buffer copy has to do it.
        """
        return self._fft_convolve

    def _fft_convolve(self, in1, in2, mode: str = "full", axes=None):
        """Convolve two device arrays over ``axes`` by transform.

        Matches ``scipy.signal.oaconvolve`` closely enough for the solvers,
        which call it as ``convolution(A_sq, kernel, mode="same",
        axes=last_axes)``. Both arguments carry the same rank; any axes
        outside ``axes`` are looped over rather than broadcast, since
        PyOpenCL arrays do not broadcast in arithmetic.

        Parameters
        ----------
        in1, in2 : pyopencl.array.Array
            Arrays of equal rank; only ``axes`` are convolved.
        mode : str
            "full" or "same". "same" returns ``in1``'s shape, centred.
        axes : tuple of int, optional
            Axes to convolve. Defaults to all of them.

        Returns
        -------
        pyopencl.array.Array
            The convolution, real where both inputs were real.
        """
        if mode not in ("full", "same"):
            raise ValueError(f"unsupported convolution mode {mode!r}")
        ndim = in1.ndim
        axes = tuple(range(ndim)) if axes is None else tuple(a % ndim for a in axes)
        if tuple(sorted(axes)) != tuple(range(ndim - len(axes), ndim)):
            raise ValueError("only trailing axes can be convolved on OpenCL")
        real = in1.dtype.kind == "f" and in2.dtype.kind == "f"
        lead1, lead2 = in1.shape[: ndim - len(axes)], in2.shape[: ndim - len(axes)]
        batch = int(np.prod(lead1)) if lead1 else 1
        conv1, conv2 = in1.shape[len(lead1) :], in2.shape[len(lead2) :]
        out_shape = (
            conv1 if mode == "same" else tuple(a + b - 1 for a, b in zip(conv1, conv2))
        )
        flat1 = in1.reshape((batch, *conv1))
        flat2 = in2.reshape((int(np.prod(lead2)) if lead2 else 1, *conv2))
        pieces = [
            self._convolve_one(
                flat1[b], flat2[b if flat2.shape[0] > 1 else 0], out_shape, mode
            )
            for b in range(batch)
        ]
        stacked = (
            pieces[0]
            if batch == 1
            else cla.concatenate([p.reshape((1, *out_shape)) for p in pieces])
        )
        out = stacked.reshape(lead1 + out_shape)
        return out.real if real else out

    def _convolve_one(self, a, b, out_shape: tuple, mode: str):
        """Convolve two arrays whose whole rank is transformed."""
        conv_a, conv_b = a.shape, b.shape
        full = [x + y - 1 for x, y in zip(conv_a, conv_b)]
        padded = tuple(_good_size(f) for f in full)
        itemsize = np.dtype(np.complex64).itemsize
        buf_a = cla.zeros(self._queue, padded, np.complex64)
        buf_b = cla.zeros(self._queue, padded, np.complex64)
        _rect_copy(self._queue, buf_a, a.astype(np.complex64), conv_a, itemsize)
        _rect_copy(self._queue, buf_b, b.astype(np.complex64), conv_b, itemsize)
        plan = _VkFFTPlan(
            padded, np.complex64, self._queue, tuple(range(len(padded))), len(padded)
        )
        plan.fft(buf_a, buf_a)
        plan.fft(buf_b, buf_b)
        buf_a *= buf_b
        plan.ifft(buf_a, buf_a)
        offset = tuple((y - 1) // 2 if mode == "same" else 0 for y in conv_b)
        out = cla.zeros(self._queue, out_shape, np.complex64)
        _rect_crop(self._queue, out, buf_a, offset, itemsize)
        return out

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build VkFFT app for OpenCL.

        Returns
        -------
        list
            List containing VkFFTApp instance (for consistency with CPU backend)

        """
        plan = _VkFFTPlan(shape, dtype, queue=self._queue, axes=axes, ndim=len(axes))
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
        """Reduce on the device where the reduction can be built.

        PyOpenCL generates its reduction kernels from a mako template, and
        mako is not a hard dependency of pyopencl. Without it every reduction
        raises ModuleNotFoundError -- not at import, but the first time one is
        asked for, which here was inside a callback partway through a
        propagation. Falling back to the base class costs the round trip this
        override exists to avoid, and is what the caller would have had
        anyway.
        """
        try:
            flat = array.reshape(-1)
            return float(np.sqrt(cla.vdot(flat, flat).get().real))
        except ModuleNotFoundError:
            return super().norm(array)

    def exp(self, array: Any) -> Any:
        """Exponentiate without leaving this backend."""
        return clmath.exp(array)

    def sum(self, array: Any) -> float:
        """Reduce on the device, falling back as ``norm`` does."""
        try:
            return float(cla.sum(array).get())
        except ModuleNotFoundError:
            return super().sum(array)

    @property
    def kernels(self) -> Any:
        """Return OpenCL kernels."""
        if self._kernels is None:
            from ..kernels.cl import OpenCLKernels

            self._kernels = OpenCLKernels(self._context, self._queue)
        return self._kernels

    def supports_double_precision(self) -> bool:
        """Check OpenCL device double precision support.

        Asked of this backend's own device, so importing NLSE does not need
        to create a context to find out.
        """
        return bool(self._context.devices[0].double_fp_config)
