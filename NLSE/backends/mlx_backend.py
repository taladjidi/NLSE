"""MLX backend implementation for Apple Silicon GPU acceleration."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..utils import __MLX_AVAILABLE__
from .backend import Backend

if not __MLX_AVAILABLE__:
    raise ImportError("MLX is not available - cannot import MLXBackend")

import mlx.core as mx

# MLX dtype mapping from numpy dtypes
_NUMPY_TO_MLX_DTYPE = {
    np.dtype(np.float32): mx.float32,
    np.dtype(np.float64): mx.float32,  # downcast
    np.dtype(np.complex64): mx.complex64,
    np.dtype(np.complex128): mx.complex64,  # downcast
}


def _fft_convolve(in1: Any, in2: Any, mode: str = "full", axes=None) -> Any:
    """Convolve two MLX arrays over ``axes`` by transform.

    Mirrors ``scipy.signal.oaconvolve`` closely enough for the solvers, which
    call it as ``convolution(A_sq, kernel, mode="same", axes=last_axes)``.
    Both arguments carry the same rank -- a batched run broadcasts the kernel
    over its leading axes -- so only ``axes`` are transformed and the rest ride
    along.

    Parameters
    ----------
    in1, in2 : mx.array
        Arrays of equal rank; only ``axes`` are convolved.
    mode : str
        "full" or "same". "same" returns ``in1``'s shape, centred.
    axes : tuple of int, optional
        Axes to convolve. Defaults to all of them.

    Returns
    -------
    mx.array
        The convolution, real where both inputs were real.
    """
    if mode not in ("full", "same"):
        raise ValueError(f"unsupported convolution mode {mode!r}")
    ndim = in1.ndim
    axes = tuple(range(ndim)) if axes is None else tuple(a % ndim for a in axes)
    s1 = [in1.shape[a] for a in axes]
    s2 = [in2.shape[a] for a in axes]
    full = [a + b - 1 for a, b in zip(s1, s2)]

    spectrum = mx.fft.fftn(in1, s=full, axes=axes) * mx.fft.fftn(in2, s=full, axes=axes)
    out = mx.fft.ifftn(spectrum, s=full, axes=axes)
    # A real convolution comes back with rounding-level imaginary parts.
    if in1.dtype != mx.complex64 and in2.dtype != mx.complex64:
        out = out.real

    if mode == "full":
        return out
    # "same": take in1's extent from the centre of the full support.
    index: list[Any] = [slice(None)] * ndim
    for axis, n1, n2 in zip(axes, s1, s2):
        start = (n2 - 1) // 2
        index[axis] = slice(start, start + n1)
    return out[tuple(index)]


class MLXBackend(Backend):
    """MLX backend for Apple Silicon GPU acceleration.

    Fuses via mx.compile, which traces a whole step into one graph. MLX
    always normalizes its inverse FFT, so the 1/N cannot be folded into
    the propagator.
    """

    has_linear_step = True
    normalizes_on_host = True
    has_fused_split_step = True
    broadcasts_parameters_natively = True
    has_fused_rk4_step = True
    has_fused_coupled_split_step = True
    has_fused_coupled_rk4_rhs = True

    @property
    def name(self) -> str:
        return "MLX"

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate complex array on MLX device."""
        mx_dtype = _NUMPY_TO_MLX_DTYPE.get(np.dtype(dtype), mx.complex64)
        return mx.zeros(shape, dtype=mx_dtype)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> Any:
        """Allocate real array on MLX device."""
        mx_dtype = _NUMPY_TO_MLX_DTYPE.get(np.dtype(dtype), mx.float32)
        return mx.zeros(shape, dtype=mx_dtype)

    @property
    def convolution(self):
        """Return an FFT convolution with scipy's ``oaconvolve`` signature.

        MLX has no convolution of its own, which is what previously kept the
        non-local interaction off this backend: ``nl_length > 0`` is gated on
        this property being non-None. The kernel is a Bessel profile of a few
        hundred cells at most, so an overlap-add scheme buys nothing over a
        single padded transform.
        """
        return _fft_convolve

    def synchronize(self, array=None) -> None:
        """Force the lazy graph for ``array``.

        Parameters
        ----------
        array : Any, optional
            The array whose value is needed. Nothing to do without one.
        """
        if array is not None:
            mx.eval(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Transfer from MLX device to CPU."""
        mx.eval(array)
        return np.array(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Transfer from CPU to MLX device.

        Downcasts float64/complex128 to float32/complex64 since MLX
        does not support double precision.
        """
        if array.dtype == np.complex128:
            array = array.astype(np.complex64)
        elif array.dtype == np.float64:
            array = array.astype(np.float32)
        return mx.array(array)

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Build FFT plan for MLX (stores axes tuple).

        MLX FFT does not use plan objects. We store the axes for use
        in fft/ifft calls.

        Returns
        -------
        list
            List containing the axes tuple.

        """
        return [axes]

    def fft(self, array: Any, plan: list) -> Any:
        """Perform forward FFT."""
        axes = plan[0]
        return mx.fft.fftn(array, axes=axes)

    def ifft(self, array: Any, plan: list, normalize: bool = True) -> Any:
        """Perform inverse FFT.

        MLX's inverse always normalizes, so this backend does not declare
        supports_unnormalized_ifft and is never asked to skip it.
        """
        assert normalize, "MLX cannot skip the inverse transform's 1/N"
        axes = plan[0]
        return mx.fft.ifftn(array, axes=axes)

    def exp(self, array: Any) -> Any:
        """Exponentiate without leaving this backend."""
        return mx.exp(array)

    def sum(self, array: Any) -> float:
        """Reduce without leaving this backend."""
        return float(mx.sum(array))

    @property
    def kernels(self) -> Any:
        """Return MLX kernels module."""
        if not hasattr(self, "_kernels"):
            from ..kernels import mlx_kernels

            self._kernels = mlx_kernels
        return self._kernels

    def supports_double_precision(self) -> bool:
        """MLX does not support double precision."""
        return False
