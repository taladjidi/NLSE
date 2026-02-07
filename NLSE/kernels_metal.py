"""Metal backend kernels for NLSE solver.

Exposes the same function signatures as kernels_cpu.py / kernels_gpu.py
so the NLSE solver can dispatch to Metal transparently.

Arrays are MetalArray objects that wrap Metal shared-memory buffers.
"""

import numpy as np
from .metal.metal_api import MetalContext, MetalBuffer, _lib


# Global context (initialized lazily)
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = MetalContext()
    return _ctx


class MetalFFTPlan:
    """FFT plan using numpy FFT for Metal backend.

    Wraps numpy FFT to provide the same .fft()/.ifft() interface
    as VkFFT plans used by GPU/CL backends.
    """

    def __init__(self, shape, ndim):
        self.axes = tuple(range(-ndim, 0))

    def fft(self, A, A_out):
        data = A.get()
        result = np.fft.fftn(data, axes=self.axes).astype(data.dtype)
        A_out[:] = result

    def ifft(self, A, A_out):
        data = A.get()
        result = np.fft.ifftn(data, axes=self.axes).astype(data.dtype)
        A_out[:] = result


class MetalArray:
    """A numpy-like array backed by a Metal shared-memory buffer.

    Supports basic operations needed by the NLSE solver:
    slicing, in-place multiply, copy, shape/dtype access.
    """

    def __init__(self, metal_buf, shape, dtype):
        self._buf = metal_buf
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @classmethod
    def from_numpy(cls, arr):
        """Create a MetalArray from a numpy array."""
        ctx = _get_ctx()
        arr = np.ascontiguousarray(arr)
        buf = MetalBuffer.from_numpy(ctx._handle, arr)
        return cls(buf, arr.shape, arr.dtype)

    @classmethod
    def zeros(cls, shape, dtype):
        ctx = _get_ctx()
        buf = MetalBuffer(ctx._handle, shape, dtype)
        arr = np.zeros(shape, dtype=dtype)
        _lib.metal_buf_copy_from(buf._handle, arr.ctypes.data, buf._nbytes)
        return cls(buf, shape, dtype)

    def get(self):
        """Copy data back to a numpy array."""
        return self._buf.to_numpy()

    @property
    def size(self):
        return self._buf.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def real(self):
        """Return real part as numpy (for host-side computation)."""
        return self.get().real

    @property
    def imag(self):
        """Return imaginary part as numpy (for host-side computation)."""
        return self.get().imag

    def copy(self):
        ctx = _get_ctx()
        new_buf = MetalBuffer(ctx._handle, self.shape, self.dtype)
        data = self._buf.to_numpy()
        _lib.metal_buf_copy_from(new_buf._handle, data.ctypes.data, new_buf._nbytes)
        return MetalArray(new_buf, self.shape, self.dtype)

    def sum(self, axis=None, dtype=None):
        """Sum reduction (host-side)."""
        arr = self.get()
        return arr.sum(axis=axis, dtype=dtype)

    def __getitem__(self, key):
        arr = self.get()
        result = arr[key]
        if isinstance(result, np.ndarray):
            return MetalArray.from_numpy(np.ascontiguousarray(result))
        return result

    def __setitem__(self, key, value):
        if isinstance(key, slice) and key == slice(None):
            if isinstance(value, MetalArray):
                data = value.get()
            else:
                data = np.ascontiguousarray(value)
            _lib.metal_buf_copy_from(
                self._buf._handle, data.ctypes.data, self._buf._nbytes
            )
        else:
            arr = self.get()
            if isinstance(value, MetalArray):
                arr[key] = value.get()
            else:
                arr[key] = value
            _lib.metal_buf_copy_from(
                self._buf._handle, arr.ctypes.data, self._buf._nbytes
            )

    def __imul__(self, other):
        if isinstance(other, MetalArray):
            ctx = _get_ctx()
            ctx.complex_multiply_inplace(self._buf, other._buf)
        else:
            arr = self.get()
            arr *= other
            _lib.metal_buf_copy_from(
                self._buf._handle,
                np.ascontiguousarray(arr).ctypes.data,
                self._buf._nbytes,
            )
        return self

    def __mul__(self, other):
        if isinstance(other, MetalArray):
            result = self.copy()
            ctx = _get_ctx()
            ctx.complex_multiply_inplace(result._buf, other._buf)
            return result
        arr = self.get() * other
        return MetalArray.from_numpy(np.ascontiguousarray(arr))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        arr = self.get()
        if isinstance(other, MetalArray):
            arr = arr + other.get()
        else:
            arr = arr + other
        return MetalArray.from_numpy(np.ascontiguousarray(arr))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        arr = self.get()
        if isinstance(other, MetalArray):
            arr += other.get()
        else:
            arr += other
        _lib.metal_buf_copy_from(
            self._buf._handle,
            np.ascontiguousarray(arr).ctypes.data,
            self._buf._nbytes,
        )
        return self

    def __sub__(self, other):
        arr = self.get()
        if isinstance(other, MetalArray):
            arr = arr - other.get()
        else:
            arr = arr - other
        return MetalArray.from_numpy(np.ascontiguousarray(arr))

    def __rsub__(self, other):
        arr = self.get()
        if isinstance(other, MetalArray):
            result = other.get() - arr
        else:
            result = other - arr
        return MetalArray.from_numpy(np.ascontiguousarray(result))

    def __neg__(self):
        return MetalArray.from_numpy(-self.get())

    def __truediv__(self, other):
        arr = self.get()
        if isinstance(other, MetalArray):
            arr = arr / other.get()
        else:
            arr = arr / other
        return MetalArray.from_numpy(np.ascontiguousarray(arr))

    def __pow__(self, exponent):
        return MetalArray.from_numpy(self.get() ** exponent)

    def __abs__(self):
        return np.abs(self.get())

    def astype(self, dtype):
        return MetalArray.from_numpy(self.get().astype(dtype))

    @property
    def T(self):
        return MetalArray.from_numpy(self.get().T)


# ============================================================
# Kernel functions matching CPU/GPU/CL API
# ============================================================


def square_mod(A, A_sq):
    """Compute |A|^2."""
    ctx = _get_ctx()
    ctx.square_mod(A._buf, A_sq._buf)


def nl_prop(A, A_sq, dz, alpha, V, g, Isat):
    """Nonlinear propagation with potential."""
    ctx = _get_ctx()
    ctx.nl_prop(A._buf, A_sq._buf, V._buf, float(dz), float(alpha), float(g), float(Isat))


def nl_prop_without_V(A, A_sq, dz, alpha, g, Isat):
    """Nonlinear propagation without potential."""
    ctx = _get_ctx()
    ctx.nl_prop_without_V(A._buf, A_sq._buf, float(dz), float(alpha), float(g), float(Isat))


def nl_prop_c(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2):
    """Coupled nonlinear propagation with potential."""
    ctx = _get_ctx()
    ctx.nl_prop_c(
        A1._buf, A_sq_1._buf, A_sq_2._buf, V._buf,
        float(dz), float(alpha), float(g11), float(g12),
        float(Isat1), float(Isat2),
    )


def nl_prop_without_V_c(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2):
    """Coupled nonlinear propagation without potential."""
    ctx = _get_ctx()
    ctx.nl_prop_without_V_c(
        A1._buf, A_sq_1._buf, A_sq_2._buf,
        float(dz), float(alpha), float(g11), float(g12),
        float(Isat1), float(Isat2),
    )


def rabi_coupling(A1, A2, dz, omega):
    """Rabi coupling between two components."""
    ctx = _get_ctx()
    scratch = MetalBuffer(ctx._handle, A1.shape, A1.dtype)
    ctx.rabi_coupling(A1._buf, A2._buf, scratch, float(dz), float(omega))


def vortex(im, i, j, ii, jj, ll):
    """Generate vortex phase pattern."""
    ctx = _get_ctx()
    ctx.vortex(im._buf, ii._buf, jj._buf, float(i), float(j), int(ll))
