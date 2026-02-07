"""Low-level CFFI bindings to the Metal compute library."""

import ctypes
import os
import numpy as np
from pathlib import Path

# ---- Load the shared library ----
_LIB_PATH = Path(__file__).parent / "libmetal_nlse.dylib"
_SHADER_PATH = Path(__file__).parent / "kernels.metal"


def _load_lib():
    if not _LIB_PATH.exists():
        raise FileNotFoundError(
            f"Metal library not found at {_LIB_PATH}. "
            "Compile with: clang -O2 -framework Metal -framework Foundation "
            "-shared -fobjc-arc -o libmetal_nlse.dylib metal_wrapper.m"
        )
    return ctypes.CDLL(str(_LIB_PATH))


_lib = _load_lib()

# ---- C type aliases ----
_ctx_p = ctypes.c_void_p  # MetalCtx*
_buf_p = ctypes.c_void_p  # MetalBuf*
_uint32 = ctypes.c_uint32
_float = ctypes.c_float
_int = ctypes.c_int
_size_t = ctypes.c_size_t

# ---- Set up function signatures ----
_lib.metal_init.restype = _ctx_p
_lib.metal_init.argtypes = [ctypes.c_char_p]

_lib.metal_free.restype = None
_lib.metal_free.argtypes = [_ctx_p]

_lib.metal_device_name.restype = ctypes.c_char_p
_lib.metal_device_name.argtypes = [_ctx_p]

_lib.metal_buf_alloc.restype = _buf_p
_lib.metal_buf_alloc.argtypes = [_ctx_p, _size_t]

_lib.metal_buf_from_ptr.restype = _buf_p
_lib.metal_buf_from_ptr.argtypes = [_ctx_p, ctypes.c_void_p, _size_t]

_lib.metal_buf_ptr.restype = ctypes.c_void_p
_lib.metal_buf_ptr.argtypes = [_buf_p]

_lib.metal_buf_size.restype = _size_t
_lib.metal_buf_size.argtypes = [_buf_p]

_lib.metal_buf_free.restype = None
_lib.metal_buf_free.argtypes = [_buf_p]

_lib.metal_buf_copy_from.restype = None
_lib.metal_buf_copy_from.argtypes = [_buf_p, ctypes.c_void_p, _size_t]

_lib.metal_buf_copy_to.restype = None
_lib.metal_buf_copy_to.argtypes = [_buf_p, ctypes.c_void_p, _size_t]

_lib.metal_square_mod.restype = None
_lib.metal_square_mod.argtypes = [_ctx_p, _buf_p, _buf_p, _uint32]

_lib.metal_nl_prop.restype = None
_lib.metal_nl_prop.argtypes = [
    _ctx_p, _buf_p, _buf_p, _buf_p,
    _float, _float, _float, _float, _uint32,
]

_lib.metal_nl_prop_without_V.restype = None
_lib.metal_nl_prop_without_V.argtypes = [
    _ctx_p, _buf_p, _buf_p,
    _float, _float, _float, _float, _uint32,
]

_lib.metal_nl_prop_c.restype = None
_lib.metal_nl_prop_c.argtypes = [
    _ctx_p, _buf_p, _buf_p, _buf_p, _buf_p,
    _float, _float, _float, _float, _float, _float, _uint32,
]

_lib.metal_nl_prop_without_V_c.restype = None
_lib.metal_nl_prop_without_V_c.argtypes = [
    _ctx_p, _buf_p, _buf_p, _buf_p,
    _float, _float, _float, _float, _float, _float, _uint32,
]

_lib.metal_rabi_coupling.restype = None
_lib.metal_rabi_coupling.argtypes = [
    _ctx_p, _buf_p, _buf_p, _buf_p,
    _float, _float, _uint32,
]

_lib.metal_vortex.restype = None
_lib.metal_vortex.argtypes = [
    _ctx_p, _buf_p, _buf_p, _buf_p,
    _float, _float, _int, _uint32,
]

_lib.metal_complex_multiply_inplace.restype = None
_lib.metal_complex_multiply_inplace.argtypes = [_ctx_p, _buf_p, _buf_p, _uint32]


# ============================================================
# High-level Python API
# ============================================================


class MetalBuffer:
    """Wrapper around a Metal shared-memory buffer.

    Provides numpy-compatible views into GPU-visible memory.
    """

    def __init__(self, ctx_handle, shape, dtype):
        self._ctx = ctx_handle
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._nbytes = int(np.prod(shape)) * self.dtype.itemsize
        self._handle = _lib.metal_buf_alloc(ctx_handle, self._nbytes)
        if not self._handle:
            raise MemoryError("Failed to allocate Metal buffer")

    @classmethod
    def from_numpy(cls, ctx_handle, arr):
        """Create a MetalBuffer and copy numpy data into it."""
        arr = np.ascontiguousarray(arr)
        buf = cls(ctx_handle, arr.shape, arr.dtype)
        _lib.metal_buf_copy_from(
            buf._handle, arr.ctypes.data, buf._nbytes
        )
        return buf

    def to_numpy(self):
        """Copy data from Metal buffer into a new numpy array."""
        arr = np.empty(self.shape, dtype=self.dtype)
        _lib.metal_buf_copy_to(self._handle, arr.ctypes.data, self._nbytes)
        return arr

    @property
    def size(self):
        """Number of elements."""
        return int(np.prod(self.shape))

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.metal_buf_free(self._handle)


class MetalContext:
    """Metal device and kernel context."""

    def __init__(self):
        shader_source = _SHADER_PATH.read_text()
        self._handle = _lib.metal_init(shader_source.encode('utf-8'))
        if not self._handle:
            raise RuntimeError("Failed to initialize Metal context")

    @property
    def device_name(self):
        return _lib.metal_device_name(self._handle).decode('utf-8')

    def alloc(self, shape, dtype):
        return MetalBuffer(self._handle, shape, dtype)

    def from_numpy(self, arr):
        return MetalBuffer.from_numpy(self._handle, arr)

    def square_mod(self, A, A_sq):
        _lib.metal_square_mod(self._handle, A._handle, A_sq._handle, A.size)

    def nl_prop(self, A, A_sq, V, dz, alpha, g, Isat):
        _lib.metal_nl_prop(
            self._handle, A._handle, A_sq._handle, V._handle,
            dz, alpha, g, Isat, A.size,
        )

    def nl_prop_without_V(self, A, A_sq, dz, alpha, g, Isat):
        _lib.metal_nl_prop_without_V(
            self._handle, A._handle, A_sq._handle,
            dz, alpha, g, Isat, A.size,
        )

    def nl_prop_c(self, A1, A_sq_1, A_sq_2, V, dz, alpha, g11, g12, Isat1, Isat2):
        _lib.metal_nl_prop_c(
            self._handle, A1._handle, A_sq_1._handle,
            A_sq_2._handle, V._handle,
            dz, alpha, g11, g12, Isat1, Isat2, A1.size,
        )

    def nl_prop_without_V_c(self, A1, A_sq_1, A_sq_2, dz, alpha,
                             g11, g12, Isat1, Isat2):
        _lib.metal_nl_prop_without_V_c(
            self._handle, A1._handle, A_sq_1._handle, A_sq_2._handle,
            dz, alpha, g11, g12, Isat1, Isat2, A1.size,
        )

    def rabi_coupling(self, A1, A2, scratch, dz, omega):
        _lib.metal_rabi_coupling(
            self._handle, A1._handle, A2._handle, scratch._handle,
            dz, omega, A1.size,
        )

    def vortex(self, im, ii, jj, i_pos, j_pos, ll):
        _lib.metal_vortex(
            self._handle, im._handle, ii._handle, jj._handle,
            float(i_pos), float(j_pos), int(ll), im.size,
        )

    def complex_multiply_inplace(self, A, B):
        _lib.metal_complex_multiply_inplace(
            self._handle, A._handle, B._handle, A.size,
        )

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.metal_free(self._handle)
