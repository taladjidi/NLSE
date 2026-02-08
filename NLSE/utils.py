import importlib.util

__BACKEND__: str = "CUPY"
__CUPY_AVAILABLE__: bool
__PYOPENCL_AVAILABLE__: bool
__METAL_AVAILABLE__: bool

if importlib.util.find_spec("cupy") is not None:
    __CUPY_AVAILABLE__ = True
else:
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


if importlib.util.find_spec("pyopencl") is not None:
    __PYOPENCL_AVAILABLE__ = True
else:
    __PYOPENCL_AVAILABLE__ = False
    __BACKEND__ = "CPU"

try:
    from .kernels.metal_native.metal_api import (
        MetalContext as MetalContext,  # noqa: F401
    )

    __METAL_AVAILABLE__ = True
except (ImportError, FileNotFoundError, OSError):
    __METAL_AVAILABLE__ = False
