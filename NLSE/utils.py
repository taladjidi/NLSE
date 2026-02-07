import importlib.util

__BACKEND__ = "CUPY"


if importlib.util.find_spec("cupy") is not None:
    __CUPY_AVAILABLE__ = True
else:
    print("CuPy not available, falling back to CPU BACKEND ...")
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


if importlib.util.find_spec("pyopencl") is not None:
    __PYOPENCL_AVAILABLE__ = True
else:
    print("PyOpenCL not available, falling back to CPU BACKEND ...")
    __PYOPENCL_AVAILABLE__ = False
    __BACKEND__ = "CPU"

try:
    from .kernels.metal_native.metal_api import MetalContext as MetalContext  # noqa: F401

    __METAL_AVAILABLE__ = True
except (ImportError, FileNotFoundError, OSError):
    __METAL_AVAILABLE__ = False
