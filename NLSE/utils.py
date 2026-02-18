import os
import warnings
from pathlib import Path

__BACKEND__ = "CUPY"
__PYOPENCL_DOUBLE_SUPPORT__ = False


try:
    import cupy

    __CUPY_AVAILABLE__ = True

except ImportError:
    warnings.warn(
        "CuPy not available, falling back to CPU backend. "
        "Install cupy for GPU acceleration.",
        ImportWarning,
        stacklevel=2,
    )
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


try:
    import mlx.core

    __MLX_AVAILABLE__ = True
    if __BACKEND__ == "CPU":
        __BACKEND__ = "MLX"
except ImportError:
    __MLX_AVAILABLE__ = False


try:
    # for OpenCL backend you need to install OpenCL first
    # sudo apt install intel-opencl-icd opencl-headers ocl-icd-opencl-dev
    # or for AMD
    # sudo apt install opencl-headers ocl-icd-opencl-dev
    import pyopencl as cl

    # Verify an OpenCL device is actually available, not just the library
    ctx = cl.create_some_context(interactive=False)
    device = ctx.devices[0]
    __PYOPENCL_AVAILABLE__ = True
    __PYOPENCL_DOUBLE_SUPPORT__ = bool(device.double_fp_config)

except ImportError:
    warnings.warn(
        "PyOpenCL not available, OpenCL backend unavailable. "
        "Install pyopencl for OpenCL support.",
        ImportWarning,
        stacklevel=2,
    )
    __PYOPENCL_AVAILABLE__ = False
except Exception:
    warnings.warn(
        "PyOpenCL installed but no OpenCL device found. OpenCL backend unavailable.",
        ImportWarning,
        stacklevel=2,
    )
    __PYOPENCL_AVAILABLE__ = False


def get_cache_dir() -> Path:
    """Get NLSE cache directory within the library directory.

    Returns
    -------
    Path
        <NLSE_package_dir>/.cache/
    """
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_benchmark_cache_path() -> Path:
    """Get path to FFT benchmark cache file."""
    return get_cache_dir() / "fft_benchmark.json"
