import os
import sys
import warnings
from pathlib import Path

__BACKEND__ = "CUPY"


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

    # A platform with at least one device, without opening a context: creating
    # one here would take a device handle on every import of NLSE, whether or
    # not OpenCL is ever used. The backend opens its own when it is built, and
    # answers questions about the device from that.
    __PYOPENCL_AVAILABLE__ = any(p.get_devices() for p in cl.get_platforms())

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


def say(message: str) -> None:
    """Print unless ``NLSE_QUIET`` asked for silence.

    Choosing a backend talks: it announces what it picked, and if it has to
    time the backends to decide, it narrates that too. Under a parameter sweep
    or in a notebook that is noise, which is what NLSE_QUIET is for -- except
    that it reached one of those lines and every other one printed regardless.

    Read per call rather than once at import, so that setting the variable
    inside a session works, and so tests can set it without reloading the
    package.

    Parameters
    ----------
    message : str
        What to say.
    """
    if os.environ.get("NLSE_QUIET", "0") != "1":
        print(message)


def get_cache_dir() -> Path:
    """Get the directory for cached backend benchmarks.

    Uses the platform cache location rather than the package directory.
    Writing inside the installed package fails outright on a read-only or
    system-wide install, and even when it succeeds it leaves the directory
    behind on uninstall, because pip only removes files it recorded at
    install time. What is left is a directory named after the package with
    no ``__init__.py``, which Python then imports as a namespace package,
    so ``import NLSE`` silently resolves to an empty module.

    Set ``NLSE_CACHE_DIR`` to override, which CI does so it can cache a
    known path across runs.

    Returns
    -------
    Path
        The cache directory, created if missing.
    """
    override = os.environ.get("NLSE_CACHE_DIR")
    if override:
        cache_dir = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        cache_dir = Path(base) / "NLSE" / "Cache"
    elif sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches" / "NLSE"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        cache_dir = Path(base) / "NLSE"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_benchmark_cache_path() -> Path:
    """Get path to FFT benchmark cache file."""
    return get_cache_dir() / "fft_benchmark.json"
