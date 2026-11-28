import json
import os
import platform
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

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
    A write there fails on a read-only install, and a directory left
    behind on uninstall is imported as a namespace package, so
    ``import NLSE`` resolves to an empty module.

    Set ``NLSE_CACHE_DIR`` to override; the test suite does, so that no
    test reads or writes the user's cache.

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


def get_blocking_cache_path() -> Path:
    """Get path to the cache of the CPU step's measured blocks."""
    return get_cache_dir() / "cpu_blocking.json"


def machine_identity() -> tuple[str, str]:
    """Return what a cached timing is a statement about: OS and processor type.

    Returns
    -------
    tuple of str
        ``(platform.system(), platform.machine())``.
    """
    return (platform.system(), platform.machine())


def write_json_atomically(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path``, all of it or none of it.

    Written beside the file and renamed over it, so a reader never sees a
    half-written file and a failed write leaves the previous one whole.

    Parameters
    ----------
    path : Path
        File to write.
    data : Any
        What ``json.dump`` can write.

    Raises
    ------
    OSError, TypeError, ValueError
        As ``open`` and ``json.dump`` raise them, after removing the partial
        file.
    """
    # Per process and thread: two writers never share a partial file.
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(partial, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
