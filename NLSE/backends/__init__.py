"""Backend implementations for NLSE solvers."""

import os

from ..utils import say
from .backend import Backend
from .cpu import CPUBackend

__all__ = [
    "Backend",
    "CPUBackend",
    "clear_backend_cache",
    "get_backend",
    "get_optimal_backend",
    "list_available_backends",
]

# Conditional imports
try:
    from .cupy_backend import CUPYBackend

    __all__.append("CUPYBackend")
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

try:
    from .opencl import OpenCLBackend

    __all__.append("OpenCLBackend")
    _OPENCL_AVAILABLE = True
except ImportError:
    _OPENCL_AVAILABLE = False

try:
    from .mlx_backend import MLXBackend

    __all__.append("MLXBackend")
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

# Environment variables for backend control
_ENV_BACKEND = os.environ.get("NLSE_BACKEND", "").upper()
_FORCE_BENCHMARK = os.environ.get("NLSE_FORCE_BENCHMARK", "0") == "1"


def get_optimal_backend(
    grid_size: tuple = (2048, 2048), force_benchmark: bool = False
) -> str:
    """Get the optimal backend for current hardware.

    Automatically benchmarks all available backends and returns
    the fastest one. Results are cached for future calls.

    Parameters
    ----------
    grid_size : tuple
        Typical grid size for benchmarking
    force_benchmark : bool
        Force re-benchmark even if cache exists

    Returns
    -------
    str
        Name of fastest backend ("CPU", "CUPY", or "CL")

    """
    from .benchmark import get_fastest_backend

    return get_fastest_backend(grid_size, force_benchmark)


# One backend instance per resolved name, shared by every solver.
#
# A backend owns a device connection, not per-simulation state: nothing
# assigns to its attributes, and its kernels are compiled once and cached on
# it. Handing out a fresh one per solver therefore bought nothing and cost a
# whole OpenCL context and command queue each time, none of which are ever
# released. A parameter sweep that builds a solver per point ran the process
# out of file descriptors ("OSError: [Errno 24] Too many open files"), which
# surfaces as every later test erroring in setup rather than as anything to
# do with the solver that leaked.
_BACKEND_CACHE: dict[str, Backend] = {}


def clear_backend_cache() -> None:
    """Drop the cached backend instances.

    Only useful to a test that needs a device connection rebuilt; ordinary
    code should let the cache do its job.
    """
    _BACKEND_CACHE.clear()


def get_backend(name: str, grid_size: tuple = (2048, 2048)) -> Backend:
    """Get backend instance by name.

    Instances are cached per backend name, so repeated calls hand back the
    same object rather than opening another device context.

    Parameters
    ----------
    name : str
        Backend name ("CPU", "CUPY", "CL", "MLX", or "auto")
    grid_size : tuple
        Grid size for auto-benchmarking (only used if name="auto")

    Returns
    -------
    Backend
        Backend instance

    Raises
    ------
    ValueError
        If backend not available

    """
    # Override from environment variable
    if _ENV_BACKEND and _ENV_BACKEND != "AUTO":
        name = _ENV_BACKEND

    name = name.upper()

    # Auto-select optimal backend
    if name == "AUTO":
        name = get_optimal_backend(grid_size, force_benchmark=_FORCE_BENCHMARK)
        say(f"Auto-selected FFT backend: {name}")

    # Validate before consulting the cache, so an unavailable backend raises
    # the same way whether or not something asked for it earlier.
    build: type[Backend]
    if name == "CPU":
        build = CPUBackend
    elif name == "CUPY":
        if not _CUPY_AVAILABLE:
            raise ValueError("CUPY backend not available - install cupy")
        build = CUPYBackend
    elif name == "CL":
        if not _OPENCL_AVAILABLE:
            raise ValueError("OpenCL backend not available - install pyopencl")
        build = OpenCLBackend
    elif name == "MLX":
        if not _MLX_AVAILABLE:
            raise ValueError("MLX backend not available - install mlx")
        build = MLXBackend
    else:
        raise ValueError(f"Unknown backend: {name}")

    if name not in _BACKEND_CACHE:
        _BACKEND_CACHE[name] = build()
    return _BACKEND_CACHE[name]


def list_available_backends() -> list[str]:
    """List available backend names.

    Returns
    -------
    list[str]
        List of available backend names

    """
    backends = ["CPU"]
    if _CUPY_AVAILABLE:
        backends.append("CUPY")
    if _OPENCL_AVAILABLE:
        backends.append("CL")
    if _MLX_AVAILABLE:
        backends.append("MLX")
    return backends
