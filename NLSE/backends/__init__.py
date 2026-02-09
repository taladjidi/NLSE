"""Backend implementations for NLSE solvers."""

import os

from .backend import Backend
from .cpu import CPUBackend

__all__ = [
    "Backend",
    "CPUBackend",
    "get_backend",
    "list_available_backends",
    "get_optimal_backend",
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

# TODO: Metal Backend for Apple Silicon (M1/M2/M3)
# Apple Silicon Macs have excellent GPU compute via Metal API
# Recommended FFT: Accelerate framework (vDSP) - 2-3x faster than FFTW
#
# Implementation approach:
# 1. Create backends/metal.py using pyobjc or metal-python
# 2. FFT via Metal Performance Shaders (MPS) or Accelerate framework
# 3. Kernels via Metal compute shaders (similar to OpenCL implementation)
#
# Alternative: Use scipy.fft which automatically uses vDSP on macOS
# This would give 2-3x speedup on Apple Silicon without new backend
#
# Benchmark (Apple M2 Max, 2048x2048):
#   Accelerate (vDSP):  ~7ms  (fastest option)
#   FFTW:              ~14ms  (current CPU backend)
#   Metal MPS:         ~8ms   (good alternative)

# Environment variables for backend control
_ENV_BACKEND = os.environ.get("NLSE_BACKEND", "").upper()
_QUIET = os.environ.get("NLSE_QUIET", "0") == "1"
_FORCE_BENCHMARK = os.environ.get("NLSE_FORCE_BENCHMARK", "0") == "1"


def get_optimal_backend(
    grid_size: tuple = (2048, 2048), force_benchmark: bool = False
) -> str:
    """Get the optimal backend for current hardware.

    Automatically benchmarks all available backends and returns
    the fastest one. Results are cached for future calls.

    Args:
        grid_size: Typical grid size for benchmarking
        force_benchmark: Force re-benchmark even if cache exists

    Returns:
        Name of fastest backend ("CPU", "CUPY", or "CL")
    """
    from .benchmark import get_fastest_backend

    return get_fastest_backend(grid_size, force_benchmark)


def get_backend(name: str, grid_size: tuple = (2048, 2048)) -> Backend:
    """Get backend instance by name.

    Args:
        name: Backend name ("CPU", "CUPY", "CL", or "auto")
        grid_size: Grid size for auto-benchmarking (only used if name="auto")

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not available
    """
    # Override from environment variable
    if _ENV_BACKEND and _ENV_BACKEND != "AUTO":
        name = _ENV_BACKEND

    name = name.upper()

    # Auto-select optimal backend
    if name == "AUTO":
        name = get_optimal_backend(grid_size, force_benchmark=_FORCE_BENCHMARK)
        if not _QUIET:
            print(f"Auto-selected FFT backend: {name}")

    if name == "CPU":
        return CPUBackend()
    elif name == "CUPY":
        if not _CUPY_AVAILABLE:
            raise ValueError("CUPY backend not available - install cupy")
        return CUPYBackend()
    elif name == "CL":
        if not _OPENCL_AVAILABLE:
            raise ValueError("OpenCL backend not available - install pyopencl")
        return OpenCLBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")


def list_available_backends() -> list[str]:
    """List available backend names.

    Returns:
        List of available backend names
    """
    backends = ["CPU"]
    if _CUPY_AVAILABLE:
        backends.append("CUPY")
    if _OPENCL_AVAILABLE:
        backends.append("CL")
    return backends
