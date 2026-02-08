"""Backend implementations for NLSE solvers."""

from .backend import Backend
from .cpu import CPUBackend

__all__ = ["Backend", "CPUBackend", "get_backend", "list_available_backends"]

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


def get_backend(name: str) -> Backend:
    """Get backend instance by name.

    Args:
        name: Backend name ("CPU", "CUPY", "CL")

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not available
    """
    name = name.upper()

    if name == "CPU":
        return CPUBackend()
    elif name in ("CUPY", "GPU"):
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
