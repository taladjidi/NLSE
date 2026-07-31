"""Backend implementations for NLSE solvers."""

import os
import warnings
from collections.abc import Callable

from ..utils import say
from .backend import Backend
from .cpu import CPUBackend

__all__ = [
    "Backend",
    "CPUBackend",
    "backends_by_speed",
    "clear_backend_cache",
    "fastest_backend_supporting",
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

    # A backend that is not installed is answered with the fastest one that
    # is, rather than by refusing: the backend is how a run goes, not what it
    # computes, and a script that names one is portable only if naming it is
    # a preference. A name that is not a backend at all is still an error --
    # that is a typo, and guessing at it would hide the typo.
    builders: dict = {"CPU": (True, CPUBackend, "")}
    if _CUPY_AVAILABLE:
        builders["CUPY"] = (True, CUPYBackend, "")
    else:
        builders["CUPY"] = (False, None, "install cupy")
    if _OPENCL_AVAILABLE:
        builders["CL"] = (True, OpenCLBackend, "")
    else:
        builders["CL"] = (False, None, "install pyopencl")
    if _MLX_AVAILABLE:
        builders["MLX"] = (True, MLXBackend, "")
    else:
        builders["MLX"] = (False, None, "install mlx")

    if name not in builders:
        raise ValueError(f"Unknown backend: {name}")
    ok, build, how = builders[name]
    if not ok:
        replacement = backends_by_speed(grid_size)[0]
        warnings.warn(
            f"The {name} backend is not installed here, so {replacement} will "
            f"be used instead — the fastest one available. {how.capitalize()} "
            f"to use {name}, or pass backend='{replacement}' to silence this.",
            stacklevel=2,
        )
        return get_backend(replacement, grid_size)

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


def backends_by_speed(grid_size: tuple = (2048, 2048)) -> list[str]:
    """Return the available backends, fastest first.

    Read from the benchmark cache when there is one, so asking this question
    is free in the common case. Without a cache the order is the static one
    ``list_available_backends`` gives, which is not a measurement and does not
    pretend to be: benchmarking every backend to answer a fallback would cost
    more than the fallback saves, and the cache fills the first time anything
    asks for ``backend="auto"``.

    Parameters
    ----------
    grid_size : tuple
        Grid the cached measurement must have been taken at to be used.

    Returns
    -------
    list[str]
        Available backend names, fastest first where that is known.
    """
    from .benchmark import load_benchmark_cache

    available = list_available_backends()
    cache = load_benchmark_cache()
    if cache is None or tuple(cache.get("grid_size", [])) != tuple(grid_size):
        return available
    times = {
        name: entry.get("time_ms")
        for name, entry in (cache.get("results") or {}).items()
        if entry.get("time_ms") is not None
    }
    return sorted(
        available, key=lambda name: (times.get(name) is None, times.get(name, 0.0))
    )


def fastest_backend_supporting(
    supports: Callable[[Backend], bool], grid_size: tuple = (2048, 2048)
) -> Backend | None:
    """Return the fastest available backend that can serve a requirement.

    Parameters
    ----------
    supports : Callable
        Called with each candidate backend; True if it can serve the run.
    grid_size : tuple
        Grid size, used to pick the right cached ranking and to build the
        backend.

    Returns
    -------
    Backend or None
        The fastest backend that passes, or None if none does.
    """
    for name in backends_by_speed(grid_size):
        try:
            candidate = get_backend(name, grid_size=grid_size)
        except Exception:  # pragma: no cover - a backend that will not build
            continue
        if supports(candidate):
            return candidate
    return None
