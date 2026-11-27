"""CPU backend implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import fft as scipy_fft

from ..kernels import cpu as kernels_cpu
from ..kernels.cpu_blocked import BlockedStep
from . import cpu_tuning
from .backend import Backend

# The axes of a 2D transform, the only one the blocked step takes.
BLOCKED_AXES = (-2, -1)

# Points along an axis of the smallest grid tune_cpu_step measures.
MIN_GRID_SIZE = 1


class CPUPlan:
    """What the CPU backend builds for a transform: its axes, and a blocked step.

    scipy plans per call, so the axes are all the transform needs. The plan
    also carries the way to the grid's blocked step, which is how
    ``kernels.linear_nonlinear_step`` reaches it.

    Parameters
    ----------
    backend : CPUBackend
        The backend, whose unblocked step the blocked one is measured against.
    shape : tuple
        Array shape.
    axes : tuple
        Axes to transform over.
    dtype : np.dtype
        Array dtype.
    """

    def __init__(self, backend: Backend, shape: tuple, axes: tuple, dtype: np.dtype):
        self.backend = backend
        self.shape = tuple(shape)
        self.axes = axes
        self.dtype = np.dtype(dtype)

    def blocked_step(self) -> BlockedStep | None:
        """Return the blocked step for this grid, or None to take the unblocked one.

        Returns
        -------
        BlockedStep or None
            None for anything but one complex 2D field transformed over both
            axes, and wherever ``cpu_tuning.blocked_step`` says so.
        """
        if (
            len(self.shape) != len(BLOCKED_AXES)
            or tuple(self.axes) != BLOCKED_AXES
            or self.dtype.kind != "c"
        ):
            return None
        return cpu_tuning.blocked_step(self.backend, self.shape, self.dtype)


class CPUBackend(Backend):
    """CPU backend using NumPy and ``scipy.fft``.

    Provides no fused entry points: the transform is driven from Python and
    the numba kernels are already single-pass, so there is no launch
    overhead to amortize.

    It does skip the inverse transform's normalization, which scipy expresses
    as ``norm="forward"`` -- the same transform with the 1/N moved to the
    other direction, where the pre-normalized propagator already carries it.

    It used to be pyFFTW, and most of this file was the apparatus that needed:
    disk wisdom to be loaded, validated by timing against scipy, and written
    back; a planning pass that overwrote the array it planned with; a detector
    for FFTW builds without vector codelets; and an import-order guard, because
    the wheel vendors an OpenMP runtime that segfaults numba's if it
    initializes first. All of it is gone, and the transform got faster: on an
    M3 Max, where no prebuilt FFTW has NEON codelets, a 2048x2048 pair went
    from 37 ms to 6.5. On x86 the FFTW wheels are vectorized, and on a Ryzen 3
    4100 pyFFTW is still ahead at 512x512 only (docs/optimization-log.md). The
    transform stays behind the Backend interface, so another library is a
    class, not a rewrite.
    """

    supports_unnormalized_ifft = True

    # Taken where the grid was measured to gain from it (cpu_tuning).
    has_linear_nonlinear_step = True

    @property
    def name(self) -> str:
        return "CPU"

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate a field array."""
        return np.zeros(shape, dtype=dtype)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate a real field array."""
        return np.zeros(shape, dtype=dtype)

    def to_numpy(self, array: np.ndarray) -> np.ndarray:
        """Already numpy, return as-is."""
        return array

    def from_numpy(self, array: np.ndarray) -> np.ndarray:
        """Convert to contiguous array."""
        return np.ascontiguousarray(array)

    def _build_fft(
        self,
        shape: tuple,
        axes: tuple,
        dtype: np.dtype,
        array: np.ndarray | None = None,
    ) -> list:
        """Record the axes to transform over.

        scipy plans per call from a cache of its own, so there is nothing
        to build here and nothing to keep warm -- which is what let the disk
        wisdom, its staleness check and the planning pass go. ``shape``,
        ``dtype`` and ``array`` are what a planner would have needed and are
        accepted so that every backend is built the same way.

        Parameters
        ----------
        shape : tuple
            Array shape. Unused.
        axes : tuple
            Axes to transform over.
        dtype : np.dtype
            Array dtype. Unused.
        array : np.ndarray or None
            The array that will be transformed. Unused, and no longer
            overwritten: FFTW_MEASURE planned by running transforms on it,
            so this used to have to save and restore the field.

        Returns
        -------
        list
            A ``CPUPlan``, in the one-element list every backend returns.
        """
        return [CPUPlan(self, shape, axes, dtype)]

    def fft(self, array: np.ndarray, plan: list) -> np.ndarray:
        """Transform forward, reusing the input buffer where scipy can."""
        return scipy_fft.fftn(array, axes=plan[0].axes, overwrite_x=True, workers=-1)

    def ifft(self, array: np.ndarray, plan: list, normalize: bool = True) -> np.ndarray:
        """Transform back, with the 1/N left to the propagator if asked.

        scipy puts the scaling where ``norm`` says rather than in a pass of
        its own, so skipping it is a different convention rather than a step
        not taken: ``norm="forward"`` is the unnormalized inverse.
        """
        return scipy_fft.ifftn(
            array,
            axes=plan[0].axes,
            overwrite_x=True,
            workers=-1,
            norm="backward" if normalize else "forward",
        )

    def exp(self, array: Any) -> Any:
        """Exponentiate without leaving this backend."""
        return np.exp(array)

    def sum(self, array: Any) -> float:
        """Reduce without leaving this backend."""
        return float(np.sum(array))

    @property
    def kernels(self) -> Any:
        """Return CPU kernels module."""
        return kernels_cpu

    def supports_double_precision(self) -> bool:
        """CPU always supports double precision."""
        return True


def tune_cpu_step(
    shape: tuple, dtype: npt.DTypeLike = np.complex64, force: bool = False
) -> dict | None:
    """Measure whether, and how, the CPU step should be blocked on a grid.

    What a run does the first time it meets a grid, callable ahead of it:
    the unblocked step is timed against blocked ones and the fastest kept in
    the cache directory, for every later run on this machine.

    Parameters
    ----------
    shape : tuple
        ``(NY, NX)`` of the field.
    dtype : npt.DTypeLike
        complex64 or complex128.
    force : bool
        Measure again even where the cache holds a fresh entry.

    Returns
    -------
    dict or None
        ``rows`` and ``threads`` of the blocked step, both None where the
        unblocked one is faster; ``unblocked_ms`` and ``blocked_ms``, the
        best per-step times; ``timestamp``. None, measuring nothing, while
        another process tunes (it holds ``cpu_blocking.lock`` in the cache
        directory): call again once it is done.

    Raises
    ------
    ValueError
        If ``shape`` is not two positive sizes or ``dtype`` not complex.
    """
    grid = tuple(int(n) for n in shape)
    width = np.dtype(dtype)
    if len(grid) != len(BLOCKED_AXES) or min(grid) < MIN_GRID_SIZE:
        raise ValueError(f"shape must be (NY, NX), got {shape}")
    if width not in (np.complex64, np.complex128):
        raise ValueError(f"dtype must be complex64 or complex128, got {width}")
    return cpu_tuning.tune(CPUBackend(), grid, width, force)
