"""CPU backend implementation."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import fft as scipy_fft, signal

from ..kernels import cpu as kernels_cpu
from .backend import Backend


class CPUBackend(Backend):
    """CPU backend using NumPy and scipy's pocketfft.

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
    from 37 ms to 6.5. On x86 the FFTW wheels are vectorized and the gap
    should be smaller or the other way round -- unmeasured, and the reason to
    keep this behind the Backend interface rather than inline it.
    """

    supports_unnormalized_ifft = True

    @property
    def name(self) -> str:
        return "CPU"

    def allocate_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate a field array."""
        return np.zeros(shape, dtype=dtype)

    def allocate_real_field(self, shape: tuple, dtype: npt.DTypeLike) -> np.ndarray:
        """Allocate a real field array."""
        return np.zeros(shape, dtype=dtype)

    @property
    def convolution(self):
        """Return scipy's overlap-add convolution."""
        return signal.oaconvolve

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

        pocketfft plans per call from a cache of its own, so there is nothing
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
            The axes, in the one-element list every backend returns.
        """
        return [axes]

    def fft(self, array: np.ndarray, plan: list) -> np.ndarray:
        """Transform forward, reusing the input buffer where scipy can."""
        return scipy_fft.fftn(array, axes=plan[0], overwrite_x=True, workers=-1)

    def ifft(self, array: np.ndarray, plan: list, normalize: bool = True) -> np.ndarray:
        """Transform back, with the 1/N left to the propagator if asked.

        scipy puts the scaling where ``norm`` says rather than in a pass of
        its own, so skipping it is a different convention rather than a step
        not taken: ``norm="forward"`` is the unnormalized inverse.
        """
        return scipy_fft.ifftn(
            array,
            axes=plan[0],
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
