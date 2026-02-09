"""OpenCL kernels using PyOpenCL array expressions.

NOTE: Most kernels have been replaced with optimized hand-written OpenCL C
implementations in cl_optimized.py for 3-5× speedup. This module only contains
functions that are not yet optimized.
"""

from pyopencl import array as cla
from pyopencl import clmath


def rabi_coupling(A, dz: float, omega: float) -> None:
    """Apply a Rabi coupling term.
    This function implements the Rabi hopping term.
    It exchanges density between the two components.

    Args:
        A (cla.Array): First field / component
        dz (float): Solver step
        omega (float): Rabi coupling strength
    """
    A1 = A[..., 0, :, :]
    A2 = A[..., 1, :, :]
    A1_old = A1.copy()
    A1[:] = clmath.cos(omega * dz) * A1 - 1j * clmath.sin(omega * dz) * A2
    A2[:] = clmath.cos(omega * dz) * A2 - 1j * clmath.sin(omega * dz) * A1_old


def vortex_cp(
    im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
) -> None:
    """Generates a vortex of charge l at a position (i,j) on the image im.

    Args:
        im (np.ndarray): Image
        i (int): position row of the vortex
        j (int): position column of the vortex
        ii (int): meshgrid position row (coordinates of the image)
        jj (int): meshgrid position column (coordinates of the image)
        ll (int): vortex charge

    Returns:
        None
    """
    # Compute complex argument raised to power ll
    # This fixes bug: was using atan() which is incorrect - should use atan2
    arg = ((ii - i) + 1j * (jj - j)) ** ll
    # Extract phase angle: atan2(imaginary_part, real_part)
    import pyopencl.clmath as clm

    im += clm.atan2(arg.imag, arg.real)
