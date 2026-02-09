"""OpenCL kernels using native OpenCL C code.

Hand-written OpenCL C kernels with fused operations for maximum performance.
"""

import pyopencl as cl
from pyopencl import array as cla
import numpy as np


# OpenCL C kernel source for fused nonlinear propagation
NL_PROP_KERNEL = """
__kernel void nl_prop_fused(
    __global float2* A,
    __global const float* A_sq,
    __global const float* V,
    const float dz,
    const float alpha,
    const float g,
    const float Isat
) {
    int idx = get_global_id(0);

    // Saturation
    float sat = 1.0f / (1.0f + A_sq[idx] / Isat);

    // Build complex argument: arg = (1j * g * A_sq * sat - alpha * sat + 1j * V) * dz
    // arg = (-alpha * sat, g * A_sq * sat + V) * dz
    float arg_real = -alpha * sat * dz;
    float arg_imag = (g * A_sq[idx] * sat + V[idx]) * dz;

    // Compute exp(arg) = exp(arg_real) * (cos(arg_imag) + i*sin(arg_imag))
    float exp_real_part = exp(arg_real);
    float cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);

    float2 exp_arg = (float2)(exp_real_part * cos_imag, exp_real_part * sin_imag);

    // Complex multiplication: A *= exp_arg
    float2 A_val = A[idx];
    A[idx] = (float2)(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
"""

NL_PROP_WITHOUT_V_KERNEL = """
__kernel void nl_prop_without_v_fused(
    __global float2* A,
    __global const float* A_sq,
    const float dz,
    const float alpha,
    const float g,
    const float Isat
) {
    int idx = get_global_id(0);

    // Saturation
    float sat = 1.0f / (1.0f + A_sq[idx] / Isat);

    // Build complex argument: arg = (1j * g * A_sq * sat - alpha * sat) * dz
    float arg_real = -alpha * sat * dz;
    float arg_imag = g * A_sq[idx] * sat * dz;

    // Compute exp(arg)
    float exp_real_part = exp(arg_real);
    float cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);

    float2 exp_arg = (float2)(exp_real_part * cos_imag, exp_real_part * sin_imag);

    // Complex multiplication: A *= exp_arg
    float2 A_val = A[idx];
    A[idx] = (float2)(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
"""

NL_PROP_C_KERNEL = """
__kernel void nl_prop_c_fused(
    __global float2* A1,
    __global const float* A_sq_1,
    __global const float* A_sq_2,
    __global const float* V,
    const float dz,
    const float alpha,
    const float g11,
    const float g12,
    const float Isat1,
    const float Isat2
) {
    int idx = get_global_id(0);

    // Saturation parameter
    float sat = 1.0f / (1.0f + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);

    // Build complex argument
    float arg_real = -alpha * sat * dz;
    float arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat + V[idx]) * dz;

    // Compute exp(arg)
    float exp_real_part = exp(arg_real);
    float cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);

    float2 exp_arg = (float2)(exp_real_part * cos_imag, exp_real_part * sin_imag);

    // Complex multiplication
    float2 A_val = A1[idx];
    A1[idx] = (float2)(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
"""

NL_PROP_C_WITHOUT_V_KERNEL = """
__kernel void nl_prop_c_without_v_fused(
    __global float2* A1,
    __global const float* A_sq_1,
    __global const float* A_sq_2,
    const float dz,
    const float alpha,
    const float g11,
    const float g12,
    const float Isat1,
    const float Isat2
) {
    int idx = get_global_id(0);

    // Saturation parameter
    float sat = 1.0f / (1.0f + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);

    // Build complex argument
    float arg_real = -alpha * sat * dz;
    float arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat) * dz;

    // Compute exp(arg)
    float exp_real_part = exp(arg_real);
    float cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);

    float2 exp_arg = (float2)(exp_real_part * cos_imag, exp_real_part * sin_imag);

    // Complex multiplication
    float2 A_val = A1[idx];
    A1[idx] = (float2)(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
"""

SQUARE_MOD_KERNEL = """
__kernel void square_mod_fused(
    __global const float2* A,
    __global float* A_sq
) {
    int idx = get_global_id(0);
    float2 A_val = A[idx];
    A_sq[idx] = A_val.x * A_val.x + A_val.y * A_val.y;
}
"""


class OpenCLKernels:
    """OpenCL C kernels for NLSE operations."""

    def __init__(self, context, queue):
        """Compile all kernels.

        Args:
            context: OpenCL context
            queue: OpenCL command queue
        """
        self.context = context
        self.queue = queue

        # Compile all kernels
        self._program = cl.Program(
            context,
            NL_PROP_KERNEL
            + NL_PROP_WITHOUT_V_KERNEL
            + NL_PROP_C_KERNEL
            + NL_PROP_C_WITHOUT_V_KERNEL
            + SQUARE_MOD_KERNEL,
        ).build()

        # Cache kernel objects
        self._nl_prop_kernel = self._program.nl_prop_fused
        self._nl_prop_without_v_kernel = self._program.nl_prop_without_v_fused
        self._nl_prop_c_kernel = self._program.nl_prop_c_fused
        self._nl_prop_c_without_v_kernel = self._program.nl_prop_c_without_v_fused
        self._square_mod_kernel = self._program.square_mod_fused

    def nl_prop(
        self,
        A: cla.Array,
        A_sq: cla.Array,
        dz: float,
        alpha: float,
        V: cla.Array,
        g: float,
        Isat: float,
    ) -> None:
        """Fused nonlinear propagation kernel (with potential).

        Args:
            A: Complex field array
            A_sq: Field intensity (modulus squared)
            dz: Propagation step
            alpha: Loss coefficient
            V: Potential array
            g: Nonlinear interaction strength
            Isat: Saturation intensity
        """
        global_size = (int(A.size),)
        self._nl_prop_kernel(
            self.queue,
            global_size,
            None,
            A.data,
            A_sq.data,
            V.data,
            np.float32(dz),
            np.float32(alpha),
            np.float32(g),
            np.float32(Isat),
        )

    def nl_prop_without_V(
        self,
        A: cla.Array,
        A_sq: cla.Array,
        dz: float,
        alpha: float,
        g: float,
        Isat: float,
    ) -> None:
        """Fused nonlinear propagation kernel (without potential).

        Args:
            A: Complex field array
            A_sq: Field intensity
            dz: Propagation step
            alpha: Loss coefficient
            g: Nonlinear interaction strength
            Isat: Saturation intensity
        """
        global_size = (int(A.size),)
        self._nl_prop_without_v_kernel(
            self.queue,
            global_size,
            None,
            A.data,
            A_sq.data,
            np.float32(dz),
            np.float32(alpha),
            np.float32(g),
            np.float32(Isat),
        )

    def nl_prop_c(
        self,
        A1: cla.Array,
        A_sq_1: cla.Array,
        A_sq_2: cla.Array,
        dz: float,
        alpha: float,
        V: cla.Array,
        g11: float,
        g12: float,
        Isat1: float,
        Isat2: float,
    ) -> None:
        """Fused coupled nonlinear propagation (with potential).

        Args:
            A1: First component field
            A_sq_1: First component intensity
            A_sq_2: Second component intensity
            dz: Propagation step
            alpha: Loss coefficient
            V: Potential array
            g11: Self-interaction strength (component 1)
            g12: Cross-interaction strength
            Isat1: Saturation intensity (component 1)
            Isat2: Saturation intensity (component 2)
        """
        global_size = (int(A1.size),)
        self._nl_prop_c_kernel(
            self.queue,
            global_size,
            None,
            A1.data,
            A_sq_1.data,
            A_sq_2.data,
            V.data,
            np.float32(dz),
            np.float32(alpha),
            np.float32(g11),
            np.float32(g12),
            np.float32(Isat1),
            np.float32(Isat2),
        )

    def nl_prop_without_V_c(
        self,
        A1: cla.Array,
        A_sq_1: cla.Array,
        A_sq_2: cla.Array,
        dz: float,
        alpha: float,
        g11: float,
        g12: float,
        Isat1: float,
        Isat2: float,
    ) -> None:
        """Fused coupled nonlinear propagation (without potential).

        Args:
            A1: First component field
            A_sq_1: First component intensity
            A_sq_2: Second component intensity
            dz: Propagation step
            alpha: Loss coefficient
            g11: Self-interaction strength
            g12: Cross-interaction strength
            Isat1: Saturation intensity (component 1)
            Isat2: Saturation intensity (component 2)
        """
        global_size = (int(A1.size),)
        self._nl_prop_c_without_v_kernel(
            self.queue,
            global_size,
            None,
            A1.data,
            A_sq_1.data,
            A_sq_2.data,
            np.float32(dz),
            np.float32(alpha),
            np.float32(g11),
            np.float32(g12),
            np.float32(Isat1),
            np.float32(Isat2),
        )

    def square_mod(self, A: cla.Array, A_sq: cla.Array) -> None:
        """Compute square modulus (intensity).

        Args:
            A: Complex field array
            A_sq: Output intensity array
        """
        global_size = (int(A.size),)
        self._square_mod_kernel(
            self.queue,
            global_size,
            None,
            A.data,
            A_sq.data,
        )

    def rabi_coupling(self, A: cla.Array, dz: float, omega: float) -> None:
        """Apply Rabi coupling term using PyOpenCL array expressions.

        Args:
            A: Field array (two-component)
            dz: Solver step
            omega: Rabi coupling strength
        """
        _rabi_coupling_impl(A, dz, omega)

    def vortex_cp(
        self, im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
    ) -> None:
        """Generate vortex of charge ll at position (i, j) using PyOpenCL.

        Args:
            im: Image array
            i: Vortex row position
            j: Vortex column position
            ii: Row coordinate meshgrid
            jj: Column coordinate meshgrid
            ll: Vortex charge
        """
        _vortex_cp_impl(im, i, j, ii, jj, ll)


# Helper functions using PyOpenCL array expressions (not yet optimized with OpenCL C)


def _rabi_coupling_impl(A, dz: float, omega: float) -> None:
    """Apply a Rabi coupling term using PyOpenCL array expressions.

    Args:
        A (cla.Array): First field / component
        dz (float): Solver step
        omega (float): Rabi coupling strength
    """
    from pyopencl import clmath

    A1 = A[..., 0, :, :]
    A2 = A[..., 1, :, :]
    A1_old = A1.copy()
    A1[:] = clmath.cos(omega * dz) * A1 - 1j * clmath.sin(omega * dz) * A2
    A2[:] = clmath.cos(omega * dz) * A2 - 1j * clmath.sin(omega * dz) * A1_old


def _vortex_cp_impl(
    im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
) -> None:
    """Generate a vortex of charge ll at position (i,j) using PyOpenCL.

    Args:
        im (cla.Array): Image
        i (int): position row of the vortex
        j (int): position column of the vortex
        ii (cla.Array): meshgrid position row (coordinates of the image)
        jj (cla.Array): meshgrid position column (coordinates of the image)
        ll (int): vortex charge
    """
    import pyopencl.clmath as clm

    # Compute complex argument raised to power ll
    # Use atan2 for correct phase angle in all quadrants
    arg = ((ii - i) + 1j * (jj - j)) ** ll
    # Extract phase angle: atan2(imaginary_part, real_part)
    im += clm.atan2(arg.imag, arg.real)
