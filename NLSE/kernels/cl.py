"""OpenCL kernels using native OpenCL C code.

Hand-written OpenCL C kernels with fused operations for maximum performance.
Supports both single (float32/complex64) and double (float64/complex128) precision
on devices that support double precision.
"""

from pathlib import Path

import numpy as np
import pyopencl as cl
from pyopencl import array as cla

# Module-level cache for compiled programs
# Key: (context_hash, precision) -> Value: compiled cl.Program
_COMPILED_PROGRAMS = {}


def _check_double_support(context):
    """Check if context devices support double precision.

    Parameters
    ----------
    context : PyOpenCL context
        PyOpenCL context

    Returns
    -------
    bool
        True if all devices support double precision

    """
    for device in context.devices:
        if device.double_fp_config == 0:
            return False
    return True


def _load_kernel_template():
    """Load OpenCL kernel template from file.

    Returns
    -------
    str
        String containing kernel template with {{placeholders}}

    """
    template_path = Path(__file__).parent / "cl_source" / "kernels.cl"
    return template_path.read_text()


def _get_kernel_source(precision="single"):
    """Generate OpenCL C kernel source for specified precision.

    Parameters
    ----------
    precision : str
        'single' for float32 or 'double' for float64

    Returns
    -------
    str
        String containing all kernel source code

    """
    if precision == "single":
        fp_type = "float"
        fp2_type = "float2"
        fp_suffix = "f"
        pragma = ""
    elif precision == "double":
        fp_type = "double"
        fp2_type = "double2"
        fp_suffix = ""
        pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n"
    else:
        raise ValueError(f"precision must be 'single' or 'double', got {precision}")

    # Load template and substitute placeholders
    template = _load_kernel_template()
    source = template.replace("{{FP_TYPE}}", fp_type)
    source = source.replace("{{FP2_TYPE}}", fp2_type)
    source = source.replace("{{FP_SUFFIX}}", fp_suffix)

    return pragma + source


def _compile_kernels(context, precision="single"):
    """Compile OpenCL kernels for specified precision.

    Uses module-level cache to compile only once per (context, precision).

    Parameters
    ----------
    context : PyOpenCL context
        PyOpenCL context
    precision : str
        'single' or 'double'

    Returns
    -------
    cl.Program
        Compiled cl.Program

    Raises
    ------
    RuntimeError
        If double precision requested but not supported

    """
    # Check double precision support
    if precision == "double" and not _check_double_support(context):
        raise RuntimeError(
            "Double precision (complex128) requested but not supported by OpenCL device. "
            "Use complex64 (single precision) instead."
        )

    # Create cache key from context hash and precision
    ctx_hash = hash(context.int_ptr)
    cache_key = (ctx_hash, precision)

    # Check cache
    if cache_key in _COMPILED_PROGRAMS:
        return _COMPILED_PROGRAMS[cache_key]

    # Compile with aggressive optimizations
    source = _get_kernel_source(precision)
    build_options = [
        "-cl-fast-relaxed-math",  # All fast math optimizations
        "-cl-mad-enable",  # Allow fused multiply-add
    ]

    program = cl.Program(context, source).build(options=" ".join(build_options))

    # Cache for future use
    _COMPILED_PROGRAMS[cache_key] = program

    return program


class OpenCLKernels:
    """OpenCL C kernels for NLSE operations.

    Supports both single and double precision (if device supports it).
    Kernels are compiled once and cached for reuse.
    """

    def __init__(self, context, queue):
        """Initialize kernel wrapper (compilation happens lazily).

        Parameters
        ----------
        context : OpenCL context
            OpenCL context
        queue : OpenCL command queue
            OpenCL command queue
        """
        self.context = context
        self.queue = queue

        # Check and cache double precision support
        self._double_supported = _check_double_support(context)

        # Compiled programs and kernel objects cache (by precision)
        self._programs = {}
        self._kernels = {}

    def _get_kernels(self, dtype):
        """Get compiled kernels for given dtype, compiling if needed.

        Parameters
        ----------
        dtype : numpy.dtype
            numpy dtype (complex64 or complex128)

        Returns
        -------
        dict
            Dictionary of compiled kernel objects

        """
        # Detect precision from dtype
        if dtype == np.complex64:
            precision = "single"
        elif dtype == np.complex128:
            precision = "double"
            if not self._double_supported:
                raise RuntimeError(
                    f"Double precision (complex128) not supported on this OpenCL device "
                    f"({self.context.devices[0].name}). Use complex64 instead."
                )
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Use complex64 or complex128."
            )

        # Get or compile program and cache kernel objects
        if precision not in self._kernels:
            if precision not in self._programs:
                self._programs[precision] = _compile_kernels(self.context, precision)

            program = self._programs[precision]

            # Cache kernel objects to avoid repeated retrieval
            self._kernels[precision] = {
                "nl_prop": cl.Kernel(program, "nl_prop_fused"),
                "nl_prop_without_v": cl.Kernel(program, "nl_prop_without_v_fused"),
                "nl_prop_c": cl.Kernel(program, "nl_prop_c_fused"),
                "nl_prop_c_without_v": cl.Kernel(program, "nl_prop_c_without_v_fused"),
                "square_mod": cl.Kernel(program, "square_mod_fused"),
                # Optimized fused kernels
                "square_mod_nl_prop": cl.Kernel(program, "square_mod_nl_prop_fused"),
                "square_mod_nl_prop_v": cl.Kernel(
                    program, "square_mod_nl_prop_v_fused"
                ),
                "apply_propagator": cl.Kernel(program, "apply_propagator"),
                "rabi_coupling": cl.Kernel(program, "rabi_coupling"),
                # RK4 nonlinear RHS kernels
                "rk4_nl_rhs": cl.Kernel(program, "rk4_nl_rhs_fused"),
                "rk4_nl_rhs_v": cl.Kernel(program, "rk4_nl_rhs_v_fused"),
                "square_mod_rk4_nl_rhs": cl.Kernel(
                    program, "square_mod_rk4_nl_rhs_fused"
                ),
                "square_mod_rk4_nl_rhs_v": cl.Kernel(
                    program, "square_mod_rk4_nl_rhs_v_fused"
                ),
                "rk4_nl_rhs_c": cl.Kernel(program, "rk4_nl_rhs_c_fused"),
                "rk4_nl_rhs_c_v": cl.Kernel(program, "rk4_nl_rhs_c_v_fused"),
            }

        return self._kernels[precision]

    def nl_prop(
        self,
        A: cla.Array,
        A_sq: cla.Array,
        dz: float,
        alpha: float,
        V: cla.Array,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused nonlinear propagation kernel (with potential).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        A_sq : cla.Array
            Intensity array (|A|^2)
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        V : cla.Array
            Potential array
        g : float
            Nonlinear interaction strength
        Isat : float
            Saturation intensity

        Returns
        -------
        cla.Array
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)

        # Cast parameters to appropriate precision
        if A.dtype == np.complex64:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float32(dz),
                np.float32(alpha),
                np.float32(g),
                np.float32(Isat),
            )
        else:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float64(dz),
                np.float64(alpha),
                np.float64(g),
                np.float64(Isat),
            )

        kernels["nl_prop"](
            self.queue,
            global_size,
            None,
            A.data,
            A_sq.data,
            V.data,
            dz_cast,
            alpha_cast,
            g_cast,
            Isat_cast,
        )
        return A

    def nl_prop_without_V(
        self,
        A: cla.Array,
        A_sq: cla.Array,
        dz: float,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused nonlinear propagation kernel (without potential).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        A_sq : cla.Array
            Intensity array (|A|^2)
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        g : float
            Nonlinear interaction strength
        Isat : float
            Saturation intensity

        Returns
        -------
        cla.Array
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)

        if A.dtype == np.complex64:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float32(dz),
                np.float32(alpha),
                np.float32(g),
                np.float32(Isat),
            )
        else:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float64(dz),
                np.float64(alpha),
                np.float64(g),
                np.float64(Isat),
            )

        kernels["nl_prop_without_v"](
            self.queue,
            global_size,
            None,
            A.data,
            A_sq.data,
            dz_cast,
            alpha_cast,
            g_cast,
            Isat_cast,
        )
        return A

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
    ) -> cla.Array:
        """Fused coupled nonlinear propagation (with potential).

        Parameters
        ----------
        A1 : cla.Array
            Complex field (component 1)
        A_sq_1 : cla.Array
            Intensity of component 1
        A_sq_2 : cla.Array
            Intensity of component 2
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        V : cla.Array
            Potential array
        g11 : float
            Self-interaction strength
        g12 : float
            Cross-interaction strength
        Isat1 : float
            Saturation intensity (component 1)
        Isat2 : float
            Saturation intensity (component 2)

        Returns
        -------
        cla.Array
            The modified field array A1.
        """
        kernels = self._get_kernels(A1.dtype)
        global_size = (int(A1.size),)

        if A1.dtype == np.complex64:
            params = [np.float32(x) for x in [dz, alpha, g11, g12, Isat1, Isat2]]
        else:
            params = [np.float64(x) for x in [dz, alpha, g11, g12, Isat1, Isat2]]

        kernels["nl_prop_c"](
            self.queue,
            global_size,
            None,
            A1.data,
            A_sq_1.data,
            A_sq_2.data,
            V.data,
            *params,
        )
        return A1

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
    ) -> cla.Array:
        """Fused coupled nonlinear propagation (without potential).

        Parameters
        ----------
        A1 : cla.Array
            Complex field (component 1)
        A_sq_1 : cla.Array
            Intensity of component 1
        A_sq_2 : cla.Array
            Intensity of component 2
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        g11 : float
            Self-interaction strength
        g12 : float
            Cross-interaction strength
        Isat1 : float
            Saturation intensity (component 1)
        Isat2 : float
            Saturation intensity (component 2)

        Returns
        -------
        cla.Array
            The modified field array A1.
        """
        kernels = self._get_kernels(A1.dtype)
        global_size = (int(A1.size),)

        if A1.dtype == np.complex64:
            params = [np.float32(x) for x in [dz, alpha, g11, g12, Isat1, Isat2]]
        else:
            params = [np.float64(x) for x in [dz, alpha, g11, g12, Isat1, Isat2]]

        kernels["nl_prop_c_without_v"](
            self.queue, global_size, None, A1.data, A_sq_1.data, A_sq_2.data, *params
        )
        return A1

    def square_mod(self, A: cla.Array, A_sq: cla.Array) -> cla.Array:
        """Compute square modulus (intensity).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        A_sq : cla.Array
            Output intensity array

        Returns
        -------
        cla.Array
            The modified intensity array A_sq.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)
        kernels["square_mod"](self.queue, global_size, None, A.data, A_sq.data)
        return A_sq

    def square_mod_nl_prop(
        self,
        A: cla.Array,
        dz: float,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused square_mod + nl_prop_without_V (eliminates kernel launch overhead).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        g : float
            Nonlinear interaction strength
        Isat : float
            Saturation intensity

        Returns
        -------
        cla.Array
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)

        if A.dtype == np.complex64:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float32(dz),
                np.float32(alpha),
                np.float32(g),
                np.float32(Isat),
            )
        else:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float64(dz),
                np.float64(alpha),
                np.float64(g),
                np.float64(Isat),
            )

        kernels["square_mod_nl_prop"](
            self.queue,
            global_size,
            None,
            A.data,
            dz_cast,
            alpha_cast,
            g_cast,
            Isat_cast,
        )
        return A

    def square_mod_nl_prop_v(
        self,
        A: cla.Array,
        V: cla.Array,
        dz: float,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused square_mod + nl_prop (with potential, eliminates kernel launch overhead).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        V : cla.Array
            Potential array
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        g : float
            Nonlinear interaction strength
        Isat : float
            Saturation intensity

        Returns
        -------
        cla.Array
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)

        if A.dtype == np.complex64:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float32(dz),
                np.float32(alpha),
                np.float32(g),
                np.float32(Isat),
            )
        else:
            dz_cast, alpha_cast, g_cast, Isat_cast = (
                np.float64(dz),
                np.float64(alpha),
                np.float64(g),
                np.float64(Isat),
            )

        kernels["square_mod_nl_prop_v"](
            self.queue,
            global_size,
            None,
            A.data,
            V.data,
            dz_cast,
            alpha_cast,
            g_cast,
            Isat_cast,
        )
        return A

    def apply_propagator(self, A: cla.Array, propagator: cla.Array) -> cla.Array:
        """Apply linear propagator (replaces slow PyOpenCL array expression).

        Parameters
        ----------
        A : cla.Array
            Complex field array (complex64 or complex128)
        propagator : cla.Array
            Pre-computed propagator array

        Returns
        -------
        cla.Array
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        global_size = (int(A.size),)
        kernels["apply_propagator"](
            self.queue, global_size, None, A.data, propagator.data
        )
        return A

    def rabi_coupling(
        self, A1: cla.Array, A2: cla.Array, dz: float, omega: float
    ) -> tuple[cla.Array, cla.Array]:
        """Apply Rabi coupling term using native OpenCL C kernel.

        Parameters
        ----------
        A1 : cla.Array
            First field component
        A2 : cla.Array
            Second field component
        dz : float
            Solver step
        omega : float
            Rabi coupling strength

        Returns
        -------
        tuple[cla.Array, cla.Array]
            The modified field arrays (A1, A2).
        """
        kernels = self._get_kernels(A1.dtype)
        global_size = (int(A1.size),)

        cos_val = np.cos(omega * dz)
        sin_val = np.sin(omega * dz)

        if A1.dtype == np.complex64:
            cos_cast, sin_cast = np.float32(cos_val), np.float32(sin_val)
        else:
            cos_cast, sin_cast = np.float64(cos_val), np.float64(sin_val)

        kernels["rabi_coupling"](
            self.queue, global_size, None, A1.data, A2.data, cos_cast, sin_cast
        )
        return A1, A2

    def _cast_params(self, dtype, *values):
        """Cast scalar parameters to appropriate precision.

        Parameters
        ----------
        dtype : numpy.dtype
            Target dtype (complex64 or complex128)
        values : float
            Scalar values to cast

        Returns
        -------
        list
            Cast values.
        """
        fp = np.float32 if dtype == np.complex64 else np.float64
        return [fp(v) for v in values]

    def rk4_nl_rhs(
        self,
        A_prop: cla.Array,
        A: cla.Array,
        A_sq: cla.Array,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Accumulate nonlinear RHS for RK4 (no potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A : cla.Array
            Original field
        A_sq : cla.Array
            Field modulus squared
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A.dtype)
        params = self._cast_params(A.dtype, alpha, g, Isat)
        kernels["rk4_nl_rhs"](
            self.queue, (int(A.size),), None, A_prop.data, A.data, A_sq.data, *params
        )
        return A_prop

    def rk4_nl_rhs_v(
        self,
        A_prop: cla.Array,
        A: cla.Array,
        A_sq: cla.Array,
        V: cla.Array,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Accumulate nonlinear RHS for RK4 (with potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A : cla.Array
            Original field
        A_sq : cla.Array
            Field modulus squared
        V : cla.Array
            Potential (pre-scaled)
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A.dtype)
        params = self._cast_params(A.dtype, alpha, g, Isat)
        kernels["rk4_nl_rhs_v"](
            self.queue,
            (int(A.size),),
            None,
            A_prop.data,
            A.data,
            A_sq.data,
            V.data,
            *params,
        )
        return A_prop

    def square_mod_rk4_nl_rhs(
        self,
        A_prop: cla.Array,
        A: cla.Array,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused |A|^2 + RK4 NL RHS (no potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A : cla.Array
            Original field
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A.dtype)
        params = self._cast_params(A.dtype, alpha, g, Isat)
        kernels["square_mod_rk4_nl_rhs"](
            self.queue, (int(A.size),), None, A_prop.data, A.data, *params
        )
        return A_prop

    def square_mod_rk4_nl_rhs_v(
        self,
        A_prop: cla.Array,
        A: cla.Array,
        V: cla.Array,
        alpha: float,
        g: float,
        Isat: float,
    ) -> cla.Array:
        """Fused |A|^2 + RK4 NL RHS (with potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A : cla.Array
            Original field
        V : cla.Array
            Potential (pre-scaled)
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A.dtype)
        params = self._cast_params(A.dtype, alpha, g, Isat)
        kernels["square_mod_rk4_nl_rhs_v"](
            self.queue, (int(A.size),), None, A_prop.data, A.data, V.data, *params
        )
        return A_prop

    def rk4_nl_rhs_c(
        self,
        A_prop: cla.Array,
        A_orig: cla.Array,
        A_sq_1: cla.Array,
        A_sq_2: cla.Array,
        alpha: float,
        g11: float,
        g12: float,
        Isat1: float,
        Isat2: float,
    ) -> cla.Array:
        """Accumulate coupled nonlinear RHS for RK4 (no potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A_orig : cla.Array
            Original field (this component)
        A_sq_1 : cla.Array
            Modulus squared of first component
        A_sq_2 : cla.Array
            Modulus squared of second component
        alpha : float
            Losses
        g11 : float
            Intra-component interactions
        g12 : float
            Inter-component interactions
        Isat1 : float
            Saturation parameter of first component
        Isat2 : float
            Saturation parameter of second component

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A_orig.dtype)
        params = self._cast_params(A_orig.dtype, alpha, g11, g12, Isat1, Isat2)
        kernels["rk4_nl_rhs_c"](
            self.queue,
            (int(A_orig.size),),
            None,
            A_prop.data,
            A_orig.data,
            A_sq_1.data,
            A_sq_2.data,
            *params,
        )
        return A_prop

    def rk4_nl_rhs_c_v(
        self,
        A_prop: cla.Array,
        A_orig: cla.Array,
        A_sq_1: cla.Array,
        A_sq_2: cla.Array,
        V: cla.Array,
        alpha: float,
        g11: float,
        g12: float,
        Isat1: float,
        Isat2: float,
    ) -> cla.Array:
        """Accumulate coupled nonlinear RHS for RK4 (with potential).

        Parameters
        ----------
        A_prop : cla.Array
            Linearly propagated field (modified in-place)
        A_orig : cla.Array
            Original field (this component)
        A_sq_1 : cla.Array
            Modulus squared of first component
        A_sq_2 : cla.Array
            Modulus squared of second component
        V : cla.Array
            Potential (pre-scaled)
        alpha : float
            Losses
        g11 : float
            Intra-component interactions
        g12 : float
            Inter-component interactions
        Isat1 : float
            Saturation parameter of first component
        Isat2 : float
            Saturation parameter of second component

        Returns
        -------
        cla.Array
            The modified A_prop.
        """
        kernels = self._get_kernels(A_orig.dtype)
        params = self._cast_params(A_orig.dtype, alpha, g11, g12, Isat1, Isat2)
        kernels["rk4_nl_rhs_c_v"](
            self.queue,
            (int(A_orig.size),),
            None,
            A_prop.data,
            A_orig.data,
            A_sq_1.data,
            A_sq_2.data,
            V.data,
            *params,
        )
        return A_prop

    def vortex_cp(
        self, im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
    ) -> None:
        """Generate vortex of charge ll at position (i, j) using PyOpenCL.

        Parameters
        ----------
        im : cla.Array
            Image array
        i : int
            Vortex row position
        j : int
            Vortex column position
        ii : cla.Array
            Row coordinate meshgrid
        jj : cla.Array
            Column coordinate meshgrid
        ll : int
            Vortex charge
        """
        _vortex_cp_impl(im, i, j, ii, jj, ll)


# Helper functions using PyOpenCL array expressions (not yet optimized with OpenCL C)


def _vortex_cp_impl(
    im: cla.Array, i: int, j: int, ii: cla.Array, jj: cla.Array, ll: int
) -> None:
    """Generate a vortex of charge ll at position (i,j) using PyOpenCL.

    Parameters
    ----------
    im : cla.Array
        Image
    i : int
        position row of the vortex
    j : int
        position column of the vortex
    ii : cla.Array
        meshgrid position row (coordinates of the image)
    jj : cla.Array
        meshgrid position column (coordinates of the image)
    ll : int
        vortex charge
    """
    import pyopencl.clmath as clm

    # Compute complex argument raised to power ll
    # Use atan2 for correct phase angle in all quadrants
    arg = ((ii - i) + 1j * (jj - j)) ** ll
    # Extract phase angle: atan2(imaginary_part, real_part)
    im += clm.atan2(arg.imag, arg.real)
