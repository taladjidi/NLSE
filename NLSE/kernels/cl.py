"""OpenCL kernels using native OpenCL C code.

Hand-written OpenCL C kernels with fused operations for maximum performance.
Supports both single (float32/complex64) and double (float64/complex128) precision
on devices that support double precision.
"""

import math
from pathlib import Path

import numpy as np
import pyopencl as cl
from pyopencl import array as cla

from . import templating
from .templating import COMPLEX_V_SUFFIX, REAL_V_SUFFIX

# How OpenCL C opens a kernel.
KERNEL_DECL = "__kernel void"

# Length of an axis numpy broadcasts against any other (numpy's broadcasting
# rules); a parameter of that many values is one value for every simulation.
BROADCAST_LENGTH = 1

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


# A batch runs one launch per simulation, with that simulation's scalars, and
# global_offset places it on its slice (a cla.Array slice cannot be passed).
# Shared grids read grid[idx - get_global_offset(0)], which the compiler still
# sees as contiguous, unlike an int argument or an idx % n_grid wrap.


def _is_batched_param(value):
    """Return True if a physical parameter differs between simulations."""
    return isinstance(value, np.ndarray) and value.size > BROADCAST_LENGTH


def _param_depth(param_shape, field_shape):
    """Return how many leading axes of the field a parameter varies over.

    The parameter follows numpy's broadcasting rules against the field: its
    shape is aligned on the field's last axes.

    Parameters
    ----------
    param_shape : tuple of int
        Shape of a parameter holding more than one value.
    field_shape : tuple of int
        Shape of the batched field.

    Returns
    -------
    int
        Index past the last axis along which the parameter changes.

    Raises
    ------
    ValueError
        If the parameter does not broadcast against the field.
    """
    shape = (BROADCAST_LENGTH,) * (len(field_shape) - len(param_shape)) + param_shape
    if len(shape) != len(field_shape) or any(
        own not in (BROADCAST_LENGTH, other) for own, other in zip(shape, field_shape)
    ):
        raise ValueError(
            f"a parameter of shape {param_shape} does not broadcast against "
            f"a field of shape {tuple(field_shape)}"
        )
    depth = len(shape)
    while shape[depth - 1] == BROADCAST_LENGTH:
        depth -= 1
    return depth


def _batch_depth(params, field_shape):
    """Return how many leading axes of the field the parameters vary over.

    Parameters
    ----------
    params : tuple
        Physical parameters, scalars or arrays.
    field_shape : tuple of int
        Shape of the field.

    Returns
    -------
    int
        0 when every parameter is one value for the whole field.
    """
    depth = 0
    for value in params:
        if _is_batched_param(value):
            depth = max(depth, _param_depth(value.shape, field_shape))
    return depth


def _launch_count(field_shape, depth, points, grid_size):
    """Return how many launches a field needs: one per simulation.

    Parameters
    ----------
    field_shape : tuple of int
        Shape of the batched field.
    depth : int
        Number of leading axes the parameters vary over.
    points : int
        Points in the field.
    grid_size : int
        Points in the grid the kernel indexes alongside the field, or 0.
        Every launch reads the grid from its start, so a launch is one grid
        long and never runs past it.

    Returns
    -------
    int
        The number of launches.

    Raises
    ------
    ValueError
        If the grid differs between simulations whose parameters differ.
    """
    members = math.prod(field_shape[:depth])
    if not 0 < grid_size <= points:
        return members
    grids = points // grid_size
    if grids % members:
        raise ValueError(
            f"on OpenCL, parameters varying over {tuple(field_shape[:depth])} "
            f"need a potential shared by those simulations: this one holds "
            f"{grid_size} points for a field of shape {tuple(field_shape)}"
        )
    return grids


def _member_values(value, field_shape, depth):
    """Return a parameter's value for each simulation it varies over.

    Parameters
    ----------
    value : float or numpy.ndarray
        A physical parameter.
    field_shape : tuple of int
        Shape of the batched field.
    depth : int
        Number of leading axes the parameters vary over.

    Returns
    -------
    sequence
        One scalar per simulation of those ``depth`` axes, in the field's
        order.
    """
    batch = field_shape[:depth]
    if not _is_batched_param(value):
        scalar = value.item() if isinstance(value, np.ndarray) else value
        return [scalar] * math.prod(batch)
    padding = (BROADCAST_LENGTH,) * (len(field_shape) - value.ndim)
    shape = (padding + value.shape)[:depth]
    return np.broadcast_to(value.reshape(shape), batch).reshape(-1)


def _load_kernel_template():
    """Load OpenCL kernel template from file.

    Returns
    -------
    str
        String containing kernel template with {{placeholders}}

    """
    template_path = Path(__file__).parent / "cl_source" / "kernels.cl"
    return template_path.read_text()


def _expand_v_blocks(source):
    """Emit the no-V, real-V and complex-V twins of each VBLOCK kernel.

    Parameters
    ----------
    source : str
        Kernel source containing VBLOCK-marked kernels.

    Returns
    -------
    str
        Source with each marked block replaced by its three twins.
    """
    return templating.expand_v_blocks(source, KERNEL_DECL, "__global ")


_KERNEL_NAMES: list = []


def _kernel_names():
    """Return every kernel name the expanded template declares."""
    if not _KERNEL_NAMES:
        _KERNEL_NAMES.extend(
            templating.kernel_names(
                _expand_v_blocks(_load_kernel_template()), KERNEL_DECL
            )
        )
    return _KERNEL_NAMES


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
    template = _expand_v_blocks(_load_kernel_template())
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
        # No unsafe math (so no -cl-fast-relaxed-math): reassociation rounded a kernel
        # and its no-V twin differently on POCL. No -cl-finite-math-only: Isat defaults
        # to np.inf. See docs/optimization-log.md.
        "-cl-mad-enable",
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

            # Cache kernel objects to avoid repeated retrieval. The key is the
            # kernel's own name, so a V-reading kernel is reached as <name> with
            # no potential, <name>_v with a real one and <name>_cv with a
            # complex one -- see _v_kernel.
            self._kernels[precision] = {
                name: cl.Kernel(program, name) for name in _kernel_names()
            }

        return self._kernels[precision]

    def _launches(self, A, params=(), grid=None):
        """Yield one launch per simulation, or a single launch when unbatched.

        Parameters
        ----------
        A : cla.Array
            The field the launches cover.
        params : tuple
            Physical parameters, any of which may vary over the leading
            (batch) axes of the field.
        grid : cla.Array or None
            A grid-shaped array the kernel indexes alongside the field. When
            it is smaller than the field, the field carries a batch axis the
            grid does not, which needs per-simulation launches even if every
            parameter is scalar. A grid that differs between simulations whose
            parameters differ is refused: each launch reads it from its start.

        Yields
        ------
        tuple
            ``(offset, global_size, local_size, params)``. ``offset`` is both
            the launch's global offset and the amount the kernel subtracts to
            address the shared grid; it is 0 for an unbatched run.
        """
        shape = A.shape
        points = int(A.size)
        depth = _batch_depth(params, shape)
        grid_size = 0 if grid is None else int(grid.size)
        n = _launch_count(shape, depth, points, grid_size)
        if n <= 1:
            yield 0, (points,), self._local_size(points), tuple(params)
            return
        size = points // n
        local = self._local_size(size)
        columns = [_member_values(p, shape, depth) for p in params]
        # A shared grid can ask for more launches than the parameters have
        # values: consecutive launches then share one set of them.
        repeat = n // math.prod(shape[:depth])
        for b in range(n):
            yield (
                b * size,
                (size,),
                local,
                tuple(column[b // repeat] for column in columns),
            )

    @staticmethod
    def _v_kernel(kernels, base, V):
        """Return the twin of this kernel that matches the potential.

        A complex V is an absorbing potential: its imaginary part is gain or
        loss rather than phase. None, real and complex each get their own
        compiled kernel, so a run pays for exactly the case it is in.

        Parameters
        ----------
        kernels : dict
            Compiled kernels for this precision.
        base : str
            Name of the kernel, without a potential suffix.
        V : cla.Array or None
            The potential.

        Returns
        -------
        cl.Kernel
            The no-V, real-V or complex-V twin.
        """
        if V is None:
            return kernels[base]
        if V.dtype.kind == "c":
            return kernels[base + COMPLEX_V_SUFFIX]
        return kernels[base + REAL_V_SUFFIX]

    @staticmethod
    def _offset(offset):
        """Return the global_offset argument, or None when there is none."""
        return (offset,) if offset else None

    def _launch_coupled(self, kerns, base, V1, V2, gs, ls, buffers, n, params):
        """Launch a coupled kernel, passing the potentials only if there are any.

        The potential-free twin takes a shorter argument list rather than a
        null pointer, so which arguments to pass is settled by the same fact
        that settles which kernel to call. Written out at the call site that
        is two things to keep in step, and it was written out three times.

        Parameters
        ----------
        kerns : dict
            Compiled kernels for this precision.
        base : str
            Kernel name, without a potential suffix; ``_v_kernel`` picks the
            twin matching a real or complex V.
        V1 : cla.Array or None
            Potential for the first component. None means the whole run has
            none, and selects the shorter argument list.
        V2 : cla.Array or None
            Potential for the second component. Passed with V1 or not at all.
        gs : tuple
            Global size.
        ls : tuple or None
            Local size.
        buffers : tuple
            The field buffers this kernel takes ahead of the potentials.
        n : np.int32
            Element count the kernel indexes with.
        params : tuple
            The physical parameters, already cast to the field's precision.

        Returns
        -------
        cl.Event
            Whatever the launch returns.
        """
        if V1 is not None:
            return self._v_kernel(kerns, base, V1)(
                self.queue, gs, ls, *buffers, V1.data, V2.data, n, *params
            )
        return kerns[base](self.queue, gs, ls, *buffers, n, *params)

    def linear_step(self, A, propagator, plan, unnorm_ifft=False):
        """Fused linear propagation: FFT + propagator multiply + IFFT.

        Parameters
        ----------
        A : cla.Array
            Complex field array (modified in-place)
        propagator : cla.Array
            Pre-computed propagator array (pre-divided by N_fft when
            unnorm_ifft is True)
        plan : _VkFFTPlan or VkFFTApp
            Pre-built FFT plan
        unnorm_ifft : bool
            If True, use unnormalized IFFT (1/N absorbed into propagator).

        Returns
        -------
        cla.Array
            The propagated field A.
        """
        plan.fft(A, A)
        self.apply_propagator(A, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(A, A)
        else:
            plan.ifft(A, A)
        return A

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
        for offset, gs, ls, params in self._launches(A, (dz, alpha, g, Isat), V):
            dz_c, alpha_c, g_c, Isat_c = self._cast_params(A.dtype, *params)
            self._v_kernel(kernels, "nl_prop", V)(
                self.queue,
                gs,
                ls,
                A.data,
                A_sq.data,
                V.data,
                dz_c,
                alpha_c,
                g_c,
                Isat_c,
                global_offset=self._offset(offset),
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
        for offset, gs, ls, params in self._launches(A, (dz, alpha, g, Isat)):
            dz_c, alpha_c, g_c, Isat_c = self._cast_params(A.dtype, *params)
            kernels["nl_prop"](
                self.queue,
                gs,
                ls,
                A.data,
                A_sq.data,
                dz_c,
                alpha_c,
                g_c,
                Isat_c,
                global_offset=self._offset(offset),
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

        params = self._cast_params(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        self._v_kernel(kernels, "nl_prop_c", V)(
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

        params = self._cast_params(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        kernels["nl_prop_c"](
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
        for offset, gs, ls, params in self._launches(A, (dz, alpha, g, Isat)):
            dz_c, alpha_c, g_c, Isat_c = self._cast_params(A.dtype, *params)
            kernels["square_mod_nl_prop"](
                self.queue,
                gs,
                ls,
                A.data,
                dz_c,
                alpha_c,
                g_c,
                Isat_c,
                global_offset=self._offset(offset),
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
        for offset, gs, ls, params in self._launches(A, (dz, alpha, g, Isat), V):
            dz_c, alpha_c, g_c, Isat_c = self._cast_params(A.dtype, *params)
            self._v_kernel(kernels, "square_mod_nl_prop", V)(
                self.queue,
                gs,
                ls,
                A.data,
                V.data,
                dz_c,
                alpha_c,
                g_c,
                Isat_c,
                global_offset=self._offset(offset),
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
        for offset, gs, ls, _ in self._launches(A, grid=propagator):
            kernels["apply_propagator"](
                self.queue,
                gs,
                ls,
                A.data,
                propagator.data,
                global_offset=self._offset(offset),
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
        cos_c, sin_c = self._cast_params(A1.dtype, cos_val, sin_val)
        kernels["rabi_coupling"](
            self.queue, global_size, None, A1.data, A2.data, cos_c, sin_c
        )
        return A1, A2

    def _cast_params(self, dtype, *values):
        """Cast scalar parameters to appropriate precision.

        Skips re-casting values that are already the correct numpy scalar type.

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
        return [v if type(v) is fp else fp(v) for v in values]

    def rk4_axpy(
        self,
        out: cla.Array,
        A: cla.Array,
        c: float,
        k: cla.Array,
    ) -> cla.Array:
        """Compute out = A + c * k element-wise for RK4 stage arguments.

        Parameters
        ----------
        out : cla.Array
            Output array (modified in-place)
        A : cla.Array
            Base field
        c : float
            Scalar coefficient
        k : cla.Array
            RK4 slope array

        Returns
        -------
        cla.Array
            The modified output array.
        """
        kernels = self._get_kernels(A.dtype)
        gs = (int(A.size),)
        c_cast = np.float32(c) if A.dtype == np.complex64 else np.float64(c)
        kernels["rk4_axpy"](
            self.queue, gs, self._local_size(A.size), out.data, A.data, c_cast, k.data
        )
        return out

    def rk4_accumulate(
        self,
        acc: cla.Array,
        w: float,
        k: cla.Array,
    ) -> cla.Array:
        """Compute acc += w * k element-wise for RK4 weighted accumulation.

        Parameters
        ----------
        acc : cla.Array
            Accumulator array (modified in-place)
        w : float
            Weight coefficient
        k : cla.Array
            RK4 slope array

        Returns
        -------
        cla.Array
            The modified accumulator array.
        """
        kernels = self._get_kernels(acc.dtype)
        gs = (int(acc.size),)
        (w_cast,) = self._cast_params(acc.dtype, w)
        kernels["rk4_accumulate"](
            self.queue, gs, self._local_size(acc.size), acc.data, w_cast, k.data
        )
        return acc

    def rk4_set_and_axpy(
        self,
        acc: cla.Array,
        out: cla.Array,
        A: cla.Array,
        k: cla.Array,
        c: float,
    ) -> tuple[cla.Array, cla.Array]:
        """Fused: acc = k, out = A + c * k (RK4 stage 1 update).

        Combines the copy-to-acc and axpy-to-A_tmp into a single kernel.

        Parameters
        ----------
        acc : cla.Array
            Accumulator (set to k, in-place)
        out : cla.Array
            Output array (set to A + c*k, in-place)
        A : cla.Array
            Base field
        k : cla.Array
            RK4 slope array
        c : float
            Scalar coefficient for A_tmp

        Returns
        -------
        tuple[cla.Array, cla.Array]
            The modified (acc, out) arrays.
        """
        kernels = self._get_kernels(A.dtype)
        gs = (int(A.size),)
        (c_cast,) = self._cast_params(A.dtype, c)
        kernels["rk4_set_and_axpy"](
            self.queue,
            gs,
            self._local_size(A.size),
            acc.data,
            out.data,
            A.data,
            k.data,
            c_cast,
        )
        return acc, out

    def rk4_acc_and_axpy(
        self,
        acc: cla.Array,
        out: cla.Array,
        A: cla.Array,
        k: cla.Array,
        w: float,
        c: float,
    ) -> tuple[cla.Array, cla.Array]:
        """Fused: acc += w * k, out = A + c * k (RK4 stages 2-3 update).

        Combines the accumulate-to-acc and axpy-to-A_tmp into a single kernel.

        Parameters
        ----------
        acc : cla.Array
            Accumulator (modified in-place)
        out : cla.Array
            Output array (set to A + c*k, in-place)
        A : cla.Array
            Base field
        k : cla.Array
            RK4 slope array
        w : float
            Accumulate weight
        c : float
            Scalar coefficient for A_tmp

        Returns
        -------
        tuple[cla.Array, cla.Array]
            The modified (acc, out) arrays.
        """
        kernels = self._get_kernels(A.dtype)
        w_cast, c_cast = self._cast_params(A.dtype, w, c)
        kernels["rk4_acc_and_axpy"](
            self.queue,
            (int(A.size),),
            self._local_size(A.size),
            acc.data,
            out.data,
            A.data,
            k.data,
            w_cast,
            c_cast,
        )
        return acc, out

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
        for offset, gs, ls, values in self._launches(A, (alpha, g, Isat)):
            kernels["rk4_nl_rhs"](
                self.queue,
                gs,
                ls,
                A_prop.data,
                A.data,
                A_sq.data,
                *self._cast_params(A.dtype, *values),
                global_offset=self._offset(offset),
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
        for offset, gs, ls, values in self._launches(A, (alpha, g, Isat), V):
            self._v_kernel(kernels, "rk4_nl_rhs", V)(
                self.queue,
                gs,
                ls,
                A_prop.data,
                A.data,
                A_sq.data,
                V.data,
                *self._cast_params(A.dtype, *values),
                global_offset=self._offset(offset),
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
        for offset, gs, ls, values in self._launches(A, (alpha, g, Isat)):
            kernels["square_mod_rk4_nl_rhs"](
                self.queue,
                gs,
                ls,
                A_prop.data,
                A.data,
                *self._cast_params(A.dtype, *values),
                global_offset=self._offset(offset),
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
        for offset, gs, ls, values in self._launches(A, (alpha, g, Isat), V):
            self._v_kernel(kernels, "square_mod_rk4_nl_rhs", V)(
                self.queue,
                gs,
                ls,
                A_prop.data,
                A.data,
                V.data,
                *self._cast_params(A.dtype, *values),
                global_offset=self._offset(offset),
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
        self._v_kernel(kernels, "rk4_nl_rhs_c", V)(
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

    # ── Fused multi-dispatch methods ─────────────────────────────────────────
    # These chain multiple GPU operations in a single Python call to eliminate
    # interpreter overhead between kernel dispatches.

    def _local_size(self, global_n):
        """Return optimal local work size for bandwidth-bound kernels."""
        if global_n >= 256 and global_n % 256 == 0:
            return (256,)
        return None

    def split_step_fused(
        self,
        A: cla.Array,
        propagator: cla.Array,
        V_scaled: cla.Array | None,
        dz: float,
        alpha: float,
        g: float,
        Isat: float,
        splitting: str,
        plan,
        unnorm_ifft: bool = False,
    ) -> cla.Array:
        """Fused split step: all GPU operations in one Python call.

        Chains FFT + apply_propagator + IFFT + nonlinear step (and an
        additional NL half-step for double precision) without returning
        to the solver between dispatches.

        Parameters
        ----------
        A : cla.Array
            Complex field array (modified in-place).
        propagator : cla.Array
            Pre-computed propagator (pre-divided by N_fft when
            unnorm_ifft is True).
        V_scaled : cla.Array or None
            Pre-scaled potential (V * k/2), or None.
        dz : float
            Nonlinear step size (full for single, half for double).
        alpha : float
            Half-loss coefficient.
        g : float
            Nonlinear interaction strength.
        Isat : float
            Saturation intensity (converted units).
        splitting : str
            "lie" or "strang".
        plan : _VkFFTPlan
            Pre-built FFT plan.
        unnorm_ifft : bool
            If True, use unnormalized IFFT.

        Returns
        -------
        cla.Array
            The propagated field A.
        """

        # Delegates the nonlinear halves and the propagator multiply to the
        # single-kernel methods so the batch handling lives in one place.
        # Each is still a direct kernel launch: nothing returns to the solver.
        def nonlinear(step):
            if V_scaled is not None:
                self.square_mod_nl_prop_v(A, V_scaled, step, alpha, g, Isat)
            else:
                self.square_mod_nl_prop(A, step, alpha, g, Isat)

        # Strang: NL half-step before the linear step
        if splitting == "strang":
            nonlinear(dz)

        # Linear step: FFT → propagator multiply → IFFT
        plan.fft(A, A)
        self.apply_propagator(A, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(A, A)
        else:
            plan.ifft(A, A)

        # Nonlinear step
        nonlinear(dz)
        return A

    def rk4_rhs_fused(
        self,
        A_in: cla.Array,
        k: cla.Array,
        V_scaled: cla.Array | None,
        propagator: cla.Array,
        plan,
        alpha: float,
        g: float,
        Isat: float,
        unnorm_ifft: bool = False,
    ) -> cla.Array:
        """Fused RK4 RHS: out-of-place FFT + propagator + IFFT + NL step.

        Eliminates the buffer copy (k[:] = A_in) by using an out-of-place
        FFT that writes directly from A_in into k.

        Parameters
        ----------
        A_in : cla.Array
            Input field (not modified).
        k : cla.Array
            Output buffer for RHS result (modified in-place).
        V_scaled : cla.Array or None
            Pre-scaled potential (V * k/2), or None.
        propagator : cla.Array
            Pre-computed propagator (pre-divided by N_fft when
            unnorm_ifft is True).
        plan : _VkFFTPlan
            Pre-built FFT plan (must support fft_oop).
        alpha : float
            Half-loss coefficient.
        g : float
            Nonlinear interaction strength.
        Isat : float
            Saturation intensity (converted units).
        unnorm_ifft : bool
            If True, use unnormalized IFFT.

        Returns
        -------
        cla.Array
            The modified buffer k.
        """
        # Out-of-place FFT: A_in → k (eliminates buffer copy)
        plan.fft_oop(A_in, k)

        # Apply propagator to k
        self.apply_propagator(k, propagator)

        # IFFT k in-place
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(k, k)
        else:
            plan.ifft(k, k)

        # Nonlinear RHS: k = k + NL(A_in)
        if V_scaled is not None:
            self.square_mod_rk4_nl_rhs_v(k, A_in, V_scaled, alpha, g, Isat)
        else:
            self.square_mod_rk4_nl_rhs(k, A_in, alpha, g, Isat)
        return k

    # ── Fused interleaved coupled methods ────────────────────────────────────

    def split_step_coupled_fused(
        self,
        A: cla.Array,
        propagator: cla.Array,
        V1_scaled: cla.Array | None,
        V2_scaled: cla.Array | None,
        dz: float,
        alpha1: float,
        alpha2: float,
        g11: float,
        g12: float,
        g21: float,
        g22: float,
        Isat1: float,
        Isat2: float,
        splitting: str,
        plan,
        omega: float | None = None,
        unnorm_ifft: bool = False,
    ) -> cla.Array:
        """Fused coupled split step on interleaved (2, N_sq) array.

        Parameters
        ----------
        A : cla.Array
            Interleaved field (2, ...) modified in-place.
        propagator : cla.Array
            Pre-computed propagator for both components.
        V1_scaled : cla.Array or None
            Pre-scaled potential for component 1.
        V2_scaled : cla.Array or None
            Pre-scaled potential for component 2.
        dz : float
            Nonlinear step size (full for single, half for double).
        alpha1 : float
            Half-loss coefficient, component 1.
        alpha2 : float
            Half-loss coefficient, component 2.
        g11 : float
            Intra-component 1 interaction.
        g12 : float
            Cross-component interaction in component 1's equation.
        g21 : float
            Cross-component interaction in component 2's equation.
        g22 : float
            Intra-component 2 interaction.
        Isat1 : float
            Saturation intensity, component 1.
        Isat2 : float
            Saturation intensity, component 2.
        splitting : str
            "lie" or "strang".
        plan : VkFFT plan
            Pre-built FFT plan.
        omega : float or None
            Rabi coupling (half). None to skip.
        unnorm_ifft : bool
            Use unnormalized IFFT.

        Returns
        -------
        cla.Array
            The propagated field A.
        """
        kerns = self._get_kernels(A.dtype)
        # Interleaved (2, N_sq) layout: one thread per component element.
        N_sq = int(A.size) // 2
        gs = (N_sq,)
        ls = self._local_size(N_sq)
        N_sq_i = np.int32(N_sq)
        params = self._cast_params(
            A.dtype, dz, alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2
        )

        # Strang: NL half-step before the linear step
        if splitting == "strang":
            self._launch_coupled(
                kerns,
                "coupled_nl_prop_c",
                V1_scaled,
                V2_scaled,
                gs,
                ls,
                (A.data,),
                N_sq_i,
                params,
            )

        # Linear step: FFT → propagator multiply → IFFT (on full 2*N_sq)
        gs_full = (int(A.size),)
        ls_full = self._local_size(A.size)
        plan.fft(A, A)
        kerns["apply_propagator"](self.queue, gs_full, ls_full, A.data, propagator.data)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(A, A)
        else:
            plan.ifft(A, A)

        # Nonlinear step
        self._launch_coupled(
            kerns,
            "coupled_nl_prop_c",
            V1_scaled,
            V2_scaled,
            gs,
            ls,
            (A.data,),
            N_sq_i,
            params,
        )

        # Rabi coupling (Lie splitting only)
        if omega is not None:
            cos_val = np.cos(omega * float(params[0]))  # omega * dz
            sin_val = np.sin(omega * float(params[0]))
            cos_c, sin_c = self._cast_params(A.dtype, cos_val, sin_val)
            kerns["rabi_coupling_interleaved"](
                self.queue,
                gs,
                ls,
                A.data,
                N_sq_i,
                cos_c,
                sin_c,
            )

        return A

    def rk4_rhs_coupled_fused(
        self,
        A_in: cla.Array,
        k: cla.Array,
        V1_scaled: cla.Array | None,
        V2_scaled: cla.Array | None,
        propagator: cla.Array,
        plan,
        alpha1: float,
        alpha2: float,
        g11: float,
        g12: float,
        g21: float,
        g22: float,
        Isat1: float,
        Isat2: float,
        unnorm_ifft: bool = False,
    ) -> cla.Array:
        """Fused coupled RK4 RHS on interleaved (2, N_sq) arrays.

        Parameters
        ----------
        A_in : cla.Array
            Input field (not modified).
        k : cla.Array
            Output buffer (modified in-place).
        V1_scaled : cla.Array or None
            Pre-scaled potential, component 1.
        V2_scaled : cla.Array or None
            Pre-scaled potential, component 2.
        propagator : cla.Array
            Pre-computed propagator.
        plan : VkFFT plan
            Pre-built FFT plan (must support fft_oop).
        alpha1 : float
            Half-loss, component 1.
        alpha2 : float
            Half-loss, component 2.
        g11 : float
            Intra-component 1 interaction.
        g12 : float
            Cross-component interaction in component 1's equation.
        g21 : float
            Cross-component interaction in component 2's equation.
        g22 : float
            Intra-component 2 interaction.
        Isat1 : float
            Saturation, component 1.
        Isat2 : float
            Saturation, component 2.
        unnorm_ifft : bool
            Use unnormalized IFFT.

        Returns
        -------
        cla.Array
            The modified buffer k.
        """
        kerns = self._get_kernels(A_in.dtype)
        # Interleaved (2, N_sq) layout: one thread per component element.
        N_sq = int(A_in.size) // 2
        gs_full = (int(A_in.size),)
        ls_full = self._local_size(A_in.size)
        gs = (N_sq,)
        ls = self._local_size(N_sq)
        N_sq_i = np.int32(N_sq)

        # Out-of-place FFT: A_in → k
        plan.fft_oop(A_in, k)

        # Apply propagator
        kerns["apply_propagator"](self.queue, gs_full, ls_full, k.data, propagator.data)

        # IFFT
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(k, k)
        else:
            plan.ifft(k, k)

        # Coupled NL RHS
        params = self._cast_params(
            A_in.dtype, alpha1, alpha2, g11, g12, g21, g22, Isat1, Isat2
        )
        self._launch_coupled(
            kerns,
            "coupled_rk4_nl_rhs_c",
            V1_scaled,
            V2_scaled,
            gs,
            ls,
            (k.data, A_in.data),
            N_sq_i,
            params,
        )

        return k
