"""CUDA C kernels for CuPy backend.

Pre-compiled CUDA C kernels with fused operations for maximum performance.
Follows the same pattern as cl.py: load template, substitute precision
placeholders, compile once via cp.RawModule, cache, and invoke directly.

Supports both single (float32/complex64) and double (float64/complex128) precision.
"""

import re
from pathlib import Path

import cupy as cp
import numpy as np

# Suffix of the twin that takes a complex (absorbing) potential.
COMPLEX_V_SUFFIX = "_cv"

# Module-level cache: precision string -> compiled cp.RawModule
_COMPILED_MODULES = {}

BLOCK_SIZE = 256


def _load_kernel_template():
    """Load CUDA kernel template from file.

    Returns
    -------
    str
        String containing kernel template with {{placeholders}}
    """
    template_path = Path(__file__).parent / "cuda_source" / "kernels.cu"
    return template_path.read_text()


_VBLOCK = re.compile(r"// \{\{VBLOCK\}\}\n(.*?)\n// \{\{END_VBLOCK\}\}", re.DOTALL)

# Real-V and complex-V spellings of what a V-reading kernel needs: the
# argument type, the phase (real part of V) and the gain/loss (imaginary
# part). V_LOSS expands to nothing in the real case, so a real V keeps the
# exact arithmetic it had before complex potentials existed.
_V_REAL_MACROS = """#define V_T {{FP_TYPE}}
#define V_RE(v, i) ((v)[i])
#define V_LOSS(v, i)
"""
_V_COMPLEX_MACROS = """#define V_T {{FP2_TYPE}}
#define V_RE(v, i) ((v)[i].x)
#define V_LOSS(v, i) + (v)[i].y
"""
_V_UNDEF = "#undef V_T\n#undef V_RE\n#undef V_LOSS\n"


def _expand_v_blocks(source):
    """Emit a real-V and a complex-V twin of every kernel marked VBLOCK.

    A complex potential is an absorbing one: its imaginary part is gain or
    loss and belongs in the real part of the exponent, not the phase. V is a
    bare pointer, so real and complex cannot share an entry point; the
    complex twin takes ``{{FP2_TYPE}}*`` and is suffixed ``_cv``. Twins
    rather than a branch, so a real V pays nothing.

    Parameters
    ----------
    source : str
        Kernel source containing VBLOCK-marked kernels.

    Returns
    -------
    str
        Source with each marked block replaced by both twins.
    """

    def expand(match):
        block = match.group(1)
        renamed = re.sub(
            r"(__global__ void )(\w+)\(",
            r"\1\2" + COMPLEX_V_SUFFIX + "(",
            block,
            count=1,
        )
        return (
            _V_REAL_MACROS
            + block
            + "\n"
            + _V_UNDEF
            + _V_COMPLEX_MACROS
            + renamed
            + "\n"
            + _V_UNDEF
        )

    return _VBLOCK.sub(expand, source)


def _get_kernel_source(precision="single"):
    """Generate CUDA C kernel source for specified precision.

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
        sincos_func = "sincosf"
    elif precision == "double":
        fp_type = "double"
        fp2_type = "double2"
        fp_suffix = ""
        sincos_func = "sincos"
    else:
        raise ValueError(f"precision must be 'single' or 'double', got {precision}")

    template = _expand_v_blocks(_load_kernel_template())
    source = template.replace("{{FP_TYPE}}", fp_type)
    source = source.replace("{{FP2_TYPE}}", fp2_type)
    source = source.replace("{{FP_SUFFIX}}", fp_suffix)
    source = source.replace("{{SINCOS_FUNC}}", sincos_func)

    return source


def _compile_kernels(precision="single"):
    """Compile CUDA kernels for specified precision.

    Uses module-level cache to compile only once per precision.

    Parameters
    ----------
    precision : str
        'single' or 'double'

    Returns
    -------
    cp.RawModule
        Compiled CUDA module
    """
    if precision in _COMPILED_MODULES:
        return _COMPILED_MODULES[precision]

    source = _get_kernel_source(precision)
    module = cp.RawModule(code=source, options=("--use_fast_math",))

    _COMPILED_MODULES[precision] = module
    return module


class CUDAKernels:
    """CUDA C kernels for NLSE operations via CuPy RawModule.

    Supports both single and double precision.
    Kernels are compiled once and cached for reuse.

    When physical parameters (alpha, g, Isat, etc.) are arrays rather than
    scalars (broadcasting mode for parallel simulations), the raw CUDA kernels
    cannot be used since they expect scalar arguments. In that case, methods
    automatically fall back to the @cp.fuse-based kernels from cupy.py which
    support broadcasting natively.
    """

    def __init__(self):
        self._kernels = {}

    @staticmethod
    def _has_array_params(*values):
        """Check if any parameter is an array (broadcasting mode).

        Parameters
        ----------
        values : float or array-like
            Parameters to check.

        Returns
        -------
        bool
            True if any parameter has ndim > 0 (is an array).
        """
        return any(getattr(v, "ndim", 0) > 0 for v in values)

    @staticmethod
    def _v_variant(kernels, name, V):
        """Return the kernel matching this potential's realness.

        A complex V is an absorbing potential: its imaginary part is gain or
        loss. V is a bare pointer, so the two cases have separate compiled
        kernels and a real V keeps the instruction stream it always had.

        Parameters
        ----------
        kernels : dict
            Compiled kernels for this precision.
        name : str
            Base kernel key.
        V : cp.ndarray or None
            The potential.

        Returns
        -------
        cp.RawKernel
            The real-V or complex-V kernel.
        """
        if V is not None and V.dtype.kind == "c":
            return kernels[name + COMPLEX_V_SUFFIX]
        return kernels[name]

    @staticmethod
    def _shares_grid_across_batch(A, *grids):
        """Check whether a grid is shared by a batch the field carries.

        A batch does not need a batched parameter: several initial conditions
        under identical physics leave every parameter scalar and only the
        field carries the extra axis. The raw kernels index the field and the
        grid with the same flat index, so an ``(NY, NX)`` potential against a
        ``(B, NY, NX)`` field reads past the end of the potential.

        Parameters
        ----------
        A : cp.ndarray
            The field array.
        grids : cp.ndarray or None
            Grid-shaped arrays the kernel indexes alongside the field.

        Returns
        -------
        bool
            True if any grid has fewer axes than the field.
        """
        return any(g is not None and g.ndim < A.ndim for g in grids)

    def _get_kernels(self, dtype):
        """Get compiled kernel functions for given dtype, compiling if needed.

        Parameters
        ----------
        dtype : numpy.dtype
            numpy dtype (complex64 or complex128)

        Returns
        -------
        dict
            Dictionary of compiled kernel functions
        """
        if dtype == np.complex64:
            precision = "single"
        elif dtype == np.complex128:
            precision = "double"
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Use complex64 or complex128."
            )

        if precision not in self._kernels:
            module = _compile_kernels(precision)
            self._kernels[precision] = {
                "nl_prop": module.get_function("nl_prop_fused"),
                "nl_prop_without_v": module.get_function("nl_prop_without_v_fused"),
                "nl_prop_c": module.get_function("nl_prop_c_fused"),
                "nl_prop_c_without_v": module.get_function("nl_prop_c_without_v_fused"),
                "square_mod": module.get_function("square_mod_fused"),
                "square_mod_nl_prop": module.get_function("square_mod_nl_prop_fused"),
                "square_mod_nl_prop_v": module.get_function(
                    "square_mod_nl_prop_v_fused"
                ),
                "apply_propagator": module.get_function("apply_propagator"),
                "rabi_coupling": module.get_function("rabi_coupling"),
                "rk4_axpy": module.get_function("rk4_axpy"),
                "rk4_accumulate": module.get_function("rk4_accumulate"),
                "rk4_nl_rhs": module.get_function("rk4_nl_rhs_fused"),
                "rk4_nl_rhs_v": module.get_function("rk4_nl_rhs_v_fused"),
                "square_mod_rk4_nl_rhs": module.get_function(
                    "square_mod_rk4_nl_rhs_fused"
                ),
                "square_mod_rk4_nl_rhs_v": module.get_function(
                    "square_mod_rk4_nl_rhs_v_fused"
                ),
                "rk4_nl_rhs_c": module.get_function("rk4_nl_rhs_c_fused"),
                "rk4_nl_rhs_c_v": module.get_function("rk4_nl_rhs_c_v_fused"),
                "nl_prop_cv": module.get_function("nl_prop_fused_cv"),
                "nl_prop_c_cv": module.get_function("nl_prop_c_fused_cv"),
                "square_mod_nl_prop_v_cv": module.get_function(
                    "square_mod_nl_prop_v_fused_cv"
                ),
                "rk4_nl_rhs_v_cv": module.get_function("rk4_nl_rhs_v_fused_cv"),
                "square_mod_rk4_nl_rhs_v_cv": module.get_function(
                    "square_mod_rk4_nl_rhs_v_fused_cv"
                ),
                "rk4_nl_rhs_c_v_cv": module.get_function("rk4_nl_rhs_c_v_fused_cv"),
            }

        return self._kernels[precision]

    @staticmethod
    def _cast(dtype, *values):
        """Cast scalar parameters to appropriate precision.

        Parameters
        ----------
        dtype : numpy.dtype
            Target dtype (complex64 or complex128)
        values : float
            Scalar values to cast

        Returns
        -------
        tuple
            Cast values as numpy scalars
        """
        fp = np.float32 if dtype == np.complex64 else np.float64
        return tuple(fp(v) for v in values)

    @staticmethod
    def _launch(kernel, size, *args):
        """Launch a CUDA kernel with appropriate grid/block config.

        Parameters
        ----------
        kernel : cp.RawKernel
            Compiled kernel function
        size : int
            Total number of elements
        args : tuple
            Kernel arguments
        """
        grid = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        block = (BLOCK_SIZE,)
        kernel(grid, block, args)

    def nl_prop(self, A, A_sq, dz, alpha, V, g, Isat):
        """Nonlinear propagation with potential.

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
        A_sq : cp.ndarray
            Intensity array (|A|^2)
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        V : cp.ndarray
            Potential array
        g : float
            Nonlinear interaction strength
        Isat : float
            Saturation intensity

        Returns
        -------
        cp.ndarray
            The modified field array A.
        """
        if self._has_array_params(alpha, g, Isat) or self._shares_grid_across_batch(
            A, V
        ):
            from .cupy import nl_prop as _fused

            return _fused(A, A_sq, dz, alpha, V, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            self._v_variant(kernels, "nl_prop", V),
            N,
            A,
            A_sq,
            V,
            dz_c,
            alpha_c,
            g_c,
            Isat_c,
            np.int32(N),
        )
        return A

    def nl_prop_without_V(self, A, A_sq, dz, alpha, g, Isat):
        """Nonlinear propagation without potential.

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
        A_sq : cp.ndarray
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
        cp.ndarray
            The modified field array A.
        """
        if self._has_array_params(alpha, g, Isat):
            from .cupy import nl_prop_without_V as _fused

            return _fused(A, A_sq, dz, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            kernels["nl_prop_without_v"],
            N,
            A,
            A_sq,
            dz_c,
            alpha_c,
            g_c,
            Isat_c,
            np.int32(N),
        )
        return A

    def nl_prop_c(self, A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2):
        """Coupled nonlinear propagation with potential.

        Parameters
        ----------
        A1 : cp.ndarray
            Complex field (component 1)
        A_sq_1 : cp.ndarray
            Intensity of component 1
        A_sq_2 : cp.ndarray
            Intensity of component 2
        dz : float
            Propagation step
        alpha : float
            Loss coefficient
        V : cp.ndarray
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
        cp.ndarray
            The modified field array A1.
        """
        if self._has_array_params(
            alpha, g11, g12, Isat1, Isat2
        ) or self._shares_grid_across_batch(A1, V):
            from .cupy import nl_prop_c as _fused

            return _fused(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2)
        kernels = self._get_kernels(A1.dtype)
        N = int(A1.size)
        params = self._cast(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            self._v_variant(kernels, "nl_prop_c", V),
            N,
            A1,
            A_sq_1,
            A_sq_2,
            V,
            *params,
            np.int32(N),
        )
        return A1

    def nl_prop_without_V_c(
        self, A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2
    ):
        """Coupled nonlinear propagation without potential.

        Parameters
        ----------
        A1 : cp.ndarray
            Complex field (component 1)
        A_sq_1 : cp.ndarray
            Intensity of component 1
        A_sq_2 : cp.ndarray
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
        cp.ndarray
            The modified field array A1.
        """
        if self._has_array_params(alpha, g11, g12, Isat1, Isat2):
            from .cupy import nl_prop_without_V_c as _fused

            return _fused(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2)
        kernels = self._get_kernels(A1.dtype)
        N = int(A1.size)
        params = self._cast(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            kernels["nl_prop_c_without_v"],
            N,
            A1,
            A_sq_1,
            A_sq_2,
            *params,
            np.int32(N),
        )
        return A1

    def square_mod(self, A, A_sq):
        """Compute square modulus (intensity).

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
        A_sq : cp.ndarray
            Output intensity array

        Returns
        -------
        cp.ndarray
            The modified intensity array A_sq.
        """
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        self._launch(kernels["square_mod"], N, A, A_sq, np.int32(N))
        return A_sq

    def square_mod_nl_prop(self, A, dz, alpha, g, Isat):
        """Fused square_mod + nl_prop_without_V.

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
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
        cp.ndarray
            The modified field array A.
        """
        if self._has_array_params(alpha, g, Isat):
            from .cupy import square_mod_nl_prop as _fused

            return _fused(A, dz, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            kernels["square_mod_nl_prop"],
            N,
            A,
            dz_c,
            alpha_c,
            g_c,
            Isat_c,
            np.int32(N),
        )
        return A

    def square_mod_nl_prop_v(self, A, V, dz, alpha, g, Isat):
        """Fused square_mod + nl_prop with potential.

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
        V : cp.ndarray
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
        cp.ndarray
            The modified field array A.
        """
        if self._has_array_params(alpha, g, Isat) or self._shares_grid_across_batch(
            A, V
        ):
            from .cupy import square_mod_nl_prop_v as _fused

            return _fused(A, V, dz, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            self._v_variant(kernels, "square_mod_nl_prop_v", V),
            N,
            A,
            V,
            dz_c,
            alpha_c,
            g_c,
            Isat_c,
            np.int32(N),
        )
        return A

    def linear_step(self, A, propagator, plan, unnorm_ifft=False):
        """Fused linear propagation: FFT + propagator multiply + IFFT.

        Parameters
        ----------
        A : cp.ndarray
            Complex field array (modified in-place)
        propagator : cp.ndarray
            Pre-computed propagator array (pre-divided by N_fft when
            unnorm_ifft is True)
        plan : _CuFFTPlan or VkFFTApp
            Pre-built FFT plan
        unnorm_ifft : bool
            If True, use unnormalized IFFT (1/N absorbed into propagator).

        Returns
        -------
        cp.ndarray
            The propagated field A.
        """
        plan.fft(A, A)
        # Goes through apply_propagator rather than launching the kernel here,
        # so the batched case (a field with an extra axis against a propagator
        # shared by the whole batch) is handled in one place.
        self.apply_propagator(A, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(A, A)
        else:
            plan.ifft(A, A)
        return A

    def apply_propagator(self, A, propagator):
        """Apply linear propagator (complex multiply A *= propagator).

        Parameters
        ----------
        A : cp.ndarray
            Complex field array
        propagator : cp.ndarray
            Pre-computed propagator array

        Returns
        -------
        cp.ndarray
            The modified field array A.
        """
        kernels = self._get_kernels(A.dtype)
        if A.ndim > propagator.ndim:
            # Batched field against a propagator shared by the whole batch.
            # The kernel indexes both with the same flat index, so launching
            # over the full field reads past the end of the propagator and
            # returns NaN and garbage for every slice after the first.
            for index in range(A.shape[0]):
                component = A[index]
                size = int(component.size)
                self._launch(
                    kernels["apply_propagator"],
                    size,
                    component,
                    propagator,
                    np.int32(size),
                )
            return A
        N = int(A.size)
        self._launch(kernels["apply_propagator"], N, A, propagator, np.int32(N))
        return A

    def rabi_coupling(self, A1, A2, dz, omega):
        """Apply Rabi coupling term (2x2 rotation).

        Parameters
        ----------
        A1 : cp.ndarray
            First field component
        A2 : cp.ndarray
            Second field component
        dz : float
            Solver step
        omega : float
            Rabi coupling strength

        Returns
        -------
        tuple[cp.ndarray, cp.ndarray]
            The modified field arrays (A1, A2).
        """
        kernels = self._get_kernels(A1.dtype)
        N = int(A1.size)

        cos_val = np.cos(omega * dz)
        sin_val = np.sin(omega * dz)
        cos_c, sin_c = self._cast(A1.dtype, cos_val, sin_val)

        self._launch(kernels["rabi_coupling"], N, A1, A2, cos_c, sin_c, np.int32(N))
        return A1, A2

    def rk4_axpy(self, out, A, c, k):
        """Compute out = A + c * k element-wise for RK4 stage arguments.

        Parameters
        ----------
        out : cp.ndarray
            Output array (modified in-place)
        A : cp.ndarray
            Base field
        c : float
            Scalar coefficient
        k : cp.ndarray
            RK4 slope array

        Returns
        -------
        cp.ndarray
            The modified output array.
        """
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        (c_c,) = self._cast(A.dtype, c)
        self._launch(kernels["rk4_axpy"], N, out, A, c_c, k, np.int32(N))
        return out

    def rk4_accumulate(self, acc, w, k):
        """Compute acc += w * k element-wise for RK4 weighted accumulation.

        Parameters
        ----------
        acc : cp.ndarray
            Accumulator array (modified in-place)
        w : float
            Weight coefficient
        k : cp.ndarray
            RK4 slope array

        Returns
        -------
        cp.ndarray
            The modified accumulator array.
        """
        kernels = self._get_kernels(acc.dtype)
        N = int(acc.size)
        (w_c,) = self._cast(acc.dtype, w)
        self._launch(kernels["rk4_accumulate"], N, acc, w_c, k, np.int32(N))
        return acc

    def rk4_nl_rhs(self, A_prop, A, A_sq, alpha, g, Isat):
        """Accumulate nonlinear RHS for RK4 (no potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A : cp.ndarray
            Original field
        A_sq : cp.ndarray
            Field modulus squared
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(alpha, g, Isat):
            from .cupy import rk4_nl_rhs as _fused

            return _fused(A_prop, A, A_sq, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(kernels["rk4_nl_rhs"], N, A_prop, A, A_sq, *params, np.int32(N))
        return A_prop

    def rk4_nl_rhs_v(self, A_prop, A, A_sq, V, alpha, g, Isat):
        """Accumulate nonlinear RHS for RK4 (with potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A : cp.ndarray
            Original field
        A_sq : cp.ndarray
            Field modulus squared
        V : cp.ndarray
            Potential (pre-scaled)
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(alpha, g, Isat) or self._shares_grid_across_batch(
            A, V
        ):
            from .cupy import rk4_nl_rhs_v as _fused

            return _fused(A_prop, A, A_sq, V, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(
            self._v_variant(kernels, "rk4_nl_rhs_v", V),
            N,
            A_prop,
            A,
            A_sq,
            V,
            *params,
            np.int32(N),
        )
        return A_prop

    def square_mod_rk4_nl_rhs(self, A_prop, A, alpha, g, Isat):
        """Fused |A|^2 + RK4 NL RHS (no potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A : cp.ndarray
            Original field
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(alpha, g, Isat):
            from .cupy import square_mod_rk4_nl_rhs as _fused

            return _fused(A_prop, A, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(
            kernels["square_mod_rk4_nl_rhs"], N, A_prop, A, *params, np.int32(N)
        )
        return A_prop

    def square_mod_rk4_nl_rhs_v(self, A_prop, A, V, alpha, g, Isat):
        """Fused |A|^2 + RK4 NL RHS (with potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A : cp.ndarray
            Original field
        V : cp.ndarray
            Potential (pre-scaled)
        alpha : float
            Losses
        g : float
            Interactions
        Isat : float
            Saturation

        Returns
        -------
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(alpha, g, Isat) or self._shares_grid_across_batch(
            A, V
        ):
            from .cupy import square_mod_rk4_nl_rhs_v as _fused

            return _fused(A_prop, A, V, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(
            self._v_variant(kernels, "square_mod_rk4_nl_rhs_v", V),
            N,
            A_prop,
            A,
            V,
            *params,
            np.int32(N),
        )
        return A_prop

    def rk4_nl_rhs_c(
        self, A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2
    ):
        """Accumulate coupled nonlinear RHS for RK4 (no potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A_orig : cp.ndarray
            Original field (this component)
        A_sq_1 : cp.ndarray
            Modulus squared of first component
        A_sq_2 : cp.ndarray
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
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(alpha, g11, g12, Isat1, Isat2):
            from .cupy import rk4_nl_rhs_c as _fused

            return _fused(A_prop, A_orig, A_sq_1, A_sq_2, alpha, g11, g12, Isat1, Isat2)
        kernels = self._get_kernels(A_orig.dtype)
        N = int(A_orig.size)
        params = self._cast(A_orig.dtype, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            kernels["rk4_nl_rhs_c"],
            N,
            A_prop,
            A_orig,
            A_sq_1,
            A_sq_2,
            *params,
            np.int32(N),
        )
        return A_prop

    def rk4_nl_rhs_c_v(
        self, A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
    ):
        """Accumulate coupled nonlinear RHS for RK4 (with potential).

        Parameters
        ----------
        A_prop : cp.ndarray
            Linearly propagated field (modified in-place)
        A_orig : cp.ndarray
            Original field (this component)
        A_sq_1 : cp.ndarray
            Modulus squared of first component
        A_sq_2 : cp.ndarray
            Modulus squared of second component
        V : cp.ndarray
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
        cp.ndarray
            The modified A_prop.
        """
        if self._has_array_params(
            alpha, g11, g12, Isat1, Isat2
        ) or self._shares_grid_across_batch(A_prop, V):
            from .cupy import rk4_nl_rhs_c_v as _fused

            return _fused(
                A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
            )
        kernels = self._get_kernels(A_orig.dtype)
        N = int(A_orig.size)
        params = self._cast(A_orig.dtype, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            self._v_variant(kernels, "rk4_nl_rhs_c_v", V),
            N,
            A_prop,
            A_orig,
            A_sq_1,
            A_sq_2,
            V,
            *params,
            np.int32(N),
        )
        return A_prop

    def vortex_cp(self, im, i, j, ii, jj, ll):
        """Generate vortex of charge ll at position (i, j) using CuPy.

        Parameters
        ----------
        im : cp.ndarray
            Image array
        i : int
            Vortex row position
        j : int
            Vortex column position
        ii : cp.ndarray
            Row coordinate meshgrid
        jj : cp.ndarray
            Column coordinate meshgrid
        ll : int
            Vortex charge
        """
        im += cp.angle(((ii - i) + 1j * (jj - j)) ** ll)
