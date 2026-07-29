"""CUDA C kernels for CuPy backend.

Pre-compiled CUDA C kernels with fused operations for maximum performance.
Follows the same pattern as cl.py: load template, substitute precision
placeholders, compile once via cp.RawModule, cache, and invoke directly.

Supports both single (float32/complex64) and double (float64/complex128) precision.
"""

from pathlib import Path

import cupy as cp
import numpy as np

from . import templating
from .templating import COMPLEX_V_SUFFIX, REAL_V_SUFFIX

# How CUDA C opens a kernel. The extern "C" that precedes it is not part of the
# match: it keeps the names unmangled so get_function can find them.
KERNEL_DECL = "__global__ void"

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
    return templating.expand_v_blocks(source, KERNEL_DECL)


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
    def _needs_broadcast(params=(), A=None, grids=()):
        """Check whether the raw kernels must be bypassed for this call.

        The raw CUDA kernels take scalar parameters and index every array
        with one flat id, so two situations force the cp.fuse fallback, which
        broadcasts natively:

        a parameter that is an array — one value per simulation, which cannot
        be passed as a scalar; and a grid with fewer axes than the field — a
        potential or propagator shared by a batch, which the flat index would
        read past the end of.

        These were two predicates called in pairs at six sites, which made it
        easy to add a call site with only one of them. A batch does not need
        a batched parameter: several initial conditions under identical
        physics leave every parameter scalar and put the extra axis on the
        field alone.

        Parameters
        ----------
        params : tuple
            Physical parameters, any of which may be batched.
        A : cp.ndarray, optional
            The field. Only needed when there are grids to compare against.
        grids : tuple
            Grid-shaped arrays the kernel indexes alongside the field.

        Returns
        -------
        bool
            True if the fused fallback is required.
        """
        if any(getattr(v, "ndim", 0) > 0 for v in params):
            return True
        return any(g is not None and g.ndim < A.ndim for g in grids)

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
        V : cp.ndarray or None
            The potential.

        Returns
        -------
        cp.RawKernel
            The no-V, real-V or complex-V twin.
        """
        if V is None:
            return kernels[base]
        if V.dtype.kind == "c":
            return kernels[base + COMPLEX_V_SUFFIX]
        return kernels[base + REAL_V_SUFFIX]

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
            # Keyed by the kernel's own name, so a V-reading kernel is reached
            # as <name> with no potential, <name>_v with a real one and
            # <name>_cv with a complex one -- see _v_kernel.
            self._kernels[precision] = {
                name: module.get_function(name) for name in _kernel_names()
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            ),
            A,
            (V,),
        ):
            from .cupy import nl_prop as _fused

            return _fused(A, A_sq, dz, alpha, V, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            self._v_kernel(kernels, "nl_prop", V),
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            )
        ):
            from .cupy import nl_prop_without_V as _fused

            return _fused(A, A_sq, dz, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            kernels["nl_prop"],
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
        if self._needs_broadcast(
            (
                alpha,
                g11,
                g12,
                Isat1,
                Isat2,
            ),
            A1,
            (V,),
        ):
            from .cupy import nl_prop_c as _fused

            return _fused(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2)
        kernels = self._get_kernels(A1.dtype)
        N = int(A1.size)
        params = self._cast(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            self._v_kernel(kernels, "nl_prop_c", V),
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
        if self._needs_broadcast(
            (
                alpha,
                g11,
                g12,
                Isat1,
                Isat2,
            )
        ):
            from .cupy import nl_prop_without_V_c as _fused

            return _fused(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2)
        kernels = self._get_kernels(A1.dtype)
        N = int(A1.size)
        params = self._cast(A1.dtype, dz, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            kernels["nl_prop_c"],
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            )
        ):
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            ),
            A,
            (V,),
        ):
            from .cupy import square_mod_nl_prop_v as _fused

            return _fused(A, V, dz, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        dz_c, alpha_c, g_c, Isat_c = self._cast(A.dtype, dz, alpha, g, Isat)
        self._launch(
            self._v_kernel(kernels, "square_mod_nl_prop", V),
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            )
        ):
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            ),
            A,
            (V,),
        ):
            from .cupy import rk4_nl_rhs_v as _fused

            return _fused(A_prop, A, A_sq, V, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(
            self._v_kernel(kernels, "rk4_nl_rhs", V),
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            )
        ):
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
        if self._needs_broadcast(
            (
                alpha,
                g,
                Isat,
            ),
            A,
            (V,),
        ):
            from .cupy import square_mod_rk4_nl_rhs_v as _fused

            return _fused(A_prop, A, V, alpha, g, Isat)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat)
        self._launch(
            self._v_kernel(kernels, "square_mod_rk4_nl_rhs", V),
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
        if self._needs_broadcast(
            (
                alpha,
                g11,
                g12,
                Isat1,
                Isat2,
            )
        ):
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
        if self._needs_broadcast(
            (
                alpha,
                g11,
                g12,
                Isat1,
                Isat2,
            ),
            A_prop,
            (V,),
        ):
            from .cupy import rk4_nl_rhs_c_v as _fused

            return _fused(
                A_prop, A_orig, A_sq_1, A_sq_2, V, alpha, g11, g12, Isat1, Isat2
            )
        kernels = self._get_kernels(A_orig.dtype)
        N = int(A_orig.size)
        params = self._cast(A_orig.dtype, alpha, g11, g12, Isat1, Isat2)
        self._launch(
            self._v_kernel(kernels, "rk4_nl_rhs_c", V),
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
