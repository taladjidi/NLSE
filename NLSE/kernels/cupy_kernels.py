"""CUDA C kernels for CuPy backend.

Pre-compiled CUDA C kernels with fused operations for maximum performance.
Follows the same pattern as cl.py: load template, substitute splitting
placeholders, compile once via cp.RawModule, cache, and invoke directly.

Supports both single (float32/complex64) and double (float64/complex128) splitting.
"""

from pathlib import Path

import cupy as cp
import numpy as np

from . import templating
from .templating import COMPLEX_V_SUFFIX, REAL_V_SUFFIX

# How CUDA C opens a kernel. The extern "C" that precedes it is not part of the
# match: it keeps the names unmangled so get_function can find them.
KERNEL_DECL = "__global__ void"

# Module-level cache: splitting string -> compiled cp.RawModule
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


def _get_kernel_source(splitting="lie"):
    """Generate CUDA C kernel source for specified splitting.

    Parameters
    ----------
    splitting : str
        'lie' for float32 or 'strang' for float64

    Returns
    -------
    str
        String containing all kernel source code
    """
    if splitting == "lie":
        fp_type = "float"
        fp2_type = "float2"
        fp_suffix = "f"
        sincos_func = "sincosf"
    elif splitting == "strang":
        fp_type = "strang"
        fp2_type = "double2"
        fp_suffix = ""
        sincos_func = "sincos"
    else:
        raise ValueError(f"splitting must be 'lie' or 'strang', got {splitting}")

    template = _expand_v_blocks(_load_kernel_template())
    source = template.replace("{{FP_TYPE}}", fp_type)
    source = source.replace("{{FP2_TYPE}}", fp2_type)
    source = source.replace("{{FP_SUFFIX}}", fp_suffix)
    source = source.replace("{{SINCOS_FUNC}}", sincos_func)

    return source


def _compile_kernels(splitting="lie"):
    """Compile CUDA kernels for specified splitting.

    Uses module-level cache to compile only once per splitting.

    Parameters
    ----------
    splitting : str
        'lie' or 'strang'

    Returns
    -------
    cp.RawModule
        Compiled CUDA module
    """
    if splitting in _COMPILED_MODULES:
        return _COMPILED_MODULES[splitting]

    source = _get_kernel_source(splitting)
    module = cp.RawModule(code=source, options=("--use_fast_math",))

    _COMPILED_MODULES[splitting] = module
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
            splitting = "lie"
        elif dtype == np.complex128:
            splitting = "strang"
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Use complex64 or complex128."
            )

        if splitting not in self._kernels:
            module = _compile_kernels(splitting)
            # Keyed by the kernel's own name, so a V-reading kernel is reached
            # as <name> with no potential, <name>_v with a real one and
            # <name>_cv with a complex one -- see _v_kernel.
            self._kernels[splitting] = {
                name: module.get_function(name) for name in _kernel_names()
            }

        return self._kernels[splitting]

    @staticmethod
    def _cast(dtype, *values):
        """Cast scalar parameters to appropriate splitting.

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

    def rk4_set_and_axpy(self, acc, out, A, k, c):
        """Compute acc = k and out = A + c * k in one launch (RK4 stage 1).

        Every argument is a whole field of the same shape and ``c`` is a
        scalar, so there is nothing here for a batch to broadcast: the flat
        index serves a batched field as well as a single one.

        Parameters
        ----------
        acc : cp.ndarray
            Accumulator, set to k (modified in-place)
        out : cp.ndarray
            Stage argument, set to A + c*k (modified in-place)
        A : cp.ndarray
            Base field
        k : cp.ndarray
            RK4 slope array
        c : float
            Scalar coefficient

        Returns
        -------
        tuple[cp.ndarray, cp.ndarray]
            The modified (acc, out).
        """
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        (c_c,) = self._cast(A.dtype, c)
        self._launch(kernels["rk4_set_and_axpy"], N, acc, out, A, k, c_c, np.int32(N))
        return acc, out

    def rk4_final_update(self, A, acc, k, w):
        """Compute A += w * (acc + k), closing the step in one launch.

        Parameters
        ----------
        A : cp.ndarray
            The field (modified in-place)
        acc : cp.ndarray
            Accumulated slopes from the first three stages
        k : cp.ndarray
            The fourth slope
        w : float
            Scalar weight, h/6

        Returns
        -------
        cp.ndarray
            The modified field A.
        """
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        (w_c,) = self._cast(A.dtype, w)
        self._launch(kernels["rk4_final_update"], N, A, acc, k, w_c, np.int32(N))
        return A

    def rk4_acc_and_axpy(self, acc, out, A, k, w, c):
        """Compute acc += w * k and out = A + c * k in one launch (stages 2-3).

        Parameters
        ----------
        acc : cp.ndarray
            Accumulator (modified in-place)
        out : cp.ndarray
            Stage argument, set to A + c*k (modified in-place)
        A : cp.ndarray
            Base field
        k : cp.ndarray
            RK4 slope array
        w : float
            Accumulation weight
        c : float
            Scalar coefficient

        Returns
        -------
        tuple[cp.ndarray, cp.ndarray]
            The modified (acc, out).
        """
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        w_c, c_c = self._cast(A.dtype, w, c)
        self._launch(
            kernels["rk4_acc_and_axpy"], N, acc, out, A, k, w_c, c_c, np.int32(N)
        )
        return acc, out

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

    def rk4_rhs_fused(
        self, A_in, k, V_scaled, propagator, plan, alpha, g, Isat, unnorm_ifft=False
    ):
        """Compute the RK4 RHS into k without copying A_in into it first.

        Parameters
        ----------
        A_in : cp.ndarray
            Input field (not modified).
        k : cp.ndarray
            Output buffer for the RHS (modified in-place).
        V_scaled : cp.ndarray or None
            Pre-scaled potential (V * k/2), or None.
        propagator : cp.ndarray
            Pre-computed propagator (pre-divided by N_fft when unnorm_ifft).
        plan : _CuFFTPlan
            Pre-built FFT plan.
        alpha : float
            Half-loss coefficient.
        g : float
            Nonlinear interaction strength.
        Isat : float
            Saturation intensity (converted units).
        unnorm_ifft : bool
            If True, the propagator carries the 1/N and the inverse
            transform skips it.

        Returns
        -------
        cp.ndarray
            The modified buffer k.
        """
        # The transform moves A_in into k, so no copy precedes it. Unlike
        # VkFFT, which the OpenCL backend gives a separate fft_oop, a cuFFT
        # plan is out-of-place already: plan.fft takes its output array.
        plan.fft(A_in, k)
        self.apply_propagator(k, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(k, k)
        else:
            plan.ifft(k, k)
        if V_scaled is not None:
            return self.square_mod_rk4_nl_rhs_v(k, A_in, V_scaled, alpha, g, Isat)
        return self.square_mod_rk4_nl_rhs(k, A_in, alpha, g, Isat)

    def _linear_into(self, A_in, k, propagator, plan, unnorm_ifft):
        """Transform A_in into k, propagate it there, and come back.

        The transform is what moves the field into the stage buffer, so no
        copy precedes it: a cuFFT plan is out-of-place already, unlike the
        VkFFT one the OpenCL backend gives a separate ``fft_oop``.
        """
        plan.fft(A_in, k)
        self.apply_propagator(k, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(k, k)
        else:
            plan.ifft(k, k)
        return k

    def rk4_stage_fused(
        self,
        A_in,
        k,
        V_scaled,
        propagator,
        plan,
        acc,
        out,
        A,
        alpha,
        g,
        Isat,
        w,
        c,
        mode,
        unnorm_ifft=False,
    ):
        """Run a whole RK4 stage: linear part, slope and stage update.

        The slope is finished in registers and spent on ``acc`` and ``out``
        without reaching memory, which is two fewer accesses per element than
        writing it and reading it back.

        Parameters
        ----------
        A_in : cp.ndarray
            Field this stage evaluates the slope at (not modified).
        k : cp.ndarray
            Scratch buffer the transform writes into (modified in-place).
        V_scaled : cp.ndarray or None
            Pre-scaled potential (V * k/2), or None.
        propagator : cp.ndarray
            Pre-computed propagator (pre-divided by N_fft when unnorm_ifft).
        plan : _CuFFTPlan
            Pre-built FFT plan.
        acc : cp.ndarray
            Slope accumulator (modified in-place unless mode is 2).
        out : cp.ndarray
            Where the stage's result goes: the next stage's argument, or the
            field itself on the last stage.
        A : cp.ndarray
            The field the step started from.
        alpha : float
            Half-loss coefficient.
        g : float
            Nonlinear interaction strength.
        Isat : float
            Saturation intensity (converted units).
        w : float
            Weight this stage's slope carries into the accumulator.
        c : float
            Coefficient of the slope in the stage's output.
        mode : int
            0 to set the accumulator, 1 to add to it, 2 to spend it.
        unnorm_ifft : bool
            If True, the propagator carries the 1/N.

        Returns
        -------
        cp.ndarray
            ``out``, the array this stage wrote.
        """
        self._linear_into(A_in, k, propagator, plan, unnorm_ifft)
        kernels = self._get_kernels(A.dtype)
        N = int(A.size)
        params = self._cast(A.dtype, alpha, g, Isat, w, c)
        args = (k, A_in) + ((V_scaled,) if V_scaled is not None else ())
        self._launch(
            self._v_kernel(kernels, "square_mod_rk4_stage", V_scaled),
            N,
            *args,
            acc,
            out,
            A,
            *params,
            np.int32(mode),
            np.int32(N),
        )
        return out

    def rk4_stage_coupled_fused(
        self,
        A_in,
        k,
        V1_scaled,
        V2_scaled,
        propagator,
        plan,
        acc,
        out,
        A,
        alpha1,
        alpha2,
        g11,
        g12,
        g22,
        Isat1,
        Isat2,
        w,
        c,
        mode,
        unnorm_ifft=False,
    ):
        """Run a whole coupled RK4 stage, both components at once.

        Parameters
        ----------
        A_in : cp.ndarray
            Coupled field this stage evaluates the slope at (not modified).
        k : cp.ndarray
            Scratch buffer the transform writes into (modified in-place).
        V1_scaled, V2_scaled : cp.ndarray or None
            Pre-scaled potentials, one per component. Both None or neither.
        propagator : cp.ndarray
            Pre-computed propagator (pre-divided by N_fft when unnorm_ifft).
        plan : _CuFFTPlan
            Pre-built FFT plan.
        acc : cp.ndarray
            Slope accumulator (modified in-place unless mode is 2).
        out : cp.ndarray
            Where the stage's result goes.
        A : cp.ndarray
            The field the step started from.
        alpha1, alpha2 : float
            Half-loss coefficients.
        g11, g22 : float
            Intra-component interactions.
        g12 : float
            Cross-component interaction.
        Isat1, Isat2 : float
            Saturation intensities (converted units).
        w : float
            Weight this stage's slope carries into the accumulator.
        c : float
            Coefficient of the slope in the stage's output.
        mode : int
            0 to set the accumulator, 1 to add to it, 2 to spend it.
        unnorm_ifft : bool
            If True, the propagator carries the 1/N.

        Returns
        -------
        cp.ndarray
            ``out``, the array this stage wrote.
        """
        self._linear_into(A_in, k, propagator, plan, unnorm_ifft)
        kernels = self._get_kernels(A.dtype)
        N_sq = int(A.size) // 2
        params = self._cast(A.dtype, alpha1, alpha2, g11, g12, g22, Isat1, Isat2, w, c)
        args = (k, A_in) + ((V1_scaled, V2_scaled) if V1_scaled is not None else ())
        self._launch(
            self._v_kernel(kernels, "coupled_rk4_stage_c", V1_scaled),
            N_sq,
            *args,
            acc,
            out,
            A,
            *params,
            np.int32(mode),
            np.int32(N_sq),
        )
        return out

    def split_step_coupled_fused(
        self,
        A,
        propagator,
        V1_scaled,
        V2_scaled,
        dz,
        alpha1,
        alpha2,
        g11,
        g12,
        g22,
        Isat1,
        Isat2,
        splitting,
        plan,
        omega=None,
        unnorm_ifft=False,
    ):
        """Take a coupled split step without separating the components.

        The interleaved kernels read both components out of the one
        ``(2, ...)`` array, so nothing is copied out and nothing written
        back. They take scalar parameters and index with one flat id, so the
        caller must not hand them a batch; ``CNLSE.split_step`` checks.

        Parameters
        ----------
        A : cp.ndarray
            Coupled field of shape (2, ...), modified in-place.
        propagator : cp.ndarray
            Pre-computed propagator for both components.
        V1_scaled, V2_scaled : cp.ndarray or None
            Pre-scaled potentials, one per component. Both None or neither.
        dz : float
            Nonlinear step (the whole step for single precision, half for
            double).
        alpha1, alpha2 : float
            Half-loss coefficients.
        g11, g22 : float
            Intra-component interactions.
        g12 : float
            Cross-component interaction.
        Isat1, Isat2 : float
            Saturation intensities (converted units).
        splitting : str
            "lie" or "strang".
        plan : _CuFFTPlan
            Pre-built FFT plan.
        omega : float or None
            Half the Rabi coupling, or None to skip it.
        unnorm_ifft : bool
            If True, the propagator carries the 1/N.

        Returns
        -------
        cp.ndarray
            The propagated field A.
        """
        kernels = self._get_kernels(A.dtype)
        # One thread per element of a component, which is half the field.
        N_sq = int(A.size) // 2
        N_sq_i = np.int32(N_sq)
        params = self._cast(A.dtype, dz, alpha1, alpha2, g11, g12, g22, Isat1, Isat2)

        def nonlinear():
            if V1_scaled is not None:
                self._launch(
                    self._v_kernel(kernels, "coupled_nl_prop_c", V1_scaled),
                    N_sq,
                    A,
                    V1_scaled,
                    V2_scaled,
                    *params,
                    N_sq_i,
                )
            else:
                self._launch(kernels["coupled_nl_prop_c"], N_sq, A, *params, N_sq_i)

        # Double precision: a nonlinear half-step before the linear one
        if splitting == "strang":
            nonlinear()

        plan.fft(A, A)
        self.apply_propagator(A, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(A, A)
        else:
            plan.ifft(A, A)

        nonlinear()

        # Rabi coupling (single precision only)
        if omega is not None:
            cos_c, sin_c = self._cast(
                A.dtype,
                np.cos(omega * float(params[0])),
                np.sin(omega * float(params[0])),
            )
            self._launch(
                kernels["rabi_coupling_interleaved"], N_sq, A, cos_c, sin_c, N_sq_i
            )

        return A

    def rk4_rhs_coupled_fused(
        self,
        A_in,
        k,
        V1_scaled,
        V2_scaled,
        propagator,
        plan,
        alpha1,
        alpha2,
        g11,
        g12,
        g22,
        Isat1,
        Isat2,
        unnorm_ifft=False,
    ):
        """Compute the coupled RK4 RHS without separating the components.

        Parameters
        ----------
        A_in : cp.ndarray
            Coupled input field of shape (2, ...), not modified.
        k : cp.ndarray
            Output buffer (modified in-place).
        V1_scaled, V2_scaled : cp.ndarray or None
            Pre-scaled potentials, one per component. Both None or neither.
        propagator : cp.ndarray
            Pre-computed propagator (pre-divided by N_fft when unnorm_ifft).
        plan : _CuFFTPlan
            Pre-built FFT plan.
        alpha1, alpha2 : float
            Half-loss coefficients.
        g11, g22 : float
            Intra-component interactions.
        g12 : float
            Cross-component interaction.
        Isat1, Isat2 : float
            Saturation intensities (converted units).
        unnorm_ifft : bool
            If True, the propagator carries the 1/N.

        Returns
        -------
        cp.ndarray
            The modified buffer k.
        """
        kernels = self._get_kernels(A_in.dtype)
        N_sq = int(A_in.size) // 2
        N_sq_i = np.int32(N_sq)

        # The transform moves A_in into k, so no copy precedes it.
        plan.fft(A_in, k)
        self.apply_propagator(k, propagator)
        if unnorm_ifft and hasattr(plan, "ifft_unnorm"):
            plan.ifft_unnorm(k, k)
        else:
            plan.ifft(k, k)

        params = self._cast(A_in.dtype, alpha1, alpha2, g11, g12, g22, Isat1, Isat2)
        if V1_scaled is not None:
            self._launch(
                self._v_kernel(kernels, "coupled_rk4_nl_rhs_c", V1_scaled),
                N_sq,
                k,
                A_in,
                V1_scaled,
                V2_scaled,
                *params,
                N_sq_i,
            )
        else:
            self._launch(
                kernels["coupled_rk4_nl_rhs_c"], N_sq, k, A_in, *params, N_sq_i
            )
        return k

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
