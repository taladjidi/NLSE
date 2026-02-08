"""Cross-backend correctness tests for kernel operations.

Each test computes a reference result using pure numpy, then verifies that
the CPU (numba), CL, and (future) Metal backends produce matching output.
"""

import numpy as np
import pytest
from NLSE.kernels_cpu import (
    nl_prop as cpu_nl_prop,
)
from NLSE.kernels_cpu import (
    nl_prop_c as cpu_nl_prop_c,
)
from NLSE.kernels_cpu import (
    nl_prop_without_V as cpu_nl_prop_without_V,
)
from NLSE.kernels_cpu import (
    nl_prop_without_V_c as cpu_nl_prop_without_V_c,
)
from NLSE.kernels_cpu import (
    rabi_coupling as cpu_rabi_coupling,
)
from NLSE.kernels_cpu import (
    square_mod as cpu_square_mod,
)
from NLSE.kernels_cpu import (
    vortex as cpu_vortex,
)

from NLSE import NLSE

if NLSE.__PYOPENCL_AVAILABLE__:
    import pyopencl as cl
    from NLSE.kernels_cl import (
        nl_prop as cl_nl_prop,
    )
    from NLSE.kernels_cl import (
        nl_prop_c as cl_nl_prop_c,
    )
    from NLSE.kernels_cl import (
        nl_prop_without_V as cl_nl_prop_without_V,
    )
    from NLSE.kernels_cl import (
        nl_prop_without_V_c as cl_nl_prop_without_V_c,
    )
    from NLSE.kernels_cl import (
        rabi_coupling as cl_rabi_coupling,
    )
    from NLSE.kernels_cl import (
        square_mod as cl_square_mod,
    )
    from NLSE.kernels_cl import (
        vortex_cp as cl_vortex_cp,
    )
    from pyopencl import array as cla

# Try loading Metal backend
try:
    from NLSE.metal.metal_api import MetalBuffer, MetalContext

    _metal_ctx = MetalContext()
    __METAL_AVAILABLE__ = True
except Exception:
    __METAL_AVAILABLE__ = False

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
N = 64  # small for fast tests


# ---- Pure numpy reference implementations ----


def ref_square_mod(A):
    """Reference: |A|^2"""
    return A.real * A.real + A.imag * A.imag


def ref_nl_prop(A, A_sq, dz, alpha, V, g, Isat):
    """Reference: nonlinear propagation with potential."""
    sat = 1 / (1 + A_sq / Isat)
    arg = -alpha * sat + 1j * g * A_sq * sat + 1j * V
    return A * np.exp(dz * arg)


def ref_nl_prop_without_V(A, A_sq, dz, alpha, g, Isat):
    """Reference: nonlinear propagation without potential."""
    sat = 1 / (1 + A_sq / Isat)
    arg = -alpha * sat + 1j * g * A_sq * sat
    return A * np.exp(dz * arg)


def ref_nl_prop_c(A1, A_sq_1, A_sq_2, dz, alpha, V, g11, g12, Isat1, Isat2):
    """Reference: coupled nonlinear propagation with potential."""
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = -alpha * sat + 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat) + 1j * V
    return A1 * np.exp(dz * arg)


def ref_nl_prop_without_V_c(A1, A_sq_1, A_sq_2, dz, alpha, g11, g12, Isat1, Isat2):
    """Reference: coupled nonlinear propagation without potential."""
    sat = 1 / (1 + A_sq_1 / Isat1 + A_sq_2 / Isat2)
    arg = -alpha * sat + 1j * (g11 * A_sq_1 * sat + g12 * A_sq_2 * sat)
    return A1 * np.exp(dz * arg)


def ref_rabi_coupling(A1, A2, dz, omega):
    """Reference: Rabi coupling between two components."""
    c = np.cos(omega * dz)
    s = np.sin(omega * dz)
    A1_new = c * A1 - 1j * s * A2
    A2_new = c * A2 - 1j * s * A1
    return A1_new, A2_new


def ref_vortex(im, i_pos, j_pos, ii, jj, ll):
    """Reference: vortex phase pattern."""
    # Compute in the same precision as the input arrays
    z = (ii - np.float32(i_pos)) + np.complex64(1j) * (jj - np.float32(j_pos))
    z = z.astype(np.complex64) ** ll
    return im + np.arctan2(z.imag, z.real)


# ---- Helpers ----


def _get_cl_queue():
    if not NLSE.__PYOPENCL_AVAILABLE__:
        pytest.skip("PyOpenCL not available")
    ctx = cl.create_some_context(interactive=False)
    return cl.CommandQueue(ctx)


def _skip_no_metal():
    if not __METAL_AVAILABLE__:
        pytest.skip("Metal not available")


def _metal_from_np(arr):
    return MetalBuffer.from_numpy(_metal_ctx._handle, np.ascontiguousarray(arr))


def _metal_alloc(shape, dtype):
    buf = MetalBuffer(_metal_ctx._handle, shape, dtype)
    arr = np.zeros(shape, dtype=dtype)
    from NLSE.metal.metal_api import _lib

    _lib.metal_buf_copy_from(buf._handle, arr.ctypes.data, buf._nbytes)
    return buf


# ---- Fixtures for random test data ----


def _random_field_2d(shape=(N, N)):
    """Generate a random complex field."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        PRECISION_COMPLEX
    )


def _random_real_2d(shape=(N, N)):
    """Generate a random real field (positive, for A_sq / potential)."""
    rng = np.random.default_rng(123)
    return np.abs(rng.standard_normal(shape)).astype(PRECISION_REAL) + 0.01


# ============================================================
# square_mod tests
# ============================================================


class TestSquareMod:
    def test_cpu_vs_ref(self):
        A = _random_field_2d()
        expected = ref_square_mod(A)
        A_sq = np.zeros((N, N), dtype=PRECISION_REAL)
        cpu_square_mod(A, A_sq)
        assert np.allclose(A_sq, expected, rtol=1e-5), "CPU square_mod != reference"

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A = _random_field_2d()
        expected = ref_square_mod(A)
        A_cl = cla.to_device(queue, A)
        A_sq_cl = cla.zeros(queue, (N, N), dtype=PRECISION_REAL)
        cl_square_mod(A_cl, A_sq_cl)
        result = A_sq_cl.get()
        assert np.allclose(result, expected, rtol=1e-5), "CL square_mod != reference"


# ============================================================
# nl_prop tests
# ============================================================


class TestNlProp:
    dz = 1e-5
    alpha = 10.0
    g = 1e-3
    Isat = 1e4

    def test_cpu_vs_ref(self):
        A = _random_field_2d()
        A_sq = ref_square_mod(A)
        V = _random_real_2d()
        expected = ref_nl_prop(A, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        # CPU kernel mutates in-place
        A_cpu = A.copy()
        A_sq_cpu = A_sq.copy()
        cpu_nl_prop(A_cpu, A_sq_cpu, self.dz, self.alpha, V, self.g, self.Isat)
        assert np.allclose(A_cpu, expected, rtol=1e-4), "CPU nl_prop != reference"

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        V = _random_real_2d()
        expected = ref_nl_prop(A, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        A_cl = cla.to_device(queue, A.copy())
        A_sq_cl = cla.to_device(queue, A_sq.copy())
        V_cl = cla.to_device(queue, V.astype(PRECISION_REAL))
        cl_nl_prop(A_cl, A_sq_cl, self.dz, self.alpha, V_cl, self.g, self.Isat)
        result = A_cl.get()
        assert np.allclose(result, expected, rtol=1e-4), "CL nl_prop != reference"

    def test_cpu_vs_cl(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        V = _random_real_2d()
        # CPU
        A_cpu = A.copy()
        cpu_nl_prop(A_cpu, A_sq.copy(), self.dz, self.alpha, V, self.g, self.Isat)
        # CL
        A_cl = cla.to_device(queue, A.copy())
        A_sq_cl = cla.to_device(queue, A_sq.copy())
        V_cl = cla.to_device(queue, V.astype(PRECISION_REAL))
        cl_nl_prop(A_cl, A_sq_cl, self.dz, self.alpha, V_cl, self.g, self.Isat)
        result_cl = A_cl.get()
        assert np.allclose(A_cpu, result_cl, rtol=1e-4), "CPU nl_prop != CL nl_prop"


# ============================================================
# nl_prop_without_V tests
# ============================================================


class TestNlPropWithoutV:
    dz = 1e-5
    alpha = 10.0
    g = 1e-3
    Isat = 1e4

    def test_cpu_vs_ref(self):
        A = _random_field_2d()
        A_sq = ref_square_mod(A)
        expected = ref_nl_prop_without_V(
            A, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        A_cpu = A.copy()
        cpu_nl_prop_without_V(
            A_cpu, A_sq.copy(), self.dz, self.alpha, self.g, self.Isat
        )
        assert np.allclose(A_cpu, expected, rtol=1e-4), (
            "CPU nl_prop_without_V != reference"
        )

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        expected = ref_nl_prop_without_V(
            A, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        A_cl = cla.to_device(queue, A.copy())
        A_sq_cl = cla.to_device(queue, A_sq.copy())
        cl_nl_prop_without_V(A_cl, A_sq_cl, self.dz, self.alpha, self.g, self.Isat)
        result = A_cl.get()
        assert np.allclose(result, expected, rtol=1e-4), (
            "CL nl_prop_without_V != reference"
        )

    def test_cpu_vs_cl(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        A_cpu = A.copy()
        cpu_nl_prop_without_V(
            A_cpu, A_sq.copy(), self.dz, self.alpha, self.g, self.Isat
        )
        A_cl = cla.to_device(queue, A.copy())
        A_sq_cl = cla.to_device(queue, A_sq.copy())
        cl_nl_prop_without_V(A_cl, A_sq_cl, self.dz, self.alpha, self.g, self.Isat)
        result_cl = A_cl.get()
        assert np.allclose(A_cpu, result_cl, rtol=1e-4), "CPU nl_prop_without_V != CL"


# ============================================================
# nl_prop_c (coupled) tests
# ============================================================


class TestNlPropCoupled:
    dz = 1e-5
    alpha = 10.0
    g11 = 1e-3
    g12 = 5e-4
    Isat1 = 1e4
    Isat2 = 2e4

    def test_cpu_vs_ref(self):
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1)
        A_sq_2 = _random_real_2d()
        V = _random_real_2d()
        expected = ref_nl_prop_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            V,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_cpu = A1.copy()
        cpu_nl_prop_c(
            A1_cpu,
            A_sq_1.copy(),
            A_sq_2.copy(),
            self.dz,
            self.alpha,
            V,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        assert np.allclose(A1_cpu, expected, rtol=1e-4), "CPU nl_prop_c != reference"

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        V = _random_real_2d()
        expected = ref_nl_prop_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            V,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_cl = cla.to_device(queue, A1.copy())
        A_sq_1_cl = cla.to_device(queue, A_sq_1.copy())
        A_sq_2_cl = cla.to_device(queue, A_sq_2.copy())
        V_cl = cla.to_device(queue, V.astype(PRECISION_REAL))
        cl_nl_prop_c(
            A1_cl,
            A_sq_1_cl,
            A_sq_2_cl,
            self.dz,
            self.alpha,
            V_cl,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        result = A1_cl.get()
        assert np.allclose(result, expected, rtol=1e-4), "CL nl_prop_c != reference"

    def test_cpu_vs_cl(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        V = _random_real_2d()
        A1_cpu = A1.copy()
        cpu_nl_prop_c(
            A1_cpu,
            A_sq_1.copy(),
            A_sq_2.copy(),
            self.dz,
            self.alpha,
            V,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_cl = cla.to_device(queue, A1.copy())
        A_sq_1_cl = cla.to_device(queue, A_sq_1.copy())
        A_sq_2_cl = cla.to_device(queue, A_sq_2.copy())
        V_cl = cla.to_device(queue, V.astype(PRECISION_REAL))
        cl_nl_prop_c(
            A1_cl,
            A_sq_1_cl,
            A_sq_2_cl,
            self.dz,
            self.alpha,
            V_cl,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        result_cl = A1_cl.get()
        assert np.allclose(A1_cpu, result_cl, rtol=1e-4), "CPU nl_prop_c != CL"


# ============================================================
# nl_prop_without_V_c (coupled, no potential) tests
# ============================================================


class TestNlPropWithoutVCoupled:
    dz = 1e-5
    alpha = 10.0
    g11 = 1e-3
    g12 = 5e-4
    Isat1 = 1e4
    Isat2 = 2e4

    def test_cpu_vs_ref(self):
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1)
        A_sq_2 = _random_real_2d()
        expected = ref_nl_prop_without_V_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_cpu = A1.copy()
        cpu_nl_prop_without_V_c(
            A1_cpu,
            A_sq_1.copy(),
            A_sq_2.copy(),
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        assert np.allclose(A1_cpu, expected, rtol=1e-4), (
            "CPU nl_prop_without_V_c != reference"
        )

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        expected = ref_nl_prop_without_V_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_cl = cla.to_device(queue, A1.copy())
        A_sq_1_cl = cla.to_device(queue, A_sq_1.copy())
        A_sq_2_cl = cla.to_device(queue, A_sq_2.copy())
        cl_nl_prop_without_V_c(
            A1_cl,
            A_sq_1_cl,
            A_sq_2_cl,
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        result = A1_cl.get()
        assert np.allclose(result, expected, rtol=1e-4), (
            "CL nl_prop_without_V_c != reference"
        )


# ============================================================
# rabi_coupling tests
# ============================================================


class TestRabiCoupling:
    dz = 1e-5
    omega = 1e3

    def test_cpu_vs_ref(self):
        A1 = _random_field_2d()
        A2 = _random_field_2d((N, N))
        # Use a different seed for A2
        rng = np.random.default_rng(99)
        A2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        expected_A1, expected_A2 = ref_rabi_coupling(A1, A2, self.dz, self.omega)
        A1_cpu = A1.copy()
        A2_cpu = A2.copy()
        cpu_rabi_coupling(A1_cpu, A2_cpu, self.dz, self.omega)
        assert np.allclose(A1_cpu, expected_A1, rtol=1e-5), (
            "CPU rabi_coupling A1 != reference"
        )
        assert np.allclose(A2_cpu, expected_A2, rtol=1e-5), (
            "CPU rabi_coupling A2 != reference"
        )

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        rng = np.random.default_rng(42)
        A1 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        rng2 = np.random.default_rng(99)
        A2 = (rng2.standard_normal((N, N)) + 1j * rng2.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        expected_A1, expected_A2 = ref_rabi_coupling(A1, A2, self.dz, self.omega)
        A1_cl = cla.to_device(queue, A1.copy())
        A2_cl = cla.to_device(queue, A2.copy())
        cl_rabi_coupling(A1_cl, A2_cl, self.dz, self.omega)
        assert np.allclose(A1_cl.get(), expected_A1, rtol=1e-5), (
            "CL rabi_coupling A1 != reference"
        )
        assert np.allclose(A2_cl.get(), expected_A2, rtol=1e-5), (
            "CL rabi_coupling A2 != reference"
        )

    def test_unitarity(self):
        """Rabi coupling should conserve total norm."""
        A1 = _random_field_2d()
        rng = np.random.default_rng(99)
        A2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        norm_before = np.sum(np.abs(A1) ** 2 + np.abs(A2) ** 2)
        A1_cpu = A1.copy()
        A2_cpu = A2.copy()
        cpu_rabi_coupling(A1_cpu, A2_cpu, self.dz, self.omega)
        norm_after = np.sum(np.abs(A1_cpu) ** 2 + np.abs(A2_cpu) ** 2)
        assert np.allclose(norm_before, norm_after, rtol=1e-6), (
            f"Rabi coupling not unitary: {norm_before} -> {norm_after}"
        )


# ============================================================
# vortex tests
# ============================================================


class TestVortex:
    def test_cpu_vs_ref(self):
        im = np.zeros((N, N), dtype=PRECISION_REAL)
        ii, jj = np.meshgrid(
            np.arange(N, dtype=PRECISION_REAL),
            np.arange(N, dtype=PRECISION_REAL),
            indexing="ij",
        )
        i_pos, j_pos = N // 2, N // 2
        ll = 1
        expected = ref_vortex(im.copy(), i_pos, j_pos, ii, jj, ll)
        cpu_vortex(im, i_pos, j_pos, ii, jj, ll)
        assert np.allclose(im, expected, rtol=1e-5), "CPU vortex != reference"

    def test_cpu_charge2(self):
        im = np.zeros((N, N), dtype=PRECISION_REAL)
        ii, jj = np.meshgrid(
            np.arange(N, dtype=PRECISION_REAL),
            np.arange(N, dtype=PRECISION_REAL),
            indexing="ij",
        )
        i_pos, j_pos = N // 4, 3 * N // 4
        ll = 2
        expected = ref_vortex(im.copy(), i_pos, j_pos, ii, jj, ll)
        cpu_vortex(im, i_pos, j_pos, ii, jj, ll)
        assert np.allclose(im, expected, rtol=1e-5), "CPU vortex charge 2 != reference"

    def test_cl_vs_ref(self):
        if not NLSE.__PYOPENCL_AVAILABLE__:
            pytest.skip("PyOpenCL not available")
        queue = _get_cl_queue()
        im = np.zeros((N, N), dtype=PRECISION_REAL)
        ii, jj = np.meshgrid(
            np.arange(N, dtype=PRECISION_REAL),
            np.arange(N, dtype=PRECISION_REAL),
            indexing="ij",
        )
        i_pos, j_pos = N // 2, N // 2
        ll = 1
        expected = ref_vortex(im.copy(), i_pos, j_pos, ii, jj, ll)
        im_cl = cla.to_device(queue, im.copy())
        ii_cl = cla.to_device(queue, ii)
        jj_cl = cla.to_device(queue, jj)
        cl_vortex_cp(im_cl, i_pos, j_pos, ii_cl, jj_cl, ll)
        result = im_cl.get()
        # Compare angles modulo 2*pi to handle atan2 branch cut (+pi vs -pi)
        diff = np.abs(result - expected)
        diff = np.minimum(diff, 2 * np.pi - diff)
        assert np.all(diff < 1e-4), "CL vortex != reference"


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_nl_prop_zero_alpha(self):
        """No losses: nl_prop should only apply phase."""
        A = _random_field_2d()
        A_sq = ref_square_mod(A)
        V = _random_real_2d()
        dz, alpha, g, Isat = 1e-5, 0.0, 1e-3, 1e4
        A_cpu = A.copy()
        cpu_nl_prop(A_cpu, A_sq.copy(), dz, alpha, V, g, Isat)
        # No losses => |A| should be preserved element-wise
        assert np.allclose(np.abs(A_cpu), np.abs(A), rtol=1e-5), (
            "nl_prop with alpha=0 changed field amplitude"
        )

    def test_nl_prop_zero_g(self):
        """No interaction: nl_prop should only apply losses + potential phase."""
        A = _random_field_2d()
        A_sq = ref_square_mod(A)
        V = np.zeros((N, N), dtype=PRECISION_REAL)
        dz, alpha, g, Isat = 1e-5, 10.0, 0.0, 1e4
        expected = ref_nl_prop(A, A_sq, dz, alpha, V, g, Isat)
        A_cpu = A.copy()
        cpu_nl_prop(A_cpu, A_sq.copy(), dz, alpha, V, g, Isat)
        assert np.allclose(A_cpu, expected, rtol=1e-4), "nl_prop with g=0 != reference"

    def test_nl_prop_high_saturation(self):
        """When Isat is very large, sat -> 1."""
        A = _random_field_2d()
        A_sq = ref_square_mod(A)
        V = _random_real_2d()
        dz, alpha, g, Isat = 1e-5, 10.0, 1e-3, 1e20
        expected = ref_nl_prop(A, A_sq, dz, alpha, V, g, Isat)
        A_cpu = A.copy()
        cpu_nl_prop(A_cpu, A_sq.copy(), dz, alpha, V, g, Isat)
        assert np.allclose(A_cpu, expected, rtol=1e-4), (
            "nl_prop with high Isat != reference"
        )

    def test_rabi_zero_omega(self):
        """Zero coupling: fields should be unchanged."""
        A1 = _random_field_2d()
        rng = np.random.default_rng(99)
        A2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        A1_cpu = A1.copy()
        A2_cpu = A2.copy()
        cpu_rabi_coupling(A1_cpu, A2_cpu, 1e-5, 0.0)
        assert np.allclose(A1_cpu, A1, rtol=1e-6), "Rabi with omega=0 changed A1"
        assert np.allclose(A2_cpu, A2, rtol=1e-6), "Rabi with omega=0 changed A2"

    def test_square_mod_1d_field(self):
        """square_mod should work on 1D fields."""
        rng = np.random.default_rng(42)
        A = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(
            PRECISION_COMPLEX
        )
        expected = A.real * A.real + A.imag * A.imag
        A_sq = np.zeros(N, dtype=PRECISION_REAL)
        cpu_square_mod(A, A_sq)
        assert np.allclose(A_sq, expected, rtol=1e-5), "CPU square_mod 1D != reference"


# ============================================================
# Metal backend tests
# ============================================================


class TestMetalKernels:
    """Test all Metal kernels against numpy reference."""

    dz = 1e-5
    alpha = 10.0
    g = 1e-3
    Isat = 1e4
    g11 = 1e-3
    g12 = 5e-4
    Isat1 = 1e4
    Isat2 = 2e4

    def test_square_mod(self):
        _skip_no_metal()
        A = _random_field_2d()
        expected = ref_square_mod(A)
        A_buf = _metal_from_np(A)
        A_sq_buf = _metal_alloc((N, N), PRECISION_REAL)
        _metal_ctx.square_mod(A_buf, A_sq_buf)
        assert np.allclose(A_sq_buf.to_numpy(), expected, rtol=1e-5), (
            "Metal square_mod != reference"
        )

    def test_nl_prop(self):
        _skip_no_metal()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        V = _random_real_2d()
        expected = ref_nl_prop(A, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        A_buf = _metal_from_np(A.copy())
        _metal_ctx.nl_prop(
            A_buf,
            _metal_from_np(A_sq),
            _metal_from_np(V),
            self.dz,
            self.alpha,
            self.g,
            self.Isat,
        )
        assert np.allclose(A_buf.to_numpy(), expected, rtol=1e-4), (
            "Metal nl_prop != reference"
        )

    def test_nl_prop_without_V(self):
        _skip_no_metal()
        A = _random_field_2d()
        A_sq = ref_square_mod(A).astype(PRECISION_REAL)
        expected = ref_nl_prop_without_V(
            A, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        A_buf = _metal_from_np(A.copy())
        _metal_ctx.nl_prop_without_V(
            A_buf,
            _metal_from_np(A_sq),
            self.dz,
            self.alpha,
            self.g,
            self.Isat,
        )
        assert np.allclose(A_buf.to_numpy(), expected, rtol=1e-4), (
            "Metal nl_prop_without_V != reference"
        )

    def test_nl_prop_c(self):
        _skip_no_metal()
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        V = _random_real_2d()
        expected = ref_nl_prop_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            V,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_buf = _metal_from_np(A1.copy())
        _metal_ctx.nl_prop_c(
            A1_buf,
            _metal_from_np(A_sq_1),
            _metal_from_np(A_sq_2),
            _metal_from_np(V),
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        assert np.allclose(A1_buf.to_numpy(), expected, rtol=1e-4), (
            "Metal nl_prop_c != reference"
        )

    def test_nl_prop_without_V_c(self):
        _skip_no_metal()
        A1 = _random_field_2d()
        A_sq_1 = ref_square_mod(A1).astype(PRECISION_REAL)
        A_sq_2 = _random_real_2d()
        expected = ref_nl_prop_without_V_c(
            A1,
            A_sq_1,
            A_sq_2,
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        A1_buf = _metal_from_np(A1.copy())
        _metal_ctx.nl_prop_without_V_c(
            A1_buf,
            _metal_from_np(A_sq_1),
            _metal_from_np(A_sq_2),
            self.dz,
            self.alpha,
            self.g11,
            self.g12,
            self.Isat1,
            self.Isat2,
        )
        assert np.allclose(A1_buf.to_numpy(), expected, rtol=1e-4), (
            "Metal nl_prop_without_V_c != reference"
        )

    def test_rabi_coupling(self):
        _skip_no_metal()
        A1 = _random_field_2d()
        rng = np.random.default_rng(99)
        A2 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))).astype(
            PRECISION_COMPLEX
        )
        omega, dz_r = 1e3, 1e-5
        expected_A1, expected_A2 = ref_rabi_coupling(A1, A2, dz_r, omega)
        A1_buf = _metal_from_np(A1.copy())
        A2_buf = _metal_from_np(A2.copy())
        scratch = MetalBuffer(_metal_ctx._handle, (N, N), PRECISION_COMPLEX)
        _metal_ctx.rabi_coupling(A1_buf, A2_buf, scratch, dz_r, omega)
        assert np.allclose(A1_buf.to_numpy(), expected_A1, rtol=1e-5), (
            "Metal rabi_coupling A1 != reference"
        )
        assert np.allclose(A2_buf.to_numpy(), expected_A2, rtol=1e-5), (
            "Metal rabi_coupling A2 != reference"
        )

    def test_complex_multiply_inplace(self):
        _skip_no_metal()
        A = _random_field_2d()
        B = _random_field_2d()
        expected = A * B
        A_buf = _metal_from_np(A.copy())
        B_buf = _metal_from_np(B)
        _metal_ctx.complex_multiply_inplace(A_buf, B_buf)
        assert np.allclose(A_buf.to_numpy(), expected, rtol=1e-5), (
            "Metal complex_multiply != reference"
        )
