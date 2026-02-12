"""MLX backend tests.

Basic tests for MLX backend operations. Skipped if MLX is not available.
"""

import numpy as np
import pytest
from NLSE.utils import __MLX_AVAILABLE__

pytestmark = pytest.mark.skipif(not __MLX_AVAILABLE__, reason="MLX not available")


@pytest.fixture
def backend():
    """Create an MLX backend instance."""
    from NLSE.backends.mlx_backend import MLXBackend

    return MLXBackend()


class TestMLXBackendBasic:
    """Basic MLX backend functionality tests."""

    def test_name(self, backend):
        """Backend name is MLX."""
        assert backend.name == "MLX"

    def test_supports_double_precision(self, backend):
        """MLX does not support double precision."""
        assert backend.supports_double_precision() is False

    def test_allocate_field(self, backend):
        """Allocate complex field on MLX device."""
        import mlx.core as mx

        arr = backend.allocate_field((64, 64), np.complex64)
        assert arr.shape == (64, 64)
        assert arr.dtype == mx.complex64

    def test_allocate_real_field(self, backend):
        """Allocate real field on MLX device."""
        import mlx.core as mx

        arr = backend.allocate_real_field((64, 64), np.float32)
        assert arr.shape == (64, 64)
        assert arr.dtype == mx.float32

    def test_from_numpy(self, backend):
        """Transfer numpy array to MLX device."""
        import mlx.core as mx

        arr_np = np.ones((4, 4), dtype=np.complex64)
        arr_mx = backend.from_numpy(arr_np)
        assert arr_mx.dtype == mx.complex64
        assert arr_mx.shape == (4, 4)

    def test_from_numpy_downcast(self, backend):
        """Double precision is downcast to single precision."""
        import mlx.core as mx

        arr_np = np.ones((4, 4), dtype=np.complex128)
        arr_mx = backend.from_numpy(arr_np)
        assert arr_mx.dtype == mx.complex64

        arr_np_real = np.ones((4, 4), dtype=np.float64)
        arr_mx_real = backend.from_numpy(arr_np_real)
        assert arr_mx_real.dtype == mx.float32

    def test_to_numpy(self, backend):
        """Transfer MLX array back to numpy."""
        arr_np = np.ones((4, 4), dtype=np.complex64) * (1 + 2j)
        arr_mx = backend.from_numpy(arr_np)
        result = backend.to_numpy(arr_mx)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, arr_np)

    def test_roundtrip(self, backend):
        """Numpy -> MLX -> numpy preserves values."""
        arr_np = np.random.random((8, 8)).astype(np.float32)
        arr_mx = backend.from_numpy(arr_np)
        result = backend.to_numpy(arr_mx)
        np.testing.assert_allclose(result, arr_np, rtol=1e-6)


class TestMLXFFT:
    """FFT tests for MLX backend."""

    def test_build_fft(self, backend):
        """build_fft returns axes list."""
        plan = backend.build_fft((64, 64), (-2, -1), np.complex64)
        assert len(plan) == 1
        assert plan[0] == (-2, -1)

    def test_fft_roundtrip(self, backend):
        """FFT then IFFT recovers original array."""
        arr_np = np.random.random((64, 64)).astype(
            np.complex64
        ) + 1j * np.random.random((64, 64)).astype(np.complex64)
        arr_mx = backend.from_numpy(arr_np)
        plan = backend.build_fft((64, 64), (-2, -1), np.complex64)

        result = backend.fft(arr_mx, plan)
        result = backend.ifft(result, plan)
        result_np = backend.to_numpy(result)

        np.testing.assert_allclose(
            result_np, arr_np, rtol=1e-5, atol=1e-6, err_msg="FFT roundtrip failed"
        )


class TestMLXKernels:
    """Smoke tests for MLX kernels (compare against CPU reference)."""

    def test_square_mod(self, backend):
        """square_mod computes |A|^2."""
        from NLSE.kernels import mlx_kernels

        A_np = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
        A = backend.from_numpy(A_np)
        A_sq = backend.allocate_real_field(A.shape, np.float32)

        mlx_kernels.square_mod(A, A_sq)
        result = backend.to_numpy(A_sq)
        expected = np.abs(A_np) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_apply_propagator(self, backend):
        """apply_propagator multiplies A by propagator in-place."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64) * (1 + 1j)
        prop_np = np.ones((4, 4), dtype=np.complex64) * (0 + 1j)
        A = backend.from_numpy(A_np)
        prop = backend.from_numpy(prop_np)

        mlx_kernels.apply_propagator(A, prop)
        result = backend.to_numpy(A)
        expected = A_np * prop_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nl_prop_without_V(self, backend):
        """nl_prop_without_V modifies A in-place."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64)
        A_sq_np = np.ones((4, 4), dtype=np.float32)
        A = backend.from_numpy(A_np)
        A_sq = backend.from_numpy(A_sq_np)

        mlx_kernels.nl_prop_without_V(A, A_sq, 1e-5, 0.1, 1.0, 1e10)
        result = backend.to_numpy(A)
        # Should be modified (not all ones anymore)
        assert not np.allclose(result, A_np), "A was not modified"
        assert np.isfinite(result).all(), "Result contains NaN/Inf"

    def test_nl_prop_with_V(self, backend):
        """nl_prop with potential modifies A in-place."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64)
        A_sq_np = np.ones((4, 4), dtype=np.float32)
        V_np = np.ones((4, 4), dtype=np.float32) * 0.01
        A = backend.from_numpy(A_np)
        A_sq = backend.from_numpy(A_sq_np)
        V = backend.from_numpy(V_np)

        mlx_kernels.nl_prop(A, A_sq, 1e-5, 0.1, V, 1.0, 1e10)
        result = backend.to_numpy(A)
        assert not np.allclose(result, A_np), "A was not modified"
        assert np.isfinite(result).all()

    def test_rabi_coupling(self, backend):
        """rabi_coupling exchanges density between components."""
        from NLSE.kernels import mlx_kernels

        A1_np = np.ones((4, 4), dtype=np.complex64)
        A2_np = np.zeros((4, 4), dtype=np.complex64)
        A1 = backend.from_numpy(A1_np)
        A2 = backend.from_numpy(A2_np)

        mlx_kernels.rabi_coupling(A1, A2, 1e-3, 1e3)
        r1 = backend.to_numpy(A1)
        r2 = backend.to_numpy(A2)

        # Power should be conserved
        total_before = np.sum(np.abs(A1_np) ** 2 + np.abs(A2_np) ** 2)
        total_after = np.sum(np.abs(r1) ** 2 + np.abs(r2) ** 2)
        np.testing.assert_allclose(total_after, total_before, rtol=1e-5)

        # Some power should have transferred to A2
        assert np.sum(np.abs(r2) ** 2) > 0, "No power transferred"

    def test_square_mod_nl_prop(self, backend):
        """Fused square_mod + nl_prop kernel."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64) * (1 + 0.5j)
        A = backend.from_numpy(A_np)

        mlx_kernels.square_mod_nl_prop(A, 1e-5, 0.1, 1.0, 1e10)
        result = backend.to_numpy(A)
        assert not np.allclose(result, A_np)
        assert np.isfinite(result).all()

    def test_square_mod_nl_prop_v(self, backend):
        """Fused square_mod + nl_prop with potential."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64) * (1 + 0.5j)
        V_np = np.ones((4, 4), dtype=np.float32) * 0.01
        A = backend.from_numpy(A_np)
        V = backend.from_numpy(V_np)

        mlx_kernels.square_mod_nl_prop_v(A, V, 1e-5, 0.1, 1.0, 1e10)
        result = backend.to_numpy(A)
        assert not np.allclose(result, A_np)
        assert np.isfinite(result).all()

    def test_nl_prop_c(self, backend):
        """Coupled nl_prop with potential."""
        from NLSE.kernels import mlx_kernels

        A1 = backend.from_numpy(np.ones((4, 4), dtype=np.complex64))
        A_sq_1 = backend.from_numpy(np.ones((4, 4), dtype=np.float32))
        A_sq_2 = backend.from_numpy(np.ones((4, 4), dtype=np.float32) * 0.5)
        V = backend.from_numpy(np.ones((4, 4), dtype=np.float32) * 0.01)

        mlx_kernels.nl_prop_c(A1, A_sq_1, A_sq_2, 1e-5, 0.1, V, 1.0, 0.5, 1e10, 1e10)
        result = backend.to_numpy(A1)
        assert np.isfinite(result).all()

    def test_nl_prop_without_V_c(self, backend):
        """Coupled nl_prop without potential."""
        from NLSE.kernels import mlx_kernels

        A1 = backend.from_numpy(np.ones((4, 4), dtype=np.complex64))
        A_sq_1 = backend.from_numpy(np.ones((4, 4), dtype=np.float32))
        A_sq_2 = backend.from_numpy(np.ones((4, 4), dtype=np.float32) * 0.5)

        mlx_kernels.nl_prop_without_V_c(
            A1, A_sq_1, A_sq_2, 1e-5, 0.1, 1.0, 0.5, 1e10, 1e10
        )
        result = backend.to_numpy(A1)
        assert np.isfinite(result).all()
