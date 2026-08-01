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

        A_sq = mlx_kernels.square_mod(A, A_sq)
        result = backend.to_numpy(A_sq)
        expected = np.abs(A_np) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_apply_propagator(self, backend):
        """apply_propagator multiplies A by propagator."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64) * (1 + 1j)
        prop_np = np.ones((4, 4), dtype=np.complex64) * (0 + 1j)
        A = backend.from_numpy(A_np)
        prop = backend.from_numpy(prop_np)

        A = mlx_kernels.apply_propagator(A, prop)
        result = backend.to_numpy(A)
        expected = A_np * prop_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nl_prop_without_V(self, backend):
        """nl_prop_without_V modifies A."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64)
        A_sq_np = np.ones((4, 4), dtype=np.float32)
        A = backend.from_numpy(A_np)
        A_sq = backend.from_numpy(A_sq_np)

        A = mlx_kernels.nl_prop_without_V(A, A_sq, 1e-5, 0.1, 1.0, 1e10)
        result = backend.to_numpy(A)
        # Should be modified (not all ones anymore)
        assert not np.allclose(result, A_np), "A was not modified"
        assert np.isfinite(result).all(), "Result contains NaN/Inf"

    def test_nl_prop_with_V(self, backend):
        """nl_prop with potential modifies A."""
        from NLSE.kernels import mlx_kernels

        A_np = np.ones((4, 4), dtype=np.complex64)
        A_sq_np = np.ones((4, 4), dtype=np.float32)
        V_np = np.ones((4, 4), dtype=np.float32) * 0.01
        A = backend.from_numpy(A_np)
        A_sq = backend.from_numpy(A_sq_np)
        V = backend.from_numpy(V_np)

        A = mlx_kernels.nl_prop(A, A_sq, 1e-5, 0.1, V, 1.0, 1e10)
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

        A1, A2 = mlx_kernels.rabi_coupling(A1, A2, 1e-3, 1e3)
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

        A = mlx_kernels.square_mod_nl_prop(A, 1e-5, 0.1, 1.0, 1e10)
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

        A = mlx_kernels.square_mod_nl_prop_v(A, V, 1e-5, 0.1, 1.0, 1e10)
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

        A1 = mlx_kernels.nl_prop_c(
            A1, A_sq_1, A_sq_2, 1e-5, 0.1, V, 1.0, 0.5, 1e10, 1e10
        )
        result = backend.to_numpy(A1)
        assert np.isfinite(result).all()

    def test_nl_prop_without_V_c(self, backend):
        """Coupled nl_prop without potential."""
        from NLSE.kernels import mlx_kernels

        A1 = backend.from_numpy(np.ones((4, 4), dtype=np.complex64))
        A_sq_1 = backend.from_numpy(np.ones((4, 4), dtype=np.float32))
        A_sq_2 = backend.from_numpy(np.ones((4, 4), dtype=np.float32) * 0.5)

        A1 = mlx_kernels.nl_prop_without_V_c(
            A1, A_sq_1, A_sq_2, 1e-5, 0.1, 1.0, 0.5, 1e10, 1e10
        )
        result = backend.to_numpy(A1)
        assert np.isfinite(result).all()


class TestMLXLosslessStep:
    """A lossless step must not pay for the solved lossy one.

    ``_nl_factor`` decides between the solved and the frozen step with
    ``mx.where``, and MLX evaluates both arms. At ``alpha = 0`` the answer is
    unchanged, so nothing was wrong -- but a lossless split step measured
    1.45x slower than before the solved step landed, on every grid tried.
    The fix picks the factor when the graph is compiled, which only works
    while ``alpha`` is read on the host.
    """

    def test_a_host_zero_is_lossless_and_a_device_scalar_is_not(self):
        """Only a value numpy can read decides the branch."""
        import mlx.core as mx
        from NLSE.kernels.mlx_kernels import _is_lossless, _loss_mode

        assert _is_lossless(0.0)
        assert _is_lossless(np.float32(0.0))
        assert not _is_lossless(20.0)
        # A device scalar would need a synchronization to read, so it takes
        # the general kernel, which is correct at any alpha.
        assert not _is_lossless(mx.array(0.0))

        # The step length decides the mode as much as alpha does: past the
        # iteration's range the frozen arm applies and only the graph has it.
        assert _loss_mode(0.0, 1e-4, -3.2, 1e5) == "lossless"
        assert _loss_mode(20.0, 1e-4, -3.2, 1e5) == "solved"
        assert _loss_mode(20.0, 1e-2, -3.2, 1e5) == "general"
        assert _loss_mode(mx.array(20.0), 1e-4, -3.2, 1e5) == "general"
        # A batched n2 makes g an array, which reaches a Metal kernel as a
        # pointer rather than a number.
        assert _loss_mode(20.0, 1e-4, mx.array([1.0, 2.0]), 1e5) == "general"

    @pytest.mark.parametrize("splitting", ["lie", "strang"])
    @pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
    def test_the_lossless_graph_answers_exactly_as_the_general_one(
        self, splitting, with_V
    ):
        """Bit-identical, or the fast path is a different solver.

        Not approximately: at ``u = 0`` the iteration returns ``sat`` and the
        decay is exactly 1, so the two graphs are the same arithmetic.
        """
        from NLSE import NLSE
        from NLSE.kernels import mlx_kernels

        waist = 2.23e-3

        def propagate():
            simu = NLSE(
                alpha=0.0,
                power=1.05,
                window=4 * waist,
                n2=-1.6e-9,
                V=None,
                L=1e-2,
                NX=64,
                NY=64,
                Isat=1e5,
                backend="MLX",
            )
            if with_V:
                simu.V = 1e-4 * np.exp(
                    -(simu.XX**2 + simu.YY**2) / (waist / 2) ** 2
                ).astype(np.float32)
            field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
            mlx_kernels._SPLIT_STEP_CACHE.clear()
            return np.asarray(
                simu.out_field(
                    field,
                    20 * 1e-4,
                    delta_z=1e-4,
                    verbose=False,
                    plot=False,
                    splitting=splitting,
                )
            )

        fast = propagate()
        keys = list(mlx_kernels._SPLIT_STEP_CACHE)
        original = mlx_kernels._loss_mode
        try:  # force the graph that carries the loss arithmetic
            mlx_kernels._loss_mode = lambda *a, **k: "general"
            general = propagate()
        finally:
            mlx_kernels._loss_mode = original
            mlx_kernels._SPLIT_STEP_CACHE.clear()

        assert np.array_equal(fast, general), (
            "the lossless graph does not answer as the general one does at "
            f"alpha=0: max |difference| = {np.max(np.abs(fast - general)):g}"
        )
        # And it really was the lossless graph: otherwise the test passes by
        # comparing the slow path against itself.
        assert keys and all(key[-1] == "lossless" for key in keys), (
            f"a lossless run compiled the lossy graph: {keys}"
        )


class TestMLXSolvedStepKernel:
    """The solved lossy step is a Metal kernel, and a sixth copy of one formula.

    ``test_every_backend_solves_the_step_the_same_way`` is what stops the
    other five drifting apart, and it skips MLX -- it scores in double
    precision and MLX has none. So the agreement has to be checked here
    instead, against the MLX graph that expresses the same iteration, in the
    only width either of them has.
    """

    @pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
    def test_the_kernel_agrees_with_the_graph_it_replaces(self, with_V):
        """Written twice, in Metal and in MLX ops, so they must agree.

        To float32 round-off, not to the bit: the kernel evaluates the
        polynomial in Horner form and keeps everything in registers, so the
        rounding differs even though the arithmetic does not.
        """
        import mlx.core as mx
        from NLSE.kernels import mlx_kernels

        rng = np.random.default_rng(0)
        shape = (64, 64)
        A = mx.array(
            (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
        )
        V = mx.array(rng.normal(size=shape).astype(np.float32)) if with_V else None
        dz, alpha, g, Isat = (
            mx.array(x, dtype=mx.float32) for x in (1e-4, 20.0, -3.2, 1e5)
        )

        kernel = np.asarray(mlx_kernels._apply_lossy(A, dz, alpha, g, Isat, V))
        A_sq = (A * mx.conj(A)).real
        graph = np.asarray(A * mlx_kernels._nl_factor(A_sq, dz, alpha, g, Isat, V))
        relative = np.max(np.abs(kernel - graph)) / np.max(np.abs(graph))
        assert relative < 1e-6, (
            f"the Metal kernel and the graph disagree by {relative:.3e}, which "
            f"is past float32 round-off: they are not the same iteration"
        )

    def test_a_lossy_run_takes_the_kernel_and_still_decays(self):
        """The mode has to be reached by a real run, not only by a unit call.

        And loss has to remain loss: the iteration is a contraction only
        inside its range, and outside it returns a larger field than it was
        given, so a lossy run whose peak grows is the failure to catch.
        """
        from NLSE import NLSE
        from NLSE.kernels import mlx_kernels

        waist = 2.23e-3
        simu = NLSE(
            alpha=20.0,
            power=1.05,
            window=4 * waist,
            n2=-1.6e-9,
            V=None,
            L=1e-2,
            NX=64,
            NY=64,
            Isat=1e5,
            backend="MLX",
        )
        field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
        mlx_kernels._SPLIT_STEP_CACHE.clear()
        out = np.asarray(
            simu.out_field(
                field.copy(),
                20 * 1e-4,
                delta_z=1e-4,
                verbose=False,
                plot=False,
                normalize=False,
            )
        )
        modes = [key[-1] for key in mlx_kernels._SPLIT_STEP_CACHE]
        mlx_kernels._SPLIT_STEP_CACHE.clear()

        assert modes == ["solved"], (
            f"a lossy run at u = {2 * 20.0 * 1e-4:g} compiled {modes}, not the "
            f"solved kernel it is inside the range for"
        )
        assert np.all(np.isfinite(out))
        # normalize=False, so the output is comparable to the input directly.
        assert np.max(np.abs(out)) < np.max(np.abs(field)), (
            f"a lossy run came back with a peak of {np.max(np.abs(out)):.4f} "
            f"against the {np.max(np.abs(field)):.4f} it started from"
        )
