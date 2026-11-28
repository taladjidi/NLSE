"""MLX backend tests.

Basic tests for MLX backend operations. Skipped if MLX is not available.
"""

import numpy as np
import pytest
from NLSE.solvers.step_size import (
    CAPPED_STEP_FRACTION,
    SPLIT_STEP_DISPERSION_PHASE_LIMIT,
)
from NLSE.utils import __MLX_AVAILABLE__

pytestmark = pytest.mark.skipif(not __MLX_AVAILABLE__, reason="MLX not available")


# The propagator as two vectors. NY != NX, so that rows and columns swapped
# cannot pass for each other.
FACTORS_NX, FACTORS_NY = 64, 32
FACTORS_STEPS = 3  # steps per run, the last a partial one
# Fraction of a step the run ends on, so the remainder's propagator is read too.
FACTORS_REMAINDER = 0.5
FACTORS_BATCH = 3
FACTORS_SEED = 0  # of the white-noise field, which fills every wavenumber
# Phase per step on the grid's shortest wave, in rad: under the limit past which
# out_field lowers the step (SPLIT_STEP_DISPERSION_PHASE_LIMIT, step_size.py), by
# its own margin. A factor off by a part in a thousand is then far past round-off.
LINEAR_PHASE = CAPPED_STEP_FRACTION * SPLIT_STEP_DISPERSION_PHASE_LIMIT
# Roundings between one application by the factors and one by the grid: each
# factor, their product, the grid's exponential. Its float32 argument adds
# the largest phase, in eps. Measured 0.24 of the bound at most.
FACTORS_ROUNDINGS = 4
APPLICATIONS = {"lie": 1, "strang": 1, "yoshida": 3}  # propagators per step
V_PEAK = 1e-4  # the potential of the lossless-graph test below
V_WAIST = 1 / 8  # of the window: the potential's 1/e radius
ABSORPTION = 0.1  # imaginary part of a complex potential, per unit of real
# No interaction: a nonlinear phase phi turns a relative error e into about
# 2 * phi * e, which the bound does not count.
NLSE_LINEAR = {"alpha": 0.0, "n2": 0.0}
# 1/m. 2 * alpha * dz is 0.023 at NLSE's step, under LOSS_PER_STEP_LIMIT
# (step_size.py), so the step is kept and the kernel is the solved one.
LOSSY_ALPHA = 0.5
# Constructor arguments of tests/integration/test_zero_potential.py, g set to 0.
GPE_ARGS = {"gamma": 0.0, "N": 1e6, "window": 100e-6, "g": 0.0, "m": 87 * 1.66e-27}
# solver, constructor arguments, splitting, batched, potential.
FACTOR_CASES = {
    "lie": ("NLSE", NLSE_LINEAR, "lie", False, None),
    "lossy": ("NLSE", NLSE_LINEAR | {"alpha": LOSSY_ALPHA}, "lie", False, None),
    "real V": ("NLSE", NLSE_LINEAR, "lie", False, "real"),
    "strang": ("NLSE", NLSE_LINEAR, "strang", False, None),
    "strang complex V": ("NLSE", NLSE_LINEAR, "strang", False, "complex"),
    "yoshida": ("NLSE", NLSE_LINEAR, "yoshida", False, None),
    "batch": ("NLSE", NLSE_LINEAR, "lie", True, None),
    "GPE": ("GPE", GPE_ARGS, "lie", False, None),
}
# A propagator the solver did not build, for this many of its steps.
FOREIGN_STEPS = 2


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

    def test_an_unnormalized_inverse_is_refused(self, backend):
        """Catches a request for no 1/N answered with it, as under python -O."""
        arr_mx = backend.from_numpy(np.ones((8, 8), dtype=np.complex64))
        plan = backend.build_fft((8, 8), (-2, -1), np.complex64)
        with pytest.raises(ValueError, match="1/N"):
            backend.ifft(arr_mx, plan, normalize=False)


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
    """A lossless step must not pay for the solved lossy one."""

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
        """The lossless graph is bit-identical to the general one."""
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
    """The solved lossy step is a Metal kernel, and a sixth copy of one formula."""

    @pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
    def test_the_kernel_agrees_with_the_graph_it_replaces(self, with_V):
        """The solved lossy step agrees between its Metal and MLX-ops versions."""
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
        """A real lossy run takes the Metal kernel, and its peak decays."""
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


def factors_solver(name, physics):
    """Return an MLX solver of FACTOR_CASES, with the step its phase asks.

    Parameters
    ----------
    name : str
        Solver class, "NLSE" or "GPE".
    physics : dict
        Constructor arguments over helpers.PHYSICS.

    Returns
    -------
    tuple
        ``(solver, step)``, the step putting LINEAR_PHASE on the shortest wave.
    """
    import NLSE
    from helpers import make

    simu = make(getattr(NLSE, name), "MLX", n=FACTORS_NX, NY=FACTORS_NY, **physics)
    operator = np.asarray(simu._compute_propagator_rk4(), np.complex128)
    return simu, LINEAR_PHASE / np.abs(operator).max()


def white_noise(shape):
    """Return a complex64 field of independent normal samples.

    Parameters
    ----------
    shape : tuple
        Shape of the field.

    Returns
    -------
    np.ndarray
        The field, the same for the same shape.
    """
    rng = np.random.default_rng(FACTORS_SEED)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )


def factors_run(case, grid=False):
    """Run one FACTOR_CASES case on MLX; return what the step needs checked.

    Parameters
    ----------
    case : str
        Key of FACTOR_CASES.
    grid : bool
        Whether to withhold the factors, so the kernel reads the grid.

    Returns
    -------
    tuple
        ``(field out, solver, step, keys of _SPLIT_STEP_CACHE)``.
    """
    from NLSE.kernels import mlx_kernels

    name, physics, splitting, batch, potential = FACTOR_CASES[case]
    simu, dz = factors_solver(name, physics)
    waist = V_WAIST * simu.window[0]
    bump = V_PEAK * np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
    if potential == "real":
        simu.V = bump.astype(np.float32)
    elif potential == "complex":
        simu.V = (bump * (1 + 1j * ABSORPTION)).astype(np.complex64)
    else:
        assert potential is None, potential
    if grid:
        simu._propagator_factors = lambda *args: None
    batch_shape = (FACTORS_BATCH,) if batch else ()
    field = white_noise(batch_shape + simu.XX.shape)
    mlx_kernels._SPLIT_STEP_CACHE.clear()
    out = simu.out_field(
        field,
        (FACTORS_STEPS - FACTORS_REMAINDER) * dz,
        delta_z=dz,
        splitting=splitting,
        verbose=False,
        plot=False,
    )
    keys = list(mlx_kernels._SPLIT_STEP_CACHE)
    mlx_kernels._SPLIT_STEP_CACHE.clear()
    return np.asarray(out), simu, dz, keys


def one_split_step(simu, field, propagator, delta_z):
    """Return ``field`` after one ``split_step`` by hand on ``propagator``.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        Field on the host.
    propagator : mx.array
        Propagator handed to the step.
    delta_z : float
        The step it was built for.

    Returns
    -------
    np.ndarray
        The field after the step, on the host.
    """
    backend = simu._backend
    A = backend.from_numpy(field)
    A_sq = backend.allocate_real_field(field.shape, np.float32)
    plans = simu._build_fft_plan(A)
    out = simu.split_step(A, A_sq, None, propagator, plans, delta_z)
    return np.asarray(backend.to_numpy(out))


# A step lowered by out_field would take more steps than the bound counts.
@pytest.mark.filterwarnings("error:delta_z=")
@pytest.mark.filterwarnings("ignore:splitting='yoshida'")
class TestMLXPropagatorFactors:
    """The fused step multiplies by one vector per axis, not by the grid."""

    @pytest.mark.parametrize("case", sorted(FACTOR_CASES))
    def test_the_factors_answer_as_the_grid(self, case):
        """Every graph the fused step compiles reads the factors, to round-off."""
        fast, simu, dz, keys = factors_run(case)
        slow, *_, grid_keys = factors_run(case, grid=True)
        assert keys and all(key[3] for key in keys), f"the grid was read: {keys}"
        assert grid_keys and not any(key[3] for key in grid_keys), grid_keys
        operator = np.asarray(simu._compute_propagator_rk4(), np.complex128)
        phase = np.abs(operator * dz).max()
        applications = APPLICATIONS[FACTOR_CASES[case][2]] * FACTORS_STEPS
        bound = applications * (FACTORS_ROUNDINGS + phase) * np.finfo(np.float32).eps
        flat = (-1, FACTORS_NY * FACTORS_NX)
        relative = np.max(
            np.linalg.norm((fast - slow).reshape(flat), axis=-1)
            / np.linalg.norm(slow.reshape(flat), axis=-1)
        )
        assert relative <= bound, (
            f"{relative:.3e} off the grid's answer, past {bound:.3e}"
        )

    def test_without_factors_the_kernel_reads_the_grid(self, monkeypatch):
        """A dispersion with a cross term runs on the grid, as it did before."""
        from NLSE import NLSE

        grid, *_ = factors_run("lie", grid=True)
        monkeypatch.setattr(NLSE, "_dispersion_axes", lambda self: None)
        declined, *_, keys = factors_run("lie")
        assert keys and not any(key[3] for key in keys), f"offered factors: {keys}"
        assert np.array_equal(declined, grid), "the fallback is not the grid's step"

    def test_a_poisoned_grid_is_not_read(self, monkeypatch):
        """With the grid's exponential zeroed, only the factors give the answer."""
        import mlx.core as mx
        from NLSE.backends.mlx_backend import MLXBackend

        reference, *_ = factors_run("lie")
        monkeypatch.setattr(MLXBackend, "exp", lambda self, array: mx.zeros_like(array))
        poisoned, *_ = factors_run("lie")
        read, *_ = factors_run("lie", grid=True)
        assert not np.array_equal(read, reference), "the poison does not reach the grid"
        assert np.array_equal(poisoned, reference), "the step read the grid"

    def test_a_propagator_it_did_not_build_is_applied(self):
        """A step handed a copy reads it, not the factors of the solver's own."""
        import mlx.core as mx

        simu, dz = factors_solver("NLSE", NLSE_LINEAR)
        simu.propagator = simu._build_propagator(np.complex64, dz)
        step = FOREIGN_STEPS * dz
        foreign = mx.array(np.asarray(simu._build_propagator(np.complex64, step)))
        field = white_noise(simu.XX.shape)
        own = one_split_step(simu, field, simu.propagator, step)
        applied = one_split_step(simu, field, foreign, step)
        simu._propagator_factors = lambda *args: None
        grid = one_split_step(simu, field, foreign, step)
        assert not np.allclose(own, grid), "the copy steps as the solver's own does"
        assert np.array_equal(applied, grid), "the step did not apply the copy"
