"""Tests for solver state that must stay consistent across out_field calls.

The solver caches a lot onto ``self`` (propagator, precomputed step constants,
device copies of arrays). These tests pin down the cases where that cached
state has to be refreshed rather than silently reused.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends

from .helpers import as_numpy

PRECISION_COMPLEX = np.complex64

AVAILABLE_BACKENDS = list_available_backends()
# Backends whose solvers bypass apply_propagator for the fused linear step.
LINEAR_STEP_BACKENDS = [
    name for name in AVAILABLE_BACKENDS if get_backend(name).has_linear_step
]

N = 64
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 10e-3
alpha = 0.0


def make_solver(backend="CPU", **kwargs):
    """Build a small NLSE solver with the module defaults."""
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": None,
        "L": L,
        "NX": N,
        "NY": N,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(kwargs)
    return NLSE(**params)


def gaussian_input():
    """Return a smooth complex input field with some transverse structure."""
    simu = make_solver()
    E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    return E


class TestPropagatorRefresh:
    """The linear propagator must track the current delta_z."""

    def test_changing_delta_z_between_runs_rebuilds_propagator(self):
        """Reusing a solver after changing delta_z must not reuse the old propagator.

        The propagator carries exp(-i K^2 dz / 2k), so a stale one silently
        applies the wrong step size for the whole run.
        """
        E = gaussian_input()
        z = 4e-3

        reused = make_solver()
        reused.delta_z = 1e-4
        reused.out_field(E.copy(), z, verbose=False, plot=False)
        reused.delta_z = 1e-5
        got = reused.out_field(E.copy(), z, verbose=False, plot=False)

        fresh = make_solver()
        fresh.delta_z = 1e-5
        expected = fresh.out_field(E.copy(), z, verbose=False, plot=False)

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-4,
            atol=1e-4 * float(np.max(np.abs(expected))),
            err_msg=(
                "Reusing a solver after changing delta_z gave a different result "
                "than a fresh solver with the same delta_z: the propagator was "
                "not rebuilt."
            ),
        )

    def test_propagator_matches_delta_z_after_second_run(self):
        """After a second out_field with a new delta_z, the propagator matches it."""
        E = gaussian_input()
        simu = make_solver()
        simu.delta_z = 1e-4
        simu.out_field(E.copy(), 2e-3, verbose=False, plot=False)
        simu.delta_z = 1e-5
        simu.out_field(E.copy(), 2e-3, verbose=False, plot=False)

        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
        ).astype(np.complex64)
        np.testing.assert_allclose(
            np.asarray(simu.propagator),
            expected,
            rtol=1e-5,
            err_msg="Propagator does not correspond to the current delta_z",
        )


class TestLinearConstruction:
    """A purely linear problem (n2 == 0) is a legal configuration."""

    def test_zero_n2_constructor(self):
        """n2=0 must not raise; it is plain linear propagation."""
        simu = make_solver(n2=0.0)
        assert np.isfinite(simu.delta_z), "delta_z must be finite for n2=0"
        assert simu.delta_z > 0, "delta_z must be positive for n2=0"

    def test_zero_n2_propagates_linearly(self):
        """With n2=0 the field must acquire no nonlinear phase."""
        E = gaussian_input()
        simu = make_solver(n2=0.0)
        simu.delta_z = 1e-4
        out = simu.out_field(E.copy(), 2e-3, verbose=False, plot=False, normalize=False)
        assert np.all(np.isfinite(out)), "linear propagation produced non-finite values"

    def test_zero_n2_matches_negligible_n2(self):
        """n2=0 must agree with a vanishingly small n2."""
        E = gaussian_input()
        z = 2e-3

        linear = make_solver(n2=0.0)
        linear.delta_z = 1e-4
        got = linear.out_field(E.copy(), z, verbose=False, plot=False, normalize=False)

        almost = make_solver(n2=-1e-30)
        almost.delta_z = 1e-4
        expected = almost.out_field(
            E.copy(), z, verbose=False, plot=False, normalize=False
        )

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-5,
            atol=1e-6 * float(np.max(np.abs(expected))),
            err_msg="n2=0 disagrees with a negligible n2",
        )


class TestNonlinearityCutoff:
    """Propagation past the medium length L continues linearly."""

    def test_beyond_L_differs_from_fully_nonlinear(self):
        """Running past L must not be the same as staying nonlinear throughout."""
        E = gaussian_input()
        z = 4 * L

        cut = make_solver(L=L)
        cut.delta_z = L / 20
        got = cut.out_field(E.copy(), z, verbose=False, plot=False)

        never_cut = make_solver(L=10 * z)
        never_cut.delta_z = L / 20
        full = never_cut.out_field(E.copy(), z, verbose=False, plot=False)

        assert not np.allclose(got, full, rtol=1e-3), (
            "Propagating past L gave the same result as a fully nonlinear run: "
            "the nonlinearity was never switched off."
        )

    def test_beyond_L_matches_two_stage_propagation(self):
        """A run past L equals nonlinear-to-L followed by a linear run."""
        E = gaussian_input()
        dz = L / 20
        z_total = 3 * L

        one_shot = make_solver(L=L)
        one_shot.delta_z = dz
        got = one_shot.out_field(E.copy(), z_total, verbose=False, plot=False)

        # Stage 1: nonlinear up to L. Stage 2: linear for the remainder,
        # feeding stage 1's output back in unnormalized.
        stage1 = make_solver(L=L)
        stage1.delta_z = dz
        mid = stage1.out_field(E.copy(), L, verbose=False, plot=False)
        stage2 = make_solver(n2=0.0, L=L)
        stage2.delta_z = dz
        expected = stage2.out_field(
            mid.astype(PRECISION_COMPLEX),
            z_total - L,
            verbose=False,
            plot=False,
            normalize=False,
        )

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-3,
            atol=1e-3 * float(np.max(np.abs(expected))),
            err_msg="Past-L propagation does not match nonlinear-then-linear",
        )

    def test_up_to_L_is_unaffected(self):
        """Propagating only up to L must be untouched by the cutoff."""
        E = gaussian_input()
        cut = make_solver(L=L)
        cut.delta_z = L / 20
        got = cut.out_field(E.copy(), L, verbose=False, plot=False)

        never_cut = make_solver(L=1e3 * L)
        never_cut.delta_z = L / 20
        expected = never_cut.out_field(E.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            got, expected, rtol=1e-6, err_msg="z <= L should be fully nonlinear"
        )

    def test_callback_loop_cuts_off_at_the_same_place(self):
        """The callback loop and the fast loop must agree past L.

        They cut off differently: the fast loop splits into two execute_loop
        segments, the callback loop switches mid-iteration on z_prop.
        """
        E = gaussian_input()
        z = 3 * L

        fast = make_solver(L=L)
        fast.delta_z = L / 20
        without_callback = fast.out_field(E.copy(), z, verbose=False, plot=False)

        seen = []

        def record(simu, A, z_, i):
            seen.append(i)

        slow = make_solver(L=L)
        slow.delta_z = L / 20
        with_callback = slow.out_field(
            E.copy(), z, verbose=False, plot=False, callback=record
        )

        assert seen, "callback never fired"
        np.testing.assert_allclose(
            with_callback,
            without_callback,
            rtol=1e-5,
            atol=1e-5 * float(np.max(np.abs(without_callback))),
            err_msg="callback loop and fast loop disagree on the past-L cutoff",
        )

    def test_nonlinearity_restored_after_run(self):
        """The coupling attributes must survive a past-L run unchanged."""
        E = gaussian_input()
        simu = make_solver(L=L)
        simu.delta_z = L / 20
        simu.out_field(E.copy(), 3 * L, verbose=False, plot=False)
        assert simu.n2 == n2, "n2 was not restored after propagating past L"

    def test_zero_L_disables_the_cutoff(self):
        """L=0 means 'no finite medium', not 'everything is linear'.

        GPE passes L=0, so a cutoff keyed on `z > L` alone would make every
        GPE run fully linear.
        """
        E = gaussian_input()
        simu = make_solver(L=0.0)
        simu.delta_z = L / 20
        got = simu.out_field(E.copy(), 2 * L, verbose=False, plot=False)

        nonlinear = make_solver(L=1e3 * L)
        nonlinear.delta_z = L / 20
        expected = nonlinear.out_field(E.copy(), 2 * L, verbose=False, plot=False)

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-6,
            err_msg="L=0 must not switch the nonlinearity off",
        )


class TestStepLimitWithBatchedParameters:
    """The step limiter must cope with per-simulation parameter arrays.

    Broadcasting a parameter across a batch makes the precomputed constants
    arrays rather than scalars, so any scalar comparison inside the limiter
    raises "truth value of an array is ambiguous".
    """

    @staticmethod
    def batched_n2(count=3):
        """Return an n2 array shaped for broadcasting over a batch."""
        n2_arr = np.zeros((count, 1, 1))
        n2_arr[:, 0, 0] = np.linspace(-1.6e-9, -1e-10, count)
        return n2_arr

    def batched_input(self, simu, count=3):
        """Return a batched input field matching the solver grid."""
        env = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
        return (np.ones((count, N, N)) * env).astype(PRECISION_COMPLEX)

    def test_split_step_max_dz_returns_a_scalar(self):
        """A batched g must reduce to one scalar step limit."""
        simu = make_solver(n2=self.batched_n2())
        E = self.batched_input(simu)
        simu._precompute_step_constants(None, "single")
        max_dz = simu._split_step_max_dz(E)
        assert np.isscalar(max_dz) or np.ndim(max_dz) == 0, (
            f"step limit must be a scalar, got {type(max_dz)} / {max_dz!r}"
        )
        assert max_dz > 0

    def test_split_step_max_dz_takes_the_most_restrictive(self):
        """The limit must come from the largest nonlinear rate in the batch."""
        simu = make_solver(n2=self.batched_n2())
        E = self.batched_input(simu)
        simu._precompute_step_constants(None, "single")
        batched = simu._split_step_max_dz(E)

        # The strongest n2 in the batch alone must give the same limit.
        strongest = make_solver(n2=float(np.min(self.batched_n2()[:, 0, 0])))
        strongest._precompute_step_constants(None, "single")
        single = strongest._split_step_max_dz(E[0])

        np.testing.assert_allclose(
            batched,
            single,
            rtol=1e-6,
            err_msg="batched limit is not the most restrictive of the batch",
        )

    @pytest.mark.parametrize("backend_name", list_available_backends())
    def test_batched_run_matches_individual_runs(self, backend_name):
        """Each slice of a batched run must equal running that case alone.

        Broadcasting is what makes a parameter sweep cheap, so the batch has
        to be equivalent to the individual runs, not merely finite. The
        earlier failure was silent: apply_propagator indexed a shared
        propagator with the batched field's flat index, so every slice after
        the first came back as NaN or garbage.
        """
        if backend_name in ("CL", "MLX"):
            pytest.skip(
                f"{backend_name} does not implement broadcasting: its kernels "
                f"take scalar parameters, so a batched run cannot be built yet"
            )
        # Keep the run short and the step well inside the accuracy limit, so
        # the limiter clamps nothing. It reduces over the batch, so a batched
        # run and a weak individual one would otherwise take different steps
        # and differ by discretisation error rather than by a bug.
        z = 1e-3
        values = self.batched_n2()[:, 0, 0]
        batched = make_solver(n2=self.batched_n2(), backend=backend_name)
        potential = (
            -1e-4 * np.exp(-(batched.XX**2 + batched.YY**2) / (70e-6) ** 2)
        ).astype(np.float32)
        batched = make_solver(n2=self.batched_n2(), V=potential, backend=backend_name)
        batched.delta_z = 1e-4
        one_field = np.exp(-(batched.XX**2 + batched.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )

        got = batched.out_field(
            np.broadcast_to(one_field, (len(values), N, N)).copy(),
            z,
            verbose=False,
            plot=False,
        )
        assert batched.delta_z == 1e-4, "the limiter clamped the batched step"
        assert np.all(np.isfinite(as_numpy(batched, got))), (
            "batched run produced non-finite values"
        )

        for index, value in enumerate(values):
            alone = make_solver(n2=float(value), V=potential, backend=backend_name)
            alone.delta_z = 1e-4
            expected = alone.out_field(one_field.copy(), z, verbose=False, plot=False)
            assert alone.delta_z == 1e-4, "the limiter clamped an individual step"
            np.testing.assert_allclose(
                np.asarray(as_numpy(batched, got))[index],
                np.asarray(as_numpy(alone, expected)),
                rtol=1e-4,
                atol=1e-5 * float(np.max(np.abs(as_numpy(alone, expected)))),
                err_msg=f"batch slice {index} (n2={value:.3e}) differs from the "
                f"same simulation run on its own",
            )

    def test_shared_propagator_is_not_indexed_past_its_end(self):
        """A batched field against a shared propagator must broadcast."""
        from NLSE.kernels import cpu as cpu_kernels

        field = (np.ones((3, 4, 4)) + 0j).astype(PRECISION_COMPLEX)
        propagator = (np.full((4, 4), 2.0) + 0j).astype(PRECISION_COMPLEX)
        out = cpu_kernels.apply_propagator(field.copy(), propagator)
        np.testing.assert_allclose(
            out,
            np.full((3, 4, 4), 2.0, dtype=PRECISION_COMPLEX),
            err_msg="propagator was not broadcast across the batch",
        )

    @pytest.mark.parametrize("backend_name", LINEAR_STEP_BACKENDS)
    def test_linear_step_broadcasts_a_shared_propagator(self, backend_name):
        """The fused linear step must broadcast like apply_propagator does.

        Backends declaring has_linear_step never reach apply_propagator from
        the solver, so fixing the batch handling there left linear_step
        reading past the end of a shared propagator. Whether that shows up as
        NaN depends on what the device allocator left behind, which is why it
        looked like test-ordering flakiness rather than a plain out-of-bounds
        read.
        """
        if backend_name == "CL":
            pytest.skip(
                "CL cannot launch over a slice of a batched field: a cla.Array "
                "slice starts at an offset and .data refuses it. It raises "
                "instead, which test_cl_linear_step_rejects_a_batched_field pins"
            )
        backend = get_backend(backend_name)
        axes = (-2, -1)
        rng = np.random.default_rng(0)
        field = (rng.random((3, 8, 8)) + 1j * rng.random((3, 8, 8))).astype(
            PRECISION_COMPLEX
        )
        propagator = (rng.random((8, 8)) + 1j * rng.random((8, 8))).astype(
            PRECISION_COMPLEX
        )
        plans = backend.build_fft(field.shape, axes, field.dtype, array=field)

        got = backend.to_numpy(
            backend.kernels.linear_step(
                backend.from_numpy(field.copy()),
                backend.from_numpy(propagator),
                plans[0],
            )
        )

        expected = np.fft.ifftn(
            np.fft.fftn(field, axes=axes) * propagator, axes=axes
        ).astype(PRECISION_COMPLEX)
        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=(
                f"{backend_name}.linear_step did not broadcast a shared "
                f"propagator across the batch"
            ),
        )

    @pytest.mark.skipif("CL" not in AVAILABLE_BACKENDS, reason="OpenCL not available")
    def test_cl_linear_step_rejects_a_batched_field(self):
        """CL must refuse a batched field rather than read out of bounds."""
        backend = get_backend("CL")
        axes = (-2, -1)
        field = np.ones((3, 8, 8), dtype=PRECISION_COMPLEX)
        propagator = np.ones((8, 8), dtype=PRECISION_COMPLEX)
        plans = backend.build_fft(field.shape, axes, field.dtype, array=field)

        with pytest.raises(
            NotImplementedError, match="does not implement broadcasting"
        ):
            backend.kernels.linear_step(
                backend.from_numpy(field),
                backend.from_numpy(propagator),
                plans[0],
            )


class TestFieldOnlyBatch:
    """A batch does not have to carry a batched parameter.

    Running several initial conditions through identical physics leaves every
    parameter scalar and puts the extra axis on the field alone. The kernels
    then index the field and any shared grid (the potential, the propagator)
    with the same flat index, so the grid is read past its end for every
    slice after the first.
    """

    @staticmethod
    def potential(simu):
        """Return a grid-shaped potential shared by the whole batch."""
        return (-1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / (70e-6) ** 2)).astype(
            np.float32
        )

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_shared_potential_matches_the_individual_run(self, backend_name, precision):
        """Every slice must equal the same simulation run on its own."""
        if backend_name in ("CL", "MLX"):
            pytest.skip(
                f"{backend_name} does not implement broadcasting: its kernels "
                f"take scalar parameters, so a batched run cannot be built yet"
            )
        z = 1e-3
        count = 3
        probe = make_solver(backend=backend_name)
        V = self.potential(probe)

        batched = make_solver(V=V, backend=backend_name)
        batched.delta_z = 1e-4
        one_field = np.exp(-(batched.XX**2 + batched.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        got = as_numpy(
            batched,
            batched.out_field(
                np.broadcast_to(one_field, (count, N, N)).copy(),
                z,
                verbose=False,
                plot=False,
                precision=precision,
            ),
        )
        assert np.all(np.isfinite(got)), (
            "a field-only batch against a shared potential produced non-finite "
            "values: the potential was indexed past its end"
        )

        alone = make_solver(V=V, backend=backend_name)
        alone.delta_z = 1e-4
        expected = np.asarray(
            as_numpy(
                alone,
                alone.out_field(
                    one_field.copy(),
                    z,
                    verbose=False,
                    plot=False,
                    precision=precision,
                ),
            )
        )
        for index in range(count):
            np.testing.assert_allclose(
                np.asarray(got)[index],
                expected,
                rtol=1e-4,
                atol=1e-5 * float(np.max(np.abs(expected))),
                err_msg=f"batch slice {index} differs from the same simulation "
                f"run on its own",
            )


class TestPrecomputedConstants:
    """Precomputed step constants must reflect the current parameters."""

    @pytest.mark.parametrize("attr,value", [("n2", -3.2e-9), ("alpha", 5.0)])
    def test_changing_parameter_between_runs_is_picked_up(self, attr, value):
        """Changing a physical parameter between runs must change the result."""
        E = gaussian_input()
        z = 2e-3

        reused = make_solver()
        reused.delta_z = 1e-4
        reused.out_field(E.copy(), z, verbose=False, plot=False)
        setattr(reused, attr, value)
        got = reused.out_field(E.copy(), z, verbose=False, plot=False)

        fresh = make_solver(**{attr: value})
        fresh.delta_z = 1e-4
        expected = fresh.out_field(E.copy(), z, verbose=False, plot=False)

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-4,
            atol=1e-4 * float(np.max(np.abs(expected))),
            err_msg=f"Changing {attr} between runs was not picked up",
        )
