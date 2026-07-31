"""Tests for solver state that must stay consistent across out_field calls.

The solver caches a lot onto ``self`` (propagator, precomputed step constants,
device copies of arrays). These tests pin down the cases where that cached
state has to be refreshed rather than silently reused.
"""

import numpy as np
import pytest
from helpers import as_numpy, make
from NLSE import CNLSE, DDGPE, GPE, NLSE
from NLSE.backends import get_backend, list_available_backends
from NLSE.callbacks import adapt_delta_z
from NLSE.solvers.nlse import DEFAULT_MIN_STEPS, DEFAULT_PHASE_PER_STEP
from scipy.constants import atomic_mass

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
    return make(NLSE, backend, n=N, **{"L": L, "alpha": alpha, **kwargs})


def gaussian_input():
    """Return a smooth complex input field with some transverse structure."""
    simu = make_solver()
    E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    return E


class TestPropagatorRefresh:
    """The linear propagator must track the current delta_z."""

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_changing_delta_z_between_runs_rebuilds_propagator(self, backend_name):
        """Reusing a solver after changing delta_z must not reuse the old propagator.

        The propagator carries exp(-i K^2 dz / 2k), so a stale one silently
        applies the wrong step size for the whole run.

        On every backend, because what could go stale is not the arithmetic
        but where the propagator is kept: the device backends hold it on the
        device and derive a pre-normalized twin beside it, and both have to be
        rebuilt when the step changes. Only the CPU was checked, which is the
        case with the least to get wrong.

        Parameters
        ----------
        backend_name : str
            Backend to run on.
        """
        E = gaussian_input()
        z = 4e-3

        reused = make_solver(backend_name)
        reused.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-4)
        got = as_numpy(
            reused,
            reused.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-5),
        )

        fresh = make_solver(backend_name)
        expected = as_numpy(
            fresh,
            fresh.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-5),
        )

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

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_propagator_matches_delta_z_after_second_run(self, backend_name):
        """After a second out_field with a new delta_z, the propagator matches it.

        Parameters
        ----------
        backend_name : str
            Backend to run on.
        """
        E = gaussian_input()
        simu = make_solver(backend_name)
        simu.out_field(E.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-4)
        simu.out_field(E.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-5)

        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * 1e-5
        ).astype(np.complex64)
        np.testing.assert_allclose(
            np.asarray(as_numpy(simu, simu.propagator)),
            expected,
            rtol=1e-5,
            err_msg="Propagator does not correspond to the current delta_z",
        )


class TestLinearConstruction:
    """A purely linear problem (n2 == 0) is a legal configuration."""

    def test_zero_n2_constructor(self):
        """n2=0 must not raise; it is plain linear propagation."""
        simu = make_solver(n2=0.0)
        dz = simu._default_delta_z()
        assert np.isfinite(dz), "the derived step must be finite for n2=0"
        assert dz > 0, "the derived step must be positive for n2=0"

    def test_zero_n2_propagates_linearly(self):
        """With n2=0 the field must acquire no nonlinear phase."""
        E = gaussian_input()
        simu = make_solver(n2=0.0)
        out = simu.out_field(
            E.copy(), 2e-3, verbose=False, plot=False, normalize=False, delta_z=1e-4
        )
        assert np.all(np.isfinite(out)), "linear propagation produced non-finite values"

    def test_zero_n2_matches_negligible_n2(self):
        """n2=0 must agree with a vanishingly small n2."""
        E = gaussian_input()
        z = 2e-3

        linear = make_solver(n2=0.0)
        got = linear.out_field(
            E.copy(), z, verbose=False, plot=False, normalize=False, delta_z=1e-4
        )

        almost = make_solver(n2=-1e-30)
        expected = almost.out_field(
            E.copy(), z, verbose=False, plot=False, normalize=False, delta_z=1e-4
        )

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-5,
            atol=1e-6 * float(np.max(np.abs(expected))),
            err_msg="n2=0 disagrees with a negligible n2",
        )


class TestPropagationDistance:
    """A run lands on z, rather than past it.

    Taking ceil(z / delta_z) whole steps overshoots unless the step divides
    z, and the error is the phase the medium imprints over the excess: the
    excess as a fraction of z, times the total nonlinear phase.

    Floating point makes it worse than it sounds. A step derived from the
    physics rarely divides z, and even one that should can fail to: for 237
    steps of this case, z / delta_z comes to 237.00000000000003, so ceil asks
    for 238 and the run goes 0.42% too far. That put a 285x error spike at one
    step count, with its neighbours unaffected.
    """

    Z = 5e-3

    def distance(self, simu, delta_z):
        """Propagate and return how far the run actually went."""
        travelled = []
        simu.out_field(
            gaussian_input(),
            self.Z,
            delta_z=delta_z,
            verbose=False,
            plot=False,
            callback=lambda s, A, z, i: travelled.append(z),
            callback_args=(),
        )
        return travelled[-1]

    @pytest.mark.parametrize("steps", [236, 237, 238, 100, 33])
    def test_a_run_stops_at_z(self, steps):
        """Whatever the step, the last callback must report z."""
        simu = make_solver()
        got = self.distance(simu, self.Z / steps)
        assert got == pytest.approx(self.Z, rel=1e-9), (
            f"a run asked for {self.Z:g} m in steps of {self.Z / steps:.4e} "
            f"went {got:.6e} m, {100 * (got - self.Z) / self.Z:+.3f}% out"
        )

    def test_an_indivisible_step_still_stops_at_z(self):
        """The case the whole-step loop could not do at all."""
        simu = make_solver()
        got = self.distance(simu, self.Z / 7.3)
        assert got == pytest.approx(self.Z, rel=1e-9), (
            f"a step that does not divide z overshot to {got:.6e}"
        )

    def test_neighbouring_step_counts_agree_without_callbacks(self):
        """The no-callback path must land on z too.

        This is where the spike was: with no callback the loop runs
        ceil(z / delta_z) whole steps through the backend, so 237 steps of
        z/237 became 238 and the result was 285x further from a converged
        reference than 236 or 238 steps were.
        """
        E = gaussian_input()
        results = {
            steps: np.asarray(
                make_solver(n2=-1.6e-8).out_field(
                    E.copy(), self.Z, delta_z=self.Z / steps, verbose=False, plot=False
                )
            )
            for steps in (236, 237, 238)
        }
        scale = float(np.linalg.norm(results[236]))
        for steps in (237, 238):
            err = float(np.linalg.norm(results[steps] - results[236])) / scale
            assert err < 1e-2, (
                f"{steps} steps differs from 236 by {err:.3e}, so one of them "
                f"is not propagating the distance it was asked for"
            )

    def test_the_derived_step_also_stops_at_z(self):
        """The default is a real number from the physics, so it rarely divides z."""
        simu = make_solver(n2=-1.6e-8)
        travelled = []
        simu.out_field(
            gaussian_input(),
            self.Z,
            verbose=False,
            plot=False,
            callback=lambda s, A, z, i: travelled.append(z),
            callback_args=(),
        )
        assert travelled[-1] == pytest.approx(self.Z, rel=1e-9), (
            f"a default run went {travelled[-1]:.6e} m instead of {self.Z:g}"
        )


class TestCallbackArguments:
    """Callbacks are handed the position the field they receive is at.

    Every callback docstring and the README call it "the current propagation
    distance", so it has to advance. No built-in callback reads it -- they all
    key off the step index -- which leaves these tests as the only check.
    """

    def test_z_advances_with_the_field(self):
        """The position must run from one step up to the whole distance."""
        simu = make_solver()
        z, dz = 1e-3, 2e-4
        seen = []
        simu.out_field(
            gaussian_input(),
            z,
            delta_z=dz,
            verbose=False,
            plot=False,
            callback=lambda s, A, z_, i: seen.append(z_),
            callback_args=(),
        )
        assert seen == pytest.approx([dz * (k + 1) for k in range(len(seen))]), (
            f"callbacks saw {seen}, not the position after each step"
        )

    def test_z_is_not_constant(self):
        """The specific failure: the same number every step."""
        simu = make_solver()
        seen = []
        simu.out_field(
            gaussian_input(),
            1e-3,
            delta_z=2e-4,
            verbose=False,
            plot=False,
            callback=lambda s, A, z_, i: seen.append(z_),
            callback_args=(),
        )
        assert len(set(seen)) == len(seen), (
            "callbacks saw one repeated value, so they were given the total "
            "distance rather than the current position"
        )


class TestAdaptiveStep:
    """A callback changes the step by returning it, and the propagator follows.

    Both halves have to follow. The propagator is built from the step, so a
    new step that reaches the nonlinear half but not a propagator rebuild
    leaves the linear half advancing the wrong distance.
    """

    Z = 2e-3
    DZ = 1e-5

    def run(self, simu, **kwargs):
        """Propagate with the adaptive callback, returning the recorded steps."""
        steps = []
        simu.out_field(
            gaussian_input(),
            self.Z,
            delta_z=self.DZ,
            verbose=False,
            plot=False,
            callback=adapt_delta_z,
            callback_args=(5, steps),
            **kwargs,
        )
        return steps

    def test_the_callback_changes_the_step(self):
        """Precondition: the callback must actually move the step."""
        simu = make_solver()
        steps = self.run(simu)
        assert len(set(steps)) > 1, (
            "the adaptive callback never changed the step, so the checks "
            "below would hold trivially"
        )

    def test_the_propagator_tracks_the_adapted_step(self):
        """The propagator must be rebuilt whenever the step changes."""
        simu = make_solver()
        self.run(simu)
        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu._current_delta_z
        ).astype(PRECISION_COMPLEX)
        np.testing.assert_allclose(
            as_numpy(simu, simu.propagator),
            expected,
            rtol=1e-5,
            err_msg=(
                "the propagator does not correspond to the step the run ended "
                "on, so the linear step advanced by a different distance from "
                "the nonlinear one"
            ),
        )

    def test_an_adapted_run_stays_finite(self):
        """The whole thing must still produce a usable field."""
        simu = make_solver()
        out = simu.out_field(
            gaussian_input(),
            self.Z,
            delta_z=self.DZ,
            verbose=False,
            plot=False,
            callback=adapt_delta_z,
            callback_args=(5, []),
        )
        assert np.all(np.isfinite(as_numpy(simu, out))), (
            "an adaptively stepped run produced non-finite values"
        )


class TestDefaultStep:
    """delta_z is derived from the field's energy unless the caller sets one.

    The step limits are ceilings, and running just under one is the largest
    step that does not fail rather than a sensible default. The default aims
    instead at a fixed phase per step, against the same rates the ceiling for
    that method is built from.
    """

    Z = 2e-3

    def propagate(self, simu, **kwargs):
        """Propagate the module's Gaussian and return the step that was used.

        Deliberately passes no delta_z: what is under test is the one the
        solver derives.
        """
        simu.out_field(gaussian_input(), self.Z, verbose=False, plot=False, **kwargs)
        return simu._current_delta_z

    def test_the_step_hits_the_target_phase_per_step(self):
        """Where the nonlinearity binds, it sets the step and nothing else."""
        simu = make_solver(n2=-1.6e-8)
        used = self.propagate(simu)
        rates = simu._energy_rates(
            simu._prepare_output_array(gaussian_input(), True)[0]
        )
        expected = DEFAULT_PHASE_PER_STEP / (rates["potential"] + rates["interaction"])
        assert used == pytest.approx(expected, rel=1e-6), (
            f"the derived step is {used:.3e} m, but a phase of "
            f"{DEFAULT_PHASE_PER_STEP} rad per step wants {expected:.3e} m"
        )

    def test_a_stronger_nonlinearity_gives_a_shorter_step(self):
        """The step must keep tracking the energy where the energy binds."""
        weak = make_solver(n2=-1.6e-8)
        strong = make_solver(n2=-1.6e-7)
        assert self.propagate(strong) < self.propagate(weak), (
            "a ten-fold stronger nonlinearity did not shorten the step"
        )

    def test_a_weak_nonlinearity_still_takes_several_steps(self):
        """A step comparable to the distance asked for overshoots it.

        The loop runs whole steps of delta_z, so nothing here may reduce to
        one or two of them however little there is to resolve.
        """
        for weak in (-1e-11, -1e-14, 0.0):
            simu = make_solver(n2=weak)
            used = self.propagate(simu)
            assert used <= self.Z / DEFAULT_MIN_STEPS * (1 + 1e-9), (
                f"n2={weak:g}: the step is {used:.3e} m over a {self.Z:g} m "
                f"propagation, which is {self.Z / used:.3g} steps"
            )

    def test_the_step_is_continuous_in_n2(self):
        """A negligible n2 must give what an exactly zero one gives."""
        linear = self.propagate(make_solver(n2=0.0))
        almost = self.propagate(make_solver(n2=-1e-20))
        assert almost == pytest.approx(linear, rel=1e-6), (
            f"n2=0 gives {linear:.3e} but n2=-1e-20 gives {almost:.3e}, "
            f"though they are the same problem"
        )

    def test_a_step_the_caller_passed_is_used(self):
        """An explicit delta_z must be used as given, and only for that run."""
        simu = make_solver()
        assert self.propagate(simu, delta_z=1e-6) == 1e-6, (
            "the caller's step was overridden"
        )
        assert self.propagate(simu) != 1e-6, (
            "the step from the previous call carried over into the next one, "
            "so it is still state on the solver rather than an argument"
        )

    def test_a_step_the_caller_passed_is_still_capped(self):
        """Passing a step is allowed only inside the region of convergence."""
        simu = make_solver(n2=-1.6e-7)
        with pytest.warns(UserWarning, match="exceeds"):
            used = self.propagate(simu, delta_z=1.0)
        assert used < 1.0, "a step past the accuracy limit was left alone"

    def test_the_step_responds_to_the_field_not_just_the_parameters(self):
        """Two fields of the same power but different peak intensity differ.

        This is what reading the field buys over the constructor's estimate,
        which had only power over the window area to go on.

        Over a long enough distance that the phase target is what sets the
        step: DEFAULT_MIN_STEPS also bounds it, and where that binds both
        fields get the same step for a reason that has nothing to do with
        either of them.
        """
        broad = make_solver()
        narrow = make_solver()
        E_broad = np.exp(-(broad.XX**2 + broad.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        E_narrow = np.exp(-(narrow.XX**2 + narrow.YY**2) / (waist / 4) ** 2).astype(
            PRECISION_COMPLEX
        )
        far = self.Z * 100
        broad.out_field(E_broad, far, verbose=False, plot=False)
        narrow.out_field(E_narrow, far, verbose=False, plot=False)
        assert narrow._current_delta_z < broad._current_delta_z, (
            "a tighter beam of the same power concentrates the intensity and "
            "should shorten the step, but the step did not move"
        )


class TestNonlinearityCutoff:
    """Propagation past the medium length L continues linearly."""

    def test_beyond_L_differs_from_fully_nonlinear(self):
        """Running past L must not be the same as staying nonlinear throughout."""
        E = gaussian_input()
        z = 4 * L

        cut = make_solver(L=L)
        got = cut.out_field(E.copy(), z, verbose=False, plot=False, delta_z=L / 20)

        never_cut = make_solver(L=10 * z)
        full = never_cut.out_field(
            E.copy(), z, verbose=False, plot=False, delta_z=L / 20
        )

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
        got = one_shot.out_field(
            E.copy(), z_total, verbose=False, plot=False, delta_z=dz
        )

        # Stage 1: nonlinear up to L. Stage 2: linear for the remainder,
        # feeding stage 1's output back in unnormalized.
        stage1 = make_solver(L=L)
        mid = stage1.out_field(E.copy(), L, verbose=False, plot=False, delta_z=dz)
        stage2 = make_solver(n2=0.0, L=L)
        expected = stage2.out_field(
            mid.astype(PRECISION_COMPLEX),
            z_total - L,
            verbose=False,
            plot=False,
            normalize=False,
            delta_z=dz,
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
        got = cut.out_field(E.copy(), L, verbose=False, plot=False, delta_z=L / 20)

        never_cut = make_solver(L=1e3 * L)
        expected = never_cut.out_field(
            E.copy(), L, verbose=False, plot=False, delta_z=L / 20
        )

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
        without_callback = fast.out_field(
            E.copy(), z, verbose=False, plot=False, delta_z=L / 20
        )

        seen = []

        def record(simu, A, z_, i):
            seen.append(i)

        slow = make_solver(L=L)
        with_callback = slow.out_field(
            E.copy(), z, verbose=False, plot=False, callback=record, delta_z=L / 20
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
        simu.out_field(E.copy(), 3 * L, verbose=False, plot=False, delta_z=L / 20)
        assert simu.n2 == n2, "n2 was not restored after propagating past L"

    def test_zero_L_disables_the_cutoff(self):
        """L=0 means 'no finite medium', not 'everything is linear'.

        GPE passes L=0, so a cutoff keyed on `z > L` alone would make every
        GPE run fully linear.
        """
        E = gaussian_input()
        simu = make_solver(L=0.0)
        got = simu.out_field(E.copy(), 2 * L, verbose=False, plot=False, delta_z=L / 20)

        nonlinear = make_solver(L=1e3 * L)
        expected = nonlinear.out_field(
            E.copy(), 2 * L, verbose=False, plot=False, delta_z=L / 20
        )

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
        simu._precompute_step_constants(None, "lie")
        max_dz = simu._split_step_max_dz(E)
        assert np.isscalar(max_dz) or np.ndim(max_dz) == 0, (
            f"step limit must be a scalar, got {type(max_dz)} / {max_dz!r}"
        )
        assert max_dz > 0

    def test_split_step_max_dz_takes_the_most_restrictive(self):
        """The limit must come from the largest nonlinear rate in the batch."""
        simu = make_solver(n2=self.batched_n2())
        E = self.batched_input(simu)
        simu._precompute_step_constants(None, "lie")
        batched = simu._split_step_max_dz(E)

        # The strongest n2 in the batch alone must give the same limit.
        strongest = make_solver(n2=float(np.min(self.batched_n2()[:, 0, 0])))
        strongest._precompute_step_constants(None, "lie")
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
        one_field = np.exp(-(batched.XX**2 + batched.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )

        got = batched.out_field(
            np.broadcast_to(one_field, (len(values), N, N)).copy(),
            z,
            verbose=False,
            plot=False,
            delta_z=1e-4,
        )
        assert batched._current_delta_z == 1e-4, "the limiter clamped the batched step"
        assert np.all(np.isfinite(as_numpy(batched, got))), (
            "batched run produced non-finite values"
        )

        for index, value in enumerate(values):
            alone = make_solver(n2=float(value), V=potential, backend=backend_name)
            expected = alone.out_field(
                one_field.copy(), z, verbose=False, plot=False, delta_z=1e-4
            )
            assert alone._current_delta_z == 1e-4, (
                "the limiter clamped an individual step"
            )
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


class TestPrecomputedConstants:
    """Precomputed step constants must reflect the current parameters."""

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("n2", -3.2e-9),
            ("alpha", 5.0),
            ("Isat", 1e3),
            # The one that goes to the device rather than staying a number,
            # so the run after it has a copy that could be the old one.
            ("V", None),
        ],
    )
    def test_changing_parameter_between_runs_is_picked_up(
        self, attr, value, backend_name
    ):
        """Changing a physical parameter between runs must change the result.

        On every backend, because a stale value is not a property of the
        arithmetic but of where the value is kept: a scalar is reread each
        step, a potential is sent to the device once and a propagator is
        cached by a key that has to mention everything it was built from.
        Only the CPU was checked, which is the case with nothing to go stale.

        Parameters
        ----------
        attr : str
            Attribute to change between the two runs.
        value : Any
            What to change it to. None means build a potential on the grid,
            which cannot be written as a constant here.
        backend_name : str
            Backend to run on.
        """
        E = gaussian_input()
        z = 2e-3

        def new_value(simu):
            if attr != "V":
                return value
            r_sq = simu.XX**2 + simu.YY**2
            return (-2e-4 * np.exp(-r_sq / (waist / 2) ** 2)).astype(np.float32)

        reused = make_solver(backend_name)
        reused.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-4)
        setattr(reused, attr, new_value(reused))
        got = as_numpy(
            reused,
            reused.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-4),
        )

        fresh = make_solver(backend_name)
        setattr(fresh, attr, new_value(fresh))
        expected = as_numpy(
            fresh, fresh.out_field(E.copy(), z, verbose=False, plot=False, delta_z=1e-4)
        )

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-4,
            atol=1e-4 * float(np.max(np.abs(expected))),
            err_msg=f"Changing {attr} between runs was not picked up on {backend_name}",
        )


class TestStepLimitEnergies:
    """Both limiters must weigh every term the integrator actually applies.

    The rates are expectation values — the energy in each term, weighted by
    where the field has support — rather than a maximum over the grid. A
    maximum is a property of the grid, not of the solution: a tall potential
    in a corner the field never reaches would otherwise set the step for a
    run it has no effect on.

    RK4 diverged to NaN under any potential because its limit counted the
    dispersion alone, and V is scaled by k/2 ~ 4e6. Split-step omitted V
    entirely, on the argument that it is applied exactly; exact is not free,
    because what aliases is the phase the step imprints.
    """

    @staticmethod
    def ring(simu, amplitude=1e-2):
        """Return a ring-shaped potential."""
        r = np.sqrt(simu.XX**2 + simu.YY**2)
        return (amplitude * np.exp(-((r - 2e-3) ** 2) / (3e-4) ** 2)).astype(np.float32)

    def prepared(self, V, backend="CPU"):
        """Return a solver with its step constants and field ready."""
        simu = make_solver(V=V, backend=backend)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu._precompute_step_constants(V, "lie")
        A, _ = simu._prepare_output_array(E, True)
        return simu, A

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_rk4_with_a_potential_does_not_diverge(self, backend_name):
        """RK4 under a potential must converge, not return NaN."""
        probe = make_solver(backend=backend_name)
        V = self.ring(probe)
        simu = make_solver(V=V, backend=backend_name)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        out = simu.out_field(
            E.copy(), 2e-4, verbose=False, plot=False, method="RK4", delta_z=1e-4
        )
        assert np.all(np.isfinite(np.asarray(as_numpy(simu, out)))), (
            f"{backend_name}: RK4 under a potential diverged. Its step limit "
            f"is not accounting for V."
        )

    def test_the_potential_enters_both_limits(self):
        """A potential must tighten the RK4 and the split-step limit alike."""
        probe = make_solver()
        V = self.ring(probe)
        bare, A_bare = self.prepared(None)
        with_V, A_V = self.prepared(V)

        assert with_V._rk4_max_dz(A_V) < bare._rk4_max_dz(A_bare), (
            "a potential must make the RK4 limit more restrictive"
        )
        assert with_V._split_step_max_dz(A_V) < bare._split_step_max_dz(A_bare), (
            "a potential must make the split-step limit more restrictive, "
            "even though the exponential applies V exactly"
        )

    def test_the_rates_are_energies_not_grid_maxima(self):
        """A potential where the field is absent must barely count.

        Two potentials of equal peak height, one on top of the beam and one
        far outside it, would give the same limit under a max reduction. The
        field-weighted rate distinguishes them, which is the whole point.
        """
        probe = make_solver()
        r = np.sqrt(probe.XX**2 + probe.YY**2)
        on_beam = np.exp(-(r**2) / (5e-4) ** 2)
        far_away = np.exp(-((r - 4e-3) ** 2) / (2e-4) ** 2)
        # Normalise so the peaks are exactly equal: the far ring is clipped by
        # the window, and the point is to vary only where the potential sits.
        on_beam = (1e-2 * on_beam / on_beam.max()).astype(np.float32)
        far_away = (1e-2 * far_away / far_away.max()).astype(np.float32)
        assert np.isclose(on_beam.max(), far_away.max()), "precondition: equal peaks"

        near, A_near = self.prepared(on_beam)
        far, A_far = self.prepared(far_away)
        assert (
            near._energy_rates(A_near)["potential"]
            > 10 * far._energy_rates(A_far)["potential"]
        ), (
            "a potential outside the beam constrains the step as much as one "
            "on top of it, so the rate is a grid maximum rather than an energy"
        )

    def test_split_step_ignores_dispersion_but_rk4_does_not(self):
        """Only RK4 is limited by the kinetic term.

        Split-step applies the linear part exactly in Fourier space, so a
        purely linear problem is solved exactly at any step. RK4 approximates
        it, so dispersion binds.
        """
        linear, A = self.prepared(None)
        linear.n2 = 0.0
        linear._precompute_step_constants(None, "lie")
        rates = linear._energy_rates(A)
        assert rates["kinetic"] > 0, "precondition: dispersion is present"
        assert linear._split_step_max_dz(A) == np.inf, (
            "split-step must not be limited by dispersion alone"
        )
        assert np.isfinite(linear._rk4_max_dz(A)), (
            "RK4 must still be limited by dispersion"
        )

    def test_absorption_counts_for_rk4_only(self):
        """A purely absorbing V limits RK4 but imprints no phase.

        Its imaginary part is gain/loss: RK4 approximates it, so it is part
        of the eigenvalue, while split-step applies it exactly and only the
        phase can alias.
        """
        probe = make_solver()
        absorbing = (1j * self.ring(probe)).astype(np.complex64)
        simu, A = self.prepared(absorbing)
        rates = simu._energy_rates(A)

        assert rates["potential"] == pytest.approx(0.0, abs=1e-9), (
            "a purely imaginary V rotates no phase"
        )
        assert rates["loss"] > 0, "a purely imaginary V must count as loss"

    def test_no_potential_leaves_a_finite_limit(self):
        """Without V both limits still have to be finite and positive."""
        simu, A = self.prepared(None)
        assert simu._energy_rates(A)["potential"] == 0.0
        for limit in (simu._rk4_max_dz(A), simu._split_step_max_dz(A)):
            assert np.isfinite(limit) and limit > 0


class TestStepConstantTable:
    """Each step constant is defined once, by ``_step_constants``.

    ``_precompute_step_constants`` writes the table onto the solver as
    fixed-splitting attributes for the kernels. Readers that run before a
    propagation go through ``_constant``, which returns the attribute once it
    exists and the table's value until then. The two must agree.
    """

    @staticmethod
    def solvers():
        """Return one small solver of each kind, by name."""
        common = {"window": window, "V": None, "NX": 32, "NY": 32, "backend": "CPU"}
        h_bar = 0.654
        return {
            "NLSE": NLSE(alpha=0, power=power, n2=n2, L=L, Isat=Isat, **common),
            "CNLSE": CNLSE(
                alpha=0, power=power, n2=n2, n12=-1e-10, L=L, Isat=Isat, **common
            ),
            "GPE": GPE(
                gamma=0.1,
                N=1e5,
                g=1e3,
                m=87 * atomic_mass,
                **{**common, "window": 1e-4},
            ),
            "DDGPE": DDGPE(
                gamma=0.1,
                power=1.0,
                g=1e-2 / h_bar,
                g12=0,
                omega=5.07 / h_bar,
                T=1,
                omega_exc=1484.44 / h_bar,
                omega_cav=1482.76 / h_bar,
                detuning=0.17 / h_bar,
                k_z=27,
                **{**common, "window": 256},
            ),
        }

    @pytest.fixture(params=["NLSE", "CNLSE", "GPE", "DDGPE"])
    def simu(self, request):
        """Return one solver of each kind."""
        return self.solvers()[request.param]

    def test_the_precompute_writes_every_name_the_table_declares(self, simu):
        """A name in the table that nothing sets reads as ``None`` in a kernel."""
        names = sorted(simu._step_constants())
        assert names, "the table is empty"
        for name in names:
            assert getattr(simu, name, None) is None, (
                f"{name} was already set before any precompute"
            )

        simu._precompute_step_constants(None, "lie")

        for name in names:
            assert getattr(simu, name, None) is not None, (
                f"{name} is declared in _step_constants but the precompute skips it"
            )

    def test_reading_a_constant_gives_the_same_value_either_side_of_a_run(self, simu):
        """``_constant`` is the table before a run and the attribute after it.

        They have to agree, or a step chosen before propagating is not the
        step the kernels then take.
        """
        before = {k: float(np.asarray(v)) for k, v in simu._step_constants().items()}
        simu._precompute_step_constants(None, "lie")

        for name, value in before.items():
            np.testing.assert_allclose(
                float(np.asarray(simu._constant(name))),
                value,
                rtol=1e-6,
                err_msg=f"{name} changed across the precompute",
            )

    def test_the_coupled_solvers_reuse_the_base_coupling(self, simu):
        """``_g11`` is the base class's ``_g``, not a second copy of it.

        Except in DDGPE, which hands the kernels unconverted couplings.
        """
        table = simu._step_constants()
        if "_g11" not in table or isinstance(simu, DDGPE):
            pytest.skip("not a solver that renames the base coupling")
        assert table["_g11"] == table["_g"]


def test_ddgpe_couplings_are_not_scaled_by_the_optical_constant():
    """DDGPE's ``g`` reaches the kernels as given, not as an optical n2.

    CNLSE converts a nonlinear index with ``k / 2 * c * epsilon_0``, and
    DDGPE's ``k`` comes from a wavelength it supplies only to satisfy the base
    constructor, about 6e30. Applying that conversion here inflates the
    interaction rate by ~1e26 and collapses the step limit with it.
    """
    h_bar = 0.654
    g = 1e-2 / h_bar
    simu = TestStepConstantTable.solvers()["DDGPE"]
    table = simu._step_constants()

    assert table["_g11"] == pytest.approx(-g), (
        "DDGPE's intra-component coupling is not its own g"
    )
    A = np.ones((2, 32, 32), dtype=PRECISION_COMPLEX)
    interaction = simu._energy_rates(A)["interaction"]
    assert interaction == pytest.approx(g, rel=0.5), (
        f"interaction rate {interaction:.3e} is not of order g = {g:.3e}"
    )


class TestArraysComeBackFromTheDevice:
    """A run leaves the solver's arrays on the host, however it ends.

    ``out_field`` moves V, nl_profile, the propagator and any batched
    parameter onto the device for the duration of a run. Left there, the next
    run is what breaks: ``from_numpy`` is handed a device array. On CL that
    surfaced as ``ValueError: setting an array element with a sequence`` from
    a call that had nothing to do with the failure.
    """

    @staticmethod
    def solver(backend_name):
        """Return a small solver with a potential, which has to travel."""
        V = (-1e-4 * np.ones((N, N))).astype(np.float32)
        return make_solver(backend_name, V=V)

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_a_failed_run_still_brings_them_back(self, backend_name):
        """The retrieve has to happen on the way out of a failure too."""
        simu = self.solver(backend_name)
        E = gaussian_input()

        def raise_midway(simu, A, z, i):
            if i == 2:
                raise RuntimeError("callback failed mid-run")

        with pytest.raises(RuntimeError, match="failed mid-run"):
            simu.out_field(
                E.copy(),
                2e-3,
                verbose=False,
                plot=False,
                delta_z=1e-4,
                callback=raise_midway,
            )

        for attr in simu._gpu_array_attrs:
            value = getattr(simu, attr, None)
            if value is not None:
                assert isinstance(value, np.ndarray), (
                    f"{attr} was left on the device as {type(value).__name__}"
                )

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_the_solver_is_usable_after_a_failed_run(self, backend_name):
        """Which is the point: the failure must not poison the next run."""
        E = gaussian_input()
        z, dz = 2e-3, 1e-4

        expected = as_numpy(
            self.solver(backend_name),
            self.solver(backend_name).out_field(
                E.copy(), z, verbose=False, plot=False, delta_z=dz
            ),
        )

        simu = self.solver(backend_name)

        def raise_midway(simu, A, z, i):
            if i == 2:
                raise RuntimeError("callback failed mid-run")

        with pytest.raises(RuntimeError):
            simu.out_field(
                E.copy(),
                z,
                verbose=False,
                plot=False,
                delta_z=dz,
                callback=raise_midway,
            )

        got = as_numpy(
            simu, simu.out_field(E.copy(), z, verbose=False, plot=False, delta_z=dz)
        )
        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-5,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=f"{backend_name}: the run after a failure disagrees with a clean one",
        )
