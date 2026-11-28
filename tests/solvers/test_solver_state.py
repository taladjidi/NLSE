"""Tests for solver state that must stay consistent across out_field calls."""

from unittest.mock import Mock

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, as_numpy, make
from NLSE import CNLSE, DDGPE, GPE, NLSE
from NLSE.backends import get_backend, list_available_backends
from NLSE.callbacks import adapt_delta_z
from NLSE.solvers.nlse import DEFAULT_MIN_STEPS, DEFAULT_PHASE_PER_STEP
from NLSE.solvers.step_size import (
    CAPPED_STEP_FRACTION,
    RK4_CUBIC_RADIUS,
    RK4_PHASE_PER_STEP,
    RK4_STABILITY_RADIUS,
    SPLIT_STEP_DISPERSION_PHASE_LIMIT,
)
from scipy.constants import atomic_mass

PRECISION_COMPLEX = np.complex64

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

# Ring potential outside the beam for the RK4 bound: radius and width in m,
# peak as an index change. Scaled by k/2 the peak is 4.0e4 rad/m, against
# 63 rad/m for this grid's shortest wave (both measured, CPU).
FAR_RING_RADIUS = 4e-3
FAR_RING_WIDTH = 2e-4
FAR_RING_PEAK = 1e-2
# Loss in 1/m for the RK4 bound test; any value above zero serves.
BOUND_ALPHA = 20.0
# Loss in 1/m and potential as an index change, each the only term of an RK4
# run without n2: their rates, alpha/2 and (k/2) V, are 400 and 403 rad/m, so
# each alone sets the default step at 5e-5 m, under the Z / DEFAULT_MIN_STEPS
# = 2e-4 m bound; without it the step is that bound.
RK4_TERM_ALPHA = 800.0
RK4_TERM_V = 1e-4
# The Gaussian's kinetic rate, 0.025 rad/m, also counts: measured shift of the
# step from the one-term closed form 6.2e-5 (CPU). Smallest effect: 2x.
RK4_TERM_RTOL = 2e-4
# Interaction for the phase-limit tests: its rate, 4.7e3 rad/m, puts the
# split-step phase limit at 6.7e-4 m, under the 1.6e-2 m dispersion limit of
# this grid, so the phase limit is the one that binds.
PHASE_LIMIT_N2 = -1.6e-8
# Against the closed form for a Gaussian, measured 2.0e-4 (CPU): the window
# cuts the beam at two waists and the field is float32. Smallest effect: 2x.
PHASE_LIMIT_RTOL = 1e-3
# A batch's rate against its strongest member's alone, float32 on both sides:
# measured 1.6e-7 (CPU). The weakest member's rate is 16x lower.
BATCH_RATE_RTOL = 1e-6
# DDGPE's interaction rate on a unit field against its g: measured equal.
RATE_RTOL = 1e-6
# A step far past every limit of this module's grid, in m.
FAR_PAST_LIMIT = 1.0
# RK4 on the cubic term alone, psi' = -i |psi|^2 psi from |psi| = 1, as a map
# of the phase per step: where its iteration first diverges (scanned in
# steps of 1e-3), how many steps are iterated and the modulus that counts as
# diverged. Bounded steps stay under 2 (scanned), and 1.547 passes 1e6 in
# under 100 steps.
CUBIC_DIVERGES_FROM = 1.547
CUBIC_MAP_STEPS = 500
CUBIC_MAP_CEILING = 1e6
CUBIC_SCAN = np.arange(0.1, RK4_CUBIC_RADIUS + 0.05, 0.1)
# An RK4 run the cubic term limits: a Kerr medium with no saturation, long
# enough for this many steps at the capped step (L in m).
RK4_CUBIC_STEPS = 200
RK4_CUBIC_L = 1.0
# Coupled RK4 runs, all linear. Uniform potential as an index change, the
# ratio k2 / k that makes the second component's potential the faster one
# (1.6e5 rad/m), and a Rabi coupling in rad/m whose half is 5e4 rad/m.
COUPLED_V = 1e-2
K2_OVER_K = 4
RABI_OMEGA = 1e5
# Their length and the step the caller asks, in m: the step is 57 and 18
# times past the bound (2.83 over those two rates), so only the cap keeps
# the run stable.
COUPLED_RUN_LENGTH = 2e-3
COUPLED_CALLER_STEP = 1e-3
# How far above its input the norm of a stable lossless run may sit: float32
# round-off over the 40 to 130 steps these runs take.
NORM_GROWTH_RTOL = 1e-4


def make_solver(backend="CPU", **kwargs):
    """Build a small NLSE solver with the module defaults."""
    return make(NLSE, backend, n=N, **{"L": L, "alpha": alpha, **kwargs})


def rk4_cubic_rate(psi):
    """Return the right-hand side of psi' = -i |psi|^2 psi."""
    return -1j * abs(psi) ** 2 * psi


def rk4_cubic_step(psi, x):
    """Take one classic RK4 step of size x on psi' = -i |psi|^2 psi."""
    k1 = rk4_cubic_rate(psi)
    k2 = rk4_cubic_rate(psi + x / 2 * k1)
    k3 = rk4_cubic_rate(psi + x / 2 * k2)
    k4 = rk4_cubic_rate(psi + x * k3)
    return psi + x / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_cubic_stays_bounded(x):
    """Whether iterating that step from |psi| = 1 stays under the ceiling."""
    psi = 1 + 0j
    for _ in range(CUBIC_MAP_STEPS):
        psi = rk4_cubic_step(psi, x)
        if not abs(psi) < CUBIC_MAP_CEILING:
            return False
    return True


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
        dz = simu._default_delta_z(gaussian_input())
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
    """A run lands on z, rather than past it."""

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
        """Neighbouring step counts land on z without callbacks too."""
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
    """Callbacks are handed the position the field they receive is at."""

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

    @pytest.mark.parametrize("none", [[], ()], ids=["list", "tuple"])
    def test_no_callbacks_is_a_run_without_one(self, none, monkeypatch):
        """An empty list or tuple of callbacks runs as no callback does."""
        per_step = Mock(side_effect=AssertionError("the per-step loop ran"))
        monkeypatch.setattr(NLSE, "_loop_with_callbacks", per_step)
        runs = [
            make_solver().out_field(
                gaussian_input(), 1e-3, delta_z=2e-4, verbose=False, callback=given
            )
            for given in (none, None)
        ]
        np.testing.assert_allclose(runs[0], runs[1], rtol=NORM_GROWTH_RTOL)

    def test_a_tuple_of_callbacks_is_called(self):
        """A tuple of callables is taken as a list of them is."""
        seen = []
        make_solver().out_field(
            gaussian_input(),
            1e-3,
            delta_z=2e-4,
            verbose=False,
            callback=(lambda s, A, z_, i: seen.append(z_),),
        )
        assert len(seen) == 5, f"the callback ran {len(seen)} times over 5 steps"

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
    """A callback changes the step by returning it, and the propagator follows."""

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
    """delta_z is derived from the field's energy unless the caller sets one."""

    Z = 2e-3

    def propagate(self, simu, **kwargs):
        """Propagate the module's Gaussian and return the step that was used."""
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

    def test_the_rk4_step_hits_its_own_target_phase_per_step(self):
        """The RK4 step hits its own target phase, weighing every rate."""
        simu = make_solver(n2=-1.6e-8)
        used = self.propagate(simu, method="RK4")
        rates = simu._energy_rates(
            simu._prepare_output_array(gaussian_input(), True)[0]
        )
        expected = RK4_PHASE_PER_STEP / sum(rates.values())
        assert used == pytest.approx(expected, rel=1e-6), (
            f"the derived RK4 step is {used:.3e} m, but a phase of "
            f"{RK4_PHASE_PER_STEP} rad per step wants {expected:.3e} m"
        )

    @pytest.mark.parametrize(
        "term",
        [{"alpha": RK4_TERM_ALPHA}, {"V": RK4_TERM_V}],
        ids=["loss", "potential"],
    )
    def test_the_rk4_step_counts_the_loss_and_the_potential(self, term):
        """Loss or the potential, as the only term, alone sets RK4's default step."""
        if "V" in term:
            term = {"V": np.full((N, N), term["V"], dtype=np.float32)}
        simu = make_solver(n2=0.0, **term)
        used = self.propagate(simu, method="RK4")
        # Loss rotates at alpha/2, a uniform potential at (k/2) V.
        rate = RK4_TERM_ALPHA / 2 if "alpha" in term else simu.k / 2 * RK4_TERM_V
        expected = RK4_PHASE_PER_STEP / rate
        assert used == pytest.approx(expected, rel=RK4_TERM_RTOL), (
            f"with {sorted(term)} alone the RK4 step is {used:.3e} m, but a "
            f"phase of {RK4_PHASE_PER_STEP} rad per step wants {expected:.3e} m"
        )

    def test_rk4_takes_a_shorter_step_than_the_split_step(self):
        """The two no longer share a number, and RK4's is the tighter one."""
        rk4 = self.propagate(make_solver(n2=-1.6e-8), method="RK4")
        split = self.propagate(make_solver(n2=-1.6e-8))
        assert rk4 < split, (
            f"RK4 derived {rk4:.3e} m against the split step's {split:.3e} m, "
            f"though it aims at {RK4_PHASE_PER_STEP} rad per step against "
            f"{DEFAULT_PHASE_PER_STEP}"
        )

    def test_a_stronger_nonlinearity_gives_a_shorter_step(self):
        """The step must keep tracking the energy where the energy binds."""
        weak = make_solver(n2=-1.6e-8)
        strong = make_solver(n2=-1.6e-7)
        assert self.propagate(strong) < self.propagate(weak), (
            "a ten-fold stronger nonlinearity did not shorten the step"
        )

    def test_a_weak_nonlinearity_still_takes_several_steps(self):
        """A weak nonlinearity still takes several steps over the distance asked for."""
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
        """Fields of equal power but different peak intensity get different steps."""
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

    @pytest.mark.parametrize("beyond", [1, 2, 5])
    def test_the_absorption_stops_at_the_medium_as_well(self, beyond):
        """Absorption stops at the end of the medium, as the nonlinearity does."""
        absorbing = make_solver(alpha=20.0, n2=0.0, L=L)
        field = np.ones((N, N), dtype=PRECISION_COMPLEX)
        out = as_numpy(
            absorbing,
            absorbing.out_field(
                field.copy(),
                beyond * L,
                delta_z=L / 100,
                verbose=False,
                plot=False,
                normalize=False,
            ),
        )
        got = float(np.abs(out[N // 2, N // 2]) ** 2)
        assert got == pytest.approx(np.exp(-20.0 * L), rel=1e-4), (
            f"after {beyond}L the intensity is {got:.6f}; the medium is {L} m "
            f"long, so Beer's law over it leaves {np.exp(-20.0 * L):.6f} and "
            f"absorbing all the way to {beyond}L would leave "
            f"{np.exp(-20.0 * beyond * L):.6f}"
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
        """The callback loop and the fast loop must agree past L."""
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
        """L=0 means 'no finite medium', not 'everything is linear'."""
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
    """The step limiter must cope with per-simulation parameter arrays."""

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
        simu._precompute_step_constants(None, np.complex64)
        max_dz = simu._split_step_max_dz(E)
        assert np.isscalar(max_dz) or np.ndim(max_dz) == 0, (
            f"step limit must be a scalar, got {type(max_dz)} / {max_dz!r}"
        )
        assert max_dz > 0

    def test_split_step_max_dz_takes_the_most_restrictive(self):
        """The limit must come from the largest nonlinear rate in the batch."""
        simu = make_solver(n2=self.batched_n2())
        E = self.batched_input(simu)
        simu._precompute_step_constants(None, np.complex64)
        batched = simu._split_step_max_dz(E)

        # The strongest n2 in the batch alone must give the same limit.
        strongest = make_solver(n2=float(np.min(self.batched_n2()[:, 0, 0])))
        strongest._precompute_step_constants(None, np.complex64)
        single = strongest._split_step_max_dz(E[0])

        np.testing.assert_allclose(
            batched,
            single,
            rtol=1e-6,
            err_msg="batched limit is not the most restrictive of the batch",
        )

    def test_the_interaction_rate_is_the_strongest_members(self):
        """A batch's interaction rate is that of its strongest n2 alone."""
        simu = make_solver(n2=self.batched_n2())
        E = self.batched_input(simu)
        simu._precompute_step_constants(None, np.complex64)
        strongest = make_solver(n2=float(np.min(self.batched_n2()[:, 0, 0])))
        strongest._precompute_step_constants(None, np.complex64)
        batched = simu._energy_rates(E)["interaction"]
        single = strongest._energy_rates(E[0])["interaction"]
        assert batched == pytest.approx(single, rel=BATCH_RATE_RTOL), (
            f"the batch's interaction rate is {batched:.4e} rad/m against "
            f"{single:.4e} for its strongest member alone"
        )

    @pytest.mark.parametrize("backend_name", list_available_backends())
    def test_batched_run_matches_individual_runs(self, backend_name):
        """Each slice of a batched run must equal running that case alone."""
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
        """The fused linear step must broadcast like apply_propagator does."""
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
    """Both limiters must weigh every term the integrator actually applies."""

    @staticmethod
    def ring(simu, amplitude=1e-2):
        """Return a ring-shaped potential."""
        r = np.sqrt(simu.XX**2 + simu.YY**2)
        return (amplitude * np.exp(-((r - 2e-3) ** 2) / (3e-4) ** 2)).astype(np.float32)

    def prepared(self, V, backend="CPU", **kwargs):
        """Return a solver with its step constants and field ready."""
        simu = make_solver(V=V, backend=backend, **kwargs)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu._precompute_step_constants(V, np.complex64)
        A, _ = simu._prepare_output_array(E, True)
        return simu, A

    def test_the_split_step_limit_is_pi_of_real_space_phase(self):
        """At the limit one step imprints pi of interaction phase."""
        simu, A = self.prepared(None, n2=PHASE_LIMIT_N2)
        ratio = 2 * simu.power / (np.pi * waist**2) / simu.I_sat
        mean = simu.I_sat**2 * (ratio - np.log1p(ratio)) * np.pi * waist**2
        rate = simu.k * abs(PHASE_LIMIT_N2) * mean / (2 * simu.power)
        limit = simu._split_step_max_dz(A)
        assert limit == pytest.approx(np.pi / rate, rel=PHASE_LIMIT_RTOL), (
            f"the split-step limit is {limit:.4e} m; pi rad at the closed-form "
            f"rate {rate:.4e} rad/m wants {np.pi / rate:.4e} m"
        )

    @pytest.mark.parametrize(
        "method, limit",
        [("split_step", "_split_step_max_dz"), ("RK4", "_rk4_max_dz")],
    )
    def test_a_capped_step_lands_inside_the_limit(self, method, limit):
        """A step past the limit is lowered strictly below it, not onto it."""
        simu, A = self.prepared(None, n2=PHASE_LIMIT_N2)
        with pytest.warns(UserWarning, match="exceeds"):
            capped = simu._capped_delta_z(FAR_PAST_LIMIT, A, method)
        bound = getattr(simu, limit)(A)
        assert 0 < capped < bound, (
            f"{method}: a {FAR_PAST_LIMIT} m step was capped to {capped:.4e} m "
            f"against a limit of {bound:.4e} m"
        )

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

    def test_rk4_takes_the_potential_at_its_grid_maximum(self):
        """A potential the field barely reaches still sets the RK4 step."""
        probe = make_solver()
        r = np.sqrt(probe.XX**2 + probe.YY**2)
        ring = np.exp(-((r - FAR_RING_RADIUS) ** 2) / FAR_RING_WIDTH**2)
        ring = (FAR_RING_PEAK * ring / ring.max()).astype(np.float32)
        simu, A = self.prepared(ring)
        rates = simu._energy_rates(A)
        peak = float(np.max(np.abs(as_numpy(simu, simu._V_scaled))))
        assert rates["potential"] < 0.1 * peak, "precondition: V is off the beam"

        cubic = RK4_STABILITY_RADIUS / RK4_CUBIC_RADIUS
        fastest = simu._max_dispersion_rate() + peak
        fastest += cubic * simu._max_interaction_rate(A) + rates["loss"]
        assert simu._rk4_max_dz(A) == pytest.approx(RK4_STABILITY_RADIUS / fastest), (
            "the RK4 bound does not take the potential at its grid maximum"
        )

    def test_rk4_counts_the_interaction_and_the_loss(self):
        """The field's own rates add to the linear ones in the RK4 bound."""
        simu, A = self.prepared(None, alpha=BOUND_ALPHA)
        rates = simu._energy_rates(A)
        assert rates["interaction"] > 0, "precondition: the medium is nonlinear"
        assert rates["loss"] > 0, "precondition: the medium absorbs"

        linear = RK4_STABILITY_RADIUS / simu._rk4_max_dz()
        full = RK4_STABILITY_RADIUS / simu._rk4_max_dz(A)
        cubic = RK4_STABILITY_RADIUS / RK4_CUBIC_RADIUS
        added = cubic * simu._max_interaction_rate(A) + rates["loss"]
        assert full - linear == pytest.approx(added), (
            "the RK4 bound does not add the peak interaction and the loss"
        )

    def test_the_cubic_radius_is_inside_where_rk4_diverges(self):
        """RK4 on the cubic term alone is bounded up to the radius, not past."""
        assert all(rk4_cubic_stays_bounded(x) for x in CUBIC_SCAN), (
            "RK4 diverges on the cubic term at a step inside RK4_CUBIC_RADIUS"
        )
        assert not rk4_cubic_stays_bounded(CUBIC_DIVERGES_FROM), (
            "precondition: the map diverges where the constant says it does"
        )
        assert RK4_CUBIC_RADIUS < CUBIC_DIVERGES_FROM < RK4_STABILITY_RADIUS

    def test_an_rk4_run_capped_at_its_bound_stays_finite(self):
        """A Kerr run asked for a step far past the bound comes back finite."""
        simu, A = self.prepared(None, n2=PHASE_LIMIT_N2, Isat=np.inf, L=RK4_CUBIC_L)
        bound = simu._rk4_max_dz(A)
        peak = simu._max_interaction_rate(A)
        assert peak > 10 * simu._max_dispersion_rate(), (
            "precondition: the interaction is what limits this run"
        )
        assert RK4_STABILITY_RADIUS / simu._energy_rates(A)["interaction"] > (
            CUBIC_DIVERGES_FROM / peak
        ), "precondition: the field-weighted rate allows a diverging step"
        z = RK4_CUBIC_STEPS * CAPPED_STEP_FRACTION * bound
        with pytest.warns(UserWarning, match="RK4 stability"):
            out = simu.out_field(
                gaussian_input(), z, delta_z=FAR_PAST_LIMIT, method="RK4", verbose=False
            )
        assert np.isfinite(as_numpy(simu, out)).all(), (
            "an RK4 run capped at its own bound diverged"
        )

    @staticmethod
    def stable_coupled_run(simu):
        """Run a linear coupled solver by RK4 and return the norm's growth."""
        beam = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
        E = np.array([beam, beam]).astype(PRECISION_COMPLEX)
        with pytest.warns(UserWarning, match="RK4 stability"):
            out = simu.out_field(
                E.copy(),
                COUPLED_RUN_LENGTH,
                delta_z=COUPLED_CALLER_STEP,
                method="RK4",
                verbose=False,
                plot=False,
                normalize=False,
            )
        out = np.asarray(as_numpy(simu, out))
        return float(np.linalg.norm(out) / np.linalg.norm(E))

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_rk4_takes_the_faster_of_the_two_potentials(self, backend_name):
        """With k2 > k the second component's potential sets the RK4 step."""
        probe = make(CNLSE, backend_name, n=N)
        V = np.full((probe.NY, probe.NX), COUPLED_V, dtype=np.float32)
        simu = make(CNLSE, backend_name, n=N, V=V, alpha=0.0, n2=0.0, n12=0.0)
        simu.k2 = K2_OVER_K * simu.k
        growth = self.stable_coupled_run(simu)
        assert growth <= 1 + NORM_GROWTH_RTOL, (
            f"{backend_name}: the norm grew {growth:.3g} times, so the RK4 "
            f"step ignores the second component's potential"
        )

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_rk4_counts_the_rabi_coupling(self, backend_name):
        """The Rabi coupling is a linear rate of omega / 2 and bounds the step."""
        simu = make(
            CNLSE, backend_name, n=N, alpha=0.0, n2=0.0, n12=0.0, omega=RABI_OMEGA
        )
        fastest = simu._max_dispersion_rate() + RABI_OMEGA / 2
        assert simu._rk4_max_dz() == pytest.approx(RK4_STABILITY_RADIUS / fastest)
        growth = self.stable_coupled_run(simu)
        assert growth <= 1 + NORM_GROWTH_RTOL, (
            f"{backend_name}: the norm grew {growth:.3g} times, so the RK4 "
            f"step ignores the Rabi coupling"
        )

    def test_the_rates_are_energies_not_grid_maxima(self):
        """A potential where the field is absent barely counts in the step limit."""
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

    def test_dispersion_limits_split_step_too_but_for_stability(self):
        """Both methods are limited by the kinetic term, for opposite reasons."""
        linear, A = self.prepared(None)
        linear.n2 = 0.0
        linear._precompute_step_constants(None, np.complex64)
        rates = linear._energy_rates(A)
        assert rates["kinetic"] > 0, "precondition: dispersion is present"
        split = linear._split_step_max_dz(A)
        assert np.isfinite(split), (
            "split-step must be limited by dispersion, for stability"
        )
        peak = float(np.max(np.abs(linear._dispersion_operator())))
        assert split * peak == pytest.approx(
            SPLIT_STEP_DISPERSION_PHASE_LIMIT, rel=1e-6
        ), "the split-step bound is a phase per step at the grid maximum"
        assert rates["kinetic"] < peak, (
            "precondition: the weighted rate is below the grid maximum, so "
            "the two limits cannot agree by accident"
        )
        assert np.isfinite(linear._rk4_max_dz(A)), (
            "RK4 must still be limited by dispersion"
        )

    def test_absorption_counts_for_rk4_only(self):
        """A purely absorbing V limits RK4 but imprints no phase."""
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
    """Each step constant is defined once, by ``_step_constants``."""

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

        simu._precompute_step_constants(None, np.complex64)

        for name in names:
            assert getattr(simu, name, None) is not None, (
                f"{name} is declared in _step_constants but the precompute skips it"
            )

    def test_reading_a_constant_gives_the_same_value_either_side_of_a_run(self, simu):
        """``_constant`` is the table before a run and the attribute after it."""
        before = {k: float(np.asarray(v)) for k, v in simu._step_constants().items()}
        simu._precompute_step_constants(None, np.complex64)

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
    """DDGPE's ``g`` reaches the kernels as given, not as an optical n2."""
    h_bar = 0.654
    g = 1e-2 / h_bar
    simu = TestStepConstantTable.solvers()["DDGPE"]
    table = simu._step_constants()

    assert table["_g11"] == pytest.approx(-g), (
        "DDGPE's intra-component coupling is not its own g"
    )
    A = np.ones((2, 32, 32), dtype=PRECISION_COMPLEX)
    interaction = simu._energy_rates(A)["interaction"]
    assert interaction == pytest.approx(g, rel=RATE_RTOL), (
        f"interaction rate {interaction:.3e} is not g = {g:.3e}"
    )


class TestArraysComeBackFromTheDevice:
    """A run leaves the solver's arrays on the host, however it ends."""

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
    def test_the_propagator_comes_back_only_when_read(self, backend_name):
        """The one array a run leaves on the device, until it is read.

        Most callers never read it. Read, it is the host array the run used,
        as it always was, and it is brought back once rather than every read.
        """
        simu = self.solver(backend_name)
        if not simu._backend.is_device_backend:
            pytest.skip("a host backend never moves the propagator")
        simu.out_field(gaussian_input(), 2e-3, verbose=False, plot=False, delta_z=1e-4)
        left = simu._propagator
        assert not isinstance(left, np.ndarray), "the run brought the propagator back"
        expected = np.asarray(simu._backend.to_numpy(left))
        read = simu.propagator
        assert isinstance(read, np.ndarray), (
            f"reading the propagator gave a {type(read).__name__}"
        )
        np.testing.assert_array_equal(read, expected)
        assert simu.propagator is read, "a second read brought it back again"

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
