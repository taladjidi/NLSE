"""Tests for solver state that must stay consistent across out_field calls.

The solver caches a lot onto ``self`` (propagator, precomputed step constants,
device copies of arrays). These tests pin down the cases where that cached
state has to be refreshed rather than silently reused.
"""

import numpy as np
import pytest
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64

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
