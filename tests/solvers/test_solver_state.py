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
