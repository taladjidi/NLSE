"""Tests for choosing the step from a measured error.

``adapt_delta_z`` reads the step off the peak nonlinear index and divides by
twelve, which is a rate rather than an error. ``adapt_delta_z_to_error``
measures instead: it takes the same distance whole and in halves and compares.

The tests worth having here are about the two ways that goes wrong, both of
which were found by running it:

- **The estimate has a floor.** Below roughly 0.8 rad per step the difference
  between one step and two halves is complex64 round-off, not splitting error.
  A tolerance above that floor therefore reads as "no error at all" and asks
  for a bigger step however big the step is, so the controller has to be
  capped by the physics or it doubles until the answer is unrecognisable.
- **A tolerance below the floor cannot be met at any step**, so the controller
  has to stop shrinking or the run never ends. It is not a divergence and
  nothing raises; the propagation simply does not return.
"""

import numpy as np
import pytest
from helpers import make
from NLSE import NLSE
from NLSE.callbacks import adapt_delta_z_to_error

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST


def beam(dtype=np.complex64):
    """Return a Gaussian beam strong enough to bend the step."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / WAIST**2).astype(dtype)


def solver():
    """Return a lossless, strongly nonlinear solver."""
    return make(NLSE, "CPU", n=N, alpha=0, power=4.0, window=WINDOW, L=L, Isat=1e6)


def adaptive(tolerance, update_every=10, min_step=None):
    """Propagate with the error-driven step; return field and steps taken."""
    simu = solver()
    taken: list = []
    out = simu.out_field(
        beam(),
        L,
        verbose=False,
        plot=False,
        precision="double",
        method="split_step",
        callback=adapt_delta_z_to_error,
        callback_args=(tolerance, update_every, (0.5, 2.0), 0.9, min_step, taken),
    )
    return np.asarray(simu._backend.to_numpy(out)), [t for t in taken if t]


def test_a_tighter_tolerance_does_not_take_a_larger_step():
    """The knob has to move the step in the direction it names."""
    _, loose = adaptive(1e-3)
    _, tight = adaptive(1e-6)
    assert max(tight) <= max(loose), (
        f"tightening the tolerance grew the step, from {max(loose):.3e} to "
        f"{max(tight):.3e}"
    )


def prepared_solver():
    """Return a solver with its constants and plans ready for one callback."""
    simu = solver()
    field = beam()
    prepared, _ = simu._prepare_output_array(field.copy(), normalize=True)
    simu._precompute_step_constants(simu.V, "double")
    simu.plans = simu._build_fft_plan(prepared)
    simu.propagator = simu._build_propagator(np.complex64, L / 100)
    return simu, prepared


def test_the_step_never_passes_the_solvers_own_ceiling():
    """The estimate cannot see past its floor, so the physics has to cap it.

    Driven directly and from a step already at the ceiling, because a
    propagation only reaches the cap if it runs long enough to double its way
    there -- a run that never gets close leaves the cap untested, which is how
    removing it first went unnoticed.

    Without the cap the controller reads round-off as "no error", doubles
    every time it fires, and three adjustments in the answer is worthless:
    0.38 relative against a converged reference, where the fixed default
    gives 8e-5.
    """
    simu, prepared = prepared_solver()
    ceiling = simu._split_step_max_dz(prepared)
    simu._current_delta_z = ceiling
    proposed = adapt_delta_z_to_error(simu, prepared, 0.0, 0, 1e-3, 1)
    assert proposed is not None
    assert proposed <= ceiling * 1.001, (
        f"asked for {proposed:.3e} from a step already at the {ceiling:.3e} "
        f"ceiling, so nothing bounds a run that keeps growing"
    )


def test_an_unreachable_tolerance_stops_shrinking():
    """It must settle at the floor rather than halve for ever.

    Driven directly: the failure this guards against is a propagation that
    never returns, and a test that can only time out is not a test.
    """
    simu, prepared = prepared_solver()
    floor = L / 1e4
    step = L / 100
    for _ in range(80):
        simu._current_delta_z = step
        proposed = adapt_delta_z_to_error(
            simu, prepared, 0.0, 0, 1e-14, 1, (0.5, 2.0), 0.9, floor
        )
        assert proposed is not None
        step = proposed
        assert step >= floor * 0.999, (
            f"the step reached {step:.3e}, below the {floor:.3e} floor"
        )
    assert step == pytest.approx(floor, rel=1e-3), (
        f"an unmeetable tolerance should settle at the floor, not {step:.3e}"
    )


def test_it_beats_the_fixed_default_it_replaces():
    """Fewer steps and no worse an answer, which is the whole point."""
    simu = solver()
    field = beam()
    prepared, _ = simu._prepare_output_array(field.copy(), normalize=True)
    simu._precompute_step_constants(simu.V, "double")
    rates = simu._energy_rates(prepared)
    rate = rates["potential"] + rates["interaction"]

    def fixed(phase, dtype=np.complex64):
        s = solver()
        out = s.out_field(
            beam(dtype),
            L,
            delta_z=phase / rate,
            verbose=False,
            plot=False,
            precision="double",
            method="split_step",
        )
        return np.asarray(s._backend.to_numpy(out)).astype(np.complex128)

    reference = fixed(1e-3, np.complex128)

    def distance(got):
        return float(
            np.linalg.norm(got.astype(np.complex128) - reference)
            / np.linalg.norm(reference)
        )

    default_steps = int(L / (0.1 / rate))
    adapted, steps = adaptive(1e-3)
    assert len(steps) < default_steps, (
        f"took {len(steps)} steps against the fixed default's {default_steps}"
    )
    assert distance(adapted) < 10 * distance(fixed(0.1)), (
        f"adaptive error {distance(adapted):.3e} against the fixed default's "
        f"{distance(fixed(0.1)):.3e}"
    )


def test_a_trial_leaves_the_run_untouched():
    """Measuring must not perturb what it measures.

    The trial propagates at another step, which needs a propagator of its
    own; the run's has to come back whichever way the trial goes.
    """
    from NLSE.callbacks import _trial_propagation

    simu = solver()
    field = beam()
    simu.out_field(
        field.copy(),
        L / 50,
        verbose=False,
        plot=False,
        precision="double",
        method="split_step",
    )
    live = simu._backend.from_numpy(beam())
    before = np.asarray(simu._backend.to_numpy(live)).copy()
    propagator = simu.propagator
    _trial_propagation(simu, live, float(simu._current_delta_z) / 2, 2)
    assert simu.propagator is propagator, "the trial kept its own propagator"
    assert np.array_equal(before, np.asarray(simu._backend.to_numpy(live))), (
        "the trial propagated the live field"
    )


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3])
def test_the_run_stays_finite(tolerance):
    """Whatever the controller does, the field must survive it."""
    got, steps = adaptive(tolerance)
    assert np.all(np.isfinite(got.view(np.float32)))
    assert steps, "no steps were taken"
