"""Tests for choosing the step from a measured error."""

import numpy as np
import pytest
from helpers import make
from NLSE import NLSE
from NLSE.callbacks import adapt_delta_z_to_error

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST
STEPS_IN_L = 4.25  # four whole steps and a quarter of one, cut at L
KEEP = (1.0, 1.0)  # bounds under which the controller returns the step it scaled


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
        splitting="strang",
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
    simu._precompute_step_constants(simu.V, np.complex64)
    simu.plans = simu._build_fft_plan(prepared)
    simu.propagator = simu._build_propagator(np.complex64, L / 100)
    return simu, prepared


def test_the_step_never_passes_the_solvers_own_ceiling():
    """The adapted step never passes the solver's own ceiling."""
    simu, prepared = prepared_solver()
    ceiling = simu._split_step_max_dz(prepared)
    simu._step_in_force = ceiling
    proposed = adapt_delta_z_to_error(simu, prepared, 0.0, 0, 1e-3, 1)
    assert proposed is not None
    assert proposed <= ceiling * 1.001, (
        f"asked for {proposed:.3e} from a step already at the {ceiling:.3e} "
        f"ceiling, so nothing bounds a run that keeps growing"
    )


def test_an_unreachable_tolerance_stops_shrinking():
    """With an unreachable tolerance the step settles at its floor."""
    simu, prepared = prepared_solver()
    floor = L / 1e4
    step = L / 100
    for _ in range(80):
        simu._step_in_force = step
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
    simu._precompute_step_constants(simu.V, np.complex64)
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
            splitting="strang",
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
    """A trial step leaves the run untouched."""
    from NLSE.callbacks import _trial_propagation

    simu = solver()
    field = beam()
    simu.out_field(
        field.copy(),
        L / 50,
        verbose=False,
        plot=False,
        splitting="strang",
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


def test_a_run_whose_step_grows_still_lands_on_z():
    """The distance asked for is the distance propagated."""
    simu = solver()
    reference = np.asarray(
        solver().out_field(
            beam(np.complex128),
            L,
            delta_z=L / 4000,
            verbose=False,
            plot=False,
            splitting="strang",
        )
    ).astype(np.complex128)

    taken: list = []
    out = simu.out_field(
        beam(),
        L,
        verbose=False,
        plot=False,
        splitting="strang",
        callback=adapt_delta_z_to_error,
        callback_args=(1e-3, 10, (0.5, 2.0), 0.9, None, taken),
    )
    got = np.asarray(simu._backend.to_numpy(out)).astype(np.complex128)
    error = np.linalg.norm(got - reference) / np.linalg.norm(reference)
    assert error < 1e-3, (
        f"adaptive run is {error:.3e} from the reference, which is the size of "
        f"an overshoot rather than of a step-size difference"
    )


def keep_step_and_record(simu, A, z, i, taken):
    """Record the step just taken, then run a controller that keeps its step."""
    taken.append(simu._current_delta_z)
    return adapt_delta_z_to_error(simu, A, z, i, 1e-3, 1, KEEP)


def test_a_cut_step_is_not_what_the_controller_scales():
    """After the step cut at L, the run goes on at the step in force."""
    simu = solver()
    taken: list = []
    simu.out_field(
        beam(),
        2 * L,
        delta_z=L / STEPS_IN_L,
        verbose=False,
        plot=False,
        splitting="strang",
        callback=keep_step_and_record,
        callback_args=(taken,),
    )
    whole = int(STEPS_IN_L)
    assert taken[whole] == pytest.approx(L - whole * taken[0]), (
        f"step {whole} was {taken[whole]:.3e} m, not the one cut at L"
    )
    assert taken[whole + 1] == pytest.approx(taken[0]), (
        f"the step after the cut was {taken[whole + 1]:.3e} m, where the step "
        f"in force is {taken[0]:.3e} m"
    )
