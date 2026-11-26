"""What drives a DDGPE run: its noise, its lasers, and the callbacks it takes.

The noise is a Wiener increment, so the intensity it adds per unit time is a
constant of the physics, ``noise**2 * gamma / (2 dX dY)`` per component, and
must not depend on the step. A temporal drive profile holds one sample per
asked step, ``F[j]`` acting during ``[j dt, (j + 1) dt)``, and each step takes
the sample at its midpoint: with no coupling and no dispersion, the cavity
field is exactly ``-i F0`` times the integral of that piecewise-constant drive.
"""

import warnings

import numpy as np
import pytest
from NLSE import DDGPE
from NLSE.solvers.step_size import DEFAULT_MIN_STEPS

N = 256
WINDOW = 256.0  # one unit per pixel, so dX dY = 1
GAMMA = 0.1  # exciton loss rate, in inverse units of time
GAMMA2 = 0.4  # cavity loss rate, distinct so each component's own is checked
NOISE = 3.0  # multiple of sqrt(gamma / 4 dX dY)
STEPS = (2e-3, 1e-3, 5e-4)
SEED = 0
# 5 sigma of the mean of N**2 exponential draws, each with std equal to its mean.
NOISE_RTOL = 5 / N
T = 1.0
F0 = 1.0  # pump amplitude, in units of the field per unit time
PUMP_DT = 0.05
SAMPLES = round(T / PUMP_DT)  # one per asked step over T
# On for the first 0.4 T, then off, so the run's last step is undriven.
PROFILE = np.r_[np.ones(8), np.zeros(SAMPLES - 8)]
PUMP_EXACT = -1j * F0 * 0.4 * T  # -i F0 times the profile's integral
# Reviewer's case: half of T at a fine step, the drive F(t) = t.
HALF = 0.5
FINE_DT = 0.01
FINE_EXACT = -1j * F0 * FINE_DT**2 * sum(range(50))  # -i dt sum_j j dt, j < 50
# Cavity energy putting the dispersion cap below PUMP_DT, in inverse time units.
CAPPED_W = 10.0
SMALL_N = 32
# Exact up to float32 sums of order-one terms over at most a few hundred steps.
PUMP_ATOL = 1e-6
# Runs that are not a whole number of CUT_DT steps, with the integral of
# CUT_PROFILE over each: two steps and a tenth, then less than one step.
CUT_DT = 0.1
CUT_PROFILE = np.array([1.0, 2.0, 4.0])
CUT_RUNS = [(0.21, 0.1 * 1.0 + 0.1 * 2.0 + 0.01 * 4.0), (0.03, 0.03 * 1.0)]
# A callback multiplies the default step by this once, after the first step:
# DEFAULT_MIN_STEPS - 1 = 9 is then three whole steps of 3.
STEP_FACTOR = 3


def quiet(simu):
    """Return an undriven noise solver, its generator seeded."""
    simu._random = np.random.default_rng(SEED).normal
    return simu


def noise_solver(backend):
    """Return a DDGPE whose two components have different loss rates."""
    simu = DDGPE(
        gamma=GAMMA,
        power=1,
        window=WINDOW,
        g=0,
        omega=0,
        T=T,
        omega_exc=0,
        omega_cav=0,
        detuning=0,
        k_z=1.0,
        NX=N,
        NY=N,
        backend=backend,
    )
    simu.gamma2 = GAMMA2
    return quiet(simu)


@pytest.mark.parametrize("dt", STEPS)
def test_noise_intensity_per_unit_time_is_step_independent(backend, dt):
    """One call adds noise**2 gamma / (2 dX dY) of intensity per unit time."""
    simu = noise_solver(backend)
    simu._current_delta_z = dt
    A = simu._backend.from_numpy(np.zeros((2, N, N), np.complex64))
    DDGPE.add_noise(simu, A, dt, 0, NOISE)
    out = np.asarray(simu._backend.to_numpy(A))
    area = simu.delta_X * simu.delta_Y
    for component, gamma in ((0, GAMMA), (1, GAMMA2)):
        rate = np.mean(np.abs(out[component]) ** 2) / dt
        expected = NOISE**2 * gamma / (2 * area)
        assert rate == pytest.approx(expected, rel=NOISE_RTOL), (
            f"component {component} gains {rate:.4g} per unit time at dt={dt}"
        )


def pump_solver(backend, omega_cav=0.0, n=SMALL_N):
    """Return a DDGPE with no coupling, loss or interaction, cavity uniform."""
    return DDGPE(
        gamma=0,
        power=1,
        window=n,
        g=0,
        omega=0,
        T=T,
        omega_exc=omega_cav,
        omega_cav=omega_cav,
        detuning=0,
        k_z=1.0,
        NX=n,
        NY=n,
        backend=backend,
    )


def pump_args(simu, profile):
    """Return laser_excitation arguments: a uniform pump, no probe."""
    shape = (simu.NY, simu.NX)
    pump = np.full(shape, F0, np.complex64)
    return [(pump, profile, np.zeros(shape, np.complex64), 0 * profile)]


def cavity(simu, out):
    """Return the mean cavity field on the host."""
    return complex(np.mean(np.asarray(simu._backend.to_numpy(out))[1]))


def record_step(simu, A, t, i, seen):
    """Append the step just taken to seen."""
    seen.append(simu._current_delta_z)


def run_pump(simu, t, profile, delta_z, callback=None, extra_args=()):
    """Run from an empty field with the pump on, and return the mean cavity."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = simu.out_field(
            np.zeros((2, simu.NY, simu.NX), np.complex64),
            t,
            None,
            delta_z=delta_z,
            verbose=False,
            callback=callback,
            callback_args=[*pump_args(simu, profile), *extra_args],
        )
    return cavity(simu, out)


def test_run_shorter_than_T_takes_one_sample_per_step(backend):
    """Half of T at dt = 0.01 against F(t) = t gives -i dt sum_j j dt."""
    simu = pump_solver(backend)
    profile = np.arange(0, HALF + FINE_DT, FINE_DT)
    got = run_pump(simu, HALF, profile, FINE_DT)
    assert got == pytest.approx(FINE_EXACT, abs=PUMP_ATOL)


def test_capped_step_samples_by_time(backend):
    """A step the solver shortens takes the sample in force at its midpoint."""
    simu = pump_solver(backend, omega_cav=CAPPED_W)
    seen = []
    got = run_pump(simu, T, PROFILE, PUMP_DT, [record_step], [(seen,)])
    step = abs(seen[0])
    assert step < PUMP_DT, "the step was not capped"
    # Only the step across the switch at 0.4 T can miss, by half of itself.
    assert got == pytest.approx(PUMP_EXACT, abs=F0 * step / 2 + PUMP_ATOL)


@pytest.mark.parametrize("t, integral", CUT_RUNS)
def test_cut_last_step_is_driven_for_its_own_length(backend, t, integral):
    """A last step cut to the end of the run takes its own sample and length."""
    got = run_pump(pump_solver(backend), t, CUT_PROFILE, CUT_DT)
    assert got == pytest.approx(-1j * F0 * integral, abs=PUMP_ATOL)


def test_short_profile_raises(backend):
    """A profile that ends before the run does is an error, not a clamp."""
    simu = pump_solver(backend)
    with pytest.raises(ValueError, match=f"has {SAMPLES - 1} samples"):
        run_pump(simu, T, PROFILE[:-1], PUMP_DT)


def step_up(simu, A, t, i, factor):
    """Ask for a step factor times longer after the first one."""
    return factor * simu._current_delta_z if i == 0 else None


def test_default_step_is_the_drive_step(backend):
    """With no step asked, the profile is sampled by the solver's default.

    The default is T / DEFAULT_MIN_STEPS here (no potential, no interaction),
    and a callback triples it after the first step: the remaining steps span
    [1, 4), [4, 7) and [7, 10) defaults, so with F[j] = j they take samples
    2, 5 and 8.
    """
    simu = pump_solver(backend)
    default = T / DEFAULT_MIN_STEPS
    profile = np.arange(DEFAULT_MIN_STEPS, dtype=float)
    got = run_pump(simu, T, profile, None, [step_up], [(STEP_FACTOR,)])
    exact = -1j * F0 * STEP_FACTOR * default * (2 + 5 + 8)
    assert got == pytest.approx(exact, abs=PUMP_ATOL)


def test_callback_list_survives_two_runs(backend):
    """The caller's lists come back unchanged, and a second run gives the same."""
    simu = pump_solver(backend)
    seen = []
    callbacks = [record_step]
    args = [*pump_args(simu, PROFILE), (seen,)]
    results = []
    for _ in range(2):
        out = simu.out_field(
            np.zeros((2, simu.NY, simu.NX), np.complex64),
            T,
            None,
            delta_z=PUMP_DT,
            verbose=False,
            callback=callbacks,
            callback_args=args,
        )
        results.append(cavity(simu, out))
    assert len(callbacks) == 1 and len(args) == 2
    assert results[0] == results[1] == pytest.approx(PUMP_EXACT, abs=PUMP_ATOL)
    assert len(seen) == 2 * SAMPLES
