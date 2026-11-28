"""The built-in callbacks work on every backend, not only the CPU one."""

import functools
import warnings
from unittest.mock import Mock

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import (
    GPE,
    NLSE,
    adapt_delta_z,
    adapt_delta_z_to_error,
    callbacks,
    evaluate_delta_n,
    norm,
    sample,
)
from NLSE.solvers.step_size import LOSS_PER_STEP_LIMIT
from scipy.constants import c, epsilon_0

N = 32
WAIST = 2.23e-3
L = 1e-3
DELTA_Z = 1e-4
SAVE_EVERY = 2
SLOTS = round(L / DELTA_Z) // SAVE_EVERY + 2

BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 4 * WAIST,
    "n2": -1.6e-9,
    "V": None,
    "L": L,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}

# The uniform field the closed-form checks below use, in W/m^2: three
# saturation intensities, where evaluate_delta_n once returned 0.0018 of the
# true index.
INTENSITY = 3 * BASE["Isat"]
# adapt_delta_z steps by this fraction of the nonlinear length 1/(k dn), as
# its docstring and adapt_delta_z_to_error's say.
NONLINEAR_LENGTH_FRACTION = 1 / 12
# adapt_delta_z_to_error: Strang splitting is second order, so its local error
# goes as step**3. The trial error is the tolerance over 2**3, so the rule asks
# for twice the step, times the safety factor: 1.8, inside the bounds.
STRANG_ORDER = 2
TOLERANCE = 1e-2
TRIAL_ERROR = TOLERANCE / 2**3
SAFETY = 0.9
BOUNDS = (0.5, 2.0)
# The step handed in is this many times below the solver's own ceiling, and
# the floor this many times below the step, so neither binds.
CEILING_MARGIN = 4
# Relative tolerances of the three closed-form checks below. Measured worst
# cases over CPU and MLX: 2.5e-8, 0 (exact) and 1.3e-5.
DELTA_N_RTOL = 1e-5
STEP_RTOL = 1e-5
RULE_RTOL = 1e-4
# A loss in 1/m whose step ceiling, LOSS_PER_STEP_LIMIT / (2 alpha) = 2.5e-5 m,
# is below the twelfth of the nonlinear length at INTENSITY, 8.6e-5 m.
CAPPING_ALPHA = 1e3
# Relative tolerance of norm against N**2 |A|^2: float32 sum of N**2 equal
# terms, measured 9.3e-8 on CPU and MLX. Dropping a quadrature misses by 0.2.
NORM_RTOL = 1e-6
# Unequal real and imaginary parts: a callback that keeps one of them, swaps
# them or conjugates the field gives another value.
QUADRATURES = 1 + 2j
# adapt_delta_z_to_error's default floor, as a fraction of L (callbacks.py
# docstring: "a hundred thousandth of the medium length").
DEFAULT_FLOOR_FRACTION = 1e-5
# A condensate for the solver without a medium length: atoms, window in m,
# interaction in J m^2, run in s. Any that leaves the floor under the cap.
GPE_ATOMS = 1e4
GPE_WINDOW = 1e-4
GPE_G = 1e-16
GPE_RUN = 1e-2
GPE_STEPS = 4
# The step in force, in units of that floor: its BOUNDS[0] shrink lands below
# the floor, so the floor is what comes back.
ABOVE_FLOOR = 1.5
# Relative error a trial reports to shrink by BOUNDS[0]: far above TOLERANCE.
LARGE_ERROR = 1.0
# adapt_delta_z_to_error measures on the steps i with i % CADENCE == 0: of the
# first CADENCE_STEPS, these. Each measure is two trials, the whole step and its
# halves (callbacks.py). The controller does not read z, in m.
CADENCE = 3
CADENCE_STEPS = 7
MEASURED_STEPS = [0, 3, 6]
TRIALS_PER_MEASURE = 2
ANY_Z = 0.0


def _buffers():
    """Return one output buffer per callback, keyed by name."""
    return {
        "sample": np.zeros((SLOTS, N, N), dtype=np.complex64),
        "norm": np.zeros(SLOTS),
        "evaluate_delta_n": np.zeros((SLOTS, N, N), dtype=np.float32),
    }


CALLBACKS = {"sample": sample, "norm": norm, "evaluate_delta_n": evaluate_delta_n}


def uniform_field(intensity):
    """Return a uniform field of the given intensity on the module's grid.

    Parameters
    ----------
    intensity : float
        Intensity ``c eps0 |A|^2 / 2`` in W/m^2.

    Returns
    -------
    np.ndarray
        The field in V/m, complex64.
    """
    amplitude = np.sqrt(2 * intensity / (c * epsilon_0))
    return np.full((N, N), amplitude, dtype=np.complex64)


def _run(backend_name, which):
    """Propagate with one built-in callback and return what it recorded.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    which : str
        Which built-in callback to pass.

    Returns
    -------
    np.ndarray
        The buffer the callback filled.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    buffer = _buffers()[which]
    simu.out_field(
        field,
        L,
        verbose=False,
        plot=False,
        delta_z=DELTA_Z,
        callback=CALLBACKS[which],
        callback_args=(SAVE_EVERY, buffer),
    )
    return buffer


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("which", sorted(CALLBACKS))
def test_a_builtin_callback_runs_on_this_backend(backend_name, which):
    """The callback must survive whatever array type the backend hands it.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    which : str
        Which built-in callback to pass.
    """
    recorded = _run(backend_name, which)
    assert np.any(recorded != 0), (
        f"{which} recorded nothing on {backend_name}; it ran but wrote no "
        f"samples, which a buffer of the wrong shape would also do"
    )
    assert np.all(np.isfinite(np.abs(recorded)))


@pytest.mark.parametrize("backend_name", [b for b in AVAILABLE_BACKENDS if b != "CPU"])
@pytest.mark.parametrize("which", sorted(CALLBACKS))
def test_a_builtin_callback_records_what_the_cpu_records(backend_name, which):
    """A built-in callback records on this backend what it records on the CPU.

    Parameters
    ----------
    backend_name : str
        Backend to compare against the CPU.
    which : str
        Which built-in callback to pass.
    """
    np.testing.assert_allclose(
        np.abs(_run(backend_name, which)),
        np.abs(_run("CPU", which)),
        rtol=2e-2,
        atol=1e-6,
        err_msg=f"{which} disagrees with the CPU on {backend_name}",
    )


ADAPTIVE = {
    "adapt_delta_z": (adapt_delta_z, (5, [])),
    "adapt_delta_z_to_error": (adapt_delta_z_to_error, ()),
}


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("which", sorted(ADAPTIVE))
def test_an_adaptive_callback_drives_a_run_on_this_backend(which, backend_name):
    """The callbacks that choose the step must also survive a device array.

    Parameters
    ----------
    which : str
        Which adaptive callback to drive the run with.
    backend_name : str
        Backend to run on.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    callback, args = ADAPTIVE[which]
    with warnings.catch_warnings():
        # The step limiter says so when it lowers the step; that is not what
        # is under test here.
        warnings.simplefilter("ignore")
        out = simu.out_field(
            field,
            L,
            verbose=False,
            plot=False,
            delta_z=DELTA_Z,
            callback=callback,
            callback_args=args,
        )
    recorded = np.asarray(
        out if isinstance(out, np.ndarray) else simu._backend.to_numpy(out)
    )
    assert np.all(np.isfinite(np.abs(recorded))), (
        f"{which} drove {backend_name} to a non-finite field"
    )


def test_sample_writes_each_step_to_its_own_slot():
    """``sample`` writes each step to its own slot."""
    simu = NLSE(backend="CPU", **{**BASE, "alpha": 20.0})
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    buffer = np.zeros((SLOTS, N, N), dtype=np.complex64)
    simu.out_field(
        field.copy(),
        L,
        verbose=False,
        plot=False,
        delta_z=DELTA_Z,
        callback=sample,
        callback_args=(SAVE_EVERY, buffer),
    )

    # The buffer holds a couple of slots more than the run fills, so the
    # samples must occupy a prefix of it. A shifted destination wraps the
    # first sample onto the end and leaves a hole at the front.
    filled = [k for k in range(SLOTS) if np.any(buffer[k])]
    assert filled == list(range(len(filled))), (
        f"slots {filled} were written out of a buffer of {SLOTS}, which is not "
        f"a prefix: the samples do not start where the run does"
    )
    # Order, not values: every slot is written whatever the arithmetic does.
    peaks = [float(np.max(np.abs(buffer[k]))) for k in filled]
    assert all(np.diff(peaks) < 0), (
        f"the sampled peaks are {peaks}, which is not the monotonic decay a "
        f"lossy run makes -- the samples are out of order or off by a slot"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("isat", [BASE["Isat"], np.inf], ids=["I=3Isat", "Isat=inf"])
def test_evaluate_delta_n_is_the_saturated_kerr_index(backend_name, isat):
    """``evaluate_delta_n`` gives ``n2 I / (1 + I/Isat)`` on a uniform field.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    isat : float
        Saturation intensity in W/m^2.
    """
    simu = NLSE(backend=backend_name, **{**BASE, "Isat": isat})
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    recorded = np.zeros((1, N, N), dtype=np.float32)
    evaluate_delta_n(simu, field, 0.0, 0, 1, recorded)
    expected = BASE["n2"] * INTENSITY / (1 + INTENSITY / isat)
    np.testing.assert_allclose(
        recorded,
        expected,
        rtol=DELTA_N_RTOL,
        err_msg=f"evaluate_delta_n is not the saturated Kerr index on {backend_name}",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_adapt_delta_z_steps_by_a_twelfth_of_the_nonlinear_length(backend_name):
    """``1 / (12 k |dn|)``, with ``dn`` the saturated Kerr index of the peak.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    """
    simu = NLSE(backend=backend_name, **BASE)
    simu._current_delta_z = DELTA_Z
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    in_force: list = []
    got = adapt_delta_z(simu, field, 0.0, 0, 1, in_force)
    delta_n = abs(BASE["n2"]) * INTENSITY / (1 + INTENSITY / BASE["Isat"])
    expected = NONLINEAR_LENGTH_FRACTION / (simu.k * delta_n)
    assert got == pytest.approx(expected, rel=STEP_RTOL), (
        f"adapt_delta_z asked for {got:.4e} m on {backend_name}, against "
        f"{expected:.4e} m, a twelfth of the nonlinear length"
    )
    assert in_force == [DELTA_Z]


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_adapt_delta_z_stops_at_the_split_step_limit(backend_name):
    """Where the loss ceiling is the tighter, it is the step asked for.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    """
    simu = NLSE(backend=backend_name, **{**BASE, "alpha": CAPPING_ALPHA})
    simu._current_delta_z = DELTA_Z
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    got = adapt_delta_z(simu, field, 0.0, 0, 1, [])
    expected = LOSS_PER_STEP_LIMIT / (2 * CAPPING_ALPHA)
    assert got == pytest.approx(expected, rel=STEP_RTOL), (
        f"adapt_delta_z asked for {got:.4e} m on {backend_name}, past the "
        f"{expected:.4e} m the lossy step is solved at"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_adapt_delta_z_keeps_the_step_past_the_medium(backend_name):
    """Past L nothing turns the phase, so the step in force stands.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    """
    simu = NLSE(backend=backend_name, **BASE)
    in_force: list = []
    out = simu.out_field(
        uniform_field(INTENSITY),
        2 * L,
        delta_z=DELTA_Z,
        verbose=False,
        normalize=False,
        callback=adapt_delta_z,
        callback_args=(1, in_force),
    )
    assert np.all(np.isfinite(in_force)), f"a step was not finite on {backend_name}"
    assert np.all(np.isfinite(np.asarray(simu._backend.to_numpy(out))))
    # The run ends on a cut step; the whole step before it is past the medium
    # and still the one the medium's last adaptation asked for.
    assert in_force[-2] == pytest.approx(in_force[1], rel=STEP_RTOL)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_adapt_delta_z_to_error_follows_the_step_rule(backend_name, monkeypatch):
    """``adapt_delta_z_to_error`` returns ``h safety (tol / error) ** (1 / 3)``.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    monkeypatch : pytest.MonkeyPatch
        Replaces the trial propagation.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    step = simu._split_step_max_dz(field) / CEILING_MARGIN
    simu._step_in_force = step
    monkeypatch.setattr(
        callbacks,
        "_trial_propagation",
        lambda simu, A, step, count: A * (1 + TRIAL_ERROR) if count == 1 else A,
    )
    got = adapt_delta_z_to_error(
        simu, field, 0.0, 0, TOLERANCE, 1, BOUNDS, SAFETY, step / CEILING_MARGIN
    )
    expected = step * SAFETY * (TOLERANCE / TRIAL_ERROR) ** (1 / (STRANG_ORDER + 1))
    assert got == pytest.approx(expected, rel=RULE_RTOL), (
        f"adapt_delta_z_to_error asked for {got / step:.4f} times the step on "
        f"{backend_name}, against the {expected / step:.4f} its rule gives"
    )


def scaled_trial(simu, A, step, count, error):
    """Return a whole step ``error`` away from its halves, as the trial does.

    Parameters
    ----------
    simu : NLSE
        Solver of the run; unused.
    A : Any
        Field the trial starts from, on the solver's backend.
    step : float
        Trial step in m; unused.
    count : int
        Number of steps: 1 for the whole step, 2 for its halves.
    error : float
        Relative error of the whole step against the halves.

    Returns
    -------
    Any
        ``A * (1 + error)`` for the whole step, ``A`` for the halves.
    """
    return A * (1 + error) if count == 1 else A


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_sample_records_the_field_as_it_is(backend_name):
    """The slot holds the field passed in, value for value."""
    simu = NLSE(backend=backend_name, **BASE)
    ramp = np.arange(N * N).reshape(N, N)
    field = (QUADRATURES * ramp).astype(np.complex64)
    recorded = np.zeros((1, N, N), dtype=np.complex64)
    sample(simu, simu._backend.from_numpy(field), 0.0, 0, 1, recorded)
    np.testing.assert_array_equal(recorded[0], field)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_norm_sums_both_quadratures(backend_name):
    """``sum |A|^2`` of a uniform field with both quadratures is ``N**2 |A|^2``."""
    simu = NLSE(backend=backend_name, **BASE)
    field = (QUADRATURES * uniform_field(INTENSITY)).astype(np.complex64)
    recorded = np.zeros(1)
    norm(simu, simu._backend.from_numpy(field), 0.0, 0, 1, recorded)
    expected = N * N * float(np.abs(field[0, 0])) ** 2
    assert recorded[0] == pytest.approx(expected, rel=NORM_RTOL)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_adapt_delta_z_waits_for_its_cadence(backend_name):
    """Between updates it asks for nothing, so the step in force is kept."""
    simu = NLSE(backend=backend_name, **BASE)
    simu._current_delta_z = DELTA_Z
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    assert adapt_delta_z(simu, field, 0.0, 1, 2, []) is None


def test_adapt_delta_z_to_error_measures_on_its_cadence_only(backend, monkeypatch):
    """Only every ``update_every``-th step runs the trials and returns a step.

    Parameters
    ----------
    backend : str
        Backend to run on.
    monkeypatch : pytest.MonkeyPatch
        Replaces the trial propagation by a counted one.
    """
    simu = NLSE(backend=backend, **BASE)
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    simu._step_in_force = simu._split_step_max_dz(field) / CEILING_MARGIN
    trial = Mock(wraps=functools.partial(scaled_trial, error=TRIAL_ERROR))
    monkeypatch.setattr(callbacks, "_trial_propagation", trial)
    measured = [
        i
        for i in range(CADENCE_STEPS)
        if adapt_delta_z_to_error(simu, field, ANY_Z, i, TOLERANCE, CADENCE) is not None
    ]
    assert measured == MEASURED_STEPS
    assert trial.call_count == TRIALS_PER_MEASURE * len(MEASURED_STEPS)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_zero_error_grows_the_step_no_further_than_the_cap(backend_name, monkeypatch):
    """At the split-step limit already, a step with no error stays there.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    monkeypatch : pytest.MonkeyPatch
        Replaces the trial propagation by one with no error.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    cap = simu._split_step_max_dz(field)
    simu._step_in_force = cap
    monkeypatch.setattr(
        callbacks,
        "_trial_propagation",
        functools.partial(scaled_trial, error=0.0),
    )
    got = adapt_delta_z_to_error(simu, field, 0.0, 0, TOLERANCE, 1, BOUNDS, SAFETY)
    assert got == pytest.approx(cap, rel=STEP_RTOL), (
        f"a zero error at the cap asked for {got / cap:.3f} times the cap"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_default_floor_is_a_hundred_thousandth_of_l(backend_name, monkeypatch):
    """A large error shrinks the step to ``L / 1e5`` and not below.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    monkeypatch : pytest.MonkeyPatch
        Replaces the trial propagation by one with a large error.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = simu._backend.from_numpy(uniform_field(INTENSITY))
    floor = DEFAULT_FLOOR_FRACTION * L
    simu._step_in_force = ABOVE_FLOOR * floor
    monkeypatch.setattr(
        callbacks,
        "_trial_propagation",
        functools.partial(scaled_trial, error=LARGE_ERROR),
    )
    got = adapt_delta_z_to_error(simu, field, 0.0, 0, TOLERANCE, 1, BOUNDS, SAFETY)
    assert got == pytest.approx(floor, rel=STEP_RTOL), (
        f"the default floor is {got / L:.1e} L, not {DEFAULT_FLOOR_FRACTION:.0e} L"
    )


def test_a_solver_without_a_medium_takes_its_floor_from_the_run(monkeypatch):
    """GPE has ``L = 0``: the floor is a hundred thousandth of the run instead.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the trial propagation by one with a large error.
    """
    simu = GPE(0.0, GPE_ATOMS, GPE_WINDOW, GPE_G, None, NX=N, NY=N, backend="CPU")
    field = np.ones((N, N), dtype=np.complex64)
    simu.out_field(
        field,
        GPE_RUN,
        delta_z=GPE_RUN / GPE_STEPS,
        verbose=False,
        callback=lambda *_: None,
    )
    floor = DEFAULT_FLOOR_FRACTION * GPE_RUN
    simu._step_in_force = ABOVE_FLOOR * floor
    monkeypatch.setattr(
        callbacks,
        "_trial_propagation",
        functools.partial(scaled_trial, error=LARGE_ERROR),
    )
    got = adapt_delta_z_to_error(simu, field, 0.0, 0, TOLERANCE, 1, BOUNDS, SAFETY)
    assert simu.L == 0
    assert got == pytest.approx(floor, rel=STEP_RTOL), (
        f"the floor without a medium is {got / GPE_RUN:.1e} of the run, not "
        f"{DEFAULT_FLOOR_FRACTION:.0e}"
    )
