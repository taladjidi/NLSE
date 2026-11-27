"""Where the nonlinear step may run inside the inverse transform, and the rest.

CUPY runs the nonlinear step as cuFFT stores the inverse transform's output.
Which runs ask for it is decided in the solver, from a backend flag, so it is
checked here on CPU with the flag set by hand; the callback itself is checked
on a CUDA machine, in tests/backends/test_fft_propagator_fusion.py.
"""

from functools import partial

import numpy as np
import pytest
from helpers import WAIST, make
from NLSE import GPE, NLSE, NLSE_1d
from NLSE.callbacks import _trial_propagation
from NLSE.solvers.nlse import YOSHIDA_WEIGHTS

N = 32  # grid points per axis
DELTA_Z = 1e-4  # m, a few steps of the 1 mm medium in helpers.PHYSICS
STEPS = 3  # steps per run: enough that the solver asks more than once
# The solver's per-step constants the callback takes, in its argument order.
CONSTANTS = ("_alpha_half", "_g", "_Isat_conv")
# Rows of the grid where the factors are checked: half its columns, so that
# rows and columns swapped cannot pass for each other.
NY = N // 2
BATCH = 3  # fields in a batch, over which the row factors repeat
# Fraction of a step a run ends on, so that it lands on L with a partial step.
REMAINDER = 0.5
# Roundings between the factors and the propagator they stand for: each
# factor, their product, the twin's exponential and its 1/N. Each is at most
# one eps of the 1/N every element has; measured 0.51 at most.
ROUNDINGS = 5
# Constructor arguments that make a GPE of helpers.PHYSICS.
GPE_PHYSICS = {"gamma": 0, "N": 1e5, "g": 1e3}
CROSS = 0.1  # weight of a K_x K_y term against the dispersion's own
# 1/m, a constant phase rate added to the dispersion: 0.1 rad over DELTA_Z,
# so that the operator is not 0 where both axes' frequencies are.
SHIFT = 1e3
OTHER_STEP = 2 * DELTA_Z  # m, a step the propagator is built for, not taken
RETUNE = 2.0  # factor on the wavenumber once the propagator is built


class Records:
    """Stand-in for ``kernels.linear_nonlinear_step`` that records each call."""

    def __init__(self, accepts):
        self.accepts = accepts
        self.calls = []

    def __call__(self, A, propagator, plan, V_scaled, dz, alpha, g, Isat, factors=None):
        """Record the arguments; take the step, or leave both to the caller."""
        self.calls.append((A, propagator, plan, V_scaled, dz, alpha, g, Isat, factors))
        return self.accepts


def asked(monkeypatch, simu, accepts):
    """Put a ``Records`` in the backend's place and return it."""
    records = Records(accepts)
    monkeypatch.setattr(
        simu._backend.kernels, "linear_nonlinear_step", records, raising=False
    )
    return records


def flagged(monkeypatch, simu):
    """Return the solver with its backend claiming the store callback."""
    monkeypatch.setattr(simu._backend, "has_nonlinear_store_callback", True)
    return simu


@pytest.mark.parametrize(
    "cls,overrides,expected",
    [
        (NLSE, {}, True),
        (GPE, GPE_PHYSICS, True),
        (NLSE_1d, {}, False),
        (NLSE, {"nl_length": 1e-4}, False),
    ],
    ids=["NLSE", "GPE", "1D", "nonlocal"],
)
def test_the_gate(monkeypatch, cls, overrides, expected):
    """2D and local only, and only where the backend says it can.

    cuFFT links callbacks into 2D plans, and a non-local response smooths the
    intensity between the transform and the step.
    """
    simu = flagged(monkeypatch, make(cls, "CPU", n=N, **overrides))
    assert simu._fuses_nonlinear_into_transform() is expected


def test_a_backend_without_the_callback_is_not_asked():
    """The flag defaults to False, and so does the gate."""
    assert make(NLSE, "CPU", n=N)._fuses_nonlinear_into_transform() is False


def beam(simu):
    """Return a Gaussian on the solver's grid."""
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)


def step_once(simu, V):
    """Run ``_linear_then_nonlinear`` once; return the field and the plans."""
    A = beam(simu)
    plans = simu._build_fft_plan(A)
    A_sq = np.empty(A.shape, np.float32)
    simu._linear_then_nonlinear(A, A_sq, V, simu.propagator, plans, DELTA_Z)
    return A, plans


@pytest.mark.parametrize(
    "V", [None, np.ones((N, N), np.float32)], ids=["no V", "real V"]
)
def test_the_callback_is_handed_the_step_owed(monkeypatch, V):
    """Each argument is the one the solver's own two steps would have used.

    The propagator is the one carrying 1/N, because the callback's inverse
    transform does not normalize.
    """
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N))
    simu.propagator = np.ones((N, N), np.complex64)
    simu._update_propagator_fft()
    records = asked(monkeypatch, simu, accepts=True)
    A, plans = step_once(simu, V)
    assert len(records.calls) == 1, f"asked {len(records.calls)} times for one step"
    field, propagator, plan, potential, dz, *constants, factors = records.calls[0]
    assert field is A, "handed another field"
    assert propagator is simu._propagator_fft, "handed the normalizing propagator"
    assert plan is plans[0], "handed another plan"
    assert potential is V, "handed another potential"
    assert dz == DELTA_Z, f"asked for a step of {dz}, not {DELTA_Z}"
    assert constants == [simu._constant(name) for name in CONSTANTS], (
        f"handed {constants} for {CONSTANTS}"
    )
    assert factors is None, "offered factors for a propagator assigned by hand"


def test_without_the_unnormalized_propagator_it_is_not_asked(monkeypatch):
    """No propagator carries 1/N, so the callback's transform cannot be used."""
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N))
    simu.propagator = np.ones((N, N), np.complex64)
    monkeypatch.setattr(simu, "_propagator_fft", None)
    records = asked(monkeypatch, simu, accepts=True)
    step_once(simu, None)
    assert not records.calls, "asked to fold the step into a normalizing transform"


def run(simu, splitting):
    """Propagate a Gaussian ``STEPS`` steps and return the field."""
    return simu.out_field(
        beam(simu),
        STEPS * DELTA_Z,
        delta_z=DELTA_Z,
        splitting=splitting,
        verbose=False,
        plot=False,
    )


@pytest.mark.parametrize("splitting", ["lie", "strang"])
def test_a_declined_callback_leaves_the_step_as_it_was(monkeypatch, splitting):
    """Declined, the solver runs both steps itself, and to the bit as before."""
    reference = np.asarray(run(make(NLSE, "CPU", n=N), splitting))
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N))
    records = asked(monkeypatch, simu, accepts=False)
    assert np.array_equal(np.asarray(run(simu, splitting)), reference)
    assert records.calls, "the solver never asked for the callback"


def exact(simu, delta_z):
    """Return the twin of the propagator over ``delta_z``, in complex128.

    Parameters
    ----------
    simu : NLSE
        Solver whose dispersion it is.
    delta_z : float
        Step the propagator advances by.

    Returns
    -------
    np.ndarray
        ``exp(operator * delta_z) / N`` over the grid.
    """
    operator = np.asarray(simu._compute_propagator_rk4(), np.complex128)
    return np.exp(operator * delta_z) / operator.size


def tolerance(simu, dtype, delta_z):
    """Return how far factors may sit from the twin they stand for.

    ROUNDINGS eps of the 1/N every element has, and the twin's exponential
    is of an argument rounded to the field's width, which adds its phase.

    Parameters
    ----------
    simu : NLSE
        Solver whose dispersion it is.
    dtype : np.dtype
        Complex width of the factors.
    delta_z : float
        Step the propagator advances by.

    Returns
    -------
    float
        Largest difference allowed, element by element.
    """
    operator = np.asarray(simu._compute_propagator_rk4(), np.complex128)
    phase = np.abs(operator * delta_z).max()
    return (ROUNDINGS + phase) * np.finfo(dtype).eps / operator.size


def product(factors, shape):
    """Return the propagator the factors stand for, in ``shape``.

    Parameters
    ----------
    factors : tuple
        ``(rows, columns)``.
    shape : tuple
        Shape of the field they multiply.

    Returns
    -------
    np.ndarray
        ``rows[:, None] * columns[None, :]``, reshaped.
    """
    rows, columns = factors
    return (rows[:, None] * columns[None, :]).reshape(shape)


def sub_steps(splitting, steps):
    """Return the step each propagator of a run is built for, in order.

    Parameters
    ----------
    splitting : str
        "lie", "strang" or "yoshida".
    steps : float
        Steps the run covers, a partial one last where it is not whole.

    Returns
    -------
    list
        One distance per call to the callback.
    """
    whole = int(steps)
    lengths = [DELTA_Z] * whole
    if steps > whole:
        lengths.append((steps - whole) * DELTA_Z)
    if splitting == "yoshida":
        return [weight * length for length in lengths for weight in YOSHIDA_WEIGHTS]
    else:
        return lengths


@pytest.mark.parametrize("shape", [(NY, N), (BATCH, NY, N)], ids=["one", "batch"])
@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
def test_each_step_is_offered_the_factors_of_its_propagator(
    monkeypatch, splitting, shape
):
    """Every call, partial step and Yoshida sub-step alike, gets its own.

    The factors must stand for the twin the call is handed, over every field
    of a batch, and for the propagator of the step it takes.
    """
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N, NY=NY))
    records = asked(monkeypatch, simu, accepts=False)
    simu.out_field(
        np.broadcast_to(beam(simu), shape).copy(),
        (STEPS - REMAINDER) * DELTA_Z,
        delta_z=DELTA_Z,
        splitting=splitting,
        verbose=False,
        plot=False,
    )
    expected = sub_steps(splitting, STEPS - REMAINDER)
    assert len(records.calls) == len(expected), (
        f"asked {len(records.calls)} times for {len(expected)} propagators"
    )
    for call, step in zip(records.calls, expected):
        field, twin, *_, factors = call
        assert factors is not None, f"no factors for the step of {step}"
        got = product(factors, field.shape)
        bound = tolerance(simu, field.dtype, step)
        assert np.abs(got - twin).max() <= bound, f"not the twin at {step}"
        assert np.abs(got - exact(simu, step)).max() <= bound, (
            f"not the propagator over {step}"
        )


def partial_step(simu, A, A_sq, plans, step):
    """Return one Strang step of ``step`` on the solver's propagator.

    Parameters
    ----------
    simu : NLSE
        Solver.
    A : np.ndarray
        Field.
    A_sq : np.ndarray
        Scratch for the intensity.
    plans : list
        FFT plans.
    step : float
        Step to take.

    Returns
    -------
    functools.partial
        The step, to call with no argument.
    """
    return partial(
        simu.split_step, A, A_sq, None, simu.propagator, plans, step, "strang"
    )


def yoshida_step(simu, A, A_sq, plans):
    """Take one Yoshida step on sub-step propagators of its own.

    Parameters
    ----------
    simu : NLSE
        Solver.
    A : np.ndarray
        Field.
    A_sq : np.ndarray
        Scratch for the intensity.
    plans : list
        FFT plans.
    """
    sub = simu._yoshida_propagators(A.dtype, DELTA_Z)
    simu.split_step_yoshida(A, A_sq, None, sub, plans, DELTA_Z)


def remainder_step(simu, A, A_sq, plans):
    """Take a partial step as a run landing on L does.

    Parameters
    ----------
    simu : NLSE
        Solver.
    A : np.ndarray
        Field.
    A_sq : np.ndarray
        Scratch for the intensity.
    plans : list
        FFT plans.
    """
    factory = partial(partial_step, simu, A, A_sq, plans)
    simu._take_partial_step(factory, REMAINDER * DELTA_Z, "split_step", A.dtype)


def trial_steps(simu, A, A_sq, plans):
    """Take the trial steps the adaptive step's callback takes.

    Parameters
    ----------
    simu : NLSE
        Solver.
    A : np.ndarray
        Field.
    A_sq : np.ndarray
        Scratch for the intensity.
    plans : list
        FFT plans.
    """
    simu.plans = plans
    _trial_propagation(simu, A, REMAINDER * DELTA_Z, STEPS)


@pytest.mark.parametrize(
    "detour",
    [yoshida_step, remainder_step, trial_steps],
    ids=["yoshida", "partial step", "trial"],
)
def test_a_step_on_its_own_propagator_puts_the_origin_back(monkeypatch, detour):
    """After one, the run's own twin is offered its factors again.

    Each swaps the propagator, its twin and the twin's origin, and the steps
    taken in between are each offered factors of their own.
    """
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N, NY=NY))
    simu.propagator = simu._build_propagator(np.complex64, DELTA_Z)
    simu._send_propagator_to_gpu()
    before = simu._linear_state()
    records = asked(monkeypatch, simu, accepts=False)
    A = beam(simu)
    plans = simu._build_fft_plan(A)
    detour(simu, A, np.empty(A.shape, np.float32), plans)
    assert records.calls, "the detour never asked for the callback"
    assert all(call[-1] is not None for call in records.calls), (
        "a step of the detour was offered no factors"
    )
    after = simu._linear_state()
    assert all(a is b for a, b in zip(after, before)), "the run's state is not back"
    assert simu._propagator_fft_factors(simu._propagator_fft, NY) is not None


def build(simu):
    """Give the solver the propagator it builds for DELTA_Z.

    Parameters
    ----------
    simu : NLSE
        Solver.
    """
    simu.propagator = simu._build_propagator(np.complex64, DELTA_Z)
    simu._send_propagator_to_gpu()


def replace_twin(simu):
    """Build, then put an equal twin of another identity in the built one's place.

    Parameters
    ----------
    simu : NLSE
        Solver.
    """
    build(simu)
    simu._propagator_fft = simu._propagator_fft.copy()


def retune(simu):
    """Build, then change the wavenumber the propagator was built for.

    Parameters
    ----------
    simu : NLSE
        Solver.
    """
    build(simu)
    simu.k *= RETUNE


def borrow(simu):
    """Give the solver a propagator another solver built.

    Parameters
    ----------
    simu : NLSE
        Solver.
    """
    simu.propagator = make(NLSE, "CPU", n=N)._build_propagator(np.complex64, DELTA_Z)
    simu._send_propagator_to_gpu()


class CrossDispersion(NLSE):
    """An NLSE whose dispersion has a K_x K_y term, so no factor per axis."""

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Add CROSS K_x K_y / k to NLSE's operator.

        Returns
        -------
        np.ndarray
            The operator.
        """
        cross = -1j * CROSS * self.Kxx * self.Kyy / self.k
        return super()._compute_propagator_rk4() + cross


class ShiftedDispersion(NLSE):
    """An NLSE whose dispersion is not 0 at the origin of the frequencies."""

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Add a constant -i SHIFT to NLSE's operator.

        Returns
        -------
        np.ndarray
            The operator.
        """
        return super()._compute_propagator_rk4() - 1j * SHIFT


class OwnBuild(NLSE):
    """An NLSE whose ``_build_propagator`` returns an array of its own."""

    def _build_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Return NLSE's propagator, negated.

        Parameters
        ----------
        dtype : np.dtype
            Complex width of the field.
        delta_z : float
            Step the propagator advances by.

        Returns
        -------
        np.ndarray
            A new array, which no cache holds.
        """
        return -super()._build_propagator(dtype, delta_z)


@pytest.mark.parametrize(
    "cls,setup",
    [
        (NLSE, replace_twin),
        (NLSE, borrow),
        (NLSE, retune),
        (CrossDispersion, build),
        (OwnBuild, build),
    ],
    ids=["twin replaced", "built elsewhere", "retuned", "cross term", "own build"],
)
def test_without_a_known_origin_no_factors_are_offered(monkeypatch, cls, setup):
    """The callback reads the full twin wherever the factors are in doubt."""
    simu = flagged(monkeypatch, make(cls, "CPU", n=N))
    setup(simu)
    records = asked(monkeypatch, simu, accepts=True)
    step_once(simu, None)
    assert len(records.calls) == 1, f"asked {len(records.calls)} times for one step"
    _, twin, *_, factors = records.calls[0]
    assert twin is simu._propagator_fft, "not handed the full twin"
    assert factors is None, "offered factors no build vouches for"


def test_the_factors_are_of_the_propagator_not_of_the_step(monkeypatch):
    """A propagator built for one step and used for another keeps its own.

    The kernel reads the twin, so the factors must describe it whatever the
    step the caller says it is taking.
    """
    simu = flagged(monkeypatch, make(NLSE, "CPU", n=N, NY=NY))
    simu.propagator = simu._build_propagator(np.complex64, OTHER_STEP)
    simu._send_propagator_to_gpu()
    records = asked(monkeypatch, simu, accepts=True)
    step_once(simu, None)
    _, twin, *_, factors = records.calls[0]
    got = product(factors, twin.shape)
    bound = tolerance(simu, np.complex64, OTHER_STEP)
    assert np.abs(got - exact(simu, OTHER_STEP)).max() <= bound, (
        "the factors are not of the propagator the kernel reads"
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("batch", [1, BATCH])
@pytest.mark.parametrize(
    "cls,physics",
    [(NLSE, {}), (GPE, GPE_PHYSICS), (ShiftedDispersion, {})],
    ids=["NLSE", "GPE", "shifted"],
)
def test_the_factors_multiply_to_the_twin(cls, physics, batch, dtype):
    """Rows carry 1/N and repeat over a batch; their product is the twin."""
    simu = make(cls, "CPU", n=N, NY=NY, **physics)
    simu.propagator = simu._build_propagator(dtype, DELTA_Z)
    simu._send_propagator_to_gpu()
    factors = simu._propagator_fft_factors(simu._propagator_fft, batch * NY)
    assert factors is not None, "no factors for a dispersion of one term per axis"
    assert all(factor.dtype == dtype for factor in factors), "not the field's width"
    got = product(factors, (batch, NY, N))
    bound = tolerance(simu, dtype, DELTA_Z)
    assert np.abs(got - simu._propagator_fft).max() <= bound, "not the twin"
    assert np.abs(got - exact(simu, DELTA_Z)).max() <= bound, "not the propagator"


def test_one_field_then_a_batch_each_get_their_rows():
    """The same twin asked for over one field, then a batch, repeats its rows."""
    simu = make(NLSE, "CPU", n=N, NY=NY)
    build(simu)
    twin = simu._propagator_fft
    assert simu._propagator_fft_factors(twin, NY) is not None
    factors = simu._propagator_fft_factors(twin, BATCH * NY)
    assert factors[0].size == BATCH * NY, "the rows of one field, for a batch"
    got = product(factors, (BATCH, NY, N))
    assert np.abs(got - twin).max() <= tolerance(simu, np.complex64, DELTA_Z)
