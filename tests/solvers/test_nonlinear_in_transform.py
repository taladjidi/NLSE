"""Where the nonlinear step may run inside the inverse transform, and the rest.

CUPY runs the nonlinear step as cuFFT stores the inverse transform's output.
Which runs ask for it is decided in the solver, from a backend flag, so it is
checked here on CPU with the flag set by hand; the callback itself is checked
on a CUDA machine, in tests/backends/test_fft_propagator_fusion.py.
"""

import numpy as np
import pytest
from helpers import WAIST, make
from NLSE import GPE, NLSE, NLSE_1d

N = 32  # grid points per axis
DELTA_Z = 1e-4  # m, a few steps of the 1 mm medium in helpers.PHYSICS
STEPS = 3  # steps per run: enough that the solver asks more than once
# The solver's per-step constants the callback takes, in its argument order.
CONSTANTS = ("_alpha_half", "_g", "_Isat_conv")


class Records:
    """Stand-in for ``kernels.linear_nonlinear_step`` that records each call."""

    def __init__(self, accepts):
        self.accepts = accepts
        self.calls = []

    def __call__(self, A, propagator, plan, V_scaled, dz, alpha, g, Isat):
        """Record the arguments; take the step, or leave both to the caller."""
        self.calls.append((A, propagator, plan, V_scaled, dz, alpha, g, Isat))
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
        (GPE, {"gamma": 0, "N": 1e5, "g": 1e3}, True),
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
    field, propagator, plan, potential, dz, *constants = records.calls[0]
    assert field is A, "handed another field"
    assert propagator is simu._propagator_fft, "handed the normalizing propagator"
    assert plan is plans[0], "handed another plan"
    assert potential is V, "handed another potential"
    assert dz == DELTA_Z, f"asked for a step of {dz}, not {DELTA_Z}"
    assert constants == [simu._constant(name) for name in CONSTANTS], (
        f"handed {constants} for {CONSTANTS}"
    )


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
