"""Tests for merging the half steps of consecutive Strang steps.

A Strang step is ``N(h/2) L(h) N(h/2)``, so a run of them is

    N(h/2) [L(h) N(h)] ... [L(h) N(h)] N(-h/2)

and the bracketed body is exactly a Lie step. Merging costs one nonlinear
application for the whole run rather than one per step, and the loop body
becomes the one ``precision="single"`` already runs.

It is exact only where ``N(a) N(b) == N(a + b)``, which needs ``|A|`` to
survive ``N``: no loss and no absorbing potential. The tests that matter here
are the ones checking the merge *declines*, because taking it where it does
not hold changes the answer without failing anything else.
"""

import numpy as np
import pytest
from helpers import make
from NLSE import CNLSE, NLSE
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST


def beam(coupled=False, dtype=np.complex64):
    """Return a Gaussian beam, two components if coupled."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    X, Y = np.meshgrid(x, x)
    single = np.exp(-(X**2 + Y**2) / WAIST**2)
    if coupled:
        return np.stack([single, np.exp(-(X**2 + Y**2) / (WAIST / 3) ** 2)]).astype(
            dtype
        )
    return single.astype(dtype)


def build(backend, cls=NLSE, **overrides):
    """Return a lossless solver, which is what the merge needs."""
    params = {"alpha": 0, "window": WINDOW, "L": L, "Isat": 1e6}
    params.update(overrides)
    return make(cls, backend, n=N, **params)


def propagate(solver, coupled=False, phase=0.1):
    """Run one propagation at a step giving this phase, in Strang splitting."""
    A = beam(coupled)
    prepared, _ = solver._prepare_output_array(A.copy(), normalize=True)
    solver._precompute_step_constants(solver.V, "double")
    rates = solver._energy_rates(prepared)
    delta_z = phase / (rates["potential"] + rates["interaction"])
    out = solver.out_field(
        A.copy(),
        L,
        delta_z=delta_z,
        verbose=False,
        plot=False,
        precision="double",
        method="split_step",
    )
    return np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_merging_does_not_change_the_answer(backend_name):
    """The merged run must agree with the unmerged one it stands for.

    To round-off rather than exactly: the two compute the same rotation with
    a different number of exponentials, and complex64 does not care that the
    algebra is an identity.
    """
    merged = propagate(build(backend_name))
    plain = build(backend_name)
    plain._lie_step_is_strang_body = False
    unmerged = propagate(plain)
    difference = np.max(np.abs(merged - unmerged)) / np.max(np.abs(unmerged))
    assert difference < 1e-3, (
        f"{backend_name}: merging changed the answer by {difference:.3e}, "
        f"which is more than the round-off it should cost"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_merging_keeps_the_accuracy_of_the_step_it_replaces(backend_name):
    """Both must sit the same distance from a converged reference.

    Agreeing with each other is not enough: two runs that made the same
    mistake would agree too.
    """
    reference = propagate(build(backend_name), phase=2e-3).astype(np.complex128)

    def distance(solver):
        return float(
            np.linalg.norm(propagate(solver).astype(np.complex128) - reference)
            / np.linalg.norm(reference)
        )

    plain = build(backend_name)
    plain._lie_step_is_strang_body = False
    merged_error = distance(build(backend_name))
    plain_error = distance(plain)
    assert merged_error == pytest.approx(plain_error, rel=0.2), (
        f"{backend_name}: merged error {merged_error:.3e} against "
        f"{plain_error:.3e} unmerged -- the merge changed the method"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_loop_body_becomes_a_lie_step(backend_name):
    """A merged run must actually take the cheaper body.

    Nothing about the answer reveals this: the unmerged run gives the same
    one, so a gate that stops matching costs only the nonlinear step it was
    supposed to save.
    """
    solver = build(backend_name)
    seen = []
    original = type(solver).split_step

    def spy(self, A, A_sq, V, propagator, plans, delta_z, precision="single"):
        seen.append(precision)
        return original(self, A, A_sq, V, propagator, plans, delta_z, precision)

    type(solver).split_step = spy
    try:
        propagate(solver)
    finally:
        type(solver).split_step = original
    assert seen, "the solver never took a split step"
    assert seen.count("single") > 1, (
        f"a merged run should take Lie steps between its two half steps; "
        f"the body ran {sorted(set(seen))}"
    )
    # The distance left over after the whole steps is covered outside the
    # bracket, so it is a Strang step of its own. There is at most one.
    assert seen.count("double") <= 1, (
        f"only the remainder may fall outside the merged run; "
        f"{seen.count('double')} steps did"
    )


@pytest.mark.parametrize(
    "reason,overrides,cls",
    [
        ("loss", {"alpha": 20}, NLSE),
        ("a complex potential", {"V": "complex"}, NLSE),
        ("non-locality", {"nl_length": 5 * WINDOW / N}, NLSE),
    ],
)
def test_the_merge_declines_where_it_is_not_exact(reason, overrides, cls):
    """N(a) N(b) is N(a+b) only while |A| survives N.

    With loss or an absorbing potential the second half step sees an
    intensity the first one changed, so the two are not one whole step.
    Non-locality convolves the intensity, which the Lie body does too, but
    the fast loop it needs is not taken there anyway.
    """
    if overrides.get("V") == "complex":
        overrides = dict(overrides)
        overrides["V"] = (np.ones((N, N)) * 1e-3 + 1j * np.ones((N, N)) * 1e-4).astype(
            np.complex64
        )
    solver = build("CPU", cls=cls, **overrides)
    solver._precompute_step_constants(solver.V, "double")
    assert not solver._merges_strang_halves("double"), (
        f"the merge must decline with {reason}"
    )


def test_declining_with_loss_is_worth_doing():
    """The guard has to earn its place, not merely exist.

    Merging a lossy run is not exact, and the cost is not subtle: the error
    doubles at every step size, because the second half step sees an
    intensity the first one has already damped. Asserting only that the guard
    returns False leaves that a structural claim -- this measures it, so
    removing the guard fails a test about the answer.
    """
    solver = build("CPU", alpha=20)
    guarded = propagate(solver).astype(np.complex128)

    forced = build("CPU", alpha=20)
    forced._merges_strang_halves = lambda precision: precision == "double"
    merged = propagate(forced).astype(np.complex128)

    reference = propagate(build("CPU", alpha=20), phase=2e-3).astype(np.complex128)

    def distance(field):
        return float(np.linalg.norm(field - reference) / np.linalg.norm(reference))

    assert distance(guarded) < distance(merged) / 1.5, (
        f"merging a lossy run should cost real accuracy: guarded "
        f"{distance(guarded):.3e} against merged {distance(merged):.3e}"
    )


def test_the_merge_declines_for_lie_splitting():
    """Lie splitting has no halves to merge."""
    solver = build("CPU")
    solver._precompute_step_constants(solver.V, "single")
    assert not solver._merges_strang_halves("single")


def test_the_merge_declines_with_a_rabi_coupling():
    """The Rabi rotation rides on the Lie step, not on the Strang one."""
    solver = build("CPU", cls=CNLSE, omega=1e4)
    solver._precompute_step_constants(solver.V, "double")
    assert not solver._merges_strang_halves("double")
    solver.omega = None
    assert solver._merges_strang_halves("double")


def test_a_driven_solver_opts_out():
    """DDGPE's step drives and adds noise, so it is not such a product."""
    from NLSE import DDGPE

    assert not DDGPE._lie_step_is_strang_body
    assert NLSE._lie_step_is_strang_body
