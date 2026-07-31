"""Tests for merging the half steps of consecutive Strang steps.

A Strang step is ``N(h/2) L(h) N(h/2)``, so a run of them is

    N(h/2) [L(h) N(h)] ... [L(h) N(h)] N(-h/2)

and the bracketed body is exactly a Lie step. Merging costs one nonlinear
application for the whole run rather than one per step, and the loop body
becomes the one ``splitting="lie"`` already runs.

It is exact only where ``N(a) N(b) == N(a + b)``, which needs ``N`` to be the
exact solution of the real-space equation over its own step: an absorbing
potential breaks that, and plain loss used to. The tests that matter here are
the ones checking the merge *declines*, because taking it where it does not
hold changes the answer without failing anything else.
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
    solver._precompute_step_constants(solver.V, np.complex64)
    rates = solver._energy_rates(prepared)
    delta_z = phase / (rates["potential"] + rates["interaction"])
    out = solver.out_field(
        A.copy(),
        L,
        delta_z=delta_z,
        verbose=False,
        plot=False,
        splitting="strang",
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

    def spy(self, A, A_sq, V, propagator, plans, delta_z, splitting="lie"):
        seen.append(splitting)
        return original(self, A, A_sq, V, propagator, plans, delta_z, splitting)

    type(solver).split_step = spy
    try:
        propagate(solver)
    finally:
        type(solver).split_step = original
    assert seen, "the solver never took a split step"
    assert seen.count("lie") > 1, (
        f"a merged run should take Lie steps between its two half steps; "
        f"the body ran {sorted(set(seen))}"
    )
    # The distance left over after the whole steps is covered outside the
    # bracket, so it is a Strang step of its own. There is at most one.
    assert seen.count("strang") <= 1, (
        f"only the remainder may fall outside the merged run; "
        f"{seen.count('strang')} steps did"
    )


@pytest.mark.parametrize(
    "reason,overrides,cls",
    [
        ("a complex potential", {"V": "complex"}, NLSE),
        ("non-locality", {"nl_length": 5 * WINDOW / N}, NLSE),
    ],
)
def test_the_merge_declines_where_it_is_not_exact(reason, overrides, cls):
    """N(a) N(b) is N(a+b) only while N solves its own step exactly.

    An absorbing potential's decay is applied frozen, so a second half step
    sees an intensity the first one changed and the two are not one whole step.
    Non-locality convolves the intensity, which the Lie body does too, but
    the fast loop it needs is not taken there anyway.

    Plain loss used to be on this list. It is not any more -- see
    ``test_the_merge_is_taken_with_loss``.
    """
    if overrides.get("V") == "complex":
        overrides = dict(overrides)
        overrides["V"] = (np.ones((N, N)) * 1e-3 + 1j * np.ones((N, N)) * 1e-4).astype(
            np.complex64
        )
    solver = build("CPU", cls=cls, **overrides)
    solver._precompute_step_constants(solver.V, np.complex64)
    assert not solver._merges_strang_halves("strang"), (
        f"the merge must decline with {reason}"
    )


def test_the_merge_is_taken_with_loss():
    """Loss is no longer a reason to decline, and this is why it was.

    A lossy run used to be refused the merge because the second half step saw
    an intensity the first had damped, which cost real accuracy -- the test
    that stood here measured it. What removed the reason is that the real-space
    step is now solved rather than frozen (``_loss_factor`` in
    kernels/cpu.py): both halves telescope, exactly, so the merged run and the
    unmerged one are the same answer and the merge saves a kernel per step.

    Measured rather than asserted structurally, in both directions: the merge
    is taken, and taking it costs nothing against the run that does not.
    """
    solver = build("CPU", alpha=20)
    solver._precompute_step_constants(solver.V, np.complex64)
    assert solver._merges_strang_halves("strang"), (
        "a lossy Strang run should merge now that its half steps telescope"
    )

    merged = propagate(solver).astype(np.complex128)

    unmerged = build("CPU", alpha=20)
    unmerged._merges_strang_halves = lambda splitting: False
    separate = propagate(unmerged).astype(np.complex128)

    reference = propagate(build("CPU", alpha=20), phase=2e-3).astype(np.complex128)

    def distance(field):
        return float(np.linalg.norm(field - reference) / np.linalg.norm(reference))

    # Not bit-for-bit: each solved step carries its own O(u^5) truncation, and
    # a half plus a half is not the same rounding as a whole. Far below the
    # step error either way, which is what "costs nothing" has to mean.
    assert distance(merged) == pytest.approx(distance(separate), rel=0.05), (
        f"merging a lossy run should cost nothing: merged {distance(merged):.3e} "
        f"against separate {distance(separate):.3e}"
    )


def test_the_merge_declines_for_lie_splitting():
    """Lie splitting has no halves to merge."""
    solver = build("CPU")
    solver._precompute_step_constants(solver.V, np.complex64)
    assert not solver._merges_strang_halves("lie")


def test_the_merge_declines_with_a_rabi_coupling():
    """The Rabi rotation rides on the Lie step, not on the Strang one."""
    solver = build("CPU", cls=CNLSE, omega=1e4)
    solver._precompute_step_constants(solver.V, np.complex64)
    assert not solver._merges_strang_halves("strang")
    solver.omega = None
    assert solver._merges_strang_halves("strang")


def test_a_driven_solver_opts_out():
    """DDGPE's step drives and adds noise, so it is not such a product."""
    from NLSE import DDGPE

    assert not DDGPE._lie_step_is_strang_body
    assert NLSE._lie_step_is_strang_body
