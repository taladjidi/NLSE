"""Tests for merging the half steps of consecutive Strang steps."""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, make
from NLSE import CNLSE, NLSE, NLSE_3d
from scipy.constants import c

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST
# Relative. At COUPLED_PHASE the scalar bracket of a merged run is off by
# 1.3e-3 (CPU and MLX, measured 2026-11-27); the run without it by 0.
COUPLED_TOLERANCE = 1e-5
# Phase per step, rad: large, so that a wrong half step shows.
COUPLED_PHASE = 0.5
# Loss of both components (1/m), so that the second one's is in the run too.
COUPLED_ALPHA = 20
# The 3D case: points per axis, steps over L, the pulse's window (s) and its
# 1/e half-width as a fraction of it, energy (J) and saturation (W/m^2) (about
# 5 rad at the peak over L: k*n2*I*L/(1 + I/Isat) with I = 6e5 W/m^2), and
# dispersion (s^2/m, that of tests/benchmarks).
N_3D = 16
STEPS_3D = 20
WINDOW_T = 1e-12
PULSE_FRACTION = 6
ENERGY_3D = 1e-12
ISAT_3D = 1e6
D0 = 1e-26
# Relative. Merged and unmerged 3D runs are 2e-6 to 5e-6 apart in complex64
# (CPU, CL and MLX, measured 2026-11-28); a wrong half step is 1e-2 and more.
MERGE_TOLERANCE_3D = 1e-4


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


def propagate(solver, coupled=False, phase=0.1, callback=None):
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
        callback=callback,
    )
    return np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_merging_does_not_change_the_answer(backend_name):
    """The merged run must agree with the unmerged one it stands for."""
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
    """Merged and unmerged runs sit the same distance from a converged reference."""
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
    """A merged run must actually take the cheaper body."""
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
    """N(a) N(b) is N(a+b) only while N solves its own step exactly."""
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
    """The merge is taken with loss, and costs nothing against the unmerged run."""
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


def run_3d(backend, merged):
    """Run a Gaussian pulse over L on a lossy NLSE_3d in Strang splitting.

    Parameters
    ----------
    backend : str
        Backend name.
    merged : bool
        Whether the run may merge its half steps.

    Returns
    -------
    tuple
        The solver and the propagated field.
    """
    solver = make(
        NLSE_3d,
        backend,
        n=N_3D,
        NZ=N_3D,
        energy=ENERGY_3D,
        window=[WINDOW, WINDOW, WINDOW_T],
        D0=D0,
        vg=c,
        L=L,
        Isat=ISAT_3D,
    )
    solver._lie_step_is_strang_body = merged
    X, Y, T = np.meshgrid(solver.X, solver.Y, solver.T)
    duration = WINDOW_T / PULSE_FRACTION
    pulse = np.exp(-(X**2 + Y**2) / WAIST**2 - T**2 / duration**2)
    delta_z = L / STEPS_3D
    out = solver.out_field(
        pulse.astype(np.complex64),
        L,
        delta_z=delta_z,
        verbose=False,
        plot=False,
        splitting="strang",
    )
    assert solver._current_delta_z == delta_z, "the solver lowered the step"
    return solver, np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_3d_run_merges_its_half_steps(backend_name):
    """NLSE_3d's ``omega`` is its frequency grid, not a Rabi coupling."""
    solver, merged = run_3d(backend_name, merged=True)
    assert solver._merges_strang_halves("strang"), (
        "NLSE_3d should merge its Strang half steps: it has no Rabi coupling"
    )
    _, unmerged = run_3d(backend_name, merged=False)
    difference = np.max(np.abs(merged - unmerged)) / np.max(np.abs(unmerged))
    assert 0 < difference < MERGE_TOLERANCE_3D, (
        f"{backend_name}: merging changed the 3D answer by {difference:.3e}: "
        f"more than the round-off it should cost, or 0 if the two runs were "
        f"the same one"
    )


def test_the_merge_declines_for_lie_splitting():
    """Lie splitting has no halves to merge."""
    solver = build("CPU")
    solver._precompute_step_constants(solver.V, np.complex64)
    assert not solver._merges_strang_halves("lie")


def no_callback(simu, A, z, i):
    """Do nothing: any callback keeps a run off the merged path."""


def run_coupled(backend_name, callback):
    """Return a lossy two-component Strang run, stepped under this callback."""
    solver = build(backend_name, cls=CNLSE, symmetric=False, alpha=COUPLED_ALPHA)
    return propagate(solver, coupled=True, phase=COUPLED_PHASE, callback=callback)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_coupled_strang_run_keeps_its_cross_coupling(backend_name):
    """A coupled run must not take the scalar half steps of a merged one."""
    assert not CNLSE._lie_step_is_strang_body
    stepped = run_coupled(backend_name, no_callback)
    merged = run_coupled(backend_name, None)
    difference = np.max(np.abs(merged - stepped)) / np.max(np.abs(stepped))
    assert difference < COUPLED_TOLERANCE, (
        f"{backend_name}: a coupled Strang run differs by {difference:.3e} from "
        f"the same run stepped under a callback"
    )


def run_rk4(backend_name, splitting):
    """Return an RK4 run of the solver's own step, asked for this splitting."""
    solver = build(backend_name)
    out = solver.out_field(
        beam(), L, verbose=False, plot=False, splitting=splitting, method="RK4"
    )
    return np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_an_rk4_run_ignores_the_splitting(backend_name):
    """RK4 does not split, so no half step may bracket its run."""
    assert np.array_equal(run_rk4(backend_name, "strang"), run_rk4(backend_name, "lie"))


def test_a_driven_solver_opts_out():
    """DDGPE's step drives and adds noise, so it is not such a product."""
    from NLSE import DDGPE

    assert not DDGPE._lie_step_is_strang_body
    assert NLSE._lie_step_is_strang_body
