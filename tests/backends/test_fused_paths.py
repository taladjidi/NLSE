"""Tests for the fused kernel paths: same answer, and actually taken."""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, gaussian as beam, make as build
from NLSE import CNLSE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.backends import _backend_builders, get_backend

# Every flag that puts a step on a shorter path than the generic one. A level
# is defined by which of these are left on, so a new capability has to be
# added here or the levels below it stop being reachable in these tests.
FUSION_FLAGS = (
    "has_fused_rk4_stage",
    "has_fused_rk4_rhs",
    "has_fused_coupled_rk4_rhs",
    "has_fused_rk4_stage_update",
    "has_fused_rk4_final_update",
    "has_fused_coupled_split_step",
)

# Level -> the flags it leaves on. Ordered most fused first.
LEVELS = {
    "stage": FUSION_FLAGS,
    "rhs": tuple(f for f in FUSION_FLAGS if f != "has_fused_rk4_stage"),
    "generic": (),
}

FUSED_BACKENDS = [
    name
    for name in AVAILABLE_BACKENDS
    if any(getattr(get_backend(name), flag, False) for flag in FUSION_FLAGS)
]

pytestmark = pytest.mark.skipif(
    not FUSED_BACKENDS, reason="no backend declares a fused path"
)

N = 64
L = 1e-3
waist = 2.23e-3
window = 4 * waist
SOLVERS = [NLSE, NLSE_1d, CNLSE, CNLSE_1d]
COUPLED = [CNLSE, CNLSE_1d]


def one_dimensional(cls):
    """Whether this solver works on a line rather than a plane."""
    return cls in (NLSE_1d, CNLSE_1d)


def coupled(cls):
    """Whether this solver carries two components."""
    return cls in COUPLED


def grid_shape(cls):
    """Shape of one component's grid."""
    return (N,) if one_dimensional(cls) else (N, N)


def make(cls, backend, **overrides):
    """Return a solver of this class, its two components not alike."""
    return build(cls, backend, n=N, symmetric=False, **overrides)


def gaussian(cls, dtype=np.complex64):
    """Return a smooth field of the shape this solver takes."""
    shape = grid_shape(cls)
    return beam((2, *shape) if coupled(cls) else shape, dtype=dtype)


def fusion(monkeypatch, backend, level):
    """Restrict a backend to one level of fusion, for the rest of the test."""
    keep = LEVELS[level]
    for flag in FUSION_FLAGS:
        if not getattr(type(backend), flag, False):
            continue
        monkeypatch.setattr(type(backend), flag, flag in keep, raising=False)


def potential(kind, cls):
    """Return a potential of this kind, shaped for this solver."""
    if kind == "none":
        return None
    shape = grid_shape(cls)
    real = np.ones(shape) * 1e-3
    if kind == "real":
        return real.astype(np.float32)
    return (real + 1j * np.ones(shape) * 1e-4).astype(np.complex64)


def propagate(cls, backend_name, level, monkeypatch, method="RK4", dtype=None, **kw):
    """Run one propagation with the backend held at this level of fusion."""
    solver = make(cls, backend_name, **kw)
    fusion(monkeypatch, solver._backend, level)
    out = solver.out_field(
        gaussian(cls, dtype or np.complex64),
        L,
        verbose=False,
        plot=False,
        splitting="lie",
        method=method,
    )
    return np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("level", ["stage", "rhs"])
@pytest.mark.parametrize("kind", ["none", "real", "complex"])
def test_each_rk4_level_matches_the_generic_one(
    backend_name, cls, level, kind, monkeypatch
):
    """Every level of RK4 fusion must give the generic answer."""
    if not getattr(get_backend(backend_name), "has_fused_rk4_stage", False):
        pytest.skip(f"{backend_name} has no fused RK4 stage")
    V = potential(kind, cls)
    fused = propagate(cls, backend_name, level, monkeypatch, V=V)
    generic = propagate(cls, backend_name, "generic", monkeypatch, V=V)
    difference = np.max(np.abs(fused - generic)) / np.max(np.abs(generic))
    assert difference < 1e-5, (
        f"{cls.__name__} at level {level!r} with a {kind} potential differs "
        f"from the generic path by {difference:.3e}"
    )
    if not coupled(cls):
        assert np.array_equal(fused, generic), (
            f"{cls.__name__} has one component, so the fused levels run the "
            f"same arithmetic as the generic path and must agree exactly"
        )


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["none", "real", "complex"])
def test_the_fused_levels_agree_bitwise(backend_name, cls, kind, monkeypatch):
    """The whole-stage kernels must not re-associate what the rhs ones do."""
    if not getattr(get_backend(backend_name), "has_fused_rk4_stage", False):
        pytest.skip(f"{backend_name} has no fused RK4 stage")
    V = potential(kind, cls)
    stage = propagate(cls, backend_name, "stage", monkeypatch, V=V)
    rhs = propagate(cls, backend_name, "rhs", monkeypatch, V=V)
    assert np.array_equal(stage, rhs), (
        f"{cls.__name__} with a {kind} potential differs between the "
        f"whole-stage and rhs levels by "
        f"{np.max(np.abs(stage - rhs)) / np.max(np.abs(rhs)):.3e}"
    )


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("level", ["stage", "rhs"])
def test_each_rk4_level_matches_in_double(backend_name, cls, level, monkeypatch):
    """The same, at the other splitting."""
    backend = get_backend(backend_name)
    if not getattr(backend, "has_fused_rk4_stage", False):
        pytest.skip(f"{backend_name} has no fused RK4 stage")
    if not backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision")
    fused = propagate(cls, backend_name, level, monkeypatch, dtype=np.complex128)
    generic = propagate(cls, backend_name, "generic", monkeypatch, dtype=np.complex128)
    difference = np.max(np.abs(fused - generic)) / np.max(np.abs(generic))
    assert difference < 1e-12, f"{cls.__name__} differs by {difference:.3e}"
    if not coupled(cls):
        assert np.array_equal(fused, generic)


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", COUPLED, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["none", "real", "complex"])
def test_the_coupled_split_step_matches_the_generic_one(
    backend_name, cls, kind, monkeypatch
):
    """The interleaved split step must give the generic answer."""
    if not getattr(get_backend(backend_name), "has_fused_coupled_split_step", False):
        pytest.skip(f"{backend_name} has no fused coupled split step")
    V = potential(kind, cls)
    fused = propagate(cls, backend_name, "stage", monkeypatch, method="split_step", V=V)
    generic = propagate(
        cls, backend_name, "generic", monkeypatch, method="split_step", V=V
    )
    assert np.max(np.abs(fused - generic)) / np.max(np.abs(generic)) < 1e-5


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
def test_the_fused_split_step_carries_the_rabi_coupling(backend_name, monkeypatch):
    """The interleaved Rabi rotation must match the per-component one."""
    if not getattr(get_backend(backend_name), "has_fused_coupled_split_step", False):
        pytest.skip(f"{backend_name} has no fused coupled split step")

    def run(level):
        solver = make(CNLSE, backend_name)
        solver.omega = 1e4
        fusion(monkeypatch, solver._backend, level)
        return np.asarray(
            solver._backend.to_numpy(
                solver.out_field(
                    gaussian(CNLSE),
                    L,
                    verbose=False,
                    plot=False,
                    splitting="lie",
                    method="split_step",
                )
            )
        )

    fused, generic = run("stage"), run("generic")
    assert np.max(np.abs(fused - generic)) / np.max(np.abs(generic)) < 1e-5


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_the_top_level_is_the_one_taken(backend_name, cls, monkeypatch):
    """The most fused path must be the one a plain run gets."""
    if not getattr(get_backend(backend_name), "has_fused_rk4_stage", False):
        pytest.skip(f"{backend_name} has no fused RK4 stage")
    solver = make(cls, backend_name)
    kernels = type(solver._backend.kernels)
    top = "rk4_stage_coupled_fused" if coupled(cls) else "rk4_stage_fused"
    lower = ("rk4_rhs_fused", "rk4_rhs_coupled_fused")
    counts: dict[str, int] = {}
    for name in (top, *lower):
        original = getattr(kernels, name, None)
        if original is None:
            continue

        def spy(self, *args, _name=name, _original=original, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(kernels, name, spy)

    solver.out_field(
        gaussian(cls), L, verbose=False, plot=False, splitting="lie", method="RK4"
    )
    assert counts.get(top, 0) > 0, (
        f"{cls.__name__} on {backend_name} never reached {top}: {counts}"
    )
    assert counts.get(top, 0) % 4 == 0, "RK4 takes four stages a step"
    for name in lower:
        assert name not in counts, (
            f"{cls.__name__} went through {name}, so the slope was written to "
            f"memory and read straight back"
        )


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", COUPLED, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_a_fused_run_splits_no_components(backend_name, cls, method, monkeypatch):
    """The point of the coupled kernels is that they split nothing."""
    if not getattr(get_backend(backend_name), "has_fused_coupled_split_step", False):
        pytest.skip(f"{backend_name} has no fused coupled path")
    solver = make(cls, backend_name)
    calls = []
    for name in ("_take_components", "_set_components"):
        original = getattr(type(solver), name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(type(solver), name, spy)

    solver.out_field(
        gaussian(cls), L, verbose=False, plot=False, splitting="lie", method=method
    )
    assert not calls, (
        f"{backend_name} declares a fused coupled path but {method} still "
        f"split the field: {sorted(set(calls))}"
    )


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_a_batch_declines_every_fused_path(backend_name, cls):
    """The kernels take scalars and one field, so a batch must fall back."""
    solver = make(cls, backend_name)
    batched = np.stack([gaussian(cls)] * 3)
    assert not solver._can_fuse_rk4_stage(batched)
    assert solver._can_fuse_rk4_stage(gaussian(cls))
    if coupled(cls):
        scalars = (20, 20, -1.6e-9, -1e-10, -1.6e-9, 10e4, 10e4)
        assert not solver._can_fuse_components(batched, scalars)
        assert solver._can_fuse_components(gaussian(cls), scalars)


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", COUPLED, ids=lambda c: c.__name__)
def test_a_batched_parameter_declines_too(backend_name, cls):
    """A per-simulation parameter cannot be passed as a scalar either."""
    solver = make(cls, backend_name)
    batched_alpha = np.array([1.0, 2.0, 3.0])
    scalars = (batched_alpha, 20, -1.6e-9, -1e-10, -1.6e-9, 10e4, 10e4)
    assert not solver._can_fuse_components(gaussian(cls), scalars)


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_a_non_local_run_declines_every_fused_path(backend_name, cls):
    """Non-locality smooths the intensity, which the fused kernels skip."""
    solver = make(cls, backend_name, nl_length=5 * window / N)
    assert solver.nl_length > 0
    assert not solver._can_fuse_rk4_stage(gaussian(cls))
    if coupled(cls):
        scalars = (20, 20, -1.6e-9, -1e-10, -1.6e-9, 10e4, 10e4)
        assert not solver._can_fuse_components(gaussian(cls), scalars)


@pytest.mark.parametrize("backend_name", FUSED_BACKENDS)
def test_a_batched_run_still_gives_the_same_answer(backend_name, monkeypatch):
    """Falling back must be a fallback, not a different answer."""
    if not _backend_builders()[backend_name][1].supports_coupled_batch:
        pytest.skip(f"{backend_name} refuses a batched coupled run outright")
    single = propagate(CNLSE, backend_name, "stage", monkeypatch)
    solver = make(CNLSE, backend_name)
    out = np.asarray(
        solver._backend.to_numpy(
            solver.out_field(
                np.stack([gaussian(CNLSE)] * 3),
                L,
                verbose=False,
                plot=False,
                splitting="lie",
                method="RK4",
            )
        )
    )
    for i in range(3):
        assert np.max(np.abs(out[i] - single)) / np.max(np.abs(single)) < 1e-5
