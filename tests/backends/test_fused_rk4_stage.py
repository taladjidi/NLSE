"""Tests for the whole-stage RK4 path.

An RK4 stage otherwise writes its slope to memory and the stage update reads
it straight back. The fused kernels finish the slope in registers and spend it
on the accumulator and the next stage's argument, which is two fewer accesses
per element.

The arithmetic is unchanged, so these runs are expected to agree with the
generic path *bit for bit*, not merely to within a tolerance: anything looser
would hide a re-association, which is how a fused kernel usually goes wrong.
"""

import numpy as np
import pytest
from NLSE import CNLSE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.backends import get_backend, list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

STAGE_BACKENDS = [
    name for name in AVAILABLE_BACKENDS if get_backend(name).has_fused_rk4_stage
]

pytestmark = pytest.mark.skipif(
    not STAGE_BACKENDS, reason="no backend declares a fused RK4 stage"
)

N = 64
L = 1e-3
window = 8.92e-3
waist = 2.23e-3
SOLVERS = [NLSE, NLSE_1d, CNLSE, CNLSE_1d]


def one_dimensional(cls):
    """Whether this solver class works on a line rather than a plane."""
    return cls in (NLSE_1d, CNLSE_1d)


def coupled(cls):
    """Whether this solver class carries two components."""
    return cls in (CNLSE, CNLSE_1d)


def grid_shape(cls):
    """Shape of one component's grid."""
    return (N,) if one_dimensional(cls) else (N, N)


def build(cls, backend, **overrides):
    """Return a solver of this class with this module's parameters."""
    params = {
        "alpha": 20,
        "power": 1.05,
        "window": window,
        "n2": -1.6e-9,
        "V": None,
        "L": L,
        "NX": N,
        "Isat": 1e5,
        "backend": backend,
    }
    if coupled(cls):
        params["n12"] = -1e-10
    if not one_dimensional(cls):
        params["NY"] = N
    params.update(overrides)
    return cls(**params)


def initial_field(cls, dtype=np.complex64):
    """Return a smooth field of the shape this solver takes."""
    x = np.linspace(-window / 2, window / 2, N)
    if one_dimensional(cls):
        r2 = x**2
    else:
        X, Y = np.meshgrid(x, x)
        r2 = X**2 + Y**2
    single = np.exp(-r2 / waist**2)
    if coupled(cls):
        return np.stack([single, single]).astype(dtype)
    return single.astype(dtype)


def propagate(cls, backend, fused, monkeypatch, dtype=np.complex64, **overrides):
    """Run one RK4 propagation with the whole-stage path on or off."""
    solver = build(cls, backend, **overrides)
    monkeypatch.setattr(
        type(solver._backend), "has_fused_rk4_stage", fused, raising=False
    )
    out = solver.out_field(
        initial_field(cls, dtype),
        L,
        verbose=False,
        plot=False,
        precision="single",
        method="RK4",
    )
    return np.asarray(solver._backend.to_numpy(out))


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("potential", ["none", "real", "complex"])
def test_the_fused_stage_is_bitwise_identical(
    backend_name, cls, potential, monkeypatch
):
    """Fusing the stage must not re-associate the arithmetic."""
    shape = grid_shape(cls)
    V = {
        "none": None,
        "real": (np.ones(shape) * 1e-3).astype(np.float32),
        "complex": (np.ones(shape) * 1e-3 + 1j * np.ones(shape) * 1e-4).astype(
            np.complex64
        ),
    }[potential]
    fused = propagate(cls, backend_name, True, monkeypatch, V=V)
    generic = propagate(cls, backend_name, False, monkeypatch, V=V)
    assert np.array_equal(fused, generic), (
        f"{cls.__name__} with a {potential} potential differs between the "
        f"fused stage and the generic path by "
        f"{np.max(np.abs(fused - generic)) / np.max(np.abs(generic)):.3e}"
    )


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_the_fused_stage_is_bitwise_identical_in_double(backend_name, cls, monkeypatch):
    """The same, at the other precision."""
    solver = build(cls, backend_name)
    if not solver._backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision")
    fused = propagate(cls, backend_name, True, monkeypatch, dtype=np.complex128)
    generic = propagate(cls, backend_name, False, monkeypatch, dtype=np.complex128)
    assert np.array_equal(fused, generic)


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_every_stage_goes_through_the_fused_kernel(backend_name, cls, monkeypatch):
    """Four stages a step, and no separate slope buffer written.

    Correctness cannot see this: the generic path gives the same answer, so a
    gate that stops matching costs only speed, silently.
    """
    solver = build(cls, backend_name)
    kernels = type(solver._backend.kernels)
    fused_name = "rk4_stage_coupled_fused" if coupled(cls) else "rk4_stage_fused"
    counts: dict[str, int] = {}
    for name in (fused_name, "rk4_rhs_fused", "rk4_rhs_coupled_fused"):
        original = getattr(kernels, name, None)
        if original is None:
            continue

        def spy(self, *args, _name=name, _original=original, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(kernels, name, spy)

    solver.out_field(
        initial_field(cls),
        L,
        verbose=False,
        plot=False,
        precision="single",
        method="RK4",
    )
    assert counts.get(fused_name, 0) > 0, (
        f"{cls.__name__} on {backend_name} declares a fused RK4 stage but "
        f"never called {fused_name}: {counts}"
    )
    assert counts.get(fused_name, 0) % 4 == 0, "RK4 takes four stages a step"
    for separate in ("rk4_rhs_fused", "rk4_rhs_coupled_fused"):
        assert separate not in counts, (
            f"{cls.__name__} still went through {separate}, so the slope was "
            f"written to memory and read back"
        )


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_a_batch_declines_the_fused_stage(backend_name, cls):
    """The kernels take scalars and one field, so a batch must fall back."""
    solver = build(cls, backend_name)
    batched = np.stack([initial_field(cls)] * 3)
    assert not solver._can_fuse_rk4_stage(batched)
    assert solver._can_fuse_rk4_stage(initial_field(cls))


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_a_non_local_run_declines_the_fused_stage(backend_name, cls):
    """The fused kernels take the intensity from registers, unconvolved."""
    backend = get_backend(backend_name)
    if backend.convolution is None:
        pytest.skip(f"{backend_name} has no convolution")
    solver = build(cls, backend_name, nl_length=5 * window / N)
    assert solver.nl_length > 0
    assert not solver._can_fuse_rk4_stage(initial_field(cls))


@pytest.mark.parametrize("backend_name", STAGE_BACKENDS)
def test_a_batched_run_still_gives_the_same_answer(backend_name, monkeypatch):
    """Falling back must be a fallback, not a different answer.

    Each simulation of a batch must match the same simulation run alone
    through the fused path.
    """
    single = propagate(CNLSE, backend_name, True, monkeypatch)
    solver = build(CNLSE, backend_name)
    batched = np.stack([initial_field(CNLSE)] * 3)
    out = np.asarray(
        solver._backend.to_numpy(
            solver.out_field(
                batched,
                L,
                verbose=False,
                plot=False,
                precision="single",
                method="RK4",
            )
        )
    )
    for i in range(3):
        assert np.max(np.abs(out[i] - single)) / np.max(np.abs(single)) < 1e-5
