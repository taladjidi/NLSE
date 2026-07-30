"""Tests for the fused coupled path: same answer, and no copies.

A coupled solver holds its two components in one ``(2, ...)`` array. Reaching
a kernel that takes one component means copying each out and the result back,
which on a device backend is real traffic and no arithmetic -- 36 complex and
10 real array copies per step at the profiled workload. The interleaved
kernels read both components from the one array instead.

Two things are worth pinning, and the second is the one a later change is
likely to break silently:

- the fused path must agree with the generic one it replaces, and
- it must actually be taken. A gate that quietly stops matching leaves every
  correctness test passing and gives the copies back.
"""

import numpy as np
import pytest
from NLSE import CNLSE, CNLSE_1d
from NLSE.backends import get_backend, list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

# A beam, not noise. A coupled RK4 run under a strong nonlinearity amplifies a
# last-bit difference into a visible one, so an adversarial field measures the
# chaos rather than the kernels.
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3
alpha = 20

N = 64

COUPLED_BACKENDS = [
    name
    for name in AVAILABLE_BACKENDS
    if get_backend(name).has_fused_coupled_split_step
    or get_backend(name).has_fused_coupled_rk4_rhs
]

pytestmark = pytest.mark.skipif(
    not COUPLED_BACKENDS, reason="no backend declares a fused coupled path"
)


def gaussian_pair(cls, n):
    """Return a two-component Gaussian field for this solver class."""
    x = np.linspace(-window / 2, window / 2, n)
    if cls is CNLSE:
        X, Y = np.meshgrid(x, x)
        r2 = X**2 + Y**2
    else:
        r2 = x**2
    return np.stack([np.exp(-r2 / waist**2), np.exp(-r2 / waist2**2)]).astype(
        np.complex64
    )


def make(cls, backend, n=N, **overrides):
    """Return a coupled solver with this module's parameters."""
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "n12": n12,
        "V": None,
        "L": L,
        "NX": n,
        "Isat": Isat,
        "backend": backend,
    }
    if cls is CNLSE:
        params["NY"] = n
    params.update(overrides)
    return cls(**params)


def propagate(cls, backend, method, fused, monkeypatch, **overrides):
    """Run one propagation with the fused coupled path on or off."""
    solver = make(cls, backend, **overrides)
    for flag in ("has_fused_coupled_split_step", "has_fused_coupled_rk4_rhs"):
        monkeypatch.setattr(type(solver._backend), flag, fused, raising=False)
    out = solver.out_field(
        gaussian_pair(cls, overrides.get("n", N)),
        L,
        verbose=False,
        plot=False,
        precision="single",
        method=method,
    )
    return np.asarray(solver._backend.to_numpy(out))


def relative(a, b):
    """Largest difference, against the largest value."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
@pytest.mark.parametrize("cls", [CNLSE, CNLSE_1d])
@pytest.mark.parametrize("method", ["split_step", "RK4"])
@pytest.mark.parametrize("potential", ["none", "real", "complex"])
def test_the_fused_path_agrees_with_the_generic_one(
    backend_name, cls, method, potential, monkeypatch
):
    """Fusing the components must not change the answer."""
    shape = (N, N) if cls is CNLSE else (N,)
    V = {
        "none": None,
        "real": (np.ones(shape) * 1e-3).astype(np.float32),
        "complex": (np.ones(shape) * 1e-3 + 1j * np.ones(shape) * 1e-4).astype(
            np.complex64
        ),
    }[potential]
    fused = propagate(cls, backend_name, method, True, monkeypatch, V=V)
    generic = propagate(cls, backend_name, method, False, monkeypatch, V=V)
    assert relative(fused, generic) < 1e-5


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
def test_the_fused_split_step_carries_the_rabi_coupling(backend_name, monkeypatch):
    """The interleaved Rabi rotation must match the per-component one."""
    solver = make(CNLSE, backend_name)
    solver.omega = 1e4
    fused = propagate(CNLSE, backend_name, "split_step", True, monkeypatch)

    def with_omega(fused_flag):
        s = make(CNLSE, backend_name)
        s.omega = 1e4
        for flag in ("has_fused_coupled_split_step", "has_fused_coupled_rk4_rhs"):
            monkeypatch.setattr(type(s._backend), flag, fused_flag, raising=False)
        return np.asarray(
            s._backend.to_numpy(
                s.out_field(
                    gaussian_pair(CNLSE, N),
                    L,
                    verbose=False,
                    plot=False,
                    precision="single",
                    method="split_step",
                )
            )
        )

    assert relative(with_omega(True), with_omega(False)) < 1e-5
    assert fused is not None


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_the_fused_path_copies_no_components(backend_name, method, monkeypatch):
    """The point of the fused path is that it splits nothing.

    Correctness tests cannot see this: the generic path gives the same
    answer, so a gate that stops matching costs only speed, silently. This
    asserts the copies are not made.
    """
    solver = make(CNLSE, backend_name)
    calls = []
    for name in ("_take_components", "_set_components"):
        original = getattr(type(solver), name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(type(solver), name, spy)

    solver.out_field(
        gaussian_pair(CNLSE, N),
        L,
        verbose=False,
        plot=False,
        precision="single",
        method=method,
    )
    assert not calls, (
        f"{backend_name} declares a fused coupled path but {method} still "
        f"split the field: {sorted(set(calls))}"
    )


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
def test_a_batch_falls_back_rather_than_being_fused(backend_name):
    """The interleaved kernels take scalars, so a batch must decline them.

    CL and MLX refuse a batched coupled run outright; CUPY serves one
    through the generic path. Either way the fused gate must say no.
    """
    solver = make(CNLSE, backend_name)
    batched = np.stack([gaussian_pair(CNLSE, N)] * 3)
    scalars = (alpha, alpha, n2, n12, n2, Isat, Isat)
    assert not solver._can_fuse_components(batched, scalars)
    assert solver._can_fuse_components(gaussian_pair(CNLSE, N), scalars)


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
def test_a_batched_parameter_falls_back_too(backend_name):
    """A per-simulation parameter cannot be passed as a scalar either."""
    solver = make(CNLSE, backend_name)
    field = gaussian_pair(CNLSE, N)
    batched_alpha = np.array([1.0, 2.0, 3.0])
    scalars = (batched_alpha, alpha, n2, n12, n2, Isat, Isat)
    assert not solver._can_fuse_components(field, scalars)


@pytest.mark.parametrize("backend_name", COUPLED_BACKENDS)
def test_a_non_local_run_falls_back(backend_name):
    """Non-locality convolves the intensity, which the fused kernels skip."""
    backend = get_backend(backend_name)
    if backend.convolution is None:
        pytest.skip(f"{backend_name} has no convolution, so no non-local run")
    solver = make(CNLSE, backend_name, nl_length=5 * window / N)
    scalars = (alpha, alpha, n2, n12, n2, Isat, Isat)
    assert solver.nl_length > 0
    assert not solver._can_fuse_components(gaussian_pair(CNLSE, N), scalars)
