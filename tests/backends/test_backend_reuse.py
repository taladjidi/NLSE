"""Backends are shared between solvers rather than reopened per solver."""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 32
WAIST = 2.23e-3
BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 4 * WAIST,
    "n2": -1.6e-9,
    "V": None,
    "L": 10e-3,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_get_backend_returns_the_same_instance(backend_name):
    """Repeated lookups must not build a second backend."""
    assert get_backend(backend_name) is get_backend(backend_name)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_solvers_share_one_backend(backend_name):
    """Building many solvers must not open a device connection each time.

    A backend owns a device context, not per-simulation state. Handing out a
    fresh one per solver leaked an OpenCL context and command queue that are
    never released, and a parameter sweep — or a long test session — then ran
    the process out of file descriptors. That surfaces as unrelated later
    work failing in setup with "Too many open files", so it has to be pinned
    here rather than left to be rediscovered.
    """
    solvers = [NLSE(backend=backend_name, **BASE) for _ in range(25)]
    backends = {id(solver._backend) for solver in solvers}
    assert len(backends) == 1, (
        f"{len(solvers)} solvers hold {len(backends)} distinct {backend_name} "
        f"backends; each one is a device context that is never released"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_shared_backend_still_propagates_correctly(backend_name):
    """Sharing must not couple solvers: each keeps its own results."""
    a = NLSE(backend=backend_name, **BASE)
    b = NLSE(backend=backend_name, **{**BASE, "n2": -1e-10})
    assert a._backend is b._backend, "precondition: the backend is shared"

    field = np.exp(-(a.XX**2 + a.YY**2) / WAIST**2).astype(np.complex64)
    out_a = a.out_field(field.copy(), 1e-3, verbose=False, plot=False, delta_z=1e-4)
    out_b = b.out_field(field.copy(), 1e-3, verbose=False, plot=False)

    def host(simu, arr):
        return arr if isinstance(arr, np.ndarray) else simu._backend.to_numpy(arr)

    assert not np.allclose(np.asarray(host(a, out_a)), np.asarray(host(b, out_b))), (
        "two solvers with different n2 gave the same result: sharing a "
        "backend leaked state between them"
    )
