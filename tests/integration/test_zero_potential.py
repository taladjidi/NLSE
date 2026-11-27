"""A zero potential must propagate exactly like no potential at all."""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import CNLSE, GPE, NLSE

N = 64
WAIST = 2.23e-3
Z = 2e-3
DELTA_Z = 1e-4

BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 4 * WAIST,
    "n2": -1.6e-9,
    "L": 10e-3,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}


def field(simu, dtype=np.complex64):
    """Return a Gaussian input field on this solver's grid."""
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(dtype)


def host(simu, out):
    """Return a propagated field as numpy, whatever backend produced it."""
    if isinstance(out, np.ndarray):
        return out
    return np.asarray(simu._backend.to_numpy(out))


def propagate(simu, E, **kwargs):
    """Propagate E and return the result as numpy."""
    out = simu.out_field(
        E.copy(),
        Z,
        verbose=False,
        plot=False,
        normalize=False,
        **kwargs,
        delta_z=DELTA_Z,
    )
    return host(simu, out)


def assert_same(without_V, with_zero_V, what):
    """Assert the two paths agree, with a message naming the two kernels."""
    scale = float(np.max(np.abs(without_V)))
    np.testing.assert_allclose(
        with_zero_V,
        without_V,
        rtol=1e-6,
        atol=1e-7 * scale,
        err_msg=(
            f"{what}: propagating under V = 0 differs from propagating with no "
            f"V at all, so the two spellings of the kernel have drifted apart"
        ),
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_nlse_zero_potential_matches_no_potential(backend_name, method):
    """NLSE's nl_prop and nl_prop_without_V must agree at V = 0."""
    bare = NLSE(V=None, backend=backend_name, **BASE)
    E = field(bare)
    zero = NLSE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **BASE)

    assert_same(
        propagate(bare, E, method=method),
        propagate(zero, E, method=method),
        f"NLSE[{backend_name}, {method}]",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_nlse_zero_potential_matches_under_saturation(backend_name):
    """A zero potential matches no potential under saturation too."""
    params = {**BASE, "Isat": 1e2}
    bare = NLSE(V=None, backend=backend_name, **params)
    E = field(bare)
    zero = NLSE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **params)

    assert_same(
        propagate(bare, E),
        propagate(zero, E),
        f"NLSE[{backend_name}] under saturation",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_nlse_zero_potential_matches_with_loss(backend_name):
    """A zero potential matches no potential under loss too."""
    params = {**BASE, "alpha": 1.0}
    bare = NLSE(V=None, backend=backend_name, **params)
    E = field(bare)
    zero = NLSE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **params)

    assert_same(
        propagate(bare, E),
        propagate(zero, E),
        f"NLSE[{backend_name}] with loss",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_gpe_zero_potential_matches_no_potential(backend_name):
    """GPE reaches the same kernels through different physical parameters."""
    params = {
        "gamma": 0.0,
        "N": 1e6,
        "window": 100e-6,
        "g": 1e-11,  # rad/s*m^2: g*n = 1e3 rad/s at 1e14 atoms/m^2
        "m": 87 * 1.66e-27,
        "NX": N,
        "NY": N,
    }
    bare = GPE(V=None, backend=backend_name, **params)
    E = field(bare)
    zero = GPE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **params)

    assert_same(
        host(
            bare,
            bare.out_field(E.copy(), 1e-6, verbose=False, plot=False, delta_z=1e-8),
        ),
        host(
            zero,
            zero.out_field(E.copy(), 1e-6, verbose=False, plot=False, delta_z=1e-8),
        ),
        f"GPE[{backend_name}]",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_cnlse_zero_potential_matches_no_potential(backend_name, method):
    """The coupled kernels have their own no-V twins, on both components."""
    params = {**BASE, "n12": -1e-10}
    bare = CNLSE(V=None, backend=backend_name, **params)
    E = np.zeros((2, N, N), dtype=np.complex64)
    E[0] = field(bare)
    E[1] = 0.5 * field(bare)
    zero = CNLSE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **params)

    assert_same(
        propagate(bare, E, method=method),
        propagate(zero, E, method=method),
        f"CNLSE[{backend_name}, {method}]",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_cnlse_zero_potential_matches_under_rabi_coupling(backend_name):
    """A zero potential matches no potential in CNLSE under Rabi coupling."""
    params = {**BASE, "n12": -1e-10, "omega": 1e3}
    bare = CNLSE(V=None, backend=backend_name, **params)
    E = np.zeros((2, N, N), dtype=np.complex64)
    E[0] = field(bare)
    E[1] = 0.5 * field(bare)
    zero = CNLSE(V=np.zeros((N, N), dtype=np.float32), backend=backend_name, **params)

    assert_same(
        propagate(bare, E),
        propagate(zero, E),
        f"CNLSE[{backend_name}] under Rabi coupling",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_nonzero_potential_does_change_the_result(backend_name):
    """Precondition: the checks above are not trivially satisfied.

    If V were ignored entirely, every assertion in this file would pass.
    """
    bare = NLSE(V=None, backend=backend_name, **BASE)
    E = field(bare)
    r = np.sqrt(bare.XX**2 + bare.YY**2)
    V = (1e-4 * np.exp(-(r**2) / (1e-3) ** 2)).astype(np.float32)
    lensed = NLSE(V=V, backend=backend_name, **BASE)

    assert not np.allclose(propagate(bare, E), propagate(lensed, E)), (
        f"{backend_name}: a nonzero potential left the field unchanged, so the "
        f"V-reading kernels are not reading V"
    )
