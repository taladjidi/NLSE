"""Tests that the rates reduced on the device are the rates.

The step comes from these, and the host version is what every step size in the
project was tuned against, so the device one has to agree with it rather than
merely look plausible: a rate wrong by a factor would change the step and
nothing would fail, it would just integrate something else.

The host path is kept for that reason -- it is the reference these are checked
against, and the fallback wherever the device one cannot apply.
"""

import numpy as np
import pytest
from helpers import make
from NLSE import CNLSE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST
SOLVERS = [NLSE, NLSE_1d, CNLSE, CNLSE_1d]


def one_dimensional(cls):
    """Whether this solver works on a line."""
    return cls in (NLSE_1d, CNLSE_1d)


def coupled(cls):
    """Whether this solver carries two components."""
    return cls in (CNLSE, CNLSE_1d)


def grid_shape(cls):
    """Shape of one component's grid."""
    return (N,) if one_dimensional(cls) else (N, N)


def prepared(cls, backend, potential=None):
    """Return a solver and a normalized field, ready for the rates."""
    solver = make(cls, backend, n=N, window=WINDOW, L=L, Isat=1e6, V=potential)
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    if one_dimensional(cls):
        r2 = x**2
    else:
        X, Y = np.meshgrid(x, x)
        r2 = X**2 + Y**2
    beam = np.exp(-r2 / WAIST**2).astype(np.complex64)
    field = np.stack([beam, beam * 0.5]) if coupled(cls) else beam
    prepared_field, _ = solver._prepare_output_array(field, normalize=True)
    solver._precompute_step_constants(solver.V, "double")
    solver.plans = solver._build_fft_plan(prepared_field)
    return solver, prepared_field


def worst_disagreement(device, host):
    """Largest relative difference between two sets of rates."""
    return max(
        abs(device[key] - host[key]) / max(abs(host[key]), 1e-30) for key in device
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["none", "real", "complex"])
def test_the_device_rates_are_the_host_rates(backend_name, cls, kind):
    """Every rate, every solver, every kind of potential."""
    shape = grid_shape(cls)
    potential = {
        "none": None,
        "real": (np.ones(shape) * 1e-3).astype(np.float32),
        "complex": (np.ones(shape) * 1e-3 + 1j * np.ones(shape) * 1e-4).astype(
            np.complex64
        ),
    }[kind]
    solver, field = prepared(cls, backend_name, potential)
    device = NLSE._energy_rates_on_device(solver, field)
    host = NLSE._energy_rates_on_host(solver, field)
    assert worst_disagreement(device, host) < 1e-5, (
        f"{cls.__name__} on {backend_name} with a {kind} potential: the rates "
        f"differ between the device and host paths by "
        f"{worst_disagreement(device, host):.2e}\n  device {device}\n  host   {host}"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_device_path_leaves_the_field_alone(backend_name):
    """It transforms a copy: the transform is in place and the field is live.

    The rates are read partway through a propagation, so a transform applied
    to the field itself would leave the run in the Fourier domain and produce
    no error at all -- just a different answer.
    """
    solver, field = prepared(NLSE, backend_name)
    before = np.asarray(solver._backend.to_numpy(field)).copy()
    NLSE._energy_rates_on_device(solver, field)
    after = np.asarray(solver._backend.to_numpy(field))
    assert np.array_equal(before, after), "the rates transformed the live field"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_batched_parameter_takes_the_host_path(backend_name):
    """The device path indexes one shape, so a value per simulation declines.

    Only CUPY broadcasts a batched parameter natively and pyopencl does not
    broadcast at all, so this is a correctness gate rather than a preference.
    """
    solver, field = prepared(NLSE, backend_name)
    assert solver._can_rate_on_device(field)
    solver._g = np.array([1.0, 2.0, 3.0])
    assert not solver._can_rate_on_device(field)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_rates_before_the_transforms_are_planned_take_the_host_path(backend_name):
    """It uses the plan the run already has, and may be asked before there is one."""
    solver, field = prepared(NLSE, backend_name)
    solver.plans = None
    assert not solver._can_rate_on_device(field)
    rates = solver._energy_rates(field)
    assert set(rates) == {"kinetic", "potential", "interaction", "loss"}
    assert all(np.isfinite(value) for value in rates.values())
