"""Tests that the rates reduced on the device are the rates."""

import warnings
from unittest import mock

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, make
from NLSE import CNLSE, NLSE, CNLSE_1d, NLSE_1d

N = 64
L = 1e-3
WAIST = 2.23e-3
WINDOW = 4 * WAIST
SOLVERS = [NLSE, NLSE_1d, CNLSE, CNLSE_1d]
# Index change of the potential that is -SIGNED_V on one half of the window
# and +SIGNED_V on the other: its mean over a centred beam vanishes.
SIGNED_V = 1e-3
# Its rate against its magnitude's: equal but for float32 sums (measured
# below 1e-6, CPU).
SIGNED_RTOL = 1e-5
# Short enough that a call is a handful of steps on any backend.
CALL_DISTANCE = L / 100


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
    solver._precompute_step_constants(solver.V, np.complex64)
    solver.plans = solver._build_fft_plan(prepared_field)
    return solver, prepared_field


def signed_potential(cls):
    """Return a potential of both signs, antisymmetric across the window."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    step = np.where(x < 0, -SIGNED_V, SIGNED_V).astype(np.float32)
    return np.broadcast_to(step, grid_shape(cls)).copy()


def worst_disagreement(device, host):
    """Largest relative difference between two sets of rates."""
    return max(
        abs(device[key] - host[key]) / max(abs(host[key]), 1e-30) for key in device
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("kind", ["none", "real", "complex", "signed"])
def test_the_device_rates_are_the_host_rates(backend_name, cls, kind):
    """Every rate, every solver, every kind of potential."""
    shape = grid_shape(cls)
    potential = {
        "none": None,
        "real": (np.ones(shape) * 1e-3).astype(np.float32),
        "complex": (np.ones(shape) * 1e-3 + 1j * np.ones(shape) * 1e-4).astype(
            np.complex64
        ),
        "signed": signed_potential(cls),
    }[kind]
    solver, field = prepared(cls, backend_name, potential)
    device = NLSE._energy_rates_on_device(solver, field)
    host = NLSE._energy_rates_on_host(solver, field)
    assert worst_disagreement(device, host) < 1e-5, (
        f"{cls.__name__} on {backend_name} with a {kind} potential: the rates "
        f"differ between the device and host paths by "
        f"{worst_disagreement(device, host):.2e}\n  device {device}\n  host   {host}"
    )


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_a_signed_potential_counts_by_its_magnitude(cls):
    """A potential whose mean over the beam vanishes still sets the rate."""
    signed = signed_potential(cls)
    rates = [
        solver._energy_rates(field)["potential"]
        for solver, field in (
            prepared(cls, "CPU", signed),
            prepared(cls, "CPU", np.abs(signed)),
        )
    ]
    assert rates[1] > 0, "precondition: the potential turns the phase"
    assert rates[0] == pytest.approx(rates[1], rel=SIGNED_RTOL), (
        f"{cls.__name__}: a signed potential rates {rates[0]:.3e} rad/m, its "
        f"magnitude {rates[1]:.3e}"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_device_path_leaves_the_field_alone(backend_name):
    """The device path transforms a copy and leaves the field alone."""
    solver, field = prepared(NLSE, backend_name)
    before = np.asarray(solver._backend.to_numpy(field)).copy()
    NLSE._energy_rates_on_device(solver, field)
    after = np.asarray(solver._backend.to_numpy(field))
    assert np.array_equal(before, after), "the rates transformed the live field"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_batched_parameter_takes_the_host_path(backend_name):
    """A batched parameter takes the host path."""
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


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", [NLSE, CNLSE], ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("method", "splitting"),
    [("split_step", "lie"), ("RK4", "lie"), ("split_step", "yoshida")],
)
@pytest.mark.parametrize("delta_z", [None, CALL_DISTANCE / 4], ids=["default", "given"])
def test_a_call_reduces_the_rates_once(backend_name, cls, method, splitting, delta_z):
    """The field does not change before the run starts, so neither do its rates.

    The default step, the method's cap and Yoshida's span all read them. Each
    used to reduce them again, and on the coupled solver every reduction
    brings the whole field back to the host.
    """
    solver, field = prepared(cls, backend_name)
    host_field = np.asarray(solver._backend.to_numpy(field))
    with (
        mock.patch.object(
            cls, "_energy_rates", autospec=True, side_effect=cls._energy_rates
        ) as rates,
        warnings.catch_warnings(),
    ):
        # Yoshida in single precision warns that it is not worth it.
        warnings.simplefilter("ignore")
        solver.out_field(
            host_field,
            CALL_DISTANCE,
            delta_z=delta_z,
            method=method,
            splitting=splitting,
            verbose=False,
        )
    assert rates.call_count == 1, (
        f"{cls.__name__} {method}/{splitting} on {backend_name} reduced the "
        f"rates {rates.call_count} times in one call"
    )
