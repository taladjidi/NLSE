"""The Rabi coupling, under every splitting and RK4."""

import warnings

import numpy as np
import pytest
from helpers import ORDERS
from NLSE import CNLSE, DDGPE, CNLSE_1d
from scipy.linalg import expm

# CNLSE: coupling in rad/m and distance in m, giving omega z / 2 = 1 rad.
OMEGA = 200.0
Z = 1e-2
WINDOW = 1e-3  # m
# DDGPE, in the units of its rates: coupling, total time, cavity and exciton
# energy W, with detuning = OMEGA_DD / 2 putting the pump at W.
OMEGA_DD = 2.0
T_DD = 1.0
W_DD = 2.0
EXACT_RATIO = float(np.tan(1.0) ** 2)  # tan(omega z / 2)**2 at omega z / 2 = 1
# Measured spread of P2 / P1 around EXACT_RATIO, float32 on MLX the widest.
RATIO_RTOL = 1e-5
# Plane-wave index along x: it sits well inside the band of either grid.
MODE_CNLSE = 6
MODE_DDGPE = 4
# The measured slope is within ORDER_TOL of each method's ORDERS entry.
ORDER_TOL = 0.1
# Step counts for the order: Yoshida reaches its asymptotic order from 16.
STEPS = {"lie": (8, 16), "strang": (8, 16), "yoshida": (16, 32), "RK4": (8, 16)}
METHODS = [
    ("lie", "split_step"),
    ("strang", "split_step"),
    ("yoshida", "split_step"),
    ("lie", "RK4"),
]
# Steps of the lower-polariton test, and its tolerance: measured worst case
# 5.4e-7 (MLX, float32). A pump frame off by omega / 4 turns the mode by
# 0.5 rad over T_DD, a change of 0.49 in each unit amplitude.
POLARITON_STEPS = 8
POLARITON_ATOL = 5e-6


def coupled(cls, backend, n):
    """Return a CNLSE-type solver with only the Rabi coupling and dispersion."""
    shape = {"NX": n} if cls is CNLSE_1d else {"NX": n, "NY": n}
    return cls(
        alpha=0,
        power=1,
        window=WINDOW,
        n2=0,
        n12=0,
        V=None,
        L=Z,
        omega=OMEGA,
        backend=backend,
        **shape,
    )


def ddgpe(backend, detuning=OMEGA_DD / 2):
    """Return a lossless, undriven, interaction-free DDGPE.

    Parameters
    ----------
    backend : str
        Backend name.
    detuning : float
        Pump detuning from the lower polariton, in rad per unit time.

    Returns
    -------
    DDGPE
        The solver.
    """
    return DDGPE(
        gamma=0,
        power=1,
        window=32,
        g=0,
        omega=OMEGA_DD,
        T=T_DD,
        omega_exc=W_DD,
        omega_cav=W_DD,
        detuning=detuning,
        k_z=1.0,
        NX=32,
        NY=32,
        backend=backend,
    )


def no_drive(simu, steps):
    """Return laser_excitation arguments that add nothing, one sample a step."""
    zero = np.zeros((simu.NY, simu.NX), np.complex64)
    return [(zero, np.zeros(steps), zero, np.zeros(steps))]


def run_coupled(simu, field, steps, splitting, method):
    """Propagate over Z and return the field on the host."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = simu.out_field(
            field,
            Z,
            delta_z=Z / steps,
            splitting=splitting,
            method=method,
            verbose=False,
            normalize=False,
        )
    return np.asarray(simu._backend.to_numpy(out)).astype(np.complex128)


def run_ddgpe(simu, field, steps, splitting):
    """Propagate over T_DD and return the field on the host."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = simu.out_field(
            field,
            T_DD,
            None,
            delta_z=T_DD / steps,
            splitting=splitting,
            verbose=False,
            callback_args=no_drive(simu, steps),
        )
    return np.asarray(simu._backend.to_numpy(out)).astype(np.complex128)


def ratio(field):
    """Return the power of the second component over the first's."""
    return float(np.mean(np.abs(field[1]) ** 2) / np.mean(np.abs(field[0]) ** 2))


def plane_wave(n, m):
    """Return the (n, n) Fourier mode of index (0, m), modulus one."""
    spectrum = np.zeros((n, n), complex)
    spectrum[0, m] = n * n
    return np.fft.ifft2(spectrum)


def field_dtype(simu):
    """Complex128 where the backend has it, so round-off does not hide the order."""
    return np.complex128 if simu._backend.supports_double_precision() else np.complex64


def relative_error(got, reference):
    """Relative L2 difference."""
    return float(np.linalg.norm(got - reference) / np.linalg.norm(reference))


@pytest.mark.parametrize("cls", [CNLSE, CNLSE_1d])
@pytest.mark.parametrize("splitting, method", METHODS)
def test_power_exchange_matches_tan_squared(backend, cls, splitting, method):
    """A uniform field in component 1 ends with P2 / P1 = tan(omega z / 2)**2."""
    simu = coupled(cls, backend, 32)
    grid = (32,) if cls is CNLSE_1d else (32, 32)
    field = np.zeros((2, *grid), np.complex64)
    field[0] = 1
    out = run_coupled(simu, field, 100, splitting, method)
    assert ratio(out) == pytest.approx(EXACT_RATIO, rel=RATIO_RTOL), (
        f"{splitting}/{method} skips or misapplies the Rabi coupling"
    )


@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
def test_ddgpe_power_exchange_matches_tan_squared(backend, splitting):
    """A uniform cavity field ends with exciton / cavity = tan(omega t / 2)**2."""
    simu = ddgpe(backend)
    field = np.zeros((2, 32, 32), np.complex64)
    field[1] = 1
    out = run_ddgpe(simu, field, 100, splitting)
    assert 1 / ratio(out) == pytest.approx(EXACT_RATIO, rel=RATIO_RTOL), (
        f"{splitting} skips or misapplies the Rabi coupling"
    )


def cnlse_reference(simu, mode):
    """Return the exact coupled plane wave after Z, component 1 lit at the start."""
    q2 = (simu.Kxx**2 + simu.Kyy**2)[0, MODE_CNLSE]
    H = np.array([[q2 / (2 * simu.k), OMEGA / 2], [OMEGA / 2, q2 / (2 * simu.k2)]])
    a1, a2 = expm(-1j * Z * H) @ np.array([1, 0])
    return np.stack([a1 * mode, a2 * mode])


@pytest.mark.parametrize("splitting, method", METHODS)
def test_coupled_order_against_closed_form(backend, splitting, method):
    """Halving the step divides the error by 2**order."""
    name = "RK4" if method == "RK4" else splitting
    errors = []
    for steps in STEPS[name]:
        simu = coupled(CNLSE, backend, 16)
        if name == "yoshida" and not simu._backend.supports_double_precision():
            pytest.skip("fourth-order error is under float32 round-off here")
        simu.k2 = simu.k / 2
        mode = plane_wave(16, MODE_CNLSE)
        field = np.stack([mode, 0 * mode]).astype(field_dtype(simu))
        out = run_coupled(simu, field, steps, splitting, method)
        errors.append(relative_error(out, cnlse_reference(simu, mode)))
    order = np.log2(errors[0] / errors[1])
    assert order == pytest.approx(ORDERS[name], abs=ORDER_TOL), (
        f"{name} converges at order {order:.2f}, errors {errors}"
    )


def ddgpe_reference(simu, mode):
    """Return the exact polariton plane wave after T_DD, cavity lit at the start."""
    q2 = (simu.Kxx**2 + simu.Kyy**2)[0, MODE_DDGPE]
    exciton = simu.omega_exc - simu.omega_pump
    cavity = simu.omega_cav * np.sqrt(1 + q2 / simu.k_z**2) - simu.omega_pump
    H = np.array([[exciton, OMEGA_DD / 2], [OMEGA_DD / 2, cavity]])
    a1, a2 = expm(-1j * T_DD * H) @ np.array([0, 1])
    return np.stack([a1 * mode, a2 * mode])


@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
def test_ddgpe_order_against_closed_form(backend, splitting):
    """Halving the step divides the error by 2**order, cavity dispersion on."""
    errors = []
    for steps in STEPS[splitting]:
        simu = ddgpe(backend)
        if splitting == "yoshida" and not simu._backend.supports_double_precision():
            pytest.skip("fourth-order error is under float32 round-off here")
        mode = plane_wave(32, MODE_DDGPE)
        field = np.stack([0 * mode, mode]).astype(field_dtype(simu))
        out = run_ddgpe(simu, field, steps, splitting)
        errors.append(relative_error(out, ddgpe_reference(simu, mode)))
    order = np.log2(errors[0] / errors[1])
    assert order == pytest.approx(ORDERS[splitting], abs=ORDER_TOL), (
        f"{splitting} converges at order {order:.2f}, errors {errors}"
    )


def test_the_lower_polariton_is_at_rest_in_the_pump_frame(backend):
    """At zero detuning the pump turns at the lower polariton's frequency."""
    simu = ddgpe(backend, detuning=0.0)
    mode = plane_wave(simu.NX, 0)
    field = (np.stack([mode, -mode]) / np.sqrt(2)).astype(field_dtype(simu))
    out = run_ddgpe(simu, field.copy(), POLARITON_STEPS, "strang")
    np.testing.assert_allclose(out, field, rtol=0, atol=POLARITON_ATOL)
