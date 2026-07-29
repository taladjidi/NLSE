import numpy as np
from NLSE import NLSE_1d
from scipy.constants import c, epsilon_0

from .helpers import as_numpy, assert_c_contiguous

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_solver(backend="CPU", n=N, **overrides):
    """Return a NLSE_1d with this module's parameters.

    Parameters
    ----------
    backend : str
        Backend name.
    n : int
        Grid size.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    NLSE_1d
        The solver.
    """
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": None,
        "L": L,
        "NX": n,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(overrides)
    return NLSE_1d(**params)


def test_build_propagator(backend) -> None:
    simu = NLSE_1d(alpha, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    prop = simu._build_propagator(PRECISION_COMPLEX, DZ_TEST)
    assert np.allclose(prop, np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * DZ_TEST)), (
        f"Propagator is wrong. (Backend {backend})"
    )


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
    A = np.ones(N, dtype=PRECISION_COMPLEX)
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    if backend == "CPU":
        assert out.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert out_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    out_np = as_numpy(simu, out)
    integral = (np.abs(out_np) ** 2 * simu.delta_X**2).sum()
    integral *= c * epsilon_0 / 2
    assert np.allclose(integral, simu.power, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
    assert out_np.shape == (N,), f"Output array has wrong shape. (Backend {backend})"
    np.testing.assert_allclose(
        out_np / np.max(np.abs(out_np)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_split_step(backend) -> None:
    simu = NLSE_1d(alpha, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    E = np.ones((N,), dtype=PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, 0, precision="double"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        np.ones((N,), dtype=PRECISION_COMPLEX),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Split step is not unitary. (Backend {backend})",
    )


def test_out_field(backend) -> None:
    simu = NLSE_1d(0, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    E0 = np.ones(N, dtype=PRECISION_COMPLEX)
    A = simu.out_field(
        E0, DZ_TEST, delta_z=DZ_TEST, verbose=False, plot=False, precision="single"
    )
    rho = A.real * A.real + A.imag * A.imag
    norm = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
    norm *= c * epsilon_0 / 2
    assert A.shape == (N,), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(norm, power, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
