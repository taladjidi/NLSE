import numpy as np
from NLSE import CNLSE_1d
from scipy.constants import c, epsilon_0

from .helpers import as_numpy, assert_c_contiguous

if CNLSE_1d.__CUPY_AVAILABLE__:
    pass
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def test_build_propagator(backend) -> None:
    simu = CNLSE_1d(
        alpha,
        power,
        window,
        n2,
        n12,
        None,
        L,
        NX=N,
        Isat=Isat,
        backend=backend,
    )
    prop = simu._build_propagator()
    prop1 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * simu.delta_z)
    prop2 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k2 * simu.delta_z)
    assert np.allclose(prop, np.array([prop1, prop2])), (
        f"Propagator is wrong. (Backend {backend})"
    )


def test_prepare_output_array(backend) -> None:
    simu = CNLSE_1d(
        alpha,
        power,
        window,
        n2,
        n12,
        None,
        L,
        NX=N,
        Isat=Isat,
        backend=backend,
    )
    A = np.ones((2, N), dtype=PRECISION_COMPLEX)
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    if backend == "CPU":
        assert out.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert out_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    out = as_numpy(simu, out)
    integral = ((out.real * out.real + out.imag * out.imag) * simu.delta_X**2).sum(
        axis=simu._last_axes
    )
    integral *= c * epsilon_0 / 2
    assert np.allclose(
        integral,
        np.array([simu.power, simu.power2]),
        rtol=1e-4,
    ), f"Normalization failed. (Backend {backend})"
    assert out.shape == (
        2,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    np.testing.assert_allclose(
        out / np.max(np.abs(out)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_split_step(backend) -> None:
    simu = CNLSE_1d(
        alpha,
        power,
        window,
        n2,
        n12,
        None,
        L,
        NX=N,
        Isat=Isat,
        backend=backend,
    )
    simu.delta_z = 0
    simu.propagator = simu._build_propagator()
    E = np.ones((2, N), dtype=PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator()
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        np.ones((2, N), dtype=PRECISION_COMPLEX),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Split step is not unitary. (Backend {backend})",
    )


def test_out_field(backend) -> None:
    simu = CNLSE_1d(
        0, power, window, n2, n12, None, L, NX=N, Isat=Isat, backend=backend
    )
    E0 = np.ones((2, N), dtype=PRECISION_COMPLEX)
    A = simu.out_field(E0, simu.delta_z, verbose=False, plot=False, precision="single")
    rho = A.real * A.real + A.imag * A.imag
    print(rho)
    integral = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
    integral *= c * epsilon_0 / 2
    assert A.shape == (
        2,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(integral, [simu.power, simu.power2], rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
