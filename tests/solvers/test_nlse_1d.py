import numpy as np
from NLSE import NLSE_1d
from scipy.constants import c, epsilon_0

if NLSE_1d.__CUPY_AVAILABLE__:
    import cupy as cp
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


def test_build_propagator(backend) -> None:
    simu = NLSE_1d(alpha, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    prop = simu._build_propagator()
    assert np.allclose(
        prop, np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * simu.delta_z)
    ), f"Propagator is wrong. (Backend {backend})"


def test_prepare_output_array(backend) -> None:
    simu = NLSE_1d(
        alpha,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        Isat=Isat,
        backend=backend,
    )
    if backend == "CUPY" and NLSE_1d.__CUPY_AVAILABLE__:
        A = cp.ones(N, dtype=PRECISION_COMPLEX)
    else:
        A = np.ones(N, dtype=PRECISION_COMPLEX)
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    # Convert CL arrays to numpy for assertions
    if backend == "CL":
        out = out.get()
        out_sq = out_sq.get()
    assert out.flags.c_contiguous, (
        f"Output array is not C-contiguous. (Backend {backend})"
    )
    assert out_sq.flags.c_contiguous, (
        f"Output array is not C-contiguous. (Backend {backend})"
    )
    if backend == "CPU":
        assert out.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert out_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    integral = (np.abs(out) ** 2 * simu.delta_X**2).sum()
    integral *= c * epsilon_0 / 2
    assert np.allclose(integral, simu.power), (
        f"Normalization failed. (Backend {backend})"
    )
    assert out.shape == (N,), f"Output array has wrong shape. (Backend {backend})"
    if backend == "CUPY" and NLSE_1d.__CUPY_AVAILABLE__:
        assert isinstance(out, cp.ndarray), (
            f"Output array type does not match backend. (Backend {backend})"
        )
        out /= cp.max(cp.abs(out))
        A /= cp.max(cp.abs(A))
        assert cp.allclose(out, A), (
            f"Output array does not match input array. (Backend {backend})"
        )
    else:
        assert isinstance(out, np.ndarray), (
            f"Output array type does not match backend. (Backend {backend})"
        )
        out /= np.max(np.abs(out))
        A /= np.max(np.abs(A))
        assert np.allclose(out, A), (
            f"Output array does not match input array. (Backend {backend})"
        )


def test_split_step(backend) -> None:
    simu = NLSE_1d(alpha, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    simu.delta_z = 0
    simu.propagator = simu._build_propagator()
    E = np.ones((N,), dtype=PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator()
    if backend in ["CUPY", "CL"]:
        simu._send_arrays_to_gpu()
    simu.split_step(A, A_sq, simu.V, simu.propagator, simu.plans, precision="double")
    if backend == "CUPY" and NLSE_1d.__CUPY_AVAILABLE__:
        assert cp.allclose(A, cp.ones((N,), dtype=PRECISION_COMPLEX)), (
            f"Split step is not unitary. (Backend {backend})"
        )
    else:
        if backend == "CL":
            A = A.get()
        assert np.allclose(A, np.ones((N,), dtype=PRECISION_COMPLEX)), (
            f"Split step is not unitary. (Backend {backend})"
        )


def test_out_field(backend) -> None:
    simu = NLSE_1d(0, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    E0 = np.ones(N, dtype=PRECISION_COMPLEX)
    A = simu.out_field(E0, simu.delta_z, verbose=False, plot=False, precision="single")
    rho = A.real * A.real + A.imag * A.imag
    norm = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
    norm *= c * epsilon_0 / 2
    assert A.shape == (N,), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(norm, power, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
