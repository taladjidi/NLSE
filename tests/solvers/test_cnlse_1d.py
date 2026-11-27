import numpy as np
from helpers import as_numpy, assert_c_contiguous, gaussian_beam, make
from NLSE import CNLSE_1d
from scipy.constants import c, epsilon_0

if CNLSE_1d.__CUPY_AVAILABLE__:
    pass
PRECISION_COMPLEX = np.complex64

N = 256
waist = 2.23e-3
window = 4 * waist

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4
# A beam well inside the window and well resolved: 16 waists across the
# window, 16 cells across the waist.
BEAM_WAIST = window / 16
# Absolute tolerance of the one-step Gaussian beam check, unit peak; the
# step lands within 2.4e-7 on CPU and MLX (float32).
BEAM_ATOL = 1e-5


def make_solver(backend="CPU", n=N, **overrides):
    """Return a CNLSE_1d with this module's parameters."""
    return make(CNLSE_1d, backend, n=n, **overrides)


def test_build_propagator(backend) -> None:
    simu = make_solver(backend)
    prop = as_numpy(simu, simu._build_propagator(np.complex64, DZ_TEST))
    prop1 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k * DZ_TEST)
    prop2 = np.exp(-1j * 0.5 * (simu.Kx**2) / simu.k2 * DZ_TEST)
    assert np.allclose(prop, np.array([prop1, prop2])), (
        f"Propagator is wrong. (Backend {backend})"
    )


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
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
    """One linear step lands each component on its own Gaussian beam.

    Lossless and linear, so the step is diffraction alone and exact. The two
    components have different wavelengths, so swapping them fails, and the
    complex field is compared, so a propagator of the wrong sign fails too.
    """
    simu = make_solver(backend, symmetric=False, n2=0, n12=0, alpha=0)
    r_sq = simu.X**2
    z_rayleigh = simu.k * BEAM_WAIST**2 / 2
    E = np.stack([gaussian_beam(r_sq, BEAM_WAIST, simu.k, 0, 1)] * 2)
    A, A_sq = simu._prepare_output_array(E.astype(PRECISION_COMPLEX), normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, z_rayleigh)
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, z_rayleigh, splitting="strang"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        [
            gaussian_beam(r_sq, BEAM_WAIST, simu.k, z_rayleigh, 1),
            gaussian_beam(r_sq, BEAM_WAIST, simu.k2, z_rayleigh, 1),
        ],
        rtol=0,
        atol=BEAM_ATOL,
        err_msg=f"One step is not the Gaussian beams at zR. (Backend {backend})",
    )


def test_out_field(backend) -> None:
    simu = make_solver(backend, alpha=0)
    E0 = np.ones((2, N), dtype=PRECISION_COMPLEX)
    A = simu.out_field(
        E0, DZ_TEST, delta_z=DZ_TEST, verbose=False, plot=False, splitting="lie"
    )
    rho = A.real * A.real + A.imag * A.imag
    integral = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
    integral *= c * epsilon_0 / 2
    assert A.shape == (
        2,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(integral, [simu.power, simu.power2], rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
