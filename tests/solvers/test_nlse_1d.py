import numpy as np
from helpers import as_numpy, assert_c_contiguous, gaussian_beam, make
from NLSE import NLSE_1d
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64

N = 256
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4
# A beam well inside the window and well resolved: 16 waists across the
# window, 16 cells across the waist.
BEAM_WAIST = window / 16
# Absolute tolerance of the one-step Gaussian beam check, unit peak; the
# step lands within 2.4e-7 on CPU and MLX (float32).
BEAM_ATOL = 1e-5


def make_solver(backend="CPU", n=N, **overrides):
    """Return a NLSE_1d with this module's parameters."""
    return make(NLSE_1d, backend, n=n, **overrides)


def test_build_propagator(backend) -> None:
    simu = NLSE_1d(alpha, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
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
    """One linear step of a Rayleigh length lands on the Gaussian beam.

    Lossless and linear, so the step is diffraction alone and exact. The
    complex field is compared, so a propagator of the wrong sign fails too.
    """
    simu = make_solver(backend, n2=0, alpha=0)
    r_sq = simu.X**2
    z_rayleigh = simu.k * BEAM_WAIST**2 / 2
    E = gaussian_beam(r_sq, BEAM_WAIST, simu.k, 0, 1).astype(PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, z_rayleigh)
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, z_rayleigh, splitting="strang"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        gaussian_beam(r_sq, BEAM_WAIST, simu.k, z_rayleigh, 1),
        rtol=0,
        atol=BEAM_ATOL,
        err_msg=f"One step is not the Gaussian beam at zR. (Backend {backend})",
    )


def test_out_field(backend) -> None:
    simu = NLSE_1d(0, power, window, n2, None, L, NX=N, Isat=Isat, backend=backend)
    E0 = np.ones(N, dtype=PRECISION_COMPLEX)
    A = simu.out_field(
        E0, DZ_TEST, delta_z=DZ_TEST, verbose=False, plot=False, splitting="lie"
    )
    rho = A.real * A.real + A.imag * A.imag
    norm = (rho * simu.delta_X**2).sum(axis=simu._last_axes)
    norm *= c * epsilon_0 / 2
    assert A.shape == (N,), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(norm, power, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
