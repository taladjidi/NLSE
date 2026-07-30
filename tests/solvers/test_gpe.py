import numpy as np
from helpers import as_numpy, assert_c_contiguous, random_field
from NLSE import GPE
from scipy.constants import atomic_mass, hbar

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
N_at = 1e6
g = 1e3 / (N_at / 1e-3**2)
waist = 1e-3
window = 1e-3
m = 87 * atomic_mass

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def test_build_propagator(backend) -> None:
    simu_gpe = GPE(
        gamma=0,
        N=N_at,
        window=window,
        g=g,
        V=None,
        m=m,
        NX=N,
        NY=N,
        backend=backend,
    )
    prop = simu_gpe._build_propagator(np.complex64, DZ_TEST)
    expected = np.exp(
        -1j * 0.5 * hbar * (simu_gpe.Kxx**2 + simu_gpe.Kyy**2) / simu_gpe.m * DZ_TEST,
        dtype=np.complex64,
    )
    assert np.allclose(
        prop,
        expected,
    ), f"Propagator is wrong. (Backend {backend})"


def test_prepare_output_array(backend) -> None:
    simu = GPE(
        gamma=0,
        N=N_at,
        window=window,
        g=g,
        V=None,
        m=m,
        NX=N,
        NY=N,
        backend=backend,
    )
    E_in = random_field((N, N))
    A, A_sq = simu._prepare_output_array(E_in, normalize=True)
    assert_c_contiguous(A, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(A_sq, f"Output array is not C-contiguous. (Backend {backend})")
    if backend == "CPU":
        assert A.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert A_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    A_np = as_numpy(simu, A)
    integral = (
        (A_np.real * A_np.real + A_np.imag * A_np.imag) * simu.delta_X * simu.delta_Y
    ).sum(axis=simu._last_axes)
    assert np.allclose(integral, simu.N, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
    np.testing.assert_allclose(
        A_np / np.max(np.abs(A_np)),
        E_in / np.max(np.abs(E_in)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_out_field(backend) -> None:
    simu = GPE(
        gamma=0,
        N=N_at,
        window=window,
        g=g,
        V=None,
        m=m,
        NX=N,
        NY=N,
        backend=backend,
    )
    psi_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    psi = simu.out_field(
        psi_0, 1e-6, verbose=True, plot=False, precision="single", delta_z=1e-8
    )
    norm = np.sum(np.abs(psi) ** 2 * simu.delta_X * simu.delta_Y)
    assert np.allclose(norm, simu.N, rtol=1e-4), (
        f"Norm not conserved. (Backend {backend})"
    )
