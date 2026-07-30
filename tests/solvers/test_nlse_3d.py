import numpy as np
import pyfftw
from helpers import as_numpy, assert_c_contiguous, random_field
from NLSE import NLSE_3d
from scipy.constants import c, epsilon_0

if NLSE_3d.__CUPY_AVAILABLE__:
    import cupy as cp
    from NLSE.backends.cupy_backend import _CuFFTPlan
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
NZ = 128
n2 = -1.6e-9
D0 = 1e-16
vg = 1e-1 * c
waist = 2.23e-3
duration = 2e-6
waist2 = 70e-6
window = np.array([4 * waist, 8 * duration])  # 4*waist transverse, 10e-6 s temporal
energy = 1.05 * duration
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-2
alpha = 20

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_solver(backend="CPU", n=N, **overrides):
    """Return an NLSE_3d with this module's parameters.

    Parameters
    ----------
    backend : str
        Backend name.
    n : int
        Transverse grid size, square. The temporal axis stays NZ.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    NLSE_3d
        The solver.
    """
    params = {
        "alpha": alpha,
        "energy": energy,
        "window": window,
        "n2": n2,
        "D0": D0,
        "vg": vg,
        "V": None,
        "L": L,
        "NX": n,
        "NY": n,
        "NZ": NZ,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(overrides)
    return NLSE_3d(**params)


def test_build_propagator(backend) -> None:
    simu = make_solver(backend)
    prop = simu._build_propagator(np.complex64, DZ_TEST)
    prop_th = np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * DZ_TEST)
    prop_th *= np.exp(-1j * simu.D0 / 2 * simu.Omega**2)
    assert np.allclose(
        prop,
        prop_th,
    ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan(backend) -> None:
    simu = make_solver(backend)
    if backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
        A = cp.random.random((N, N, NZ)).astype(PRECISION_REAL) + 1j * cp.random.random(
            (N, N, NZ)
        ).astype(PRECISION_REAL)
    else:
        A = np.random.random((N, N, NZ)).astype(PRECISION_REAL) + 1j * np.random.random(
            (N, N, NZ)
        ).astype(PRECISION_REAL)
    plans = simu._build_fft_plan(A)
    if backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
        assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
        assert isinstance(plans[0], _CuFFTPlan), (
            f"Plan type is wrong. (Backend {backend})"
        )
    elif backend == "CPU":
        assert len(plans) == 2, f"Number of plans is wrong. (Backend {backend})"
        assert isinstance(plans[0], pyfftw.FFTW), (
            f"Plan type is wrong. (Backend {backend})"
        )
        assert plans[0].output_shape == (
            N,
            N,
            NZ,
        ), f"Plan shape is wrong. (Backend {backend})"


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
    A = random_field((N, N, NZ))
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    if backend == "CPU":
        assert out.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert out_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    out_np = as_numpy(simu, out)
    integral = (
        (out_np.real * out_np.real + out_np.imag * out_np.imag)
        * simu.delta_X
        * simu.delta_Y
        * simu.delta_T
    ).sum(axis=simu._last_axes)
    integral *= c * epsilon_0 / 2
    assert np.allclose(integral, simu.energy, rtol=1e-4), (
        f"Normalization failed. (Backend {backend})"
    )
    assert out_np.shape == (
        N,
        N,
        NZ,
    ), f"Output array has wrong shape. (Backend {backend})"
    np.testing.assert_allclose(
        out_np / np.max(np.abs(out_np)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_send_arrays_to_gpu() -> None:
    if NLSE_3d.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = 1e-4 * np.random.random((N, N, NZ)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, V=V, Isat=Isat)
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        simu._send_arrays_to_gpu()
        assert isinstance(simu.propagator, cp.ndarray), (
            "propagator is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.V, cp.ndarray), "V is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.alpha, cp.ndarray), (
            "alpha is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.n2, cp.ndarray), "n2 is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.I_sat, cp.ndarray), (
            "I_sat is not a cp.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if NLSE_3d.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = 1e-4 * np.random.random((N, N, NZ)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, V=V, Isat=Isat)
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(simu.propagator, np.ndarray), (
            "propagator is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.V, np.ndarray), "V is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.alpha, np.ndarray), (
            "alpha is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.n2, np.ndarray), "n2 is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.I_sat, np.ndarray), (
            "I_sat is not a np.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_split_step(backend) -> None:
    simu = make_solver(backend)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    E = np.ones((N, N, NZ), dtype=PRECISION_COMPLEX)
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
        np.ones((N, N, NZ), dtype=PRECISION_COMPLEX),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Split step is not unitary. (Backend {backend})",
    )


# tests for convergence of the solver : the norm of the field should be
# conserved
def test_out_field(backend) -> None:
    simu = make_solver(backend)
    E0 = np.ones((N, N, NZ), dtype=PRECISION_COMPLEX)
    # alpha is 20 here, so the norm is only conserved to 1e-4 over a distance
    # short enough for the (saturated) absorption to be negligible. Pin the
    # step rather than propagate over whatever the solver would choose.
    E = simu.out_field(
        E0, L / 1000, delta_z=L / 1000, verbose=False, plot=False, precision="single"
    )
    norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y * simu.delta_T)
    norm *= c * epsilon_0 / 2
    assert E.shape == (
        N,
        N,
        NZ,
    ), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(norm, simu.energy, rtol=1e-4), (
        f"Norm not conserved. (Backend {backend})"
    )
