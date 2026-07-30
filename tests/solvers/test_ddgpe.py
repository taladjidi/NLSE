import numpy as np
import pytest
from helpers import as_numpy, assert_c_contiguous
from NLSE import DDGPE

if DDGPE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
T = 1
h_bar = 0.654  # (meV*ps)
omega = 5.07 / h_bar  # (meV/h_bar) linear coupling (Rabi split)
omega_exc = 1484.44 / h_bar  # (meV/h_bar) exciton energy
omega_cav = 1482.76 / h_bar  # (meV/h_bar) cavity energy
detuning = 0.17 / h_bar
k_z = 27
gamma = 0 * 0.07 / h_bar
waist = 50
window = 256
g = 1e-2 / h_bar
puiss = detuning / g

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_solver(backend="CPU", n=N, **overrides):
    """Return a DDGPE with this module's parameters.

    Parameters
    ----------
    backend : str
        Backend name.
    n : int
        Grid size, square.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    DDGPE
        The solver.
    """
    params = {
        "gamma": gamma,
        "power": puiss,
        "window": window,
        "g": g,
        "omega": omega,
        "T": T,
        "omega_exc": omega_exc,
        "omega_cav": omega_cav,
        "detuning": detuning,
        "k_z": k_z,
        "NX": n,
        "NY": n,
        "backend": backend,
    }
    params.update(overrides)
    return DDGPE(**params)


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
    A = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    out, out_sq = simu._prepare_output_array(A, normalize=False)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    out = as_numpy(simu, out)
    assert out.shape == (
        2,
        N,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    np.testing.assert_allclose(
        out / np.max(np.abs(out)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_send_arrays_to_gpu() -> None:
    if DDGPE.__CUPY_AVAILABLE__:
        omega_exc_s = 1484.44 / h_bar
        omega_cav_s = 1482.76 / h_bar
        detuning_s = 0.17 / h_bar
        k_z_s = 27
        gamma_s = 0 * 0.07 / h_bar
        omega_s = 5.07 / h_bar
        g_s = 1e-2 / h_bar
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        simu = DDGPE(
            gamma_s,
            puiss,
            window,
            g_s,
            omega_s,
            T,
            omega_exc_s,
            omega_cav_s,
            detuning_s,
            k_z_s,
            V=V,
            NX=N,
            NY=N,
            backend="CUPY",
        )
        # Build propagator while params are still scalars
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        # Now broadcast params for GPU transfer test
        simu.gamma = np.repeat(gamma_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.g = np.repeat(g_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.omega = np.repeat(omega_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.omega_exc = np.repeat(omega_exc_s, 2)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        simu.omega_cav = np.repeat(omega_cav_s, 2)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        simu._send_arrays_to_gpu()
        assert isinstance(simu.propagator, cp.ndarray), (
            "propagator is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.V, cp.ndarray), "V is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.gamma, cp.ndarray), (
            "gamma is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.g, cp.ndarray), "g is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.omega, cp.ndarray), (
            "omega is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.omega_cav, cp.ndarray), (
            "omega cav is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.omega_exc, cp.ndarray), (
            "omega exc is not a cp.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if DDGPE.__CUPY_AVAILABLE__:
        omega_exc_s = 1484.44 / h_bar
        omega_cav_s = 1482.76 / h_bar
        detuning_s = 0.17 / h_bar
        k_z_s = 27
        gamma_s = 0 * 0.07 / h_bar
        g_s = 1e-2 / h_bar
        omega_s = 5.07 / h_bar
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        simu = DDGPE(
            gamma_s,
            puiss,
            window,
            g_s,
            omega_s,
            T,
            omega_exc_s,
            omega_cav_s,
            detuning_s,
            k_z_s,
            V=V,
            NX=N,
            NY=N,
            backend="CUPY",
        )
        # Build propagator while params are still scalars
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        # Now broadcast params for GPU transfer test
        simu.gamma = np.repeat(gamma_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.g = np.repeat(g_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.omega = np.repeat(omega_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
        simu.omega_exc = np.repeat(omega_exc_s, 2)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        simu.omega_cav = np.repeat(omega_cav_s, 2)[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(simu.propagator, np.ndarray), (
            "propagator is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.gamma, np.ndarray), (
            "gamma is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.g, np.ndarray), "g is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.omega, np.ndarray), (
            "omega is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.omega_cav, np.ndarray), (
            "omega cav is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.omega_exc, np.ndarray), (
            "omega exc is not a np.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_build_propagator(backend) -> None:
    simu = make_solver()
    prop = as_numpy(simu, simu._build_propagator(np.complex64, DZ_TEST))
    assert np.allclose(
        prop[0],
        np.exp(
            -1j * (simu.omega_exc * (1 + 0 * simu.Kxx**2) - simu.omega_pump) * DZ_TEST
        ),
    ), f"Propagator1 is wrong. (Backend {backend})"
    # The cavity branch is dispersive, unlike the exciton one. This used to
    # assert the exciton formula for both, which only passed because the step
    # was small enough to make either of them indistinguishable from 1.
    assert np.allclose(
        prop[1],
        np.exp(
            -1j
            * (
                simu.omega_cav * np.sqrt(1 + (simu.Kxx**2 + simu.Kyy**2) / simu.k_z**2)
                - simu.omega_pump
            )
            * DZ_TEST
        ),
    ), f"Propagator2 is wrong. (Backend {backend})"


def test_take_components(backend) -> None:
    simu = make_solver(backend)
    # create a larger array to test the fancy indexing
    A = np.ones((3, 2, N, N), dtype=PRECISION_COMPLEX)
    A1, A2 = simu._take_components(A)
    assert A1.shape[-2:] == (
        N,
        N,
    ), f"A1 has wrong last dimensions. (Backend {backend})"
    assert A2.shape[-2:] == (
        N,
        N,
    ), f"A2 has wrong last dimensions. (Backend {backend})"
    assert A1.shape == A2.shape, f"A1 and A2 have different shapes. (Backend {backend})"
    assert A1.shape[0] == 3, f"A1 has wrong first dimensions. (Backend {backend})"
    assert A2.shape[0] == 3, f"A2 has wrong first dimensions. (Backend {backend})"


def callback_sample(
    simu: DDGPE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    sample1: list,
    sample2: list,
    sample3: list,
) -> None:
    if i % save_every == 0:
        # Convert CL arrays to numpy for computation
        if hasattr(A, "get"):
            A_np = A.get()
        else:
            A_np = A
        sum_exc = (A_np[..., 0, :, :].real ** 2 + A_np[..., 0, :, :].imag ** 2).sum()
        sum_cav = (A_np[..., 1, :, :].real ** 2 + A_np[..., 1, :, :].imag ** 2).sum()
        sum_tot = sum_exc + sum_cav
        sample1[i // save_every] = sum_exc
        sample2[i // save_every] = sum_cav
        sample3[i // save_every] = sum_tot


def turn_on(
    F_laser_t: np.ndarray,
    time: np.ndarray,
    t_up=10,
):
    """Turn on the pump more or less adiabatically.

    Parameters
    ----------
    F_laser_t : np.ndarray
        self.F_pump_t as defined in class ggpe,
        cp.ones((int(self.t_max//self.dt)), dtype=cp.complex64)
    time : np.ndarray
        array with the value of the time at each discretized
        step.
    t_up : int, optional
        time taken to reach the maximum intensity (=F).
        Defaults to 10.
    """
    F_laser_t[time < t_up] = np.exp(
        -1 * (time[time < t_up] - t_up) ** 2 / (t_up / 2) ** 2
    )
    F_laser_t[time >= t_up] = 1


# CL backend is too slow for DDGPE propagation (unoptimized array-expression kernels)
@pytest.mark.parametrize(
    "ddgpe_backend",
    list(["CPU"] + (["CUPY"] if DDGPE.__CUPY_AVAILABLE__ else [])),
)
def test_out_field(ddgpe_backend) -> None:
    backend = ddgpe_backend
    simu = make_solver(backend)
    dt = 0.1 / 32
    time = np.arange(0, T + dt, step=dt, dtype=np.float32)
    save_every = 1  # np.argwhere(time == 1)[0][0]
    sample1 = np.zeros(time.size // save_every, dtype=np.float32)
    sample2 = np.zeros(time.size // save_every, dtype=np.float32)
    sample3 = np.zeros(time.size // save_every, dtype=np.float32)
    E0 = np.zeros((2, simu.NY, simu.NX), dtype=np.complex64)
    F_pump = 0
    F_pump_r = F_pump * np.exp(-((simu.XX**2 + simu.YY**2) / waist**2)).astype(
        np.complex64
    )
    F_pump_t = np.zeros(time.shape, dtype=np.complex64)
    F_probe = 0
    F_probe_r = F_probe * np.exp(-((simu.XX**2 + simu.YY**2) / waist**2)).astype(
        np.complex64
    )
    F_probe_t = np.zeros(time.shape, dtype=np.complex64)
    turn_on(F_pump_t, time, t_up=20)
    callback = [callback_sample]
    if backend == "CUPY" and DDGPE.__CUPY_AVAILABLE__:
        callback_args = [
            [
                cp.asarray(F_pump_r),
                F_pump_t,
                cp.asarray(F_probe_r),
                F_probe_t,
            ],
            [save_every, sample1, sample2, sample3],
        ]
    else:
        callback_args = [
            [
                F_pump_r,
                F_pump_t,
                F_probe_r,
                F_probe_t,
            ],
            [save_every, sample1, sample2, sample3],
        ]
    simu.out_field(
        E0,
        T,
        simu.laser_excitation,
        plot=False,
        callback=callback,
        callback_args=callback_args,
        delta_z=0.1 / 32,  # need to be adjusted automatically,
    )
    # test stationarity here


def test_the_dispersion_operator_is_a_pair_of_grids(backend) -> None:
    """Both branches must be grids, even the one with no dispersion in it.

    The exciton branch does not depend on k, so the expression for it is a
    scalar. Paired with the cavity grid by np.array that is an inhomogeneous
    array, not the (2, NY, NX) it looks like, and numpy has raised on it since
    1.24. It was unreachable while only RK4 read it and DDGPE exposes no RK4,
    but the split step's propagator is built from it now.
    """
    simu = make_solver(backend)
    operator = as_numpy(simu, simu._build_propagator_rk4(PRECISION_COMPLEX))
    assert operator.shape == (2, N, N), (
        f"the dispersion operator is {operator.shape}, not a pair of grids"
    )
    assert np.all(np.isfinite(operator.view(np.float32)))
    # The exciton branch is flat, which is why it needed broadcasting at all.
    assert np.allclose(operator[0], operator[0].flat[0])
    assert not np.allclose(operator[1], operator[1].flat[0]), (
        "the cavity branch should vary across the grid"
    )


def test_the_propagator_is_the_exponential_of_the_operator(backend) -> None:
    """One statement of the linear physics, exponentiated for the step.

    If these two ever disagree, the split step and anything built on the
    operator are integrating different equations -- which is exactly what had
    happened in NLSE_3d, where the split step left the step out of its
    temporal dispersion entirely.

    At the precision every backend has, like its neighbour above: this asks
    whether two expressions agree, which is not a question about float width,
    and asking it in double excluded CL and MLX from an identity that holds
    for them too.
    """
    simu = make_solver(backend)
    operator = as_numpy(simu, simu._build_propagator_rk4(PRECISION_COMPLEX))
    propagator = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
    assert np.allclose(propagator, np.exp(operator * DZ_TEST), atol=1e-6), (
        f"the propagator is not exp(operator * dz). (Backend {backend})"
    )
