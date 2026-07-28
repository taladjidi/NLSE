import numpy as np
import pytest
from NLSE import DDGPE

from .helpers import as_numpy, assert_c_contiguous

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


def test_prepare_output_array(backend) -> None:
    simu = DDGPE(
        gamma,
        puiss,
        window,
        g,
        omega,
        T,
        omega_exc,
        omega_cav,
        detuning,
        k_z,
        NX=N,
        NY=N,
        backend=backend,
    )
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
        simu.propagator = simu._build_propagator()
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
        simu.propagator = simu._build_propagator()
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
    simu = DDGPE(
        gamma,
        puiss,
        window,
        g,
        omega,
        T,
        omega_exc,
        omega_cav,
        detuning,
        k_z,
        NX=N,
        NY=N,
    )
    prop = simu._build_propagator()
    assert np.allclose(
        prop[0],
        np.exp(
            -1j
            * (simu.omega_exc * (1 + 0 * simu.Kxx**2) - simu.omega_pump)
            * simu.delta_z
        ),
    ), f"Propagator1 is wrong. (Backend {backend})"
    assert np.allclose(
        prop[1],
        np.exp(
            -1j
            * (simu.omega_exc * (1 + 0 * simu.Kxx**2) - simu.omega_pump)
            * simu.delta_z
        ),
    ), f"Propagator2 is wrong. (Backend {backend})"


def test_take_components(backend) -> None:
    simu = DDGPE(
        gamma,
        puiss,
        window,
        g,
        omega,
        T,
        omega_exc,
        omega_cav,
        detuning,
        k_z,
        NX=N,
        NY=N,
        backend=backend,
    )
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
    simu = DDGPE(
        gamma,
        puiss,
        window,
        g,
        omega,
        T,
        omega_exc,
        omega_cav,
        detuning,
        k_z,
        NX=N,
        NY=N,
        backend=backend,
    )
    simu.delta_z = 0.1 / 32  # need to be adjusted automatically
    time = np.arange(0, T + simu.delta_z, step=simu.delta_z, dtype=np.float32)
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
    )
    # test stationarity here
