import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, as_numpy, assert_c_contiguous
from NLSE import DDGPE

if DDGPE.__CUPY_AVAILABLE__:
    import cupy as cp


PRECISION_COMPLEX = np.complex64

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

# Uniform runs with no Rabi coupling and the pump at both components' energy,
# so only the interaction or the damping acts. Units of time and density.
UNIFORM_N = 16
UNIFORM_W = 2.0  # exciton and cavity energy, rad per unit time
UNIFORM_T = 1.0
# Steps of 0.04: the cavity's kernel loss u = 2 (CAVITY_GAMMA / 2) 0.04 = 0.024,
# inside the solved range (u < 0.1); float32 error grows with the step count,
# measured 1.5e-6 at 25 steps and 1.4e-5 at 100 (MLX).
UNIFORM_STEPS = 25
EXCITON_DENSITY = 1.0
CAVITY_DENSITY = 2.0
# Cross coupling, rad per unit time and density: phases of 0.5 and 1 rad.
CROSS_G = 0.5
# Damping rates, per unit time, distinct so each component's own is checked.
EXCITON_GAMMA = 0.2
CAVITY_GAMMA = 0.6
# Measured worst case 1.5e-6 (float32, CPU and MLX). Smallest effects guarded:
# the cavity damped at the exciton's rate, 49% of its density, and a cross
# phase of the wrong sign on one component, 2 in relative terms.
UNIFORM_RTOL = 1e-5
UNIFORM_PARAMS = {
    "window": UNIFORM_N,
    "gamma": 0,
    "g": 0,
    "omega": 0,
    "T": UNIFORM_T,
    "omega_exc": UNIFORM_W,
    "omega_cav": UNIFORM_W,
    "detuning": 0,
    "k_z": 1.0,
}
# Probe amplitude, field per unit time, and the step it is applied over.
PROBE = 3.0
PROBE_DT = 1e-3

# Driven steady state: a uniform pump on the cavity and no interaction, so the
# closed form of two coupled, damped, driven modes applies.
# 20 times the example's decay (0.07 meV), so that the transient, exp(-10.7)
# by DRIVEN_T, is under the tolerance in a short run.
DRIVEN_GAMMA = 20 * 0.07 / h_bar  # 1/ps
DRIVEN_T = 10.0  # ps
DRIVEN_STEPS = 1600
DRIVE = 1.0  # field per ps
# The Lie step with a drive is first order: measured 8.0e-3 at 1600 steps and
# 4.0e-3 at 3200 (CPU, CL, MLX). Smallest convention error guarded: the
# detuning's sign in the solver, measured 0.46; gamma for gamma / 2 gives 0.49.
DRIVEN_RTOL = 1.5e-2


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


@pytest.mark.skipif(not DDGPE.__CUPY_AVAILABLE__, reason="CuPy not available")
def test_retrieve_arrays_from_gpu() -> None:
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
    simu.omega_exc = np.repeat(omega_exc_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
    simu.omega_cav = np.repeat(omega_cav_s, 2)[..., np.newaxis, np.newaxis, np.newaxis]
    simu._send_arrays_to_gpu()
    for name in ("propagator", "V", "gamma", "g", "omega", "omega_cav", "omega_exc"):
        assert isinstance(getattr(simu, name), cp.ndarray), f"{name} not on the GPU"
    simu._retrieve_arrays_from_gpu()
    for name in ("propagator", "gamma", "g", "omega", "omega_cav", "omega_exc"):
        assert isinstance(getattr(simu, name), np.ndarray), f"{name} not back on host"


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


# Every backend: excluding CL as too slow once hid laser_excitation
# destroying the cavity component there.
@pytest.mark.parametrize("ddgpe_backend", AVAILABLE_BACKENDS)
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
    """Both branches of the dispersion operator are grids, the flat one too."""
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
    """The propagator is the exponential of the dispersion operator."""
    simu = make_solver(backend)
    operator = as_numpy(simu, simu._build_propagator_rk4(PRECISION_COMPLEX))
    propagator = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
    assert np.allclose(propagator, np.exp(operator * DZ_TEST), atol=1e-6), (
        f"the propagator is not exp(operator * dz). (Backend {backend})"
    )


def test_add_noise_reaches_the_field(backend) -> None:
    """The noise callback must actually perturb the field on every backend.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu = make_solver(backend, n=32, gamma=0.07 / h_bar)
    simu._current_delta_z = 1e-3
    added = {}
    for label, amplitude in (("quiet", 0.0), ("noisy", 1.0)):
        np.random.seed(99)
        field = simu._backend.from_numpy(np.zeros((2, 32, 32), dtype=PRECISION_COMPLEX))
        DDGPE.add_noise(simu, field, 0.0, 0, amplitude)
        added[label] = np.abs(np.asarray(as_numpy(simu, field))).max()

    assert added["quiet"] == 0.0, (
        f"asking for no noise perturbed the field anyway. (Backend {backend})"
    )
    assert added["noisy"] > 1e-6, (
        f"add_noise left the field unchanged on {backend}: it ran without "
        f"raising and wrote nothing, which is what an in-place add on a slice "
        f"does on a backend that quietly drops it"
    )


def test_laser_excitation_of_zero_leaves_the_field_alone(backend) -> None:
    """Subtracting a pump of zero must not change anything.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu = make_solver(backend, n=32)
    simu._current_delta_z = 1e-3
    before = np.ones((2, 32, 32), dtype=PRECISION_COMPLEX)
    field = simu._backend.from_numpy(before.copy())
    flat_r = np.zeros((32, 32), dtype=PRECISION_COMPLEX)
    flat_t = np.zeros(4, dtype=PRECISION_COMPLEX)

    DDGPE.laser_excitation(simu, field, 0.0, 0, flat_r, flat_t, flat_r, flat_t)

    after = np.asarray(as_numpy(simu, field))
    np.testing.assert_allclose(
        after,
        before,
        err_msg=(
            f"a zero pump changed the field on {backend}; the cavity "
            f"component is what an in-place subtract on a slice destroys"
        ),
    )


def test_the_total_density_decays_at_the_damping_rate(backend) -> None:
    """The total density decays at the damping rate of the analytic answer.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    gamma = 0.07 / h_bar
    total_time = 1e-2
    n = 32

    simu = make_solver(backend, n=n, gamma=gamma)
    simu.gamma2 = gamma
    field = np.full((2, n, n), 0.9, dtype=PRECISION_COMPLEX)
    steps = np.arange(0, total_time, 1e-3, dtype=np.float32)
    flat_r = np.zeros((n, n), dtype=PRECISION_COMPLEX)
    flat_t = np.zeros(steps.shape, dtype=PRECISION_COMPLEX)

    before = float(np.sum(np.abs(field) ** 2))
    out = simu.out_field(
        field.copy(),
        total_time,
        simu.laser_excitation,
        plot=False,
        verbose=False,
        delta_z=1e-3,
        callback=[],
        callback_args=[[flat_r, flat_t, flat_r, flat_t]],
    )
    after = float(np.sum(np.abs(np.asarray(as_numpy(simu, out))) ** 2))

    assert after / before == pytest.approx(np.exp(-gamma * total_time), rel=1e-5), (
        f"total density fell to {after / before:.8f} of its value where "
        f"{np.exp(-gamma * total_time):.8f} was due. (Backend {backend})"
    )


def uniform_run(backend, duration=UNIFORM_T, **overrides):
    """Run two uniform components for a duration and return them on the host.

    Parameters
    ----------
    backend : str
        Backend name.
    duration : float
        Time to run for, a whole number of UNIFORM_T / UNIFORM_STEPS steps.
    **overrides
        Constructor arguments; ``gamma2`` is set after construction.

    Returns
    -------
    np.ndarray
        The exciton and cavity values at one point.
    """
    gamma2 = overrides.pop("gamma2", None)
    simu = make_solver(backend, n=UNIFORM_N, **(UNIFORM_PARAMS | overrides))
    if gamma2 is not None:
        simu.gamma2 = gamma2
    field = np.empty((2, UNIFORM_N, UNIFORM_N), dtype=PRECISION_COMPLEX)
    field[0] = np.sqrt(EXCITON_DENSITY)
    field[1] = np.sqrt(CAVITY_DENSITY)
    zero_r = np.zeros((UNIFORM_N, UNIFORM_N), dtype=PRECISION_COMPLEX)
    steps = round(UNIFORM_STEPS * duration / UNIFORM_T)
    zero_t = np.zeros(steps, dtype=PRECISION_COMPLEX)
    out = simu.out_field(
        field,
        duration,
        None,
        delta_z=UNIFORM_T / UNIFORM_STEPS,
        verbose=False,
        callback_args=[(zero_r, zero_t, zero_r, zero_t)],
    )
    return np.asarray(as_numpy(simu, out))[:, 0, 0]


def test_the_cross_interaction_is_mutual(backend) -> None:
    """Each component is phased by g12 times the other's density."""
    exciton, cavity = uniform_run(backend, g12=CROSS_G)
    per_density = (
        np.angle(exciton) / CAVITY_DENSITY,
        np.angle(cavity) / EXCITON_DENSITY,
    )
    assert per_density[0] == pytest.approx(per_density[1], rel=UNIFORM_RTOL), (
        f"the exciton turns by {per_density[0]:+.6f} and the cavity by "
        f"{per_density[1]:+.6f} rad per unit of the other's density over T. "
        f"(Backend {backend})"
    )
    assert abs(per_density[0]) == pytest.approx(CROSS_G * UNIFORM_T, rel=UNIFORM_RTOL)


def test_each_component_decays_at_its_own_rate(backend) -> None:
    """The exciton density falls as exp(-gamma t), the cavity's as exp(-gamma2 t)."""
    out = uniform_run(backend, gamma=EXCITON_GAMMA, gamma2=CAVITY_GAMMA)
    densities = np.abs(out) ** 2 / np.array([EXCITON_DENSITY, CAVITY_DENSITY])
    for density, rate in zip(densities, (EXCITON_GAMMA, CAVITY_GAMMA), strict=True):
        assert density == pytest.approx(np.exp(-rate * UNIFORM_T), rel=UNIFORM_RTOL), (
            f"a component damped at {rate} kept {density:.6f} of its density. "
            f"(Backend {backend})"
        )


def test_neither_component_decays_past_t(backend) -> None:
    """Past T both losses stop: a run to 2 T keeps what was left at T."""
    out = uniform_run(
        backend, duration=2 * UNIFORM_T, gamma=EXCITON_GAMMA, gamma2=CAVITY_GAMMA
    )
    densities = np.abs(out) ** 2 / np.array([EXCITON_DENSITY, CAVITY_DENSITY])
    for density, rate in zip(densities, (EXCITON_GAMMA, CAVITY_GAMMA), strict=True):
        assert density == pytest.approx(np.exp(-rate * UNIFORM_T), rel=UNIFORM_RTOL), (
            f"a component damped at {rate} until T kept {density:.6f} of its "
            f"density at 2 T. (Backend {backend})"
        )


def test_the_probe_reaches_the_cavity(backend) -> None:
    """A probe and no pump adds ``-i F dt`` to the cavity, nothing to the exciton.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu = make_solver(backend, n=UNIFORM_N)
    simu._current_delta_z = PROBE_DT
    shape = (UNIFORM_N, UNIFORM_N)
    field = simu._backend.from_numpy(np.zeros((2, *shape), dtype=PRECISION_COMPLEX))
    zero_r = np.zeros(shape, dtype=PRECISION_COMPLEX)
    probe_r = np.full(shape, PROBE, dtype=PRECISION_COMPLEX)
    profile_t = np.ones(1, dtype=PRECISION_COMPLEX)

    DDGPE.laser_excitation(
        simu, field, 0.0, 0, zero_r, 0 * profile_t, probe_r, profile_t
    )

    after = np.asarray(as_numpy(simu, field))
    expected = np.stack([zero_r, -1j * PROBE * PROBE_DT * np.ones(shape)])
    np.testing.assert_allclose(after, expected, rtol=UNIFORM_RTOL, atol=0)


def test_a_uniform_pump_reaches_the_linear_steady_state(backend) -> None:
    """Both components settle on the fixed point of the driven linear system.

    ``i dX/dt = (d_x - i gamma / 2) X + (omega / 2) C`` and
    ``i dC/dt = (d_c - i gamma / 2) C + (omega / 2) X + F``, with ``d`` each
    component's frequency less the pump's, give
    ``C = -F / (d_c - i gamma / 2 - (omega / 2)^2 / (d_x - i gamma / 2))`` and
    ``X = -(omega / 2) C / (d_x - i gamma / 2)``.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu = make_solver(backend, n=UNIFORM_N, gamma=DRIVEN_GAMMA, g=0, T=DRIVEN_T)
    shape = (UNIFORM_N, UNIFORM_N)
    pump_r = np.full(shape, DRIVE, dtype=PRECISION_COMPLEX)
    pump_t = np.ones(DRIVEN_STEPS, dtype=PRECISION_COMPLEX)
    out = simu.out_field(
        np.zeros((2, *shape), dtype=PRECISION_COMPLEX),
        DRIVEN_T,
        simu.laser_excitation,
        plot=False,
        verbose=False,
        delta_z=DRIVEN_T / DRIVEN_STEPS,
        callback=[],
        callback_args=[[pump_r, pump_t, pump_r, 0 * pump_t]],
    )
    out = np.asarray(as_numpy(simu, out))

    lower_polariton = (omega_exc + omega_cav) / 2 - 0.5 * np.hypot(
        omega_exc - omega_cav, omega
    )
    pump = lower_polariton + detuning
    exciton_rate = omega_exc - pump - 0.5j * DRIVEN_GAMMA
    cavity_rate = omega_cav - pump - 0.5j * DRIVEN_GAMMA
    cavity = -DRIVE / (cavity_rate - (omega / 2) ** 2 / exciton_rate)
    exciton = -(omega / 2) * cavity / exciton_rate
    for got, want, name in zip(
        out, (exciton, cavity), ("exciton", "cavity"), strict=True
    ):
        np.testing.assert_allclose(
            got,
            want,
            rtol=DRIVEN_RTOL,
            atol=0,
            err_msg=f"the {name} is off its steady state. (Backend {backend})",
        )
