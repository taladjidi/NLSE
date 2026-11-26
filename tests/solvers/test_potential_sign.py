"""The sign and scale of the potential, in DDGPE and in CNLSE.

DDGPE's equation is ``i dpsi/dt = (V + g |psi|**2) psi + ...``
(docs/solvers_overview.md), so on a uniform exciton field with nothing else
acting the phase after a time t is ``-(V + g n) t``. CNLSE's is the paraxial
``i dE/dz = -(k / 2) V E + ...`` (docs/physics_problem.md), a phase of
``+(k / 2) V z`` per component, each with its own k.
"""

import warnings

import numpy as np
import pytest
from NLSE import CNLSE, DDGPE

N = 32
T = 0.2  # total time, in the units of DDGPE's rates
DT = 0.01
V0 = 0.3  # potential, a rate
G = 0.5  # interaction, a rate per unit density; the field has n = 1
# Measured worst error 1.2e-7 rad (yoshida; CUPY on an RTX 3050, and MLX) and
# 6.6e-8 on CPU; a wrong sign misses by 0.1.
PHASE_ATOL = 1e-4
# A direct split_step: a DDGPE potential and step giving -0.05 rad.
V_DIRECT = 0.5
DT_DIRECT = 0.1
# CNLSE: a step in m and the phase (k / 2) V dz it gives component 1, in rad.
DZ_CNLSE = 1e-3
PHASE_CNLSE = 0.1


def ddgpe(backend, V=None, g=0.0):
    """Return a lossless, uncoupled DDGPE with flat dispersion."""
    return DDGPE(
        gamma=0,
        power=1,
        window=N,
        g=g,
        omega=0,
        T=T,
        omega_exc=0,
        omega_cav=0,
        detuning=0,
        k_z=1.0,
        NX=N,
        NY=N,
        backend=backend,
        V=V,
    )


def exciton_phase(simu, splitting):
    """Run a uniform exciton field over T; return its phase and the last step.

    A capped step is an error here: at the asked DT the run is exact, and a
    cap that binds means the potential was scaled as an NLSE wavenumber.
    """
    field = np.zeros((2, N, N), np.complex64)
    field[0] = 1
    zero = np.zeros((N, N), np.complex64)
    steps = round(T / DT)
    drive = [(zero, np.zeros(steps), zero, np.zeros(steps))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error", message="delta_z=")
        out = simu.out_field(
            field,
            T,
            None,
            delta_z=DT,
            splitting=splitting,
            verbose=False,
            callback_args=drive,
        )
    exciton = np.asarray(simu._backend.to_numpy(out))[0]
    return float(np.angle(np.mean(exciton))), simu._current_delta_z


@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
def test_ddgpe_potential_phase(backend, splitting):
    """A uniform potential turns the exciton by -V t."""
    simu = ddgpe(backend, V=np.full((N, N), V0, np.float32))
    phase, _ = exciton_phase(simu, splitting)
    assert phase == pytest.approx(-V0 * T, abs=PHASE_ATOL)


@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
def test_ddgpe_interaction_phase(backend, splitting):
    """A uniform density n = 1 turns the exciton by -g n t."""
    simu = ddgpe(backend, g=G)
    phase, _ = exciton_phase(simu, splitting)
    assert phase == pytest.approx(-G * T, abs=PHASE_ATOL)


def test_ddgpe_potential_keeps_the_asked_step(backend):
    """A potential does not cap the step: it is a rate, not k / 2 times one."""
    simu = ddgpe(backend, V=np.full((N, N), V0, np.float32))
    _, step = exciton_phase(simu, "lie")
    assert step == DT


def direct_split_step(simu, V, dz):
    """Take one Lie split_step on a uniform field, as a direct caller would.

    Returns the phase of each component on the host.
    """
    field = np.ones((2, N, N), np.complex64)
    A, A_sq = simu._prepare_output_array(field, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, dz)
    # out_field sends arrays for every device backend, so mirror that here or
    # the kernels receive a host propagator.
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    potential = simu._backend.from_numpy(np.full((N, N), V, np.float32))
    A = simu.split_step(
        A, A_sq, potential, simu.propagator, simu.plans, dz, splitting="lie"
    )
    out = np.asarray(simu._backend.to_numpy(A))
    return [float(np.angle(np.mean(component))) for component in out]


def test_ddgpe_direct_split_step_potential(backend):
    """Called directly, split_step still turns both components by -V dt."""
    phases = direct_split_step(ddgpe(backend), V_DIRECT, DT_DIRECT)
    assert phases == pytest.approx([-V_DIRECT * DT_DIRECT] * 2, abs=PHASE_ATOL)


def test_cnlse_direct_split_step_potential(backend):
    """Called directly, split_step turns each component by +(k / 2) V dz."""
    simu = CNLSE(
        alpha=0,
        power=1,
        window=1e-3,
        n2=0,
        n12=0,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        backend=backend,
    )
    simu.k2 = simu.k / 2
    V = PHASE_CNLSE / (simu.k / 2 * DZ_CNLSE)
    phases = direct_split_step(simu, V, DZ_CNLSE)
    assert phases == pytest.approx([PHASE_CNLSE, PHASE_CNLSE / 2], abs=PHASE_ATOL)
