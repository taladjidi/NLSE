"""A lossy real-space step must be solved, not frozen, or nothing keeps order."""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS, make
from NLSE import CNLSE, DDGPE, NLSE
from NLSE.backends import get_backend
from NLSE.kernels.cpu import _LOSS_SOLVED_LIMIT
from NLSE.solvers.step_size import LOSS_PER_STEP_LIMIT
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq

N = 32
WAIST = 2.23e-3
WINDOW = 4 * WAIST
ALPHA = 20.0
PHYSICS = {
    "power": 4.0,
    "window": WINDOW,
    "n2": -1.6e-9,
    "V": None,
    "L": 5e-3,
    "Isat": 1e6,
}
# Coarse enough that the splitting error is well above complex128 round-off,
# fine enough that it is well below saturation. Halved twice: a first-order
# method quarters its drift over the pair, a second-order one divides it by 16.
PHASES = (0.8, 0.4, 0.2)
# A step past the solved range, in kernel units: u = 2 * alpha * dz = 0.6,
# six times _LOSS_SOLVED_LIMIT. No input is 1 or equals another, so a kernel
# that drops or inverts one lands elsewhere: |A|^2 = 2.5, sat = 4/9, the
# frozen step decays by exp(-2/15) and turns by 2 (1/3 + 1/2) = 1.67 rad.
FROZEN_DZ = 2.0
FROZEN_ALPHA = 0.15
FROZEN_G = 0.3
FROZEN_V = 0.5
FROZEN_ISAT = 2.0
FROZEN_FIELD = 1.5 + 0.5j
FROZEN_N = 8
# Float32 exponentials of order-one arguments: measured 2.5e-8 (CPU), 1.2e-7 (MLX).
FROZEN_RTOL = 1e-6
# The step's closed form holds |A|^2 at its entry value: 2.5, so sat = 4/9.
FROZEN_INTENSITY = abs(FROZEN_FIELD) ** 2
FROZEN_SAT = 1 / (1 + FROZEN_INTENSITY / FROZEN_ISAT)
# The two other cases the same closed form covers: no loss (1/m), where holding
# |A|^2 is exact, and no potential (rad/m), which takes the kernels without V.
LOSSLESS = 0.0
NO_POTENTIAL = 0.0
# A CNLSE lossy on its second component only, 1/m, on a uniform field at the
# saturation intensity. The step asked takes u = 2 * alpha2 * dz = 1 a step,
# twenty times LOSS_PER_STEP_LIMIT; the coupled kernels freeze the step.
SECOND_ALPHA = 5e3
SECOND_N = 16
SECOND_L = 1e-3  # m
SECOND_DZ = 1e-4  # m
# Relative error of the cavity density against the closed form. Measured
# 1.8e-3 at the capped step and 4.1e-2 uncapped (CPU, complex128).
SECOND_RTOL = 1e-2
# Smallest intensity the closed form is searched down to, relative to y0.
DECAY_FLOOR = 1e-30


def solver(alpha, backend="CPU"):
    """Return a solver with this much loss."""
    return make(NLSE, backend, n=N, alpha=alpha, **PHYSICS)


def beam():
    """Return a smooth Gaussian, in double precision to see past round-off."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / WAIST**2).astype(np.complex128)


def propagate(simu, phase, splitting):
    """Propagate at a step imprinting this phase, and return the field."""
    A = beam()
    prepared, _ = simu._prepare_output_array(A.copy(), normalize=True)
    simu._precompute_step_constants(simu.V, np.complex128)
    rates = simu._energy_rates(prepared)
    delta_z = phase / (rates["potential"] + rates["interaction"])
    out = simu.out_field(
        A.copy(),
        PHYSICS["L"],
        delta_z=delta_z,
        verbose=False,
        plot=False,
        splitting=splitting,
    )
    return np.asarray(simu._backend.to_numpy(out)).astype(np.complex128)


def drift_ratio(alpha, splitting, backend="CPU"):
    """Return how much halving the step divides the change in the answer by.

    Two for a first-order method, four for second, sixteen for fourth.
    """
    fields = [propagate(solver(alpha, backend), phase, splitting) for phase in PHASES]

    def distance(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    coarse = distance(fields[0], fields[1])
    fine = distance(fields[1], fields[2])
    assert fine > 0, f"{splitting} stopped moving; the drifts are at the floor"
    return coarse / fine


@pytest.mark.parametrize(
    "splitting,expected", [("lie", 2.0), ("strang", 4.0), ("yoshida", 16.0)]
)
def test_loss_does_not_cost_the_splitting_its_order(splitting, expected):
    """Each composition must converge at its own order, loss or no loss."""
    ratio = drift_ratio(ALPHA, splitting)
    assert ratio == pytest.approx(expected, rel=0.25), (
        f"{splitting} with loss converges at a ratio of {ratio:.2f} where "
        f"{expected:.0f} is its order; a ratio of 2 means the real-space step "
        f"is being frozen again"
    )


@pytest.mark.parametrize("splitting", ["strang", "yoshida"])
def test_loss_converges_like_no_loss(splitting):
    """The lossy and lossless orders must be the same order."""
    lossy = drift_ratio(ALPHA, splitting)
    lossless = drift_ratio(0.0, splitting)
    assert lossy == pytest.approx(lossless, rel=0.25), (
        f"{splitting} converges at {lossy:.2f} with loss against "
        f"{lossless:.2f} without: loss is still costing it order"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_every_backend_solves_the_step_the_same_way(backend_name):
    """The five implementations of the lossy step's iteration agree."""
    if not solver(ALPHA, backend_name)._backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision to agree in")
    reference = propagate(solver(ALPHA, "CPU"), PHASES[-1], "strang")
    got = propagate(solver(ALPHA, backend_name), PHASES[-1], "strang")
    difference = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
    assert difference < 1e-6, (
        f"{backend_name} disagrees with CPU on a lossy step by {difference:.3e}"
    )


def test_a_step_too_lossy_to_solve_still_decays():
    """A step too lossy for the iteration's range still decays rather than grows."""
    from NLSE.kernels import cpu as cpu_kernels

    field = np.ones((8, 8), dtype=np.complex64)
    A_sq = np.ones((8, 8), dtype=np.float32)
    # u = 2*alpha*dz, swept from inside the solved range to far outside it.
    for alpha, dz in ((20.0, 1e-3), (50.0, 1e-2), (500.0, 1e-1), (5e3, 1.0)):
        out = cpu_kernels.nl_prop_without_V(
            field.copy(),
            A_sq,
            np.float32(dz),
            np.float32(alpha),
            np.float32(1e-3),
            np.float32(1e5),
        )
        amplitude = float(np.abs(out[0, 0]))
        assert amplitude <= 1.0, (
            f"a lossy step with u = {2 * alpha * dz:.3g} returned an amplitude "
            f"of {amplitude}, so the iteration is being used outside its range"
        )


@pytest.mark.parametrize("requested", [1e-3, 1e-2, 1.0])
def test_the_solver_caps_a_step_that_would_leave_the_solved_range(requested):
    """The caller's step is lowered until ``u = 2*alpha*dz`` is in range."""
    simu = make(NLSE, "CPU", n=N, alpha=ALPHA, **{**PHYSICS, "n2": -1e-15})
    simu.out_field(
        beam().astype(np.complex64),
        PHYSICS["L"],
        delta_z=requested,
        verbose=False,
        plot=False,
    )
    used = simu._current_delta_z
    u = 2 * ALPHA * used
    assert u <= _LOSS_SOLVED_LIMIT, (
        f"asked for delta_z={requested:g} m with alpha={ALPHA}, the solver used "
        f"{used:.3e} m, which takes out u = {u:.3g} of the intensity per step "
        f"-- past {_LOSS_SOLVED_LIMIT}, where the kernel stops solving the step"
    )


def test_a_lossless_run_is_not_capped_by_the_loss_ceiling():
    """The ceiling is infinite without loss, or every run pays for it."""
    simu = solver(0.0)
    assert np.isinf(simu._loss_max_dz()), (
        "a lossless medium reported a finite loss ceiling, which would shorten "
        "the step of every run that has no loss at all"
    )


def step_once(implementation, fused, alpha, potential):
    """Take one real-space step of a uniform ``FROZEN_FIELD``.

    Parameters
    ----------
    implementation : Backend
        Backend whose kernels take the step.
    fused : bool
        Whether the kernel computes ``|A|^2`` itself (``square_mod_nl_prop*``)
        or is handed it (``nl_prop*``).
    alpha : float
        Loss, in the kernel's 1/m.
    potential : float
        Uniform potential, rad/m; ``NO_POTENTIAL`` takes the kernel without V.

    Returns
    -------
    np.ndarray
        The field after the step, on the host.
    """
    shape = (FROZEN_N, FROZEN_N)
    kernels = implementation.kernels
    field = implementation.from_numpy(np.full(shape, FROZEN_FIELD, dtype=np.complex64))
    V = implementation.from_numpy(np.full(shape, potential, dtype=np.float32))
    A_sq = implementation.from_numpy(np.full(shape, FROZEN_INTENSITY, np.float32))
    scalars = (FROZEN_DZ, alpha, FROZEN_G, FROZEN_ISAT)
    if fused and potential == NO_POTENTIAL:
        out = kernels.square_mod_nl_prop(field, *scalars)
    elif fused:
        out = kernels.square_mod_nl_prop_v(field, V, *scalars)
    elif potential == NO_POTENTIAL:
        out = kernels.nl_prop_without_V(field, A_sq, *scalars)
    else:
        out = kernels.nl_prop(field, A_sq, FROZEN_DZ, alpha, V, FROZEN_G, FROZEN_ISAT)
    return np.asarray(implementation.to_numpy(out))


@pytest.mark.parametrize("fused", [False, True], ids=["nl_prop", "square_mod"])
@pytest.mark.parametrize("potential", [NO_POTENTIAL, FROZEN_V], ids=["no_V", "V"])
@pytest.mark.parametrize("alpha", [LOSSLESS, FROZEN_ALPHA], ids=["lossless", "frozen"])
def test_a_frozen_step_applies_loss_interaction_and_potential(
    backend, fused, alpha, potential
):
    """Unsolved, the step multiplies by ``exp(dz (-alpha s + i g |A|^2 s + i V))``.

    Exact without loss, where ``|A|^2`` is constant; past the solved range, the
    step that holds ``|A|^2`` at its entry value. Every sign and term shows.

    Parameters
    ----------
    backend : str
        Backend to run on.
    fused : bool
        Whether the kernel computes ``|A|^2`` itself.
    alpha : float
        Loss, 1/m: ``LOSSLESS`` or ``FROZEN_ALPHA``, past the solved range.
    potential : float
        Uniform potential, rad/m.
    """
    out = step_once(get_backend(backend), fused, alpha, potential)
    decay = np.exp(-alpha * FROZEN_SAT * FROZEN_DZ)
    phase = FROZEN_DZ * (FROZEN_G * FROZEN_INTENSITY * FROZEN_SAT + potential)
    np.testing.assert_allclose(
        np.abs(out), abs(FROZEN_FIELD) * decay, rtol=FROZEN_RTOL, err_msg="loss"
    )
    np.testing.assert_allclose(
        np.angle(out / FROZEN_FIELD), phase, rtol=FROZEN_RTOL, err_msg="phase"
    )


def saturable_decay(y0, y_other, isat, alpha, z):
    """Return the intensity a uniform component keeps after a saturable loss.

    ``dy/dz = -alpha y / (1 + y_other/isat + y/isat)``, with ``y_other`` held,
    integrates to ``a ln y + y/isat = a ln y0 + y0/isat - alpha z``, where
    ``a = 1 + y_other/isat``.

    Parameters
    ----------
    y0 : float
        Intensity entering, as ``|A|^2``.
    y_other : float
        The other component's intensity, constant.
    isat : float
        Saturation, in the units of ``y0``.
    alpha : float
        Loss rate of the intensity, 1/m.
    z : float
        Distance, m.

    Returns
    -------
    float
        The intensity at ``z``.
    """
    a = 1 + y_other / isat
    target = a * np.log(y0) + y0 / isat - alpha * z
    return brentq(lambda y: a * np.log(y) + y / isat - target, DECAY_FLOOR * y0, y0)


def test_the_second_component_loss_caps_the_step(backend):
    """``alpha2`` bounds the step as ``alpha`` does, or its decay comes out wrong.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu = make(CNLSE, backend, n=SECOND_N, alpha=0.0)
    simu.alpha2 = SECOND_ALPHA
    isat = 2 * simu.I_sat / (epsilon_0 * c)
    field = np.full((2, SECOND_N, SECOND_N), np.sqrt(isat), dtype=np.complex64)
    with pytest.warns(UserWarning, match="exceeds split-step accuracy limit"):
        out = simu.out_field(
            field, SECOND_L, delta_z=SECOND_DZ, verbose=False, normalize=False
        )
    got = float(np.mean(np.abs(np.asarray(simu._backend.to_numpy(out))[1]) ** 2))
    expected = saturable_decay(isat, isat, isat, SECOND_ALPHA, SECOND_L)
    assert got == pytest.approx(expected, rel=SECOND_RTOL), (
        f"the lossy component kept {got / isat:.5f} of its intensity where "
        f"{expected / isat:.5f} was due. (Backend {backend})"
    )


def one_component_lossy(backend, lossy):
    """Return a CNLSE that loses on this component (0 or 1) only, and its field."""
    simu = make(CNLSE, backend, n=SECOND_N, alpha=SECOND_ALPHA * (1 - lossy))
    simu.alpha2 = SECOND_ALPHA * lossy
    isat = 2 * simu.I_sat / (epsilon_0 * c)
    return simu, np.full((2, SECOND_N, SECOND_N), np.sqrt(isat), dtype=np.complex64)


def second_component_lossy(backend):
    """Return a CNLSE that loses on its second component only, and its field."""
    return one_component_lossy(backend, 1)


def rk4_bound_in_a_run(backend, lossy):
    """Return the loss rate and the RK4 bound as a run reads them.

    A run has planned its transforms, which puts the rates on the device path.

    Parameters
    ----------
    backend : str
        Backend to run on.
    lossy : int
        The component that loses, 0 or 1.

    Returns
    -------
    tuple of float
        The loss rate in 1/m and the largest stable RK4 step in m.
    """
    simu, host = one_component_lossy(backend, lossy)
    field = simu._backend.from_numpy(host)
    simu.plans = simu._build_fft_plan(field)
    assert simu._can_rate_on_device(field)
    return simu._energy_rates(field)["loss"], simu._rk4_max_dz(field)


def test_the_second_component_loss_enters_the_loss_rate(backend):
    """The amplitude of the lossy component decays at ``alpha2 / 2``.

    RK4's stability bound and the adaptive step read this rate.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu, field = second_component_lossy(backend)
    assert simu._energy_rates(field)["loss"] == pytest.approx(SECOND_ALPHA / 2)


def test_the_second_component_loss_bounds_rk4_in_a_run(backend):
    """The same loss gives the same RK4 bound on either component.

    9.56e-4 m here; 6.15e-3 m, 6.4 times looser, if only ``alpha`` is read.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    loss, bound = rk4_bound_in_a_run(backend, 1)
    assert loss == pytest.approx(SECOND_ALPHA / 2)
    assert bound == pytest.approx(rk4_bound_in_a_run(backend, 0)[1])


def test_yoshida_warns_about_the_second_component_loss(backend):
    """A loss on either component runs backwards in Yoshida's middle sub-step.

    Parameters
    ----------
    backend : str
        Backend to run on.
    """
    simu, field = second_component_lossy(backend)
    with pytest.warns(UserWarning, match="splitting='yoshida' with loss"):
        simu.out_field(
            field, SECOND_L, splitting="yoshida", verbose=False, normalize=False
        )


def test_the_cavity_loss_caps_the_step():
    """DDGPE's ``gamma2`` bounds the step: ``LOSS_PER_STEP_LIMIT / (2 gamma2)``."""
    simu = DDGPE(
        gamma=0.0,
        power=1.0,
        window=float(SECOND_N),
        g=0.0,
        omega=0.0,
        T=1.0,
        omega_exc=1.0,
        omega_cav=1.0,
        detuning=0.0,
        k_z=1.0,
        NX=SECOND_N,
        NY=SECOND_N,
        backend="CPU",
    )
    simu.gamma2 = SECOND_ALPHA
    assert simu._loss_max_dz() == pytest.approx(
        LOSS_PER_STEP_LIMIT / (2 * SECOND_ALPHA)
    )
