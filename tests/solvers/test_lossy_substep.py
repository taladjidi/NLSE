"""A lossy real-space step must be solved, not frozen, or nothing keeps order.

The real-space step applies ``exp(-alpha*s*dz + i*g*|A|^2*s*dz)`` with ``|A|^2``
read once, entering. That is the exact solution of the real-space equation only
while the step preserves ``|A|^2`` -- true of a pure rotation, false the moment
there is loss, because the amplitude decays *inside* the step while the
interaction goes on turning the phase at the rate the step began with.

Frozen, the sub-step is O(dz^2) locally and O(dz) over a run, and that ceiling
lands on the composition wrapped around it: Strang and Yoshida both came out
**first order** on a lossy problem, which is what these tests would catch a
return of. See ``_loss_factor`` in kernels/cpu.py for what replaced it.

The order tests use a smooth beam rather than anything turbulent. Order is only
readable where the error is neither saturated nor at the round-off floor, and a
chaotic problem is in the first of those within a few steps.
"""

import numpy as np
import pytest
from helpers import make
from NLSE import NLSE
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

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
    """Each composition must converge at its own order, loss or no loss.

    The number is the claim: before the sub-step was solved, Strang and Yoshida
    both returned 2.0 here -- first order -- while their lossless twins
    returned 4 and 16.
    """
    ratio = drift_ratio(ALPHA, splitting)
    assert ratio == pytest.approx(expected, rel=0.25), (
        f"{splitting} with loss converges at a ratio of {ratio:.2f} where "
        f"{expected:.0f} is its order; a ratio of 2 means the real-space step "
        f"is being frozen again"
    )


@pytest.mark.parametrize("splitting", ["strang", "yoshida"])
def test_loss_converges_like_no_loss(splitting):
    """The lossy and lossless orders must be the same order.

    A tighter statement than the one above and the reason it holds: what the
    solved step buys is that loss stops being visible to the composition at
    all.
    """
    lossy = drift_ratio(ALPHA, splitting)
    lossless = drift_ratio(0.0, splitting)
    assert lossy == pytest.approx(lossless, rel=0.25), (
        f"{splitting} converges at {lossy:.2f} with loss against "
        f"{lossless:.2f} without: loss is still costing it order"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_every_backend_solves_the_step_the_same_way(backend_name):
    """Five implementations of one iteration have to agree.

    The formula is written out once per dialect -- numba, CUDA C, OpenCL C,
    MLX, and the fused CuPy kernels for broadcasting -- so this is the test
    that stops one of them drifting from the others.

    In double precision, because the agreement worth checking is closer than
    float32 round-off. That rules the comparison out where there is no double
    to be had: MLX is single-precision throughout, and Apple's OpenCL ships no
    fp64, so both would be scored against a reference of another width.
    """
    if not solver(ALPHA, backend_name)._backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision to agree in")
    reference = propagate(solver(ALPHA, "CPU"), PHASES[-1], "strang")
    got = propagate(solver(ALPHA, backend_name), PHASES[-1], "strang")
    difference = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
    assert difference < 1e-6, (
        f"{backend_name} disagrees with CPU on a lossy step by {difference:.3e}"
    )


def test_a_step_too_lossy_to_solve_still_decays():
    """The iteration has a range, and outside it the field must not grow.

    It is a contraction only while the step takes out a small fraction of the
    intensity. Past that it walks away and returns a *larger* field than it was
    given -- which is worse than an inaccurate answer, because a growing field
    does not look like a step-size problem. Outside its range the kernels fall
    back to the frozen step, which at least only ever decays.
    """
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
