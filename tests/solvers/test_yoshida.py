"""The fourth-order splitting, and when it is the wrong tool.

Yoshida composes three Strang sub-steps with weights summing to one, the
middle one backwards. Two things are worth pinning: that it really converges
at fourth order rather than merely running, and that it says so when the float
width or the physics makes it the wrong choice — a negative sub-step in a
lossy medium amplifies, and in complex64 the order is bought with three
transform pairs for accuracy round-off will not let the field hold.
"""

import warnings

import numpy as np
import pytest
from helpers import make
from NLSE import NLSE
from NLSE.solvers.nlse import YOSHIDA_WEIGHTS

N = 64
WAIST = 2.23e-3
WINDOW = 4 * WAIST
L = 2e-3


def beam(dtype=np.complex128):
    """Return a beam strong enough that the splitting error is visible."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, N)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / WAIST**2).astype(dtype)


def solver(**kwargs):
    """Return a lossless, strongly nonlinear solver."""
    params = {"alpha": 0, "power": 4.0, "window": WINDOW, "L": L, "Isat": 1e6}
    params.update(kwargs)
    return make(NLSE, "CPU", n=N, **params)


def propagate(splitting, dz, dtype=np.complex128):
    """Propagate with one splitting, ignoring the advice it gives."""
    simu = solver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = simu.out_field(
            beam(dtype), L, delta_z=dz, verbose=False, plot=False, splitting=splitting
        )
    return np.asarray(simu._backend.to_numpy(out)).astype(np.complex128)


def error(got, reference):
    """Relative L2 difference."""
    return float(np.linalg.norm(got - reference) / np.linalg.norm(reference))


def test_the_weights_compose_a_whole_step():
    """A step made of the three sub-steps advances by exactly one step."""
    assert sum(YOSHIDA_WEIGHTS) == pytest.approx(1.0, abs=1e-15)
    assert min(YOSHIDA_WEIGHTS) < 0, (
        "the middle weight has to be negative; a composition of three "
        "positive sub-steps cannot exceed second order"
    )


def test_it_converges_at_fourth_order():
    """Halving the step must cut the error by about sixteen, not four.

    Measured against Strang on the same problem, which is second order: if
    the composition were wrong in a way that still ran, the most likely
    outcome is that it stays second order.
    """
    reference = propagate("yoshida", L / 512)
    coarse, fine = L / 8, L / 16
    orders = {}
    for splitting in ("strang", "yoshida"):
        e_coarse = error(propagate(splitting, coarse), reference)
        e_fine = error(propagate(splitting, fine), reference)
        orders[splitting] = np.log2(e_coarse / e_fine)

    assert orders["strang"] == pytest.approx(2.0, abs=0.4), (
        f"Strang should be second order, measured {orders['strang']:.2f}"
    )
    assert orders["yoshida"] > 3.5, (
        f"Yoshida should be fourth order, measured {orders['yoshida']:.2f}"
    )


def test_it_is_more_accurate_than_strang_at_the_same_step():
    """At one step size it must beat Strang, or it is not worth three pairs."""
    reference = propagate("yoshida", L / 512)
    dz = L / 16
    assert error(propagate("yoshida", dz), reference) < error(
        propagate("strang", dz), reference
    )


def test_it_warns_on_a_single_precision_field():
    """complex64 cannot hold the accuracy the extra transforms buy."""
    simu = solver()
    with pytest.warns(UserWarning, match="complex64"):
        simu.out_field(
            beam(np.complex64),
            L,
            delta_z=L / 8,
            verbose=False,
            plot=False,
            splitting="yoshida",
        )


def test_it_warns_when_a_backwards_step_would_amplify():
    """The middle sub-step runs backwards, which gain does not survive."""
    simu = solver(alpha=1.0)
    with pytest.warns(UserWarning, match="backwards"):
        simu.out_field(
            beam(), L, delta_z=L / 8, verbose=False, plot=False, splitting="yoshida"
        )


@pytest.mark.parametrize("splitting", ["lie", "strang"])
def test_a_double_precision_run_is_pointed_at_yoshida(splitting):
    """The splitting is what is left once round-off is out of the way."""
    simu = solver()
    with pytest.warns(UserWarning, match="yoshida"):
        simu.out_field(
            beam(), L, delta_z=L / 8, verbose=False, plot=False, splitting=splitting
        )


def test_a_single_precision_run_is_left_alone():
    """No advice for the combination that is already right."""
    simu = solver()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simu.out_field(
            beam(np.complex64),
            L,
            delta_z=L / 8,
            verbose=False,
            plot=False,
            splitting="strang",
        )


def test_an_unknown_splitting_is_refused():
    """It used to fall through to Lie, silently."""
    simu = solver()
    with pytest.raises(ValueError, match="splitting must be"):
        simu.out_field(
            beam(), L, delta_z=L / 8, verbose=False, plot=False, splitting="ruth"
        )


@pytest.mark.parametrize("gone", ["single", "double"])
def test_the_old_spelling_is_gone(gone):
    """It named a float width to every reader, so it is not kept alive."""
    simu = solver()
    with pytest.raises(ValueError, match="splitting must be"):
        simu.out_field(
            beam(), L, delta_z=L / 8, verbose=False, plot=False, splitting=gone
        )


def test_the_old_keyword_is_gone():
    """precision= is not quietly accepted either."""
    simu = solver()
    with pytest.raises(TypeError, match="precision"):
        simu.out_field(
            beam(), L, delta_z=L / 8, verbose=False, plot=False, precision="double"
        )
