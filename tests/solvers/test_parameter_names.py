"""GPE and DDGPE name their parameters as their own physics does."""

import numpy as np
import pytest
from NLSE import DDGPE, GPE
from scipy.constants import atomic_mass, c, epsilon_0

N = 32
H_BAR = 0.654
# On a unit field the interaction rate is the couplings themselves: measured
# 4e-9 off, float32 rounding. At 0.5 a rate off by a third passed.
RATE_RTOL = 1e-6

GPE_PARAMS = {
    "gamma": 0.1,
    "N": 1e5,
    "window": 100e-6,
    "g": 1e3,
    "V": None,
    "m": 87 * atomic_mass,
    "NX": N,
    "NY": N,
    "sat": 1e5,
    "backend": "CPU",
}

DDGPE_PARAMS = {
    "gamma": 0.1,
    "power": 1.0,
    "window": 256,
    "g": 1e-2 / H_BAR,
    "g12": 3e-3 / H_BAR,
    "omega": 5.07 / H_BAR,
    "T": 1,
    "omega_exc": 1484.44 / H_BAR,
    "omega_cav": 1482.76 / H_BAR,
    "detuning": 0.17 / H_BAR,
    "k_z": 27,
    "V": None,
    "NX": N,
    "NY": N,
    "backend": "CPU",
}


@pytest.mark.parametrize("name", ["gamma", "N", "m", "g"])
def test_gpe_reports_what_it_was_given(name):
    """Each physical name must read back the constructor argument."""
    simu = GPE(**GPE_PARAMS)
    assert getattr(simu, name) == GPE_PARAMS[name], (
        f"GPE was given {name}={GPE_PARAMS[name]} but reports {getattr(simu, name)}"
    )


def test_gpe_saturation_round_trips():
    """The saturation is stored converted, so it has to convert back."""
    simu = GPE(**GPE_PARAMS)
    assert simu.sat == pytest.approx(GPE_PARAMS["sat"], rel=1e-12)
    assert simu.I_sat == pytest.approx(GPE_PARAMS["sat"] * epsilon_0 * c / 2, rel=1e-12)


@pytest.mark.parametrize(
    "name,slot",
    [("gamma", "alpha"), ("N", "power"), ("m", "k"), ("g", "n2")],
)
def test_gpe_names_write_through(name, slot):
    """Assigning the physical name must move the storage the solver reads."""
    simu = GPE(**GPE_PARAMS)
    setattr(simu, name, 3.0)
    assert getattr(simu, slot) == 3.0, (
        f"setting GPE.{name} left {slot} alone, so the solver would keep "
        f"running on the old value"
    )


@pytest.mark.parametrize("name", ["gamma", "g", "g12", "T"])
def test_ddgpe_reports_what_it_was_given(name):
    """Each physical name must read back the constructor argument."""
    simu = DDGPE(**DDGPE_PARAMS)
    assert getattr(simu, name) == DDGPE_PARAMS[name], (
        f"DDGPE was given {name}={DDGPE_PARAMS[name]} but reports {getattr(simu, name)}"
    )


def test_ddgpe_storage_keeps_the_kernel_convention():
    """The sign flip has to survive where the kernels read it."""
    simu = DDGPE(**DDGPE_PARAMS)
    assert simu.n2 == -DDGPE_PARAMS["g"], (
        "the storage no longer holds the negated coupling, so the kernels "
        "would receive the wrong sign"
    )


@pytest.mark.parametrize(
    "name,slot,sign",
    [("gamma", "alpha", 1), ("g", "n2", -1), ("g12", "n12", 1), ("T", "L", 1)],
)
def test_ddgpe_names_write_through(name, slot, sign):
    """Assigning the physical name must move the storage, sign included."""
    simu = DDGPE(**DDGPE_PARAMS)
    setattr(simu, name, 2.0)
    assert getattr(simu, slot) == sign * 2.0, (
        f"setting DDGPE.{name} did not put {sign * 2.0} in {slot}"
    )


def test_ddgpe_interaction_rate_is_its_own_coupling():
    """The step limits must read DDGPE's coupling, not an optical conversion."""
    simu = DDGPE(**DDGPE_PARAMS)
    A, _ = simu._prepare_output_array(np.ones((2, N, N), dtype=np.complex64), False)
    simu._precompute_step_constants(simu.V, np.complex64)
    rates = simu._energy_rates(A)

    assert rates["interaction"] == pytest.approx(
        abs(DDGPE_PARAMS["g"]) + abs(DDGPE_PARAMS["g12"]), rel=RATE_RTOL
    ), (
        f"the interaction rate is {rates['interaction']:.3g} for couplings of "
        f"order {abs(DDGPE_PARAMS['g']):.3g}, so it is being scaled by "
        f"something that does not belong to DDGPE"
    )


def test_ddgpe_keeps_a_reasonable_step():
    """A step a caller passes must survive the limiter."""
    simu = DDGPE(**DDGPE_PARAMS)
    A, _ = simu._prepare_output_array(np.ones((2, N, N), dtype=np.complex64), False)
    simu._precompute_step_constants(simu.V, np.complex64)
    assert simu._capped_delta_z(1e-3, A, "split_step") == 1e-3, (
        "the limiter reduced a reasonable step, so a DDGPE run would not finish"
    )


def test_ddgpe_propagates_with_the_sign_it_was_given():
    """End to end: the coupling the kernels use must not have flipped."""
    simu = DDGPE(**DDGPE_PARAMS)
    E = np.ones((2, N, N), dtype=np.complex64)
    out = simu.out_field(
        E,
        2e-3,
        delta_z=1e-3,
        laser_excitation=lambda s, A, t, i: None,
        verbose=False,
        callback=[],
        callback_args=[()],
    )
    assert np.all(np.isfinite(np.asarray(out)))
    # The step reads the storage, which carries the kernel's sign convention.
    assert simu.n2 == -simu.g
