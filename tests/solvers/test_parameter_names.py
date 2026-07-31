"""GPE and DDGPE name their parameters as their own physics does.

Both solve NLSE's equation with different physics attached to each term, so
they reuse its storage: a mass in the wavenumber slot, an interaction energy in
n2, a total time in L. The physical names are ``Parameter`` views onto that one
storage, so reading and assigning either name reaches the same value, and the
scaled ones (DDGPE's ``g``, GPE's ``sat``) round-trip.
"""

import numpy as np
import pytest
from NLSE import DDGPE, GPE
from scipy.constants import atomic_mass, c, epsilon_0

N = 32
H_BAR = 0.654

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
    """Each physical name must read back the constructor argument.

    ``g`` is the one that did not: the kernels want the opposite sign, the
    constructor stores it negated, and the old copy was taken from the negated
    storage.
    """
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
    """The step limits must read DDGPE's coupling, not an optical conversion.

    DDGPE hands its couplings to the kernels as they are. CNLSE's precompute
    scales them by ``k / 2 * c * epsilon_0``, and DDGPE's ``k`` comes from a
    wavelength it only supplies to keep the base class happy -- 1e-30 m, so
    ``k`` is 6e30. Left at CNLSE's, the interaction rate came out around 1e26
    times too large, the accuracy limit collapsed to 1e-26, and a run over any
    field of order one turned into 1e23 steps.
    """
    simu = DDGPE(**DDGPE_PARAMS)
    A, _ = simu._prepare_output_array(np.ones((2, N, N), dtype=np.complex64), False)
    simu._precompute_step_constants(simu.V, np.complex64)
    rates = simu._energy_rates(A)

    assert rates["interaction"] == pytest.approx(
        abs(DDGPE_PARAMS["g"]) + abs(DDGPE_PARAMS["g12"]), rel=0.5
    ), (
        f"the interaction rate is {rates['interaction']:.3g} for couplings of "
        f"order {abs(DDGPE_PARAMS['g']):.3g}, so it is being scaled by "
        f"something that does not belong to DDGPE"
    )


def test_ddgpe_keeps_a_reasonable_step():
    """A step a caller passes must survive the limiter.

    The end-to-end form of the above: with the rate wrong, every DDGPE run
    was clamped to a step that would take 1e23 of them.
    """
    simu = DDGPE(**DDGPE_PARAMS)
    A, _ = simu._prepare_output_array(np.ones((2, N, N), dtype=np.complex64), False)
    simu._precompute_step_constants(simu.V, np.complex64)
    assert simu._capped_delta_z(1e-3, A, "split_step") == 1e-3, (
        "the limiter reduced a reasonable step, so a DDGPE run would not finish"
    )


def test_ddgpe_propagates_with_the_sign_it_was_given():
    """End to end: the coupling the kernels use must not have flipped.

    The names being views rather than copies moved which attribute the step
    reads. This pins the value that actually reaches it.
    """
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
