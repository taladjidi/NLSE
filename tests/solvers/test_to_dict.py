"""A solver rebuilt from its to_dict runs as it does."""

import inspect
import json
from typing import Any

import numpy as np
import pytest
from helpers import (
    ARGUMENTS,
    AVAILABLE_BACKENDS,
    SMALL,
    assert_same_state,
    random_field,
)
from NLSE import CNLSE, DDGPE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.utils import __BACKEND__
from scipy.constants import atomic_mass

# Each solver's field, and the run it is put through: (z, delta_z), ten steps.
FIELDS = {
    NLSE: (SMALL, SMALL),
    NLSE_1d: (SMALL,),
    NLSE_3d: (SMALL, SMALL, SMALL),
    CNLSE: (2, SMALL, SMALL),
    CNLSE_1d: (2, SMALL),
    GPE: (SMALL, SMALL),
    DDGPE: (2, SMALL, SMALL),
}
RUNS = dict.fromkeys(FIELDS, (0.001, 0.0001)) | {GPE: (1e-4, 1e-5), DDGPE: (0.1, 0.01)}
# Samples of DDGPE's temporal drive: one per asked step (ddgpe.py,
# _drive_sample), and RUNS[DDGPE] asks for 10, so twice that is to spare.
DRIVE_SAMPLES = 20
# Attributes changed after construction, as factors: n2 and GPE's m are
# arguments, the rest are set only by assignment and so must travel as
# attributes. One at zero, DDGPE's g2, is set to its factor instead.
CHANGES = {
    "n2": 2.0,
    "m": 0.5,
    "n22": 0.5,
    "alpha2": 0.25,
    "k2": 1.5,
    "power2": 3.0,
    "I_sat2": 2.0,
    "g2": 1e-3,
    "omega_pump": 1.01,
}
# A backend the solver does not default to, so that one rebuilt without it
# would land elsewhere; None where the default is the only one installed.
OTHER_BACKEND = next((name for name in AVAILABLE_BACKENDS if name != __BACKEND__), None)
COUPLED = (CNLSE, CNLSE_1d, DDGPE)
# A wavelength other than ARGUMENTS', in m, set after the rebuild.
NEW_WAVELENGTH = 532e-9
# wvl and m read back through k = 2 pi / wvl, an ulp or so off.
RTOL = 1e-12
# A k and a mass whose trip through wvl = 2 pi / k misses their last bit, found
# by trying values: a glass's index times ARGUMENTS' k (795 nm), and sodium-23.
GLASS_INDEX = 1.48
SODIUM_MASS = 23 * atomic_mass  # kg
OFF_ROUND_TRIP = {
    NLSE: ("k", GLASS_INDEX * 2 * np.pi / ARGUMENTS[NLSE]["wvl"]),
    GPE: ("m", SODIUM_MASS),
}
# The argument that sets k, and the edit made to it in a saved dict.
K_ARGUMENTS = {NLSE: ("wvl", NEW_WAVELENGTH), GPE: ("m", SODIUM_MASS)}


class Relabelled(NLSE):
    """An NLSE with an argument of its own, handing the rest on unnamed."""

    def __init__(self, *args: Any, label: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label = label


def run(simu, field):
    """Return the field a short run of this solver gives.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        The input field.

    Returns
    -------
    np.ndarray
        The output field.
    """
    z, delta_z = RUNS[type(simu)]
    if isinstance(simu, DDGPE):
        flat = np.ones(field.shape[1:], dtype=field.dtype)
        pulse = np.ones(DRIVE_SAMPLES, dtype=field.dtype)
        drive = [(flat, pulse, flat, pulse)]
        return simu.out_field(
            field.copy(), z, None, delta_z, verbose=False, callback_args=drive
        )
    else:
        return simu.out_field(field.copy(), z, delta_z, verbose=False)


def changed(value, factor):
    """Return an attribute's new value: scaled, or the factor where it is 0.

    Parameters
    ----------
    value : float
        The attribute as the constructor set it.
    factor : float
        Its factor in CHANGES.

    Returns
    -------
    float
        The value to assign.
    """
    return factor * value if value != 0 else factor


@pytest.mark.parametrize("cls", ARGUMENTS, ids=lambda cls: cls.__name__)
def test_from_dict_rebuilds_the_run(cls) -> None:
    """The rebuilt solver holds the same state and propagates identically.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    simu = cls(**ARGUMENTS[cls])
    for name in (name for name in CHANGES if hasattr(simu, name)):
        setattr(simu, name, changed(getattr(simu, name), CHANGES[name]))
    rebuilt = cls.from_dict(simu.to_dict())
    assert_same_state(simu, rebuilt)
    assert getattr(rebuilt, "k2", None) == getattr(simu, "k2", None)
    field = random_field(FIELDS[cls])
    assert np.array_equal(run(simu, field), run(rebuilt, field))


@pytest.mark.parametrize("cls", ARGUMENTS, ids=lambda cls: cls.__name__)
def test_to_dict_gives_back_every_constructor_argument(cls) -> None:
    """Every parameter of the signature comes back, at the value given.

    So the argument-to-attribute mapping and the attributes to_dict adds both
    name attributes that exist, and the mapping only arguments that exist.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    arguments = ARGUMENTS[cls]
    parameters = cls(**arguments).to_dict()
    names = set(inspect.signature(cls).parameters)
    assert set(parameters) == names | set(cls._attribute_parameters) - {"k2"} | {"k"}
    assert set(vars(cls).get("_argument_attributes", {})) <= names
    for name in names:
        given, back = arguments[name], parameters[name]
        if isinstance(given, str):
            assert back == given, name
        else:
            np.testing.assert_allclose(back, given, rtol=RTOL, err_msg=name)


@pytest.mark.filterwarnings("ignore:splitting='lie' on a complex128 field")
@pytest.mark.parametrize("cls", OFF_ROUND_TRIP, ids=lambda cls: cls.__name__)
def test_a_k_off_its_round_trip_rebuilds_the_same_run(cls) -> None:
    """The dict's k restores the last bit a rebuild from wvl or m alone loses.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    simu = cls(**ARGUMENTS[cls])
    name, value = OFF_ROUND_TRIP[cls]
    setattr(simu, name, value)
    parameters = simu.to_dict()
    field = random_field(FIELDS[cls], dtype=np.complex128)
    without_k = cls.from_dict({n: v for n, v in parameters.items() if n != "k"})
    assert without_k.k != simu.k
    assert not np.array_equal(run(simu, field), run(without_k, field))
    rebuilt = cls.from_dict(parameters)
    assert rebuilt.k == simu.k
    assert np.array_equal(run(simu, field), run(rebuilt, field))


@pytest.mark.parametrize("cls", K_ARGUMENTS, ids=lambda cls: cls.__name__)
def test_an_edit_that_moves_k_with_the_old_k_kept_raises(cls) -> None:
    """A dict whose wvl (or m) was edited but whose k was not is refused.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    parameters = cls(**ARGUMENTS[cls]).to_dict()
    name, value = K_ARGUMENTS[cls]
    parameters[name] = value
    with pytest.raises(ValueError, match="remove the key 'k'"):
        cls.from_dict(parameters)


@pytest.mark.parametrize("cls", K_ARGUMENTS, ids=lambda cls: cls.__name__)
def test_a_dict_without_k_builds_at_its_edited_argument(cls) -> None:
    """Without k, the edited wvl (or m) sets k as the constructor does.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    parameters = cls(**ARGUMENTS[cls]).to_dict()
    name, value = K_ARGUMENTS[cls]
    parameters[name] = value
    del parameters["k"]
    rebuilt = cls.from_dict(parameters)
    assert rebuilt.k == cls(**{**ARGUMENTS[cls], name: value}).k


@pytest.mark.skipif(OTHER_BACKEND is None, reason="no backend but the default")
def test_from_dict_keeps_a_backend_other_than_the_default() -> None:
    """A solver off the default backend is rebuilt on its own backend."""
    simu = NLSE(**{**ARGUMENTS[NLSE], "backend": OTHER_BACKEND})
    assert NLSE.from_dict(simu.to_dict()).backend == OTHER_BACKEND


@pytest.mark.parametrize("cls", COUPLED, ids=lambda cls: cls.__name__)
def test_an_unassigned_k2_follows_k_after_a_rebuild(cls) -> None:
    """Left alone, k2 is not in the dict, and the rebuilt one follows k.

    Parameters
    ----------
    cls : type
        Coupled solver class.
    """
    simu = cls(**ARGUMENTS[cls])
    rebuilt = cls.from_dict(simu.to_dict())
    for solver in (simu, rebuilt):
        solver.wl = NEW_WAVELENGTH
    assert (rebuilt.k, rebuilt.k2) == (simu.k, simu.k2) == (simu.k, simu.k)


@pytest.mark.parametrize("cls", COUPLED, ids=lambda cls: cls.__name__)
def test_an_assigned_k2_stays_after_a_rebuild(cls) -> None:
    """Assigned, k2 travels in the dict and stays when the wavelength moves.

    Parameters
    ----------
    cls : type
        Coupled solver class.
    """
    simu = cls(**ARGUMENTS[cls])
    simu.k2 = CHANGES["k2"] * simu.k
    rebuilt = cls.from_dict(simu.to_dict())
    for solver in (simu, rebuilt):
        solver.wl = NEW_WAVELENGTH
    assert (rebuilt.k, rebuilt.k2) == (simu.k, simu.k2)
    assert simu.k2 != simu.k


def test_to_dict_copies_the_window() -> None:
    """Editing the dict's window leaves the solver's alone."""
    simu = NLSE(**ARGUMENTS[NLSE])
    parameters = simu.to_dict()
    parameters["window"][0] = 0.0
    assert simu.window[0] == ARGUMENTS[NLSE]["window"]


def test_to_dict_skips_a_subclass_s_args_and_kwargs() -> None:
    """A subclass's own argument comes back; its *args and **kwargs do not.

    So the arguments it forwards are missing, and from_dict on its dict
    raises the constructor's TypeError for them.
    """
    simu = Relabelled(label="probe", **ARGUMENTS[NLSE])
    parameters = simu.to_dict()
    assert parameters["label"] == "probe"
    assert not {"args", "kwargs"} & set(parameters)
    with pytest.raises(TypeError, match="missing 6 required keyword-only"):
        Relabelled.from_dict(parameters)


def test_a_dict_without_arrays_goes_through_json() -> None:
    """Numbers, a list and None are all JSON has to carry."""
    simu = NLSE(**{**ARGUMENTS[NLSE], "V": None})
    text = json.dumps(simu.to_dict())
    assert NLSE.from_dict(json.loads(text)).to_dict() == simu.to_dict()
