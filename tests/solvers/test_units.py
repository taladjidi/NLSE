"""A solver takes a pint quantity for every argument with a unit."""

import inspect
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
from helpers import ARGUMENTS, assert_same_state
from NLSE import CNLSE, DDGPE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.solvers.units import to_units

pint = pytest.importorskip("pint")
UREG = pint.UnitRegistry()
# A registry of its own: its quantities are not UREG's.
OTHER_REGISTRY = pint.UnitRegistry()

# Arguments with no unit: the grid sizes and the backend's name, and DDGPE's
# power, which no DDGPE run reads (DDGPE.UNITS leaves it out).
NOT_PHYSICAL = {"NX", "NY", "NZ", "backend"}
UNREAD = {DDGPE: {"power"}}
# Each unit a solver declares, and another of its dimension to give it in.
OTHER_UNITS = {
    "1/m": "1/cm",
    "W": "mW",
    "W/m": "mW/cm",
    "m": "mm",
    "m**2/W": "cm**2/W",
    "dimensionless": "percent",
    "W/m**2": "W/cm**2",
    "J": "mJ",
    "s**2/m": "fs**2/mm",
    "m/s": "km/s",
    "s": "ps",
    "rad/m": "rad/cm",
    "1/s": "1/ms",
    "kg": "g",
    "rad*m**2/s": "rad*um**2/ms",
    "1/m**2": "1/um**2",
    "rad/s": "rad/ms",
    "1/ps": "1/ns",
    "um": "mm",
    "rad*um**2/ps": "rad*mm**2/ns",
    "rad/ps": "rad/ns",
    "ps": "fs",
    "1/um": "1/mm",
    "1/um**2": "1/mm**2",
}
# Every argument with a unit, read off the signatures: one missing from a
# solver's UNITS gets a quantity it cannot take, and its case fails.
CASES = [
    pytest.param(cls, name, id=f"{cls.__name__}-{name}")
    for cls in ARGUMENTS
    for name in inspect.signature(cls).parameters
    if name not in NOT_PHYSICAL | UNREAD.get(cls, set())
]
# A float in mm and back to m is off by an ulp at most.
RTOL = 1e-12

# Each solver's units, written out by hand rather than read from UNITS: a
# quantity and the plain number its constructor must turn it into. The optical
# solvers are SI, GPE SI with rates in rad/s, DDGPE ps and um (ddgpe.py).
OPTICS_LITERALS = {
    "alpha": ("0.2 / cm", 20.0),
    "power": ("500 mW", 0.5),
    "window": ("8 mm", 8e-3),
    "n2": ("-1.6e-5 cm**2 / W", -1.6e-9),
    "V": ("5 percent", 0.05),
    "L": ("1 cm", 1e-2),
    "Isat": ("10 W / cm**2", 1e5),
    "nl_length": ("20 um", 2e-5),
    "wvl": ("780 nm", 7.8e-7),
}
LINE_POWER_LITERAL = {"power": ("5 mW / cm", 0.5)}
RABI_LITERALS = {
    "n12": ("-1e-5 cm**2 / W", -1e-9),
    "omega": ("1 rad / cm", 100.0),
}
LITERALS = {
    NLSE: OPTICS_LITERALS,
    NLSE_1d: {**OPTICS_LITERALS, **LINE_POWER_LITERAL},
    NLSE_3d: {
        **{k: v for k, v in OPTICS_LITERALS.items() if k not in {"power", "window"}},
        "energy": ("2 mJ", 2e-3),
        "window": (("8 mm", "4 mm", "2 ps"), (8e-3, 4e-3, 2e-12)),
        "D0": ("1 fs**2 / mm", 1e-27),
        "vg": ("2e5 km / s", 2e8),
    },
    CNLSE: {**OPTICS_LITERALS, **RABI_LITERALS},
    CNLSE_1d: {**OPTICS_LITERALS, **RABI_LITERALS, **LINE_POWER_LITERAL},
    GPE: {
        "gamma": ("1 / ms", 1e3),
        "N": ("50 percent", 0.5),
        "window": ("100 um", 1e-4),
        "g": ("1 rad * um**2 / ms", 1e-9),
        "V": ("1 rad / ms", 1e3),
        "m": ("1 g", 1e-3),
        "sat": ("1 / um**2", 1e12),
        "nl_length": ("1 um", 1e-6),
    },
    DDGPE: {
        "gamma": ("1 / ns", 1e-3),
        "window": ("1 mm", 1e3),
        "g": ("1 rad * mm**2 / ns", 1e3),
        "omega": ("1 rad / ns", 1e-3),
        "T": ("1 ns", 1e3),
        "omega_exc": ("1 rad / fs", 1e3),
        "omega_cav": ("2 rad / fs", 2e3),
        "detuning": ("4 rad / ns", 4e-3),
        "k_z": ("1 / mm", 1e-3),
        "V": ("3 rad / ns", 3e-3),
        "g12": ("2 rad * mm**2 / ns", 2e3),
        "Isat": ("1 / mm**2", 1e-6),
        "nl_length": ("1 mm", 1e3),
    },
}
LITERAL_CASES = [
    pytest.param(cls, name, id=f"{cls.__name__}-{name}")
    for cls, literals in LITERALS.items()
    for name in literals
]
# A window given as a namedtuple, the sequence to_units must hand back as is.
Window = namedtuple("Window", ["x", "y"])
# A window of 4 mm by 2 mm, in m, and in the forms it can be given in.
WINDOW_IN_M = [4e-3, 2e-3]
WINDOW_FORMS = {
    "array": np.array(WINDOW_IN_M),
    "array quantity": UREG.Quantity([4, 2], "mm"),
    "object array": np.array(
        [UREG.Quantity("4 mm"), UREG.Quantity("2 mm")], dtype=object
    ),
}


def quantity(value, unit):
    """Return a value as a quantity in the other unit of its dimension.

    Parameters
    ----------
    value : Any
        Number, array, or sequence of them for a pair of units.
    unit : str or tuple
        The unit it is in.

    Returns
    -------
    Any
        The quantity in ``OTHER_UNITS[unit]``, or a tuple of quantities.
    """
    if isinstance(value, (list, tuple)):
        last = len(value) - 1
        return tuple(quantity(item, unit[i == last]) for i, item in enumerate(value))
    # A float32 array would round on the way to the other unit and back.
    exact = value.astype(np.float64) if isinstance(value, np.ndarray) else value
    return UREG.Quantity(exact, unit).to(OTHER_UNITS[unit])


def parsed(text):
    """Return a literal's quantity, or a tuple of them.

    Parameters
    ----------
    text : str or tuple
        A quantity as pint parses it, or a tuple of such.

    Returns
    -------
    Any
        The quantity, or a tuple of quantities.
    """
    if isinstance(text, tuple):
        return tuple(UREG.Quantity(item) for item in text)
    return UREG.Quantity(text)


@pytest.mark.parametrize("cls", LITERALS, ids=lambda cls: cls.__name__)
def test_every_unit_has_a_literal(cls) -> None:
    """Each entry of a solver's UNITS is checked against a literal below.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    assert set(LITERALS[cls]) == set(cls.UNITS)


@pytest.mark.parametrize(("cls", "name"), LITERAL_CASES)
def test_a_quantity_becomes_the_number_written_here(cls, name) -> None:
    """A quantity arrives as the plain number its unit, by hand, gives.

    Parameters
    ----------
    cls : type
        Solver class.
    name : str
        Argument given as a quantity.
    """
    text, expected = LITERALS[cls][name]
    converted = to_units(cls.UNITS, {name: parsed(text)})[name]
    # abs=0: approx's default 1e-12 would pass any n2 or D0 (1e-9, 1e-27).
    assert converted == pytest.approx(expected, rel=RTOL, abs=0)


@pytest.mark.parametrize(("cls", "name"), CASES)
def test_a_quantity_in_another_unit_is_converted(cls, name) -> None:
    """The solver given a quantity holds what it holds given the number.

    Parameters
    ----------
    cls : type
        Solver class.
    name : str
        Argument given as a quantity.
    """
    arguments = ARGUMENTS[cls]
    given = quantity(arguments[name], cls.UNITS[name])
    simu = cls(**{**arguments, name: given})
    assert_same_state(cls(**arguments), simu, rtol=RTOL)


@pytest.mark.parametrize(("cls", "name"), CASES)
def test_a_wrong_dimension_is_refused(cls, name) -> None:
    """A current where the argument wants another dimension raises.

    Parameters
    ----------
    cls : type
        Solver class.
    name : str
        Argument given in amperes.
    """
    value = ARGUMENTS[cls][name]
    amperes = UREG.Quantity(1.0, "A")
    wrong = tuple(amperes for _ in value) if isinstance(value, tuple) else amperes
    with pytest.raises(pint.DimensionalityError, match=f"{name} takes a quantity"):
        cls(**{**ARGUMENTS[cls], name: wrong})


@pytest.mark.parametrize(("cls", "name"), [(NLSE, "NX"), (DDGPE, "power")])
def test_an_argument_without_a_unit_refuses_a_quantity(cls, name) -> None:
    """A grid size, or DDGPE's unread power, takes no quantity at all.

    Parameters
    ----------
    cls : type
        Solver class.
    name : str
        Argument given as a dimensionless quantity.
    """
    given = ARGUMENTS[cls][name] * UREG.dimensionless
    with pytest.raises(TypeError, match=f"{name} takes a plain number"):
        cls(**{**ARGUMENTS[cls], name: given})


def test_an_array_quantity_is_converted_and_keeps_its_dtype() -> None:
    """A float32 potential in percent arrives as float32, a hundredth of it."""
    V = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    simu = NLSE(**{**ARGUMENTS[NLSE], "V": UREG.Quantity(V, "percent")})
    assert simu.V.dtype == np.float32
    np.testing.assert_allclose(simu.V, V / 100, rtol=1e-6)


@pytest.mark.parametrize("cls", ARGUMENTS, ids=lambda cls: cls.__name__)
def test_plain_numbers_pass_through_untouched(cls) -> None:
    """Every plain argument reaches the constructor as the very same object.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    arguments = ARGUMENTS[cls]
    converted = to_units(cls.UNITS, arguments)
    for name, value in arguments.items():
        assert converted[name] is value, name


def test_a_sequence_without_a_quantity_is_returned_untouched() -> None:
    """A namedtuple of numbers is handed on as it is, and builds a solver."""
    window = Window(4e-3, 2e-3)
    assert to_units(NLSE.UNITS, {"window": window})["window"] is window
    simu = NLSE(**{**ARGUMENTS[NLSE], "window": window})
    assert simu.window is window


def test_a_sequence_of_quantities_becomes_a_tuple_of_numbers() -> None:
    """A namedtuple of quantities comes back a plain tuple, in metres."""
    window = Window(UREG.Quantity("4 mm"), UREG.Quantity("2 mm"))
    assert to_units(NLSE.UNITS, {"window": window})["window"] == (4e-3, 2e-3)


def test_a_list_of_quantities_becomes_a_list_of_numbers() -> None:
    """A list of quantities comes back a new list, in metres."""
    window = [UREG.Quantity("4 mm"), UREG.Quantity("2 mm")]
    converted = to_units(NLSE.UNITS, {"window": window})["window"]
    assert type(converted) is list
    assert converted == pytest.approx(WINDOW_IN_M, rel=RTOL, abs=0)


def test_an_object_array_of_quantities_becomes_numbers() -> None:
    """An object array of quantities is converted as the sequence it is."""
    converted = to_units(NLSE.UNITS, {"window": WINDOW_FORMS["object array"]})
    assert converted["window"].dtype == np.float64
    np.testing.assert_allclose(converted["window"], WINDOW_IN_M, rtol=RTOL)


@pytest.mark.parametrize("window", WINDOW_FORMS.values(), ids=WINDOW_FORMS)
def test_an_array_window_builds_what_a_list_builds(window) -> None:
    """A length-2 array, plain or of quantities, is a window like a list.

    Parameters
    ----------
    window : Any
        The window, 4 mm by 2 mm.
    """
    simu = NLSE(**{**ARGUMENTS[NLSE], "window": window})
    assert_same_state(
        NLSE(**{**ARGUMENTS[NLSE], "window": WINDOW_IN_M}), simu, rtol=RTOL
    )


def test_a_frequency_in_hz_is_taken_without_its_2_pi() -> None:
    """5 Hz, where GPE's V asks for rad/s, arrives as 5 rad/s, as documented."""
    converted = to_units(GPE.UNITS, {"V": UREG.Quantity("5 Hz")})["V"]
    assert converted == pytest.approx(5.0, rel=RTOL, abs=0)


def test_a_quantity_from_another_registry_is_converted() -> None:
    """A quantity of a second UnitRegistry converts as one of the first."""
    given = OTHER_REGISTRY.Quantity("1 cm")
    assert to_units(NLSE.UNITS, {"L": given})["L"] == pytest.approx(
        1e-2, rel=RTOL, abs=0
    )
    simu = NLSE(**{**ARGUMENTS[NLSE], "L": given})
    assert_same_state(NLSE(**{**ARGUMENTS[NLSE], "L": 1e-2}), simu, rtol=RTOL)


def test_plain_numbers_need_no_pint() -> None:
    """With pint failing to import, every solver still builds from numbers."""
    script = (
        "import sys\n"
        "sys.modules['pint'] = None\n"  # import pint now raises ImportError
        f"sys.path.insert(0, {str(Path(__file__).parents[1])!r})\n"
        "from helpers import ARGUMENTS\n"
        "for cls, arguments in ARGUMENTS.items():\n"
        "    cls(**arguments)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
