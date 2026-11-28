"""Taking a solver's arguments as pint quantities.

Each solver declares the unit of every physical constructor argument once, in
its ``UNITS`` table. A plain number is taken to be in that unit already, as it
always was. A pint quantity is converted to it and stripped before the
constructor's body runs. Every constructor in the chain does this: the
outermost converts, and the ``super().__init__`` it calls gets plain numbers,
which pass through. An attribute assigned after construction is not converted.

pint is optional and never imported here: a quantity cannot exist unless the
caller imported pint, so without it in ``sys.modules`` there is nothing to
convert. Quantities from any registry are recognised, each converted in its
own registry.
"""

import functools
import sys
from collections.abc import Callable, Mapping
from typing import Any, Concatenate, ParamSpec, TypeVar

import numpy as np

# A unit as pint parses it, or a pair for a sequence whose last element is in
# the second unit and the others in the first: NLSE_3d's window, (m, s).
Unit = str | tuple[str, str]

P = ParamSpec("P")
S = TypeVar("S")


def in_units(
    init: Callable[Concatenate[S, P], None],
) -> Callable[Concatenate[S, P], None]:
    """Convert the quantities a constructor is given to its class's ``UNITS``.

    Each call converts, at every level of a constructor chain: a subclass's
    ``super().__init__`` receives plain numbers, which it passes on as they
    are. An attribute assigned after construction is not converted, and
    takes a plain number in its ``UNITS`` unit.

    Parameters
    ----------
    init : callable
        The ``__init__`` to wrap.

    Returns
    -------
    callable
        The same ``__init__``, called with its quantities converted.

    Notes
    -----
    pint counts a radian as 1, so a frequency in Hz has the dimension of an
    angular frequency in rad/s and is not refused: ``5 Hz`` arrives as
    5 rad/s, without its 2 pi. Pass ``2 * np.pi * f`` where ``UNITS`` says
    rad/s.
    """

    @functools.wraps(init)
    def init_in_units(self: Any, *args: P.args, **kwargs: P.kwargs) -> None:
        init(self, *args, **to_units(type(self).UNITS, kwargs))

    return init_in_units


def to_units(units: Mapping[str, Unit], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the arguments with every pint quantity converted and stripped.

    Parameters
    ----------
    units : Mapping
        Unit of each argument that has one.
    arguments : dict
        Argument name to value.

    Returns
    -------
    dict
        The arguments, quantities replaced by their magnitudes in their units.
        The same dict when pint has not been imported.
    """
    pint = sys.modules.get("pint")
    if pint is None:
        return arguments
    return {
        name: to_unit(value, units.get(name), name, pint)
        for name, value in arguments.items()
    }


def to_unit(value: Any, unit: Unit | None, name: str, pint: Any) -> Any:
    """Return one argument in its unit, a plain number left as it is.

    Parameters
    ----------
    value : Any
        The argument: a number, an array, a pint quantity, or a list, tuple
        or object array of them.
    unit : str, tuple or None
        Its unit, None for an argument that has none (a grid size).
    name : str
        The argument's name, for the error.
    pint : module
        The pint module.

    Returns
    -------
    Any
        The value as a plain number or array, in ``unit``.

    Raises
    ------
    TypeError
        If a quantity is given for an argument without a unit.
    pint.DimensionalityError
        If the quantity's dimension is not the unit's.
    """
    if isinstance(value, (list, tuple)):
        return sequence_to_units(value, unit, name, pint)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return object_array_to_units(value, unit, name, pint)
    if not isinstance(value, pint.Quantity):
        return value
    if unit is None:
        raise TypeError(f"{name} takes a plain number, not a quantity in {value.units}")
    try:
        return value.m_as(unit)
    except pint.DimensionalityError as error:
        raise pint.DimensionalityError(
            error.units1,
            error.units2,
            error.dim1,
            error.dim2,
            extra_msg=f": {name} takes a quantity in {unit}",
        ) from None


def sequence_to_units(
    values: list | tuple, unit: Unit | None, name: str, pint: Any
) -> list | tuple:
    """Return a list or tuple with its quantities converted element by element.

    Parameters
    ----------
    values : list or tuple
        The argument, its elements numbers or quantities.
    unit : str, tuple or None
        Its unit; a pair gives the last element's unit second.
    name : str
        The argument's name, for the error.
    pint : module
        The pint module.

    Returns
    -------
    list or tuple
        ``values`` itself when it holds no quantity, a namedtuple included.
        Otherwise a new list for a list and a plain tuple for any tuple.
    """
    if not any(isinstance(item, pint.Quantity) for item in values):
        return values
    units = unit if isinstance(unit, tuple) else (unit, unit)
    last = len(values) - 1
    converted = [
        to_unit(item, units[1] if i == last else units[0], name, pint)
        for i, item in enumerate(values)
    ]
    return converted if isinstance(values, list) else tuple(converted)


def object_array_to_units(
    values: np.ndarray, unit: Unit | None, name: str, pint: Any
) -> np.ndarray:
    """Return an object array with its quantities converted, as numbers.

    ``np.array([q1, q2], dtype=object)`` is a sequence of quantities, so its
    elements are converted as those of a list, in its flat order.

    Parameters
    ----------
    values : np.ndarray
        The argument, an array of dtype object.
    unit : str, tuple or None
        Its unit; a pair gives the last element's unit second.
    name : str
        The argument's name, for the error.
    pint : module
        The pint module.

    Returns
    -------
    np.ndarray
        ``values`` itself when it holds no quantity. Otherwise an array of the
        magnitudes, of the same shape.
    """
    flat = list(values.flat)
    converted = sequence_to_units(flat, unit, name, pint)
    if converted is flat:
        return values
    return np.asarray(converted).reshape(values.shape)
