"""Naming a solver's parameters after its own physics.

Every solver here integrates the same equation; they differ in what each term
*means*. NLSE reads it optically — a wavenumber, a nonlinear index, a medium
length — while GPE reads the same terms as a cold gas: a mass, an interaction
energy, no medium at all. The parametrisations also disagree on sign, since the
two fields write the interaction term with opposite conventions.

Rather than keep a second copy of every value under its other name, a solver
declares the mapping and shares the storage:

    class GPE(NLSE):
        m = Parameter("k", "Mass of one atom in kg.")
        g = Parameter("n2", "Interaction energy in Hz*m^2.")

Copies were what made this worth changing. They were taken once in __init__,
so assigning the documented name afterwards moved the copy and left the solver
running on the original, and a copy taken from converted storage reported the
converted value rather than the one its owner was given.
"""

from typing import Any


class Parameter:
    """A solver's own name for one of the base class's storage slots.

    Parameters
    ----------
    slot : str
        Attribute the value actually lives in.
    doc : str
        What this parameter means, in this solver's terms.
    scale : float
        Conversion from this parameter to the stored value, as
        ``stored = scale * parameter``. Defaults to 1. Use -1 where the two
        parametrisations write the term with opposite signs.
    """

    def __init__(self, slot: str, doc: str, scale: float = 1.0) -> None:
        self.slot = slot
        self.scale = scale
        self.__doc__ = doc

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the name this parameter was declared under."""
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Return the stored value, converted back to this parameter's units."""
        if obj is None:
            return self
        stored = getattr(obj, self.slot)
        # Left alone at unit scale, so a batched parameter is handed back as
        # the array the caller set rather than a copy of it.
        return stored if self.scale == 1 else stored / self.scale

    def __set__(self, obj: Any, value: Any) -> None:
        """Write through to the storage, converting on the way."""
        setattr(obj, self.slot, value if self.scale == 1 else value * self.scale)
