"""Every solver takes its arguments by keyword only."""

import inspect

import pytest
from helpers import ARGUMENTS


@pytest.mark.parametrize("cls", ARGUMENTS, ids=lambda cls: cls.__name__)
def test_a_positional_argument_is_refused(cls) -> None:
    """Every argument given in signature order, none named, raises.

    Parameters
    ----------
    cls : type
        Solver class.
    """
    arguments = ARGUMENTS[cls]
    positional = [arguments[name] for name in inspect.signature(cls).parameters]
    with pytest.raises(TypeError, match="positional argument"):
        cls(*positional)
