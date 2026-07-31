"""Naming a backend is a preference; naming a wrong one is a typo.

A backend decides how a run goes, not what it computes, so a script that asks
for one that is not installed here is answered with the fastest one that is.
Every example in this repository names a backend, and most name CUPY, which
means the alternative is that most of them fail everywhere but one machine.

A name that is not a backend at all still raises: that is a typo, and guessing
at it would hide it.
"""

import warnings

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import (
    backends_by_speed,
    fastest_backend_supporting,
    get_backend,
    list_available_backends,
)

ALL_NAMES = ("CPU", "CUPY", "CL", "MLX")
MISSING = [n for n in ALL_NAMES if n not in list_available_backends()]


@pytest.mark.skipif(not MISSING, reason="every backend is installed here")
@pytest.mark.parametrize("name", MISSING)
def test_an_uninstalled_backend_falls_back_and_says_so(name):
    """The run happens, on something that exists, and the swap is announced."""
    with pytest.warns(UserWarning, match=f"{name} backend is not installed"):
        backend = get_backend(name, grid_size=(64, 64))
    assert backend.name in list_available_backends()


@pytest.mark.skipif(not MISSING, reason="every backend is installed here")
def test_the_fallback_is_the_fastest_available():
    """Falling back to the slowest would be a poor way to keep a promise."""
    with pytest.warns(UserWarning):
        backend = get_backend(MISSING[0], grid_size=(64, 64))
    assert backend.name == backends_by_speed((64, 64))[0]


def test_a_name_that_is_not_a_backend_still_raises():
    """Guessing at a typo hides the typo."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("CUDA")


@pytest.mark.skipif(not MISSING, reason="every backend is installed here")
def test_a_solver_asking_for_a_missing_backend_still_propagates():
    """The end of the story is a field, not an exception."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simu = NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=1e-3,
            NX=64,
            NY=64,
            Isat=1e5,
            backend=MISSING[0],
        )
        field = np.ones((64, 64), dtype=np.complex64)
        out = simu.out_field(field, 1e-3, verbose=False, plot=False)
    assert np.all(np.isfinite(np.asarray(simu._backend.to_numpy(out)).view(np.float32)))


def test_every_available_backend_is_ranked():
    """The ranking is what the fallback picks from, so it must be complete."""
    assert sorted(backends_by_speed((64, 64))) == sorted(list_available_backends())


def test_a_requirement_nothing_meets_returns_nothing():
    """The caller has to be able to tell 'none' from 'the first one'."""
    assert fastest_backend_supporting(lambda backend: False, (64, 64)) is None
