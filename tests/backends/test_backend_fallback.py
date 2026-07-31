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
    """Falling back to the slowest would be a poor way to keep a promise.

    Self-consistent on purpose -- it pins the fallback to the ranking. What the
    ranking itself has to get right is below, because this passes either way:
    when the ranking opened with CPU, so did the fallback, and this agreed.
    """
    with pytest.warns(UserWarning):
        backend = get_backend(MISSING[0], grid_size=(64, 64))
    assert backend.name == backends_by_speed((64, 64))[0]


DEVICES = [n for n in list_available_backends() if n != "CPU"]


@pytest.fixture
def unmeasured(monkeypatch):
    """Rank with no benchmark cache, whatever this machine has cached."""
    monkeypatch.setattr(
        "NLSE.backends.benchmark.load_benchmark_cache", lambda *args, **kwargs: None
    )


@pytest.mark.skipif(not DEVICES, reason="no device backend installed here")
def test_the_host_is_ranked_last_when_nothing_is_measured(unmeasured):
    """A device backend outranks the host, measurement or none.

    The bug this is here for: the ranking used to be the availability list,
    which opens with CPU, so a fallback with no cache to read landed on the
    host -- on a CUDA box, asking for MLX ran on the CPU past two device
    backends that were installed and faster.
    """
    ranked = backends_by_speed((64, 64))
    assert ranked[-1] == "CPU", (
        f"the host should be the last resort, and this ranking is {ranked}"
    )


@pytest.mark.skipif(not DEVICES, reason="no device backend installed here")
def test_an_untimed_device_is_not_stranded_behind_a_timed_host(monkeypatch):
    """A half-filled cache must not rank the host above an untimed device.

    Backends the cache has no time for sort together, and whatever breaks that
    tie decides the fallback. Left to the availability list it was the host.
    """
    monkeypatch.setattr(
        "NLSE.backends.benchmark.load_benchmark_cache",
        lambda *args, **kwargs: {
            "grid_size": [64, 64],
            "results": {"CPU": {"time_ms": 12.0}},
        },
    )
    ranked = backends_by_speed((64, 64))
    assert ranked[0] == "CPU", "the only timed backend should lead on its measurement"
    assert ranked[1:] == sorted(
        DEVICES, key=lambda n: ("CUPY", "MLX", "CL").index(n)
    ), (
        f"untimed devices should fall in the assumed order, not the availability "
        f"one; got {ranked}"
    )


@pytest.mark.skipif(
    not MISSING or not DEVICES, reason="needs a missing backend and a device"
)
def test_the_fallback_reaches_a_device_and_not_the_host(unmeasured):
    """The whole point, end to end: the symptom was a run on the CPU."""
    with pytest.warns(UserWarning, match="fastest one available"):
        backend = get_backend(MISSING[0], grid_size=(64, 64))
    assert backend.name != "CPU", (
        f"asked for {MISSING[0]} with {DEVICES} installed and got the host"
    )


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
