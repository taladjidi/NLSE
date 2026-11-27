"""The dispersion eigenvalues are built once, not on every call.

The step limiters read them on every ``out_field`` call and built them twice
per call on the host, 4.8 ms each at 2048x2048: a large part of the fixed
cost of a short run. They depend only on the grid and the wavenumber, the
key they are kept under.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 32
WAIST = 2.23e-3  # m
DELTA_Z = 1e-4  # m
# out_field calls per test: a rebuild on any call after the first counts.
RUNS = 3
# Factor the wavenumber is changed by between two runs of one solver.
WAVENUMBER_CHANGE = 0.5
# The rate against K^2 / (2 k), both float64 on the host: measured 0 apart
# (CPU, MLX). The previous wavenumber's rate is off by 1 / WAVENUMBER_CHANGE.
DISPERSION_RATE_RTOL = 1e-12


def make_solver(backend):
    """Return the small solver both tests run.

    Parameters
    ----------
    backend : str
        Backend name, one of ``AVAILABLE_BACKENDS``.

    Returns
    -------
    NLSE
        A 2D solver on that backend.
    """
    return NLSE(
        alpha=0,
        power=1.05,
        window=4 * WAIST,
        n2=-1.6e-9,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        Isat=1e5,
        backend=backend,
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("delta_z", [None, DELTA_Z])
def test_repeated_runs_build_the_dispersion_eigenvalues_once(
    backend, delta_z, monkeypatch
):
    """``RUNS`` calls, with the step chosen or given, build them once."""
    simu = make_solver(backend)
    builds = []
    build = simu._dispersion_operator
    monkeypatch.setattr(
        simu, "_dispersion_operator", lambda: builds.append(1) or build()
    )
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    for _ in range(RUNS):
        simu.out_field(field, DELTA_Z, delta_z=delta_z, verbose=False, plot=False)
    assert len(builds) == 1


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_new_wavenumber_rebuilds_them(backend):
    """Built once per wavenumber: after k changes, the rate is ``K^2 / (2 k)``."""
    simu = make_solver(backend)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    simu.out_field(field, DELTA_Z, delta_z=DELTA_Z, verbose=False, plot=False)
    simu.k = simu.k * WAVENUMBER_CHANGE
    expected = float(np.max(simu.Kxx**2 + simu.Kyy**2)) / (2 * simu.k)
    assert simu._max_dispersion_rate() == pytest.approx(
        expected, rel=DISPERSION_RATE_RTOL
    ), "the dispersion rate kept the eigenvalues of the previous wavenumber"
