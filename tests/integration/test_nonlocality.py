import warnings

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import GPE, NLSE, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend
from scipy.constants import atomic_mass

PRECISION_COMPLEX = np.complex64


# About a radian of nonlinear phase over the cell: enough for the smoothing to
# matter, not so much that backends diverge (at 1 W they differ by 100%).
RESPONSE_WINDOW = 1e-3  # m
RESPONSE_N = 64
RESPONSE_PARAMS = {
    "alpha": 0,
    "power": 1e-3,
    "window": RESPONSE_WINDOW,
    "n2": -5e-9,
    "V": None,
    "L": 1e-3,
    "NX": RESPONSE_N,
    "NY": RESPONSE_N,
    "Isat": 1e10,
}
# The input Gaussian's waist, in m: well inside the window.
RESPONSE_WAIST = RESPONSE_WINDOW / 6
# Non-local lengths of one and five cells, in m: 0.41 of the peak apart in the
# output (measured, CPU and MLX), so the smoothing changes the answer.
SHORT_LENGTH = RESPONSE_WINDOW / RESPONSE_N
LONG_LENGTH = 5 * RESPONSE_WINDOW / RESPONSE_N
# One backend run twice at one length: measured 0 apart.
RESPONSE_ATOL_FRACTION = 1e-6

# NLSE_1d and GPE through out_field with a response five cells long. The GPE:
# rubidium 87, 1e5 atoms in a Gaussian a sixth of the window wide, 2 rad of
# interaction phase at its centre over the run.
GPE_WINDOW = 100e-6  # m
GPE_ATOMS = 1e5
GPE_G = 1e-11  # rad/s*m^2
GPE_MASS = 87 * atomic_mass  # kg
GPE_TIME = 1e-3  # s
RUN_CELLS = 5
# Smoothed against local, over the peak: measured 1.97 (NLSE_1d) and 1.10
# (GPE). The norm of the two: 1.3e-6 apart at most (MLX), no loss in either.
RUN_APART_FLOOR = 0.5
RUN_NORM_RTOL = 1e-5


def test_a_non_local_run_stays_on_the_backend_asked_for(backend):
    """No backend lacks what non-locality needs, so none is swapped, silently or not."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simu = NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=2e-4,
            NX=256,
            NY=256,
            Isat=10e4,
            nl_length=60e-6,
            backend=backend,
        )
    assert simu._backend.name == backend
    assert simu.nl_length == 60e-6


def test_a_length_below_one_cell_is_kept(backend):
    """The response is exact at any length, so a sub-cell one is not dropped."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simu = NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=2e-4,
            NX=64,
            NY=64,
            Isat=10e4,
            nl_length=60e-6,  # smaller than this grid's 139 um cell
            backend=backend,
        )
    assert simu.nl_length == 60e-6, "the sub-cell length was dropped"


def run_1d(backend, cells):
    """Return NLSE_1d's output for a Gaussian, the response ``cells`` cells long."""
    params = {key: RESPONSE_PARAMS[key] for key in ("alpha", "power", "n2", "V", "L")}
    simu = NLSE_1d(
        window=RESPONSE_WINDOW,
        NX=RESPONSE_N,
        Isat=RESPONSE_PARAMS["Isat"],
        nl_length=cells * RESPONSE_WINDOW / RESPONSE_N,
        backend=backend,
        **params,
    )
    field = np.exp(-(simu.X**2) / RESPONSE_WAIST**2).astype(PRECISION_COMPLEX)
    out = simu.out_field(field, simu.L, verbose=False, plot=False)
    return np.asarray(simu._backend.to_numpy(out))


def run_gpe(backend, cells):
    """Return GPE's output for a Gaussian, the response ``cells`` cells long."""
    simu = GPE(
        gamma=0,
        N=GPE_ATOMS,
        window=GPE_WINDOW,
        g=GPE_G,
        V=None,
        m=GPE_MASS,
        NX=RESPONSE_N,
        NY=RESPONSE_N,
        nl_length=cells * GPE_WINDOW / RESPONSE_N,
        backend=backend,
    )
    waist = GPE_WINDOW / 6
    field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    out = simu.out_field(field, GPE_TIME, verbose=False, plot=False)
    return np.asarray(simu._backend.to_numpy(out))


@pytest.mark.parametrize("run", [run_1d, run_gpe], ids=["NLSE_1d", "GPE"])
def test_a_non_local_run_moves_the_field_and_keeps_its_norm(backend, run):
    """The response changes the phase a run puts on, and only the phase."""
    local, smoothed = run(backend, 0), run(backend, RUN_CELLS)
    apart = np.max(np.abs(smoothed - local)) / np.max(np.abs(local))
    assert apart > RUN_APART_FLOOR, f"the non-local run is the local one to {apart:.2e}"
    assert np.sum(np.abs(smoothed) ** 2) == pytest.approx(
        np.sum(np.abs(local) ** 2), rel=RUN_NORM_RTOL
    )


def test_a_three_dimensional_run_can_be_non_local(backend):
    """NLSE_3d smooths across the beam and runs to the end."""
    simu = NLSE_3d(
        alpha=0,
        energy=1e-6,
        window=(4 * 2.23e-3, 4 * 2.23e-3, 1e-9),
        n2=-1e-9,
        D0=1e-27,
        vg=3e8 / 1.5,
        V=None,
        L=2e-4,
        NX=32,
        NY=32,
        NZ=16,
        Isat=10e4,
        nl_length=1e-3,
        backend=backend,
    )
    field = np.exp(
        -(simu.XX**2 + simu.YY**2) / (2.23e-3) ** 2 - simu.TT**2 / (1e-9 / 4) ** 2
    ).astype(PRECISION_COMPLEX)
    out = np.asarray(
        get_backend(backend).to_numpy(
            simu.out_field(field, 2e-4, verbose=False, plot=False, delta_z=1e-5)
        )
    )
    assert out.shape == field.shape
    assert np.all(np.isfinite(out)), "the non-local 3D run returned non-finite values"


@pytest.mark.parametrize("splitting", ["lie", "RK4"])
@pytest.mark.parametrize("backend", [b for b in AVAILABLE_BACKENDS if b != "CPU"])
def test_a_non_local_run_agrees_with_the_cpu(backend, splitting):
    """Each backend's smoothing has to be the same smoothing."""
    fields = [
        smoothed_run(
            NLSE(backend=name, nl_length=LONG_LENGTH, **RESPONSE_PARAMS), splitting
        )
        for name in ("CPU", backend)
    ]
    cpu, other = fields
    difference = np.max(np.abs(other - cpu)) / np.max(np.abs(cpu))
    assert difference < 2e-4, (
        f"{backend} non-local run differs from the CPU by {difference:.2e}, "
        f"which is more than single precision explains"
    )


def smoothed_run(simu, splitting="lie"):
    """Propagate a Gaussian over L and return the field on the host.

    Parameters
    ----------
    simu : NLSE
        Solver built from RESPONSE_PARAMS.
    splitting : str
        ``splitting`` for ``out_field``.

    Returns
    -------
    np.ndarray
        The output field.
    """
    E = np.exp(-(simu.XX**2 + simu.YY**2) / RESPONSE_WAIST**2)
    out = simu.out_field(
        E.astype(PRECISION_COMPLEX), simu.L, verbose=False, splitting=splitting
    )
    return np.asarray(simu._backend.to_numpy(out))


def test_a_new_length_is_a_new_response(backend):
    """Changing ``nl_length`` between runs smooths at the new length."""
    simu = NLSE(backend=backend, nl_length=SHORT_LENGTH, **RESPONSE_PARAMS)
    smoothed_run(simu)
    simu.nl_length = LONG_LENGTH
    got = smoothed_run(simu)

    expected = smoothed_run(
        NLSE(backend=backend, nl_length=LONG_LENGTH, **RESPONSE_PARAMS)
    )
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=RESPONSE_ATOL_FRACTION * np.abs(expected).max()
    )
