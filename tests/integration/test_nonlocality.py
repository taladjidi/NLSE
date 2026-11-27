import warnings

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64


# The medium of the non-local runs below: about a radian of nonlinear phase
# over the cell. Enough that the smoothing changes the answer, little enough
# that the run is not chaotic: at the couple of hundred radians these
# parameters reach if the power is left at 1 W, two backends disagree by 100%
# on a LOCAL run and a comparison says nothing about either smoothing.
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
    """The response is exact at any length, so a sub-cell one is not dropped.

    The old real-space kernel was a whole number of cells wide and collapsed
    to the identity below one, so the solver warned and ran local. The
    response ``1 / (1 + l^2 K^2)`` has no such floor: it smooths a little.
    """
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


def test_nonlocality():
    # Reduced N and L for faster testing while still exercising nonlocality
    N = 512  # Reduced from 2048
    n2 = -1e-9
    n12 = -1e-10
    waist = 2.23e-3
    waist2 = 70e-6
    window = 4 * waist
    power = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 2e-4  # Reduced from 1e-3 for faster testing
    alpha = 0
    nl_length = 60e-6
    for backend in AVAILABLE_BACKENDS:
        simu_c_1d = CNLSE_1d(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_c_2d = CNLSE(
            alpha,
            power,
            window,
            n2,
            n12,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_1d = NLSE_1d(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_2d = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_gpe = GPE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            sat=Isat,
            nl_length=nl_length,
            backend=backend,
        )
        simu_c_1d.power2 = 10e-3
        simu_c_1d.n22 = 1e-10
        simu_c_1d.k2 = 2 * np.pi / 795e-9
        simu_c_2d.power2 = simu_c_1d.power2
        simu_c_2d.n22 = simu_c_1d.n22
        simu_c_2d.k2 = simu_c_1d.k2
        E_0 = np.exp(-(simu_c_2d.XX**2 + simu_c_2d.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        V0 = np.exp(-(simu_c_2d.XX**2 + simu_c_2d.YY**2) / waist2**2).astype(
            PRECISION_COMPLEX
        )
        E, _ = simu_c_1d.out_field(
            np.array([E_0[N // 2, :], V0[N // 2, :]]),
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=1e-5,
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_c_1d.delta_X**2
        norm = arr.sum(simu_c_1d._last_axes)
        assert np.allclose(norm, simu_c_1d.power, rtol=1e-3), (
            f"CNLSE_1d : Norm is not conserved ! (Backend {backend})"
        )
        E = simu_1d.out_field(
            E_0[N // 2, :],
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=1e-5,
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_1d.delta_X**2
        norm = arr.sum(simu_c_1d._last_axes)
        assert np.allclose(norm, simu_1d.power, rtol=1e-3), (
            f"NLSE_1d : Norm is not conserved ! (Backend {backend})"
        )
        E, _ = simu_c_2d.out_field(
            np.array([E_0, V0]),
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=1e-5,
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_c_2d.delta_X * simu_c_2d.delta_Y
        norm = arr.sum(simu_c_2d._last_axes)
        assert np.allclose(norm, simu_c_2d.power, rtol=1e-3), (
            f"CNLSE : Norm is not conserved ! (Backend {backend})"
        )
        E = simu_2d.out_field(
            E_0,
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=1e-5,
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= c * epsilon_0 / 2 * simu_2d.delta_X * simu_2d.delta_Y
        norm = arr.sum(simu_2d._last_axes)
        assert np.allclose(norm, simu_2d.power, rtol=1e-3), (
            f"NLSE : Norm is not conserved ! (Backend {backend})"
        )
        E = simu_gpe.out_field(
            E_0,
            L,
            verbose=False,
            plot=False,
            splitting="lie",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= simu_gpe.delta_X * simu_gpe.delta_Y
        norm = arr.sum(simu_gpe._last_axes)
        assert np.allclose(norm, simu_gpe.N, rtol=1e-3), (
            f"CNLSE : Norm is not conserved ! (Backend {backend})"
        )


def test_a_three_dimensional_run_can_be_non_local(backend):
    """NLSE_3d smooths across the beam and runs to the end.

    The closed form of the transverse response is checked in
    tests/physics/test_nonlocal_response.py; this is the whole run.
    """
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
    """Each backend's smoothing has to be the same smoothing.

    The closed-form tests check one nonlinear step. This runs the physics,
    split-step and RK4 alike, and compares the field.
    """
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
    """Changing ``nl_length`` between runs smooths at the new length.

    The response is cached with the solver's dispersion; a cache that did not
    key it on the length would hand the second run the first run's response.
    """
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
