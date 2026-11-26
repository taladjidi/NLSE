import warnings

import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend, list_available_backends
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# The non-local response is a product in Fourier space, which every backend
# has: none of them hands a non-local run to another.
AVAILABLE_BACKENDS = list_available_backends()


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
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


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
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


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
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
    # About a radian of nonlinear phase over the cell. Enough that the
    # smoothing changes the answer, little enough that the run is not
    # chaotic: at the couple of hundred radians these parameters reach if the
    # power is left at 1 W, two backends disagree by 100% on a LOCAL run and
    # the comparison says nothing about either smoothing.
    window, n, nl = 1e-3, 64, 5 * 1e-3 / 64
    kwargs = {
        "alpha": 0,
        "power": 1e-3,
        "window": window,
        "n2": -5e-9,
        "V": None,
        "L": 1e-3,
        "NX": n,
        "NY": n,
        "Isat": 1e10,
        "nl_length": nl,
    }
    fields = []
    for name in ("CPU", backend):
        simu = NLSE(backend=name, **kwargs)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / (window / 6) ** 2).astype(
            PRECISION_COMPLEX
        )
        fields.append(
            np.asarray(
                simu._backend.to_numpy(
                    simu.out_field(E, 1e-3, verbose=False, splitting=splitting)
                )
            )
        )
    cpu, other = fields
    difference = np.max(np.abs(other - cpu)) / np.max(np.abs(cpu))
    assert difference < 2e-4, (
        f"{backend} non-local run differs from the CPU by {difference:.2e}, "
        f"which is more than single precision explains"
    )
