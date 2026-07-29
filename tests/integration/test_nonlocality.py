import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.backends import list_available_backends
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Non-locality is a convolution, which OpenCL and MLX have no kernel for, so
# their solvers refuse a positive nl_length outright. Derived rather than
# listed, so a backend that gains the kernel is picked up here by removing it
# from one place.
NO_NONLOCALITY = ("CL", "MLX")
AVAILABLE_BACKENDS = [b for b in list_available_backends() if b not in NO_NONLOCALITY]
UNSUPPORTED = [b for b in list_available_backends() if b in NO_NONLOCALITY]


@pytest.mark.parametrize("backend", UNSUPPORTED)
def test_unsupported_backends_refuse_nonlocality(backend):
    """A backend without the convolution must say so, not compute silently.

    The list above is what keeps the rest of this file off those backends, so
    it has to match what the solvers actually do.
    """
    with pytest.raises(NotImplementedError, match=r"[Nn]on-local"):
        NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=2e-4,
            NX=64,
            NY=64,
            Isat=10e4,
            nl_length=60e-6,
            backend=backend,
        )


def test_supported_backends_accept_nonlocality():
    """And the ones on the list must accept it.

    The grid has to resolve nl_length: the non-local kernel spans
    ``nl_length // delta_X`` cells, so on a coarser grid than that it collapses
    to a single point and the run is local again, silently.
    """
    assert AVAILABLE_BACKENDS, "no backend can run the tests below"
    for backend in AVAILABLE_BACKENDS:
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
        assert simu.nl_profile.shape[0] > 1, (
            f"{backend} accepted nl_length but built no non-local profile"
        )


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
            precision="single",
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
            precision="single",
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
            precision="single",
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
            precision="single",
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
            precision="single",
        )
        arr = E.real * E.real + E.imag * E.imag
        arr *= simu_gpe.delta_X * simu_gpe.delta_Y
        norm = arr.sum(simu_gpe._last_axes)
        assert np.allclose(norm, simu_gpe.N, rtol=1e-3), (
            f"CNLSE : Norm is not conserved ! (Backend {backend})"
        )
