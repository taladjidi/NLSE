import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend, list_available_backends
from scipy.constants import c, epsilon_0

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Non-locality is a convolution, and a backend without one refuses a positive
# nl_length outright. Asked of the backend rather than listed here, so a
# backend that gains the kernel moves between these lists on its own.
AVAILABLE_BACKENDS = [
    b for b in list_available_backends() if get_backend(b).convolution is not None
]
UNSUPPORTED = [
    b for b in list_available_backends() if get_backend(b).convolution is None
]


@pytest.mark.parametrize("backend", UNSUPPORTED)
def test_unsupported_backends_fall_back_for_nonlocality(backend):
    """A backend without the convolution must hand the run to one that has it.

    The backend is a performance choice and the non-locality is physics, so
    the physics wins and the swap is announced. What must not happen is a
    silent local run, which is a different problem returning a plausible
    field.

    The grid has to resolve nl_length, or there is nothing to fall back for:
    below one cell the solver drops the non-locality and runs local, which
    these backends do perfectly well.
    """
    with pytest.warns(UserWarning, match=r"non-local"):
        NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=2e-4,
            NX=512,
            NY=512,
            Isat=10e4,
            nl_length=60e-6,
            backend=backend,
        )


@pytest.mark.parametrize("backend", UNSUPPORTED)
def test_the_fallback_lands_on_a_backend_that_can_convolve(backend):
    """Announcing a swap is no use if the solver stays where it was."""
    with pytest.warns(UserWarning, match=r"non-local"):
        simu = NLSE(
            alpha=0,
            power=1.05,
            window=4 * 2.23e-3,
            n2=-1e-9,
            V=None,
            L=2e-4,
            NX=512,
            NY=512,
            Isat=10e4,
            nl_length=60e-6,
            backend=backend,
        )
    assert simu._backend.convolution is not None
    assert simu._backend.name != backend


@pytest.mark.parametrize("backend", list_available_backends())
def test_an_unresolvable_nl_length_falls_back_to_local(backend):
    """Below one grid cell there is no non-locality to model.

    The kernel spans ``nl_length // delta_X`` cells, so on a coarser grid it
    is a single point: the identity, and a local run paying for a convolution
    every step. The solver warns and drops it instead, which also lets the
    backends without a convolution kernel accept it.
    """
    with pytest.warns(UserWarning, match="below one grid cell"):
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
    assert simu.nl_length == 0, "the unresolvable length was not dropped"
    assert simu.nl_profile.shape == (64, 64), (
        "a local run should have the flat profile, not a 1x1 kernel"
    )


@pytest.mark.parametrize("backend", list_available_backends())
def test_an_nl_length_longer_than_the_window_is_refused(backend):
    """Past the window there is no kernel to build, only a machine to fill.

    The kernel is six interaction lengths wide, so its size grows as the
    square of nl_length / delta_X with nothing bounding it above. Only the
    lower end was checked. A length in the wrong unit -- 5 m instead of 5 um
    on a 8.9 mm window -- asks for a 215281 x 215281 kernel, and the process
    is killed part way through building it: no traceback, no message, exit
    137, and on a shared machine possibly something else killed instead.

    Refined grids do not help here and coarser ones make it worse, so this
    raises rather than falling back the way an unresolvable length does.

    Parameters
    ----------
    backend : str
        Backend to run on. The check is on the grid, so it is the same on
        every one, and it has to come before the convolution refusal or the
        backends without a convolution would report the wrong problem.
    """
    with pytest.raises(ValueError, match="longer than the"):
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
            nl_length=5.0,  # metres, on a 8.9 mm window
            backend=backend,
        )


def test_a_resolved_nl_length_is_kept():
    """The fallback must not fire on a grid that does resolve the length."""
    simu = NLSE(
        alpha=0,
        power=1.05,
        window=4 * 2.23e-3,
        n2=-1e-9,
        V=None,
        L=2e-4,
        NX=512,
        NY=512,
        Isat=10e4,
        nl_length=60e-6,
        backend="CPU",
    )
    assert simu.nl_length == 60e-6
    assert simu.nl_profile.shape[0] > 1, "no non-local kernel was built"


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
    """NLSE_3d has taken an nl_length since it existed and never used one.

    The base builds a transverse kernel and ``NLSE_1d`` narrows it to its own
    rank; the three-dimensional solver convolves over three axes and inherited
    the two-dimensional kernel unchanged, so every non-local run raised out of
    the convolution -- ``in1 and in2 should have the same dimensionality`` --
    batched or not. Nothing here covered it: this file's cases are all 2D or
    1D.

    The non-locality is transverse, so the time axis convolves with a delta.
    That is the modelling choice the fix makes: the index diffuses across the
    beam, not along the pulse.
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
    assert simu.nl_length > 0, "the grid dropped the length before anything ran"
    assert simu.nl_profile.ndim == len(simu._last_axes), (
        f"the kernel is rank {simu.nl_profile.ndim} against {len(simu._last_axes)} "
        f"convolved axes, which is what the convolution refuses"
    )
    assert simu.nl_profile.shape[-1] == 1, (
        "the kernel spans more than one sample in time, so the non-locality is "
        "not the transverse one it is documented to be"
    )
    assert float(np.asarray(simu.nl_profile).sum()) == pytest.approx(1.0, rel=1e-5), (
        "the kernel stopped being normalized, so it changes the intensity it "
        "is only meant to redistribute"
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


@pytest.mark.parametrize("backend", [b for b in AVAILABLE_BACKENDS if b != "CPU"])
def test_a_non_local_run_agrees_with_the_cpu(backend):
    """Each backend's convolution has to be the same convolution.

    The capability test only asks that a box kernel over ones comes back as
    the box's sum, which a circular convolution would also pass. This runs the
    physics and compares the field, so a backend that wrapped at the edges
    instead of zero-padding, or that dropped the centring offset, is caught.
    """
    # About a radian of nonlinear phase over the cell. Enough that the
    # convolution changes the answer, little enough that the run is not
    # chaotic: at the couple of hundred radians these parameters reach if the
    # power is left at 1 W, two backends disagree by 100% on a LOCAL run and
    # the comparison says nothing about either convolution.
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
            np.asarray(simu._backend.to_numpy(simu.out_field(E, 1e-3, verbose=False)))
        )
    cpu, other = fields
    difference = np.max(np.abs(other - cpu)) / np.max(np.abs(cpu))
    assert difference < 2e-4, (
        f"{backend} non-local run differs from the CPU by {difference:.2e}, "
        f"which is more than single precision explains"
    )
