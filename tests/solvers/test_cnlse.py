import numpy as np
import pytest
from helpers import as_numpy, assert_c_contiguous, make
from NLSE import CNLSE
from scipy.constants import c, epsilon_0

if CNLSE.__CUPY_AVAILABLE__:
    import cupy as cp

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
n2 = -1.6e-9
n12 = -1e-10
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
Isat2 = waist / waist2 * Isat
L = 1e-3
alpha = 20

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_solver(backend="CPU", n=N, **overrides):
    """Return a CNLSE with this module's parameters."""
    return make(CNLSE, backend, n=n, **overrides)


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
    A = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    out = as_numpy(simu, out)
    integral = (
        (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
    ).sum(axis=simu._last_axes)
    integral *= c * epsilon_0 / 2
    assert np.allclose(
        integral,
        np.array([simu.power, simu.power2]),
        rtol=1e-4,
    ), f"Normalization failed. (Backend {backend})"
    assert out.shape == (
        2,
        N,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    np.testing.assert_allclose(
        out / np.max(np.abs(out)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_each_component_normalizes_to_its_own_power(backend) -> None:
    """``power2`` defaults to ``power``, which hides a shared target.

    With the two equal, normalizing both components to the first one's power
    gives the right answer for the wrong reason, so this sets them apart.
    """
    simu = make_solver(backend)
    simu.power2 = 4 * simu.power
    A = np.ones((2, N, N), dtype=PRECISION_COMPLEX)

    out = as_numpy(simu, simu._prepare_output_array(A, normalize=True)[0])
    integral = (
        (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
    ).sum(axis=simu._last_axes) * (c * epsilon_0 / 2)

    np.testing.assert_allclose(
        integral,
        [simu.power, simu.power2],
        rtol=1e-4,
        err_msg=f"each component must carry its own power (backend {backend})",
    )


def test_send_arrays_to_gpu() -> None:
    if CNLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        n12 = -1e-10
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis, np.newaxis]
        n12 = np.repeat(n2, 2)
        n12 = n12[..., np.newaxis, np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis, np.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, n12=n12, V=V, Isat=Isat)
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        simu._send_arrays_to_gpu()
        assert isinstance(simu.propagator, cp.ndarray), (
            "propagator is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.V, cp.ndarray), "V is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.alpha, cp.ndarray), (
            "alpha is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.n2, cp.ndarray), "n2 is not a cp.ndarray. (Backend GPU)"
        assert isinstance(simu.n12, cp.ndarray), (
            "n12 is not a cp.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.I_sat, cp.ndarray), (
            "I_sat is not a cp.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if CNLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        n12 = -1e-10
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis, np.newaxis]
        n12 = np.repeat(n2, 2)
        n12 = n12[..., np.newaxis, np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis, np.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, n12=n12, V=V, Isat=Isat)
        simu.propagator = simu._build_propagator(np.complex64, DZ_TEST)
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(simu.propagator, np.ndarray), (
            "propagator is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.V, np.ndarray), "V is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.alpha, np.ndarray), (
            "alpha is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.n2, np.ndarray), "n2 is not a np.ndarray. (Backend GPU)"
        assert isinstance(simu.n12, np.ndarray), (
            "n12 is not a np.ndarray. (Backend GPU)"
        )
        assert isinstance(simu.I_sat, np.ndarray), (
            "I_sat is not a np.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_take_components(backend) -> None:
    simu = make_solver(backend)
    # create a larger array to test the fancy indexing
    A = np.ones((3, 2, N, N), dtype=PRECISION_COMPLEX)
    A1, A2 = simu._take_components(A)
    assert A1.shape[-2:] == (
        N,
        N,
    ), f"A1 has wrong last dimensions. (Backend {backend})"
    assert A2.shape[-2:] == (
        N,
        N,
    ), f"A2 has wrong last dimensions. (Backend {backend})"
    assert A1.shape == A2.shape, f"A1 and A2 have different shapes. (Backend {backend})"
    assert A1.shape[0] == 3, f"A1 has wrong first dimensions. (Backend {backend})"
    assert A2.shape[0] == 3, f"A2 has wrong first dimensions. (Backend {backend})"


def test_split_step(backend) -> None:
    simu = make_solver(backend)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A,
        A_sq,
        simu.V,
        simu.propagator,
        simu.plans,
        0,
        splitting="strang",
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        np.ones((2, N, N), dtype=PRECISION_COMPLEX),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Split-step is not unitary. (Backend {backend})",
    )


# tests for convergence of the solver : the norm of the field should be
# conserved
def test_out_field(backend) -> None:
    E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    simu = make_solver(backend, alpha=0)
    E = simu.out_field(
        E, L, verbose=False, plot=False, splitting="lie", delta_z=DZ_TEST
    )
    norm = np.sum(
        np.abs(E) ** 2 * simu.delta_X * simu.delta_Y * c * epsilon_0 / 2,
        axis=simu._last_axes,
    )
    assert np.allclose(norm, [simu.power, simu.power2], rtol=1e-4), (
        "Norm not conserved."
    )


def test_components_round_trip_when_taken_by_copy(monkeypatch) -> None:
    """Components taken by copy must be written back.

    On a device backend ``_take_components`` copies, so the result of a
    nonlinear step reaches the field only through ``_set_components``.

    On the CPU it depends on the shape, which is not what this said and is
    the reason a bug got past it. An unbatched component is contiguous and
    comes back as a view, so the write-back is a no-op there and dropping it
    would not show. A batched one is strided, and since the numba kernels
    flatten with ravel -- a view of a contiguous array, a copy of a strided
    one -- it has to be copied as well, or the step is applied to a temporary
    and lost. Both are exercised below.

    Monkeypatching is_device_backend forces the copy branch, which covers
    CUPY from a machine without it. It also means this test passed throughout
    the period when the host branch was returning strided views and every
    batched coupled run on the CPU was dropping its real-space step: the
    branch under test was not the broken one.
    """
    from NLSE.backends.backend import Backend

    simu = make_solver("CPU")
    monkeypatch.setattr(Backend, "is_device_backend", property(lambda self: True))

    for shape in ((2, N, N), (3, 2, N, N)):
        A = np.zeros(shape, dtype=PRECISION_COMPLEX)
        A1, A2 = simu._take_components(A)
        assert not np.shares_memory(A1, A), "expected copies on a device backend"

        A1[:] = 1.0
        A2[:] = 2.0
        simu._set_components(A, A1, A2)

        got1, got2 = simu._take_components(A)
        assert np.all(got1 == 1.0), f"component 1 was not written back for {shape}"
        assert np.all(got2 == 2.0), f"component 2 was not written back for {shape}"


def test_an_array_norm_target_reaches_the_backend(monkeypatch) -> None:
    """A per-component target must be moved to the device before it is used.

    ``_norm_target`` is one power per component, a numpy array, and it divides
    an integral that lives wherever the field does. CuPy rejects a numpy
    operand against a device array outright; on CPU ``from_numpy`` is the
    identity, so nothing here would notice the missing conversion. Asserting
    the conversion happens is what covers CuPy from a machine without it.
    """
    simu = make_solver("CPU")
    seen = []
    original = simu._backend.from_numpy

    def spy(array):
        seen.append(np.shape(array))
        return original(array)

    monkeypatch.setattr(simu._backend, "from_numpy", spy)
    simu._prepare_output_array(
        np.ones((2, N, N), dtype=PRECISION_COMPLEX), normalize=True
    )

    assert (2,) in seen, (
        f"the per-component target never went through the backend; "
        f"shapes converted were {seen}"
    )


def test_each_component_propagates_at_its_own_wavenumber(backend) -> None:
    """Both propagators must read the wavenumber of their own component.

    ``CNLSE.__init__`` sets ``k2 = k``, so a propagator built from ``k``
    where it should use ``k2`` gives the right answer for every test that
    leaves the default alone -- and the tests that check the formula take
    their own expectation from ``simu.k2``, so they agree with it. Both
    mutations survived the suite. This gives the second component a
    different wavenumber, which is the only way the difference shows.
    """
    simu = make_solver(backend)
    simu.k2 = 2 * np.pi / 795e-9
    assert simu.k2 != simu.k, "the components must differ for this to test anything"

    split = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
    assert np.allclose(
        split[1], np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k2 * DZ_TEST)
    ), f"the split-step propagator ignores k2. (Backend {backend})"

    rk4 = as_numpy(simu, simu._build_propagator_rk4(PRECISION_COMPLEX))
    assert np.allclose(rk4[1], -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k2), (
        f"the RK4 dispersion operator ignores k2. (Backend {backend})"
    )


@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_each_component_cross_phases_at_its_own_wavenumber(backend, method) -> None:
    """The index one component writes must phase the other at that other's k.

    ``n12`` is one index coefficient, but it enters two equations, and each
    turns an index into a phase rate at its own wavenumber: ``k * n12 * I2``
    for the first component, ``k2 * n12 * I1`` for the second. The solver
    built one constant from ``k`` and handed it to both, so the second
    component cross-phased at the first's wavenumber.

    Nothing else in the suite could see it. ``k2 = k`` by construction, the
    asymmetry helper varies ``n22``, ``alpha2`` and ``I_sat2`` but not the
    cross coupling, and every backend carried the same wrong constant, so the
    cross-backend comparisons agreed with each other.

    Two uniform components with the self-interactions and the losses switched
    off leave only the cross terms, and a uniform field neither diffracts nor
    saturates unevenly, so each component's phase is the closed form above.
    """
    simu = make_solver(backend, n2=0.0, alpha=0.0, Isat=np.inf)
    simu.n22 = 0.0
    simu.alpha2 = 0.0
    simu.I_sat2 = np.inf
    simu.power2 = simu.power
    simu.k2 = 2 * np.pi / 390e-9
    assert simu.k2 != simu.k, "the components must differ for this to test anything"

    E = np.ones((2, N, N), dtype=PRECISION_COMPLEX)
    out = as_numpy(
        simu,
        simu.out_field(E, L, delta_z=DZ_TEST, method=method, verbose=False, plot=False),
    )

    # Losses are off and the terms left are pure phase, so the amplitudes are
    # the ones propagation started from and the intensity can be read off the
    # output rather than re-deriving what normalization did.
    intensity = 0.5 * c * epsilon_0 * np.abs(out) ** 2
    for i, k in ((0, simu.k), (1, simu.k2)):
        other = 1 - i
        expected = k * n12 * intensity[other].mean() * L
        assert np.allclose(np.angle(out[i]), expected, rtol=1e-3), (
            f"component {i + 1} cross-phases at the wrong wavenumber: got "
            f"{np.angle(out[i]).mean():.6e}, expected {expected:.6e} from "
            f"component {other + 1}. (Backend {backend}, {method})"
        )
