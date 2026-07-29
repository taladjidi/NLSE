import numpy as np
from NLSE import CNLSE
from scipy.constants import c, epsilon_0

from .helpers import as_numpy, assert_c_contiguous

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
    """Return a CNLSE with this module's parameters.

    Parameters
    ----------
    backend : str
        Backend name.
    n : int
        Grid size, square.
    **overrides
        Any constructor argument, by keyword.

    Returns
    -------
    CNLSE
        The solver.
    """
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "n12": n12,
        "V": None,
        "L": L,
        "NX": n,
        "NY": n,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(overrides)
    return CNLSE(**params)


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
        precision="double",
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
        E, L, verbose=False, plot=False, precision="single", delta_z=DZ_TEST
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
    nonlinear step reaches the field only through ``_set_components``. On CPU the components are views and the write-back is a no-op, which
    leaves CUPY as the only backend where dropping it would show. Forcing the
    copy branch puts that under test here too.
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
