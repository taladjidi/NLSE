import numpy as np
import pyfftw
import pytest
from NLSE import NLSE
from scipy.constants import c, epsilon_0

from .helpers import as_numpy, assert_c_contiguous, random_field

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp
    from NLSE.backends.cupy_backend import _CuFFTPlan
if NLSE.__PYOPENCL_AVAILABLE__:
    from NLSE.backends.opencl import _VkFFTPlan
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_solver(backend="CPU", n=N, **overrides):
    """Return a NLSE with this module's parameters.

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
    NLSE
        The solver.
    """
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": None,
        "L": L,
        "NX": n,
        "NY": n,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(overrides)
    return NLSE(**params)


def test_build_propagator(backend) -> None:
    simu = make_solver(backend)
    prop = simu._build_propagator(PRECISION_COMPLEX, DZ_TEST)
    assert np.allclose(
        prop,
        np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * DZ_TEST),
    ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan(backend) -> None:
    simu = make_solver(backend)
    A = random_field((N, N))
    plans = simu._build_fft_plan(A)
    if backend == "CPU":
        assert len(plans) == 2, f"Number of plans is wrong. (Backend {backend})"
        assert isinstance(plans[0], pyfftw.FFTW), (
            f"Plan type is wrong. (Backend {backend})"
        )
        assert plans[0].output_shape == (
            N,
            N,
        ), f"Plan shape is wrong. (Backend {backend})"
    elif backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
        assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
        assert isinstance(plans[0], _CuFFTPlan), (
            f"Plan type is wrong. (Backend {backend})"
        )
    elif backend == "CL" and NLSE.__PYOPENCL_AVAILABLE__:
        assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
        assert isinstance(plans[0], _VkFFTPlan), (
            f"Plan type is wrong. (Backend {backend})"
        )
        # The plan's spec, not its VkFFTApp: the app is compiled on first
        # use, so reading it here passed only when some earlier test in the
        # session had already triggered a transform on this cached plan.
        assert plans[0]._spec["shape"] == (
            N,
            N,
        ), f"Plan shape is wrong. (Backend {backend})"


def test_prepare_output_array(backend) -> None:
    simu = make_solver(backend)
    A = random_field((N, N))
    out, out_sq = simu._prepare_output_array(A, normalize=True)
    assert_c_contiguous(out, f"Output array is not C-contiguous. (Backend {backend})")
    assert_c_contiguous(
        out_sq, f"Output array is not C-contiguous. (Backend {backend})"
    )
    if backend == "CPU":
        assert out.flags.aligned, f"Output array is not aligned. (Backend {backend})"
        assert out_sq.flags.aligned, f"Output array is not aligned. (Backend {backend})"
    out_np = as_numpy(simu, out)
    integral = (
        (out_np.real * out_np.real + out_np.imag * out_np.imag)
        * simu.delta_X
        * simu.delta_Y
    ).sum(axis=simu._last_axes)
    integral = integral * c * epsilon_0 / 2
    error_string = f"Normalization failed. (Backend {backend})"
    error_string += f" : {integral} != {simu.power}"
    assert np.allclose(integral, simu.power, rtol=1e-4), error_string
    assert out_np.shape == (
        N,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    # Normalization only rescales, so direction must be preserved.
    np.testing.assert_allclose(
        out_np / np.max(np.abs(out_np)),
        A / np.max(np.abs(A)),
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Output array does not match input array. (Backend {backend})",
    )


def test_send_arrays_to_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., cp.newaxis, cp.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., cp.newaxis, cp.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., cp.newaxis, cp.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, V=V, Isat=Isat)
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
        assert isinstance(simu.I_sat, cp.ndarray), (
            "I_sat is not a cp.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = 1e-4 * np.random.random((N, N)).astype(np.float32)
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., cp.newaxis, cp.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., cp.newaxis, cp.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., cp.newaxis, cp.newaxis]
        simu = make_solver("CUPY", alpha=alpha, n2=n2, V=V, Isat=Isat)
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
        assert isinstance(simu.I_sat, np.ndarray), (
            "I_sat is not a np.ndarray. (Backend GPU)"
        )
    else:
        pass


def test_split_step(backend) -> None:
    simu = make_solver(backend)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, 0)
    # out_field sends arrays for every device backend, so mirror that here or
    # the kernels receive a host propagator.
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, 0, precision="double"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        np.ones((N, N), dtype=PRECISION_COMPLEX),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Split step is not unitary. (Backend {backend})",
    )


# tests for convergence of the solver : the norm of the field should be
#  conserved
def test_out_field(backend) -> None:
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    simu = make_solver(backend, alpha=0)
    E = simu.out_field(
        E, L, verbose=False, plot=False, precision="single", delta_z=DZ_TEST
    )
    norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y)
    norm *= c * epsilon_0 / 2
    assert E.shape == (
        N,
        N,
    ), f"Output array has wrong shape. (Backend {backend})"
    assert np.allclose(norm, simu.power, rtol=1e-4), (
        f"Norm not conserved. (Backend {backend})"
    )


@pytest.mark.skipif(not NLSE.__CUPY_AVAILABLE__, reason="CuPy not available")
def test_cuda_graph() -> None:
    """CUPY backend (graph-accelerated) must conserve norm for all methods."""
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    V = -1e-4 * np.ones((N, N), dtype=np.float32)

    for has_V in [False, True]:
        potential = V if has_V else None
        for method in ["split_step", "RK4"]:
            simu = make_solver("CUPY", alpha=0, V=potential)
            E_out = simu.out_field(
                E.copy(),
                L,
                verbose=False,
                plot=False,
                precision="single",
                method=method,
                delta_z=DZ_TEST,
            )
            norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
            norm *= c * epsilon_0 / 2
            assert np.allclose(norm, simu.power, rtol=1e-4), (
                f"Norm not conserved with CUDA graph "
                f"(V={'yes' if has_V else 'no'}, method={method})"
            )


def test_lossless_kernels_match_the_general_formula():
    """Alpha == 0 takes a branch that drops the exponential.

    exp(0) is 1, so the lossless step is a pure rotation and the branch is
    exact rather than an approximation. It is worth ~1.5x on the CPU kernels,
    which is only worth having if it computes the same thing: this checks the
    branch against the formula it replaces, evaluated in double precision.
    """
    from NLSE.kernels import cpu as cpu_kernels

    n = 128
    rng = np.random.default_rng(0)
    field = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(
        np.complex64
    )
    A_sq = (np.abs(field) ** 2).astype(np.float32)
    dz, g, Isat = np.float32(1e-4), np.float32(1e-3), np.float32(1e5)

    got = cpu_kernels.nl_prop_without_V(
        field.copy(), A_sq, dz, np.float32(0.0), g, Isat
    )

    sat = 1.0 / (1.0 + A_sq.astype(np.float64) / np.float64(Isat))
    expected = field.astype(np.complex128) * np.exp(
        np.float64(dz) * 1j * np.float64(g) * A_sq.astype(np.float64) * sat
    )

    scale = float(np.max(np.abs(expected)))
    assert np.max(np.abs(got - expected)) / scale < 1e-6, (
        "the lossless branch does not agree with the exponential it replaces"
    )


def test_a_lossy_run_still_loses_power():
    """And the branch must not be taken when there are losses to apply."""
    from NLSE.kernels import cpu as cpu_kernels

    n = 64
    field = np.ones((n, n), dtype=np.complex64)
    A_sq = np.ones((n, n), dtype=np.float32)
    args = (np.float32(1e-2), np.float32(1e-3), np.float32(1e5))

    lossless = cpu_kernels.nl_prop_without_V(
        field.copy(), A_sq, args[0], np.float32(0.0), *args[1:]
    )
    lossy = cpu_kernels.nl_prop_without_V(
        field.copy(), A_sq, args[0], np.float32(50.0), *args[1:]
    )

    assert np.abs(lossless[0, 0]) == pytest.approx(1.0, rel=1e-5), (
        "a lossless step must preserve the amplitude"
    )
    assert np.abs(lossy[0, 0]) < 0.999, (
        f"a lossy step must reduce it, got {np.abs(lossy[0, 0])}"
    )
