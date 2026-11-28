import numpy as np
import pytest
from helpers import as_numpy, assert_c_contiguous, gaussian_beam, make, random_field
from NLSE import CNLSE, DDGPE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from scipy.constants import c, epsilon_0

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp
    from NLSE.backends.cupy_backend import _CuFFTPlan
if NLSE.__PYOPENCL_AVAILABLE__:
    from NLSE.backends.opencl import _VkFFTPlan
PRECISION_COMPLEX = np.complex64

N = 256
waist = 2.23e-3
window = 4 * waist
L = 10e-3

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4
# A beam well inside the window and well resolved: 16 waists across the
# window, 16 cells across the waist.
BEAM_WAIST = window / 16
# Absolute tolerance of the one-step Gaussian beam check, unit peak; the
# step lands within 3.2e-7 on CPU and MLX (float32).
BEAM_ATOL = 1e-5
# The propagator cache is keyed by the step, the dtype and k (with the grid):
# each case changes one of the three, by these factors or to this dtype.
CACHE_N = 32  # grid points: any grid has a propagator to cache
# m: turns the fastest mode of that grid, 15.8 rad/m (measured), by 0.79 rad,
# so that a propagator built for a stale step or k is far from the closed form.
CACHE_DZ = 5e-2
SAME = 1.0  # factor that leaves the step or k as it is
STEP_CHANGE = 2.0
WAVENUMBER_CHANGE = 1.5
OTHER_DTYPE = np.complex128
# Round-off allowed the normalized power, in units of the field's epsilon: over
# the seven solvers, batched or not, it lands within 1.7 (complex64, every
# backend) and 2.0 (complex128, CPU) of them (measured); float32 constants left
# complex128 up to 2.5e8 off.
NORM_EPS = 10
# What the shared physics in helpers.make leaves out of the three solvers with
# parameters of their own; any values normalize. NLSE_3d's window is
# (space in m, time in s), its pulse energy in J; GPE's N is atoms.
NORM_PARAMETERS = {
    NLSE_3d: {
        "energy": 1e-6,
        "D0": 0.0,
        "vg": 1e8,
        "window": [window, 1e-12],
        "NZ": CACHE_N,
    },
    GPE: {"gamma": 0.0, "N": 1e5, "g": 1e-3},
    DDGPE: {
        "gamma": 0.0,
        "g": 0.0,
        "omega": 1.0,
        "T": 1.0,
        "omega_exc": 1.0,
        "omega_cav": 1.0,
        "detuning": 0.0,
        "k_z": 1.0,
    },
}
NORM_SOLVERS = [NLSE, NLSE_1d, NLSE_3d, CNLSE, CNLSE_1d, GPE, DDGPE]
NORM_BATCH = 3  # fields side by side in a batched run
CACHE_CHANGES = [
    (SAME, WAVENUMBER_CHANGE, PRECISION_COMPLEX),
    (STEP_CHANGE, SAME, PRECISION_COMPLEX),
    (SAME, SAME, OTHER_DTYPE),
]


def make_solver(backend="CPU", n=N, **overrides):
    """Return a NLSE with this module's parameters."""
    return make(NLSE, backend, n=n, **{"L": L, **overrides})


def test_build_propagator(backend) -> None:
    simu = make_solver(backend)
    prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
    assert np.allclose(
        prop,
        np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * DZ_TEST),
    ), f"Propagator is wrong. (Backend {backend})"


@pytest.mark.parametrize(
    "step_factor, k_factor, dtype", CACHE_CHANGES, ids=["k", "step", "dtype"]
)
def test_a_new_step_dtype_or_k_gets_its_own_propagator(
    backend, step_factor, k_factor, dtype
) -> None:
    """Each of the three builds a propagator of its own; the first stays cached.

    Parameters
    ----------
    backend : str
        Backend to run on.
    step_factor : float
        Factor on ``CACHE_DZ`` for the second propagator.
    k_factor : float
        Factor on the wavenumber for the second propagator.
    dtype : np.dtype
        Complex dtype of the second propagator.
    """
    simu = make_solver(backend, n=CACHE_N)
    wider = np.dtype(dtype).itemsize > np.dtype(PRECISION_COMPLEX).itemsize
    if wider and not simu._backend.supports_double_precision():
        pytest.skip(f"{backend} has no double precision")
    cached = simu._build_propagator(PRECISION_COMPLEX, CACHE_DZ)
    k = simu.k
    simu.k = k * k_factor
    step = CACHE_DZ * step_factor
    built = simu._build_propagator(dtype, step)
    expected = np.exp(-1j * (simu.Kxx**2 + simu.Kyy**2) * step / (2 * simu.k))
    simu.k = k
    assert built is not cached, "the changed parameter got the cached propagator"
    assert np.allclose(as_numpy(simu, built), expected), "not the new closed form"
    assert simu._build_propagator(PRECISION_COMPLEX, CACHE_DZ) is cached


def test_build_fft_plan(backend) -> None:
    simu = make_solver(backend)
    A = random_field((N, N))
    plans = simu._build_fft_plan(A)
    if backend == "CPU":
        assert len(plans) == 1, f"Number of plans is wrong. (Backend {backend})"
        assert plans[0].axes == (-2, -1), f"Plan axes are wrong. (Backend {backend})"
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
        assert isinstance(out, np.ndarray), "Output array type wrong"
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


@pytest.mark.skipif(not NLSE.__CUPY_AVAILABLE__, reason="CuPy not available")
def test_retrieve_arrays_from_gpu() -> None:
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
    for name in ("propagator", "V", "alpha", "n2", "I_sat"):
        assert isinstance(getattr(simu, name), cp.ndarray), f"{name} not on the GPU"
    simu._retrieve_arrays_from_gpu()
    for name in ("propagator", "V", "alpha", "n2", "I_sat"):
        assert isinstance(getattr(simu, name), np.ndarray), f"{name} not back on host"


def test_split_step(backend) -> None:
    """One linear step of a Rayleigh length lands on the Gaussian beam."""
    simu = make_solver(backend, n2=0, alpha=0)
    r_sq = simu.XX**2 + simu.YY**2
    z_rayleigh = simu.k * BEAM_WAIST**2 / 2
    E = gaussian_beam(r_sq, BEAM_WAIST, simu.k, 0, 2).astype(PRECISION_COMPLEX)
    A, A_sq = simu._prepare_output_array(E, normalize=False)
    simu.plans = simu._build_fft_plan(A)
    simu.propagator = simu._build_propagator(np.complex64, z_rayleigh)
    # out_field sends arrays for every device backend, so mirror that here or
    # the kernels receive a host propagator.
    if simu._backend.is_device_backend:
        simu._send_arrays_to_gpu()
    A = simu.split_step(
        A, A_sq, simu.V, simu.propagator, simu.plans, z_rayleigh, splitting="strang"
    )
    np.testing.assert_allclose(
        as_numpy(simu, A),
        gaussian_beam(r_sq, BEAM_WAIST, simu.k, z_rayleigh, 2),
        rtol=0,
        atol=BEAM_ATOL,
        err_msg=f"One step is not the Gaussian beam at zR. (Backend {backend})",
    )


# tests for convergence of the solver : the norm of the field should be
#  conserved. A beam, not a plane wave: a plane wave has no transverse
#  wavenumber for a propagator to lose norm at.
def test_out_field(backend) -> None:
    simu = make_solver(backend, alpha=0)
    E = gaussian_beam(simu.XX**2 + simu.YY**2, BEAM_WAIST, simu.k, 0, 2).astype(
        PRECISION_COMPLEX
    )
    E = simu.out_field(
        E, L, verbose=False, plot=False, splitting="lie", delta_z=DZ_TEST
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
                splitting="lie",
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
    """The lossless kernel branch (alpha == 0) matches the general formula."""
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


def test_a_field_with_no_power_survives_normalization(backend) -> None:
    """Normalizing a zero field must leave it zero, not make it NaN."""
    import warnings

    simu = make_solver(backend, n=32)
    empty = np.zeros((32, 32), dtype=PRECISION_COMPLEX)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = as_numpy(simu, simu.out_field(empty, 1e-3, verbose=False, plot=False))
    assert np.all(np.isfinite(out.view(np.float32))), (
        f"an empty field came back with non-finite values. (Backend {backend})"
    )
    assert np.all(out == 0), f"an empty field should stay empty. (Backend {backend})"


def normalized_field(simu, batched, dtype):
    """Return a random field for ``simu``, normalized, on the host.

    Parameters
    ----------
    simu : NLSE
        Solver of any of the seven classes.
    batched : bool
        Whether the field has a leading batch axis of ``NORM_BATCH``.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    np.ndarray
        The field ``normalize=True`` returns.
    """
    grid = (simu.NX,) if len(simu._last_axes) == 1 else simu.XX.shape
    leading = (NORM_BATCH,) * batched + (2,) * simu._component_axes
    field = random_field(leading + grid, dtype)
    return as_numpy(simu, simu._prepare_output_array(field, True)[0])


def exact_power(simu, A):
    """Return each field's power, from the grid cell and float64 sums.

    Parameters
    ----------
    simu : NLSE
        Solver of any of the seven classes.
    A : np.ndarray
        Field on the host.

    Returns
    -------
    np.ndarray
        Power, or energy or atoms, of each field in ``A``.
    """
    steps = (simu.delta_X, getattr(simu, "delta_Y", 1.0), getattr(simu, "delta_T", 1.0))
    cell = np.prod(steps[: len(simu._last_axes)])
    # A condensate's integral counts atoms; a beam's is a power, c eps_0 |E|^2 / 2.
    scale = 1.0 if isinstance(simu, GPE) else c * epsilon_0 / 2
    intensity = np.abs(A.astype(np.complex128)) ** 2
    return scale * cell * np.sum(intensity, axis=simu._last_axes)


@pytest.mark.parametrize("batched", [False, True], ids=["one", "batched"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("cls", NORM_SOLVERS, ids=lambda cls: cls.__name__)
def test_normalization_holds_the_power_to_the_field_precision(
    backend, cls, dtype, batched
):
    """``normalize=True`` sets the power to the precision of the field's dtype."""
    simu = make(cls, backend, n=CACHE_N, **NORM_PARAMETERS.get(cls, {}))
    if dtype == np.complex128 and not simu._backend.supports_double_precision():
        pytest.skip(f"{backend} has no fp64")
    power = exact_power(simu, normalized_field(simu, batched, dtype))
    target = np.broadcast_to(simu._norm_target, power.shape)
    rtol = float(NORM_EPS * np.finfo(dtype).eps)
    assert power == pytest.approx(target, rel=rtol), (
        f"{backend}, {cls.__name__}, {np.dtype(dtype).name}: normalized to "
        f"{power} against the {simu._norm_target} asked, off by more than "
        f"{rtol:.1e}"
    )


@pytest.mark.parametrize("power_type", [float, np.float64])
def test_a_complex64_field_is_normalized_in_float32_as_written(power_type):
    """The bits: float32 sums, the factor at the target's precision, one product.

    NEP 50: a Python float target takes the float32 integral's precision, a
    numpy.float64 one keeps float64. The product is taken in complex128 and
    rounded to complex64 once.
    """
    simu = make(NLSE, "CPU", n=CACHE_N)
    simu.power = power_type(simu.power)
    E = random_field(simu.XX.shape, np.complex64)
    cell = np.float32(simu.delta_X * simu.delta_Y)
    integral = np.sum((E * E.conj()).real * cell) * np.float32(c * epsilon_0 / 2)
    factor = np.float64((simu.power / integral) ** 0.5)
    expected = (factor * E.astype(np.complex128)).astype(np.complex64)
    A = simu._backend.to_numpy(simu._prepare_output_array(E, True)[0])
    assert np.array_equal(A, expected), (
        f"{power_type.__name__} power: the normalized field differs from the "
        f"written-out expression in {np.count_nonzero(A != expected)} of "
        f"{A.size} points"
    )
