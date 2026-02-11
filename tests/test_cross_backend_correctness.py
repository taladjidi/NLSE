"""Cross-backend correctness tests.

CPU is the reference backend. All other backends are compared against it.
Also validates CPU backend internals (FFT plans, propagator, conservation laws).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for plot tests

import numpy as np
import pyfftw
import pytest
from scipy.constants import c, epsilon_0

from NLSE import CNLSE, NLSE
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()
GPU_BACKENDS = [b for b in AVAILABLE_BACKENDS if b != "CPU"]

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Physical parameters matching reference
N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4
L = 10e-3
alpha = 20


# ---------------------------------------------------------------------------
# Section 1: CPU reference backend correctness tests (always run)
# ---------------------------------------------------------------------------


class TestCPUCorrectness:
    """Tests that validate CPU backend internals."""

    def test_build_fft_plan(self):
        """FFTW plans have correct type, shape, and count."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
        plans = simu._build_fft_plan(A)

        assert len(plans) == 2, "Number of plans is wrong"
        assert isinstance(plans[0], pyfftw.FFTW), "Forward plan type is wrong"
        assert isinstance(plans[1], pyfftw.FFTW), "Inverse plan type is wrong"
        assert plans[0].output_shape == (N, N), "Plan shape is wrong"

    def test_fft_roundtrip(self):
        """FFT then IFFT returns the original array."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        A_orig = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A_orig = A_orig.astype(PRECISION_COMPLEX)

        A, _ = simu._prepare_output_array(A_orig.copy(), normalize=False)
        plans = simu._build_fft_plan(A)

        A_saved = A.copy()
        simu._backend.fft(A, plans)
        simu._backend.ifft(A, plans)

        np.testing.assert_allclose(
            A, A_saved, rtol=1e-5, atol=1e-6,
            err_msg="FFT roundtrip failed to recover original array",
        )

    def test_fft_plan_performance(self):
        """FFT plans are not degraded by stale wisdom."""
        import time

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
        A, _ = simu._prepare_output_array(A, normalize=False)
        plans = simu._build_fft_plan(A)

        # Warmup
        for _ in range(3):
            simu._backend.fft(A, plans)
            simu._backend.ifft(A, plans)

        t0 = time.perf_counter()
        for _ in range(10):
            simu._backend.fft(A, plans)
            simu._backend.ifft(A, plans)
        t_per_roundtrip = (time.perf_counter() - t0) / 10

        # A 2048x2048 FFT+IFFT roundtrip should be well under 200ms
        assert t_per_roundtrip < 0.2, (
            f"FFT roundtrip too slow: {t_per_roundtrip*1e3:.0f}ms "
            f"(stale wisdom file?)"
        )

    def test_split_step_unitarity(self):
        """Split step with delta_z=0 preserves the field."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()

        E = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()

        simu.split_step(
            A, A_sq, simu.V, simu.propagator, simu.plans, precision="double",
        )

        np.testing.assert_allclose(
            A, np.ones((N, N), dtype=PRECISION_COMPLEX),
            rtol=1e-6, atol=1e-7,
            err_msg="Split step is not unitary (delta_z=0 should preserve field)",
        )

    def test_propagator_correctness(self):
        """Propagator matches analytical formula."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        prop = simu._build_propagator()
        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
        )
        assert np.allclose(prop, expected), "Propagator is wrong (CPU)"

    def test_prepare_output_array(self):
        """Output array is normalized, aligned, and contiguous."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        A_in = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A_in = A_in.astype(PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(A_in, normalize=True)

        assert out.flags.c_contiguous, "Output array is not C-contiguous"
        assert out_sq.flags.c_contiguous, "Output sq array is not C-contiguous"
        assert out.flags.aligned, "Output array is not aligned"
        assert out_sq.flags.aligned, "Output sq array is not aligned"
        assert out.shape == (N, N), "Output array has wrong shape"
        assert isinstance(out, np.ndarray), "Output array type wrong"

        # Check normalization
        integral = (
            (out.real * out.real + out.imag * out.imag)
            * simu.delta_X
            * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral = integral * c * epsilon_0 / 2
        np.testing.assert_allclose(
            integral, simu.power, rtol=1e-4,
            err_msg="Normalization failed",
        )

    def test_power_conservation(self):
        """Propagation with alpha=0 conserves the norm."""
        simu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )

        E_in = np.ones((N, N), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="single")

        norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2

        np.testing.assert_allclose(
            norm, simu.power, rtol=1e-4,
            err_msg="Norm not conserved on CPU (alpha=0)",
        )

    def test_power_decay(self):
        """Propagation with alpha>0 gives expected exponential power decay."""
        test_alpha = 10.0
        simu = NLSE(
            test_alpha, power, window, 1e-10, None, L,
            NX=64, NY=64, Isat=1e20, backend="CPU",
        )

        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, normalize=False)

        power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
        power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y

        expected_ratio = np.exp(-test_alpha * L)
        actual_ratio = power_out / power_in
        rel_error = np.abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_error < 0.01, (
            f"Power decay incorrect on CPU: {rel_error:.2%} error"
        )

    def test_propagation_with_potential(self):
        """CPU propagation with V produces a valid result."""
        NX = NY = 64
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=NX, NY=NY, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (NX, NY)
        assert np.isfinite(E_out).all(), "Output contains NaN/Inf"

    def test_coupled_propagation(self):
        """CPU CNLSE propagation produces a valid result."""
        NX = NY = 64
        n12 = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12, None, L,
            NX=NX, NY=NY, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (2, NX, NY)
        assert np.isfinite(E_out).all(), "CNLSE output contains NaN/Inf"


# ---------------------------------------------------------------------------
# Section 2: Cross-backend comparison — each backend vs CPU reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestNLSEvsReference:
    """Compare each backend against the CPU reference."""

    def test_propagation_without_potential(self, backend):
        """Results without potential match CPU reference."""
        simu_ref = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        simu_test = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test, E_ref, rtol=1e-5, atol=1e-6,
            err_msg=f"{backend} does not match CPU reference (no potential)",
        )

    def test_propagation_with_potential(self, backend):
        """Results with potential match CPU reference."""
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, N),
            np.linspace(-window / 2, window / 2, N),
        )
        V = 1e4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ref = NLSE(
            alpha, power, window, n2, V, L,
            NX=N, NY=N, Isat=Isat, backend="CPU",
        )
        simu_test = NLSE(
            alpha, power, window, n2, V, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test, E_ref, rtol=1e-5, atol=1e-6,
            err_msg=f"{backend} does not match CPU reference (with potential)",
        )

    def test_power_conservation(self, backend):
        """Power conservation (alpha=0) matches CPU reference."""
        simu = NLSE(
            0, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )

        E_in = np.ones((N, N), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="single")

        norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2

        np.testing.assert_allclose(
            norm, simu.power, rtol=1e-4,
            err_msg=f"Norm not conserved on {backend} (alpha=0)",
        )

    def test_propagator_correctness(self, backend):
        """Propagator matches analytical formula on each backend."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=N, NY=N, Isat=Isat, backend=backend,
        )
        prop = simu._build_propagator()
        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
        )
        assert np.allclose(prop, expected), (
            f"Propagator is wrong (Backend {backend})"
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestCNLSEvsReference:
    """Compare CNLSE on each backend against the CPU reference."""

    def test_coupled_propagation(self, backend):
        """Coupled NLSE results match CPU reference."""
        n12 = 0.5e-9
        NX = NY = 64

        simu_ref = CNLSE(
            alpha, power, window, n2, n12, None, L,
            NX=NX, NY=NY, Isat=Isat, backend="CPU",
        )
        simu_test = CNLSE(
            alpha, power, window, n2, n12, None, L,
            NX=NX, NY=NY, Isat=Isat, backend=backend,
        )

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test, E_ref, rtol=1e-5, atol=1e-6,
            err_msg=f"CNLSE {backend} does not match CPU reference",
        )


# ---------------------------------------------------------------------------
# Section 3: Extended CPU coverage tests
# ---------------------------------------------------------------------------

# Small grid for fast tests
S = 64


class TestCPUBackendMethods:
    """Tests for CPUBackend methods not covered by propagation tests."""

    def test_from_numpy(self):
        """from_numpy returns contiguous array."""
        from NLSE.backends.cpu import CPUBackend

        backend = CPUBackend()
        # Create non-contiguous array via transpose
        arr = np.ones((4, 8), dtype=np.float32).T
        assert not arr.flags.c_contiguous
        result = backend.from_numpy(arr)
        assert result.flags.c_contiguous

    def test_supports_double_precision(self):
        """CPU always supports double precision."""
        from NLSE.backends.cpu import CPUBackend

        backend = CPUBackend()
        assert backend.supports_double_precision() is True

    def test_allocate_field(self):
        """allocate_field returns aligned zeros."""
        from NLSE.backends.cpu import CPUBackend

        backend = CPUBackend()
        arr = backend.allocate_field((S, S), np.complex64)
        assert arr.shape == (S, S)
        assert arr.dtype == np.complex64
        assert arr.flags.aligned
        assert np.all(arr == 0)

    def test_to_numpy_passthrough(self):
        """to_numpy is identity for CPU arrays."""
        from NLSE.backends.cpu import CPUBackend

        backend = CPUBackend()
        arr = np.ones((S, S))
        assert backend.to_numpy(arr) is arr


class TestNLSEConstructor:
    """Tests for NLSE constructor branches."""

    def test_window_as_tuple(self):
        """Window parameter accepts tuple for asymmetric grids."""
        win = (window, window * 1.5)
        simu = NLSE(
            alpha, power, win, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        assert simu.window[0] == win[0]
        assert simu.window[1] == win[1]
        # X and Y grids should span different ranges
        assert not np.isclose(simu.X[-1] - simu.X[0], simu.Y[-1] - simu.Y[0])

    def test_window_as_list(self):
        """Window parameter accepts list."""
        win = [window, window * 2]
        simu = NLSE(
            alpha, power, win, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        assert simu.window[0] == win[0]
        assert simu.window[1] == win[1]

    def test_backend_property(self):
        """Backend property returns name string."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        assert simu.backend == "CPU"

    def test_backend_setter(self):
        """Backend can be changed via setter."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        assert simu.backend == "CPU"
        # Re-set to same backend (always available)
        simu.backend = "CPU"
        assert simu.backend == "CPU"

    def test_nl_length_positive(self):
        """Positive nl_length creates Bessel non-local profile."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        assert simu.nl_length > 0
        # nl_profile should be a small kernel, not full grid
        assert simu.nl_profile.shape[0] < S
        # nl_profile should be normalized
        np.testing.assert_allclose(simu.nl_profile.sum(), 1.0, rtol=1e-5)


class TestNLSEPropagator:
    """Tests for propagator building and caching."""

    def test_propagator_caching(self):
        """Propagator is cached and reused."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop1 = simu._build_propagator()
        prop2 = simu._build_propagator()
        assert prop1 is prop2  # same object from cache

    def test_propagator_double_precision(self):
        """Double precision propagator has complex128 dtype."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop = simu._build_propagator(precision="double")
        assert prop.dtype == np.complex128

    def test_propagator_rk4(self):
        """RK4 propagator does not include delta_z exponential."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop = simu._build_propagator(precision="RK4")
        expected = -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k
        np.testing.assert_allclose(
            prop, expected.astype(np.complex64), rtol=1e-5,
            err_msg="RK4 propagator is wrong",
        )


class TestNLSESplitStep:
    """Tests for split_step code paths."""

    def test_double_precision_without_V(self):
        """Double precision split step without potential."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator(precision="double")

        simu.split_step(A, A_sq, None, prop, plans, precision="double")
        assert np.isfinite(A).all(), "Double precision split step produced NaN/Inf"

    def test_double_precision_with_V(self):
        """Double precision split step with potential."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator(precision="double")

        simu.split_step(A, A_sq, V, prop, plans, precision="double")
        assert np.isfinite(A).all()

    def test_single_precision_with_V(self):
        """Single precision split step with potential."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator()

        simu.split_step(A, A_sq, V, prop, plans, precision="single")
        assert np.isfinite(A).all()

    def test_nonlocal_propagation(self):
        """Split step with nl_length > 0 uses convolution."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator()

        # Single precision path with nl_length
        simu.split_step(A, A_sq, None, prop, plans, precision="single")
        assert np.isfinite(A).all()

    def test_nonlocal_with_V(self):
        """Split step with nl_length > 0 and potential."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator()

        simu.split_step(A, A_sq, V, prop, plans, precision="single")
        assert np.isfinite(A).all()

    def test_nonlocal_double_precision(self):
        """Double precision split step with nl_length > 0."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator(precision="double")

        simu.split_step(A, A_sq, None, prop, plans, precision="double")
        assert np.isfinite(A).all()


class TestNLSERK4:
    """Tests for RK4 propagation scheme."""

    def test_rk4_propagation(self):
        """RK4 scheme produces valid output."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(
            E_in, L, verbose=False, plot=False, precision="RK4",
        )
        assert E_out.shape == (S, S)
        assert np.isfinite(E_out).all(), "RK4 output contains NaN/Inf"

    def test_rk4_single_step_with_potential(self):
        """RK4 single step with potential exercises the V code path."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e2 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator(precision="RK4")

        # Just one step to exercise the code path
        A_before = A.copy()
        simu.split_step_RK4(A, V, prop, plans)
        # Field should have changed
        assert not np.allclose(A, A_before)

    def test_rk4_nonlocal(self):
        """RK4 scheme with nl_length > 0."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(
            E_in, L, verbose=False, plot=False, precision="RK4",
        )
        assert np.isfinite(E_out).all()


class TestNLSEOutField:
    """Tests for out_field code paths."""

    def test_verbose_output(self, capsys):
        """Verbose mode prints timing info."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        simu.out_field(E_in, L, verbose=True, plot=False)
        captured = capsys.readouterr()
        assert "Time spent to solve" in captured.out

    def test_callback_single(self):
        """Single callback is invoked during propagation."""
        call_count = [0]

        def my_callback(solver, A, z, i):
            call_count[0] += 1

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        simu.out_field(E_in, L, verbose=False, plot=False, callback=my_callback)
        assert call_count[0] > 0, "Callback was never called"

    def test_callback_list(self):
        """List of callbacks is invoked during propagation."""
        counts = [0, 0]

        def cb1(solver, A, z, i, idx):
            counts[idx] += 1

        def cb2(solver, A, z, i, idx):
            counts[idx] += 1

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        simu.out_field(
            E_in, L, verbose=False, plot=False,
            callback=[cb1, cb2],
            callback_args=[(0,), (1,)],
        )
        assert counts[0] > 0, "First callback was never called"
        assert counts[1] > 0, "Second callback was never called"

    def test_callback_invalid_raises(self):
        """Invalid callback type raises ValueError."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        E_in = np.ones((S, S), dtype=PRECISION_COMPLEX)

        with pytest.raises(ValueError, match="callbacks should be a callable"):
            simu.out_field(
                E_in, L, verbose=False, plot=False,
                callback="not_a_callable",
            )

    def test_normalize_false(self):
        """normalize=False skips power normalization."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, normalize=False)
        assert E_out.shape == (S, S)
        assert np.isfinite(E_out).all()

    def test_double_precision_out_field(self):
        """Double precision propagation produces valid result."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="double")
        assert np.isfinite(E_out).all()

    def test_double_precision_with_potential(self):
        """Double precision propagation with V."""
        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="double")
        assert np.isfinite(E_out).all()


class TestCNLSEExtended:
    """Extended CNLSE coverage tests on CPU."""

    def test_build_propagator(self):
        """CNLSE propagator has two components."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop = simu._build_propagator()
        assert prop.shape == (2, S, S)
        assert np.isfinite(prop).all()

    def test_propagator_caching(self):
        """CNLSE propagator is cached."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop1 = simu._build_propagator()
        prop2 = simu._build_propagator()
        assert prop1 is prop2

    def test_propagator_rk4(self):
        """CNLSE RK4 propagator."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        prop = simu._build_propagator(precision="RK4")
        assert prop.shape == (2, S, S)
        # RK4 propagator should not be exponential (no delta_z)
        assert not np.allclose(np.abs(prop), 1.0)

    def test_take_components_cpu(self):
        """_take_components returns views on CPU."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        A = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        A[0] *= 2.0
        A[1] *= 3.0
        A1, A2 = simu._take_components(A)
        assert A1.shape == (S, S)
        assert A2.shape == (S, S)
        np.testing.assert_allclose(A1, 2.0)
        np.testing.assert_allclose(A2, 3.0)

    def test_double_precision_without_V(self):
        """CNLSE double precision split step without V."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="double")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_single_precision_with_V(self):
        """CNLSE single precision with potential."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="single")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_double_precision_with_V(self):
        """CNLSE double precision with potential."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="double")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_rabi_coupling(self):
        """CNLSE with Rabi coupling (omega != None)."""
        n12_local = 0.5e-9
        omega = 1e3
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, omega=omega, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="single")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()
        # Rabi coupling should transfer some power between components
        power_0 = np.sum(np.abs(E_out[0]) ** 2)
        power_1 = np.sum(np.abs(E_out[1]) ** 2)
        assert power_0 > 0, "Component 0 has zero power"
        assert power_1 > 0, "Component 1 has zero power"

    def test_cnlse_prepare_output_normalize(self):
        """CNLSE _prepare_output_array normalizes both components."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        E_in = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(E_in, normalize=True)

        assert out.shape == (2, S, S)
        assert out_sq.shape == (2, S, S)
        # Check normalization for each component
        integral = (
            (out.real * out.real + out.imag * out.imag)
            * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        np.testing.assert_allclose(
            integral, [simu.power, simu.power2], rtol=1e-4,
            err_msg="CNLSE normalization failed",
        )

    def test_nonlocal_propagation(self):
        """CNLSE with nl_length > 0."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_nonlocal_double_precision(self):
        """CNLSE with nl_length > 0 and double precision."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, precision="double")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_nonlocal_with_V(self):
        """CNLSE with nl_length > 0 and potential."""
        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, nl_length=1e-3, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_cnlse_power_conservation(self):
        """CNLSE with alpha=0 conserves total power."""
        n12_local = 0.5e-9
        simu = CNLSE(
            0, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        E_in = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False)

        norm = np.sum(
            np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y,
            axis=simu._last_axes,
        )
        norm *= c * epsilon_0 / 2
        np.testing.assert_allclose(
            norm, [simu.power, simu.power2], rtol=1e-4,
            err_msg="CNLSE norm not conserved (alpha=0)",
        )


class TestPlotField:
    """Tests for plot_field methods (uses Agg backend)."""

    def test_nlse_plot_field(self):
        """NLSE plot_field runs without error."""
        import matplotlib.pyplot as plt

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        A = np.ones((S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_nlse_plot_field_3d(self):
        """NLSE plot_field with >2D array drops dims."""
        import matplotlib.pyplot as plt

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        A = np.ones((3, S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_cnlse_plot_field(self):
        """CNLSE plot_field runs without error."""
        import matplotlib.pyplot as plt

        n12_local = 0.5e-9
        simu = CNLSE(
            alpha, power, window, n2, n12_local, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        A = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_nlse_out_field_with_plot(self):
        """out_field with plot=True runs without error."""
        import matplotlib.pyplot as plt

        simu = NLSE(
            alpha, power, window, n2, None, L,
            NX=S, NY=S, Isat=Isat, backend="CPU",
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=True)
        assert np.isfinite(E_out).all()
        plt.close("all")
