"""Cross-backend correctness tests.

CPU is the reference backend. All other backends are compared against it.
Also validates CPU backend internals (FFT plans, propagator, conservation laws).
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for plot tests

import numpy as np
import pytest
from helpers import as_numpy, make
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import list_available_backends
from scipy.constants import c, epsilon_0

AVAILABLE_BACKENDS = list_available_backends()
GPU_BACKENDS = [b for b in AVAILABLE_BACKENDS if b != "CPU"]

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Physical parameters matching reference
N = 128
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3
alpha = 20
n12 = 0.5e-9

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


def make_nlse(backend="CPU", n=N, **overrides):
    """Return an NLSE with this module's parameters.

    Ninety-odd solver constructions in this file differed only in backend,
    grid size and one or two parameters; spelling each out in full made the
    difference between two tests the hardest thing to see in them.
    """
    return make(NLSE, backend, n=n, **overrides)


def make_cnlse(backend="CPU", n=N, **overrides):
    """Return a CNLSE with this module's parameters.

    Ninety-odd solver constructions in this file differed only in backend,
    grid size and one or two parameters; spelling each out in full made the
    difference between two tests the hardest thing to see in them.
    """
    return make(CNLSE, backend, n=n, **{"n12": n12, **overrides})


class TestCPUCorrectness:
    """Tests that validate CPU backend internals."""

    def test_build_fft_plan(self):
        """The plan is the axes to transform over, and nothing else."""
        simu = make_nlse()
        A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
        plans = simu._build_fft_plan(A)

        assert len(plans) == 1, "Number of plans is wrong"
        assert plans[0] == (-2, -1), "Plan axes are wrong"

    def test_fft_roundtrip(self):
        """FFT then IFFT returns the original array."""
        simu = make_nlse()
        A_orig = np.random.random((N, N)) + 1j * np.random.random((N, N))
        A_orig = A_orig.astype(PRECISION_COMPLEX)

        A, _ = simu._prepare_output_array(A_orig.copy(), normalize=False)
        plans = simu._build_fft_plan(A)

        A_saved = A.copy()
        simu._backend.fft(A, plans)
        simu._backend.ifft(A, plans)

        np.testing.assert_allclose(
            A,
            A_saved,
            rtol=1e-5,
            atol=1e-6,
            err_msg="FFT roundtrip failed to recover original array",
        )

    def test_fft_plan_performance(self):
        """FFT plans are not degraded by stale wisdom."""
        import time

        simu = make_nlse()
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
            f"FFT roundtrip too slow: {t_per_roundtrip * 1e3:.0f}ms "
            f"(stale wisdom file?)"
        )

    def test_split_step_unitarity(self):
        """Split step with delta_z=0 preserves the field."""
        simu = make_nlse()
        E = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        # A zero step is the identity, which is what unitarity is checked against.
        simu.propagator = simu._build_propagator(PRECISION_COMPLEX, 0)

        simu.split_step(
            A,
            A_sq,
            simu.V,
            simu.propagator,
            simu.plans,
            0,
            splitting="strang",
        )

        np.testing.assert_allclose(
            A,
            np.ones((N, N), dtype=PRECISION_COMPLEX),
            rtol=1e-6,
            atol=1e-7,
            err_msg="Split step is not unitary (delta_z=0 should preserve field)",
        )

    def test_propagator_correctness(self):
        """Propagator matches analytical formula."""
        simu = make_nlse()
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        expected = np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * DZ_TEST)
        assert np.allclose(prop, expected), "Propagator is wrong (CPU)"

    def test_prepare_output_array(self):
        """Output array is normalized, aligned, and contiguous."""
        simu = make_nlse()
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
            (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral = integral * c * epsilon_0 / 2
        np.testing.assert_allclose(
            integral,
            simu.power,
            rtol=1e-4,
            err_msg="Normalization failed",
        )

    def test_power_conservation(self):
        """Propagation with alpha=0 conserves the norm."""
        simu = make_nlse("CPU", alpha=0)

        E_in = np.ones((N, N), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="lie")

        norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2

        np.testing.assert_allclose(
            norm,
            simu.power,
            rtol=1e-4,
            err_msg="Norm not conserved on CPU (alpha=0)",
        )

    def test_power_decay(self):
        """Propagation with alpha>0 gives expected exponential power decay."""
        test_alpha = 10.0
        simu = make_nlse("CPU", n=64, alpha=test_alpha, n2=1e-10, Isat=1e20)

        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, normalize=False)

        power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
        power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y

        expected_ratio = np.exp(-test_alpha * L)
        actual_ratio = power_out / power_in
        rel_error = np.abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_error < 0.01, f"Power decay incorrect on CPU: {rel_error:.2%} error"

    def test_propagation_with_potential(self):
        """CPU propagation with V produces a valid result."""
        NX = NY = 64
        simu = make_nlse("CPU", n=NX)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (NX, NY)
        assert np.isfinite(E_out).all(), "Output contains NaN/Inf"

    def test_coupled_propagation(self):
        """CPU CNLSE propagation produces a valid result."""
        NX = NY = 64
        simu = make_cnlse("CPU", n=NX)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (2, NX, NY)
        assert np.isfinite(E_out).all(), "CNLSE output contains NaN/Inf"


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestNLSEvsReference:
    """Compare each backend against the CPU reference for split-step."""

    def test_propagation_without_potential(self, backend):
        """Results without potential match CPU reference."""
        simu_ref = make_nlse()
        simu_test = make_nlse(backend)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1.5e-2,
            atol=1e-4,
            err_msg=f"{backend} does not match CPU reference (no potential)",
        )

    def test_propagation_with_potential(self, backend):
        """Results with potential match CPU reference."""
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, N),
            np.linspace(-window / 2, window / 2, N),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ref = make_nlse("CPU", V=V)
        simu_test = make_nlse(backend, V=V)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1.5e-2,
            atol=1e-4,
            err_msg=f"{backend} does not match CPU reference (with potential)",
        )

    def test_power_conservation(self, backend):
        """Power conservation (alpha=0) matches CPU reference."""
        simu = make_nlse(backend, alpha=0)

        E_in = np.ones((N, N), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="lie")

        norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2

        np.testing.assert_allclose(
            norm,
            simu.power,
            rtol=1e-4,
            err_msg=f"Norm not conserved on {backend} (alpha=0)",
        )

    def test_propagator_correctness(self, backend):
        """Propagator matches analytical formula on each backend."""
        simu = make_nlse(backend)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        expected = np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * DZ_TEST)
        assert np.allclose(prop, expected), f"Propagator is wrong (Backend {backend})"


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestCNLSEvsReference:
    """Compare CNLSE on each backend against the CPU reference."""

    def test_coupled_propagation(self, backend):
        """Coupled NLSE results match CPU reference."""
        NX = NY = 64

        simu_ref = make_cnlse("CPU", n=NX)
        simu_test = make_cnlse(backend, n=NX)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE {backend} does not match CPU reference",
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestNLSERK4vsReference:
    """Compare RK4 on each backend against the CPU reference."""

    def test_rk4_without_potential(self, backend):
        """RK4 results without potential match CPU reference."""
        simu_ref = make_nlse()
        simu_test = make_nlse(backend)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )
        E_test = simu_test.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1.5e-2,
            atol=1e-4,
            err_msg=f"RK4 {backend} does not match CPU reference (no potential)",
        )

    def test_rk4_with_potential(self, backend):
        """RK4 results with potential match CPU reference."""
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, N),
            np.linspace(-window / 2, window / 2, N),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ref = make_nlse("CPU", V=V)
        simu_test = make_nlse(backend, V=V)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ref = simu_ref.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )
        E_test = simu_test.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1.5e-2,
            atol=1e-4,
            err_msg=f"RK4 {backend} does not match CPU reference (with potential)",
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestCNLSERK4vsReference:
    """Compare CNLSE RK4 on each backend against the CPU reference."""

    def test_cnlse_rk4(self, backend):
        """CNLSE RK4 results match CPU reference."""
        NX = NY = 64

        simu_ref = make_cnlse("CPU", n=NX)
        simu_test = make_cnlse(backend, n=NX)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )
        E_test = simu_test.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE RK4 {backend} does not match CPU reference",
        )

    def test_cnlse_rk4_with_potential(self, backend):
        """CNLSE RK4 with potential matches CPU reference."""
        NX = NY = 64
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, NX),
            np.linspace(-window / 2, window / 2, NY),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ref = make_cnlse("CPU", n=NX, V=V)
        simu_test = make_cnlse(backend, n=NX, V=V)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )
        E_test = simu_test.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE RK4 {backend} with V does not match CPU reference",
        )


@pytest.mark.parametrize("backend", GPU_BACKENDS)
class TestCNLSEvsReferenceExtended:
    """Extended CNLSE cross-backend tests (potential, omega, double precision)."""

    def test_coupled_propagation_with_potential(self, backend):
        """CNLSE with potential matches CPU reference."""
        NX = NY = 64
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, NX),
            np.linspace(-window / 2, window / 2, NY),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ref = make_cnlse("CPU", n=NX, V=V)
        simu_test = make_cnlse(backend, n=NX, V=V)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE {backend} with V does not match CPU reference",
        )

    def test_coupled_propagation_with_omega(self, backend):
        """CNLSE with Rabi coupling matches CPU reference."""
        NX = NY = 64
        omega = 1e3

        simu_ref = make_cnlse("CPU", n=NX, omega=omega)
        simu_test = make_cnlse(backend, n=NX, omega=omega)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False)
        E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False)

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE {backend} with omega does not match CPU reference",
        )

    def test_coupled_propagation_double_split_step(self, backend):
        """CNLSE with the double-order split step matches the CPU reference.

        splitting="strang" is the *splitting order* — the nonlinear step is
        applied around the linear one rather than once per step. It is not
        float64, and with a complex64 field it needs no fp64 support, so
        every backend runs it.

        Skipping it on fp64 support would be wrong, and a NaN result is a
        real failure rather than a driver limitation: the GPU kernels pick
        their precision from the field, then read the propagator with it, so
        a complex128 propagator under a complex64 field returns NaN.
        """
        NX = NY = 64

        simu_test = make_cnlse(backend, n=NX)
        simu_ref = make_cnlse("CPU", n=NX)

        XX, YY = np.meshgrid(simu_ref.X, simu_ref.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ref = simu_ref.out_field(
            E_in.copy(), L, verbose=False, plot=False, splitting="strang"
        )
        E_test = simu_test.out_field(
            E_in.copy(), L, verbose=False, plot=False, splitting="strang"
        )

        assert not np.any(np.isnan(E_test)), (
            f"{backend} returned NaN: the propagator dtype no longer matches "
            f"the field, so the kernels are reading it at the wrong width"
        )

        np.testing.assert_allclose(
            E_test,
            E_ref,
            rtol=1e-2,
            atol=1e-4,
            err_msg=f"CNLSE {backend} double precision does not match CPU reference",
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
class TestNLSECrossMethod:
    """Compare split_step vs RK4 on same backend (both solve same equation)."""

    def test_split_step_vs_rk4_without_potential(self, backend):
        """Split-step and RK4 converge to same result (no potential)."""
        simu_ss = make_nlse(backend)
        simu_rk = make_nlse(backend)

        XX, YY = np.meshgrid(simu_ss.X, simu_ss.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ss = simu_ss.out_field(
            E_in.copy(), L, verbose=False, plot=False, splitting="lie"
        )
        E_rk = simu_rk.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_rk,
            E_ss,
            rtol=5e-2,
            atol=1e-3,
            err_msg=f"RK4 vs split_step mismatch on {backend} (no potential)",
        )

    def test_split_step_vs_rk4_with_potential(self, backend):
        """Split-step and RK4 converge to same result (with potential)."""
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, N),
            np.linspace(-window / 2, window / 2, N),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ss = make_nlse(backend, V=V)
        simu_rk = make_nlse(backend, V=V)

        XX, YY = np.meshgrid(simu_ss.X, simu_ss.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_ss = simu_ss.out_field(
            E_in.copy(), L, verbose=False, plot=False, splitting="lie"
        )
        E_rk = simu_rk.out_field(
            E_in.copy(), L, verbose=False, plot=False, method="RK4"
        )

        np.testing.assert_allclose(
            E_rk,
            E_ss,
            rtol=5e-2,
            atol=1e-3,
            err_msg=f"RK4 vs split_step mismatch on {backend} (with potential)",
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
class TestCNLSECrossMethod:
    """Compare CNLSE split_step vs RK4 on same backend.

    Uses weak nonlinearity so both O(dz) split-step and O(dz^4) RK4 converge
    to the same result within tolerance.
    """

    # Weak NL for method convergence
    n2_weak = -1e-10
    n12_weak = 1e-10
    dz_fine = 1e-5

    def test_cnlse_split_step_vs_rk4(self, backend):
        """CNLSE split-step and RK4 converge to same result."""
        NX = NY = 64

        simu_ss = make_cnlse(backend, n=NX, n2=self.n2_weak, n12=self.n12_weak)
        simu_rk = make_cnlse(backend, n=NX, n2=self.n2_weak, n12=self.n12_weak)

        XX, YY = np.meshgrid(simu_ss.X, simu_ss.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ss = simu_ss.out_field(
            E_in.copy(),
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=self.dz_fine,
        )
        E_rk = simu_rk.out_field(
            E_in.copy(),
            L,
            verbose=False,
            plot=False,
            method="RK4",
            delta_z=self.dz_fine,
        )

        np.testing.assert_allclose(
            E_rk,
            E_ss,
            rtol=5e-2,
            atol=1e-3,
            err_msg=f"CNLSE RK4 vs split_step mismatch on {backend}",
        )

    def test_cnlse_split_step_vs_rk4_with_potential(self, backend):
        """CNLSE split-step and RK4 with potential converge to same result."""
        NX = NY = 64
        XX_v, YY_v = np.meshgrid(
            np.linspace(-window / 2, window / 2, NX),
            np.linspace(-window / 2, window / 2, NY),
        )
        V = 1e-4 * np.exp(-(XX_v**2 + YY_v**2) / (2e-3) ** 2)

        simu_ss = make_cnlse(backend, n=NX, n2=self.n2_weak, n12=self.n12_weak, V=V)
        simu_rk = make_cnlse(backend, n=NX, n2=self.n2_weak, n12=self.n12_weak, V=V)

        XX, YY = np.meshgrid(simu_ss.X, simu_ss.Y)
        E_in = np.zeros((2, NX, NY), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_ss = simu_ss.out_field(
            E_in.copy(),
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            delta_z=self.dz_fine,
        )
        E_rk = simu_rk.out_field(
            E_in.copy(),
            L,
            verbose=False,
            plot=False,
            method="RK4",
            delta_z=self.dz_fine,
        )

        np.testing.assert_allclose(
            E_rk,
            E_ss,
            rtol=5e-2,
            atol=1e-3,
            err_msg=f"CNLSE RK4 vs split_step with V mismatch on {backend}",
        )


# Small grid for fast tests
S = 64

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4


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
        simu = make_nlse("CPU", n=S, window=win)
        assert simu.window[0] == win[0]
        assert simu.window[1] == win[1]
        # X and Y grids should span different ranges
        assert not np.isclose(simu.X[-1] - simu.X[0], simu.Y[-1] - simu.Y[0])

    def test_window_as_list(self):
        """Window parameter accepts list."""
        win = [window, window * 2]
        simu = make_nlse("CPU", n=S, window=win)
        assert simu.window[0] == win[0]
        assert simu.window[1] == win[1]

    def test_backend_property(self):
        """Backend property returns name string."""
        simu = make_nlse("CPU", n=S)
        assert simu.backend == "CPU"

    def test_backend_setter(self):
        """Backend can be changed via setter."""
        simu = make_nlse("CPU", n=S)
        assert simu.backend == "CPU"
        # Re-set to same backend (always available)
        simu.backend = "CPU"
        assert simu.backend == "CPU"

    def test_nl_length_positive(self):
        """Positive nl_length creates Bessel non-local profile."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        assert simu.nl_length > 0
        # nl_profile should be a small kernel, not full grid
        assert simu.nl_profile.shape[0] < S
        # nl_profile should be normalized
        np.testing.assert_allclose(simu.nl_profile.sum(), 1.0, rtol=1e-5)


class TestNLSEPropagator:
    """Tests for propagator building and caching."""

    def test_propagator_caching(self):
        """Propagator is cached and reused."""
        simu = make_nlse("CPU", n=S)
        prop1 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        prop2 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        assert prop1 is prop2  # same object from cache

    def test_propagator_double_precision(self):
        """Double precision propagator has complex128 dtype.

        This one really is about the dtype asked for, unlike the split-step
        tests nearby, which passed complex128 here and a complex64 field to
        the kernel: splitting="strang" is the order of the splitting, not the
        width of anything.
        """
        simu = make_nlse("CPU", n=S)
        prop = as_numpy(simu, simu._build_propagator(np.complex128, DZ_TEST))
        assert prop.dtype == np.complex128

    def test_propagator_rk4(self):
        """RK4 propagator does not include delta_z exponential."""
        simu = make_nlse("CPU", n=S)
        prop = simu._build_propagator_rk4(np.complex64)
        expected = -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k
        np.testing.assert_allclose(
            prop,
            expected.astype(np.complex64),
            rtol=1e-5,
            err_msg="RK4 propagator is wrong",
        )


class TestNLSESplitStep:
    """Tests for split_step code paths."""

    def test_double_precision_without_V(self):
        """Double precision split step without potential."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        simu.split_step(A, A_sq, None, prop, plans, DZ_TEST, splitting="strang")
        assert np.isfinite(A).all(), "Double precision split step produced NaN/Inf"

    def test_double_precision_with_V(self):
        """Double precision split step with potential."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        simu.split_step(A, A_sq, V, prop, plans, DZ_TEST, splitting="strang")
        assert np.isfinite(A).all()

    def test_single_precision_with_V(self):
        """Single precision split step with potential."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        simu.split_step(A, A_sq, V, prop, plans, DZ_TEST, splitting="lie")
        assert np.isfinite(A).all()

    def test_nonlocal_propagation(self):
        """Split step with nl_length > 0 uses convolution."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        # Single precision path with nl_length
        simu.split_step(A, A_sq, None, prop, plans, DZ_TEST, splitting="lie")
        assert np.isfinite(A).all()

    def test_nonlocal_with_V(self):
        """Split step with nl_length > 0 and potential."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        simu.split_step(A, A_sq, V, prop, plans, DZ_TEST, splitting="lie")
        assert np.isfinite(A).all()

    def test_nonlocal_double_precision(self):
        """Double precision split step with nl_length > 0."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))

        simu.split_step(A, A_sq, None, prop, plans, DZ_TEST, splitting="strang")
        assert np.isfinite(A).all()


class TestNLSERK4:
    """Tests for RK4 propagation scheme."""

    def test_rk4_propagation(self):
        """RK4 scheme produces valid output."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(
            E_in,
            L,
            verbose=False,
            plot=False,
            splitting="RK4",
        )
        assert E_out.shape == (S, S)
        assert np.isfinite(E_out).all(), "RK4 output contains NaN/Inf"

    def test_rk4_single_step_with_potential(self):
        """RK4 single step with potential exercises the V code path."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        A, _A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator_rk4(np.complex64)

        # Just one step to exercise the code path
        A_before = A.copy()
        simu.split_step_RK4(A, V, prop, plans, DZ_TEST)
        # Field should have changed
        assert not np.allclose(A, A_before)

    def test_rk4_nonlocal(self):
        """RK4 scheme with nl_length > 0."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(
            E_in,
            L,
            verbose=False,
            plot=False,
            splitting="RK4",
        )
        assert np.isfinite(E_out).all()

    def test_rk4_power_conservation(self):
        """RK4 with alpha=0 conserves the norm."""
        simu = make_nlse("CPU", n=S, alpha=0)
        E_in = np.ones((S, S), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False, method="RK4")

        norm = np.sum(np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2

        np.testing.assert_allclose(
            norm,
            simu.power,
            rtol=1e-4,
            err_msg="Norm not conserved with RK4 (alpha=0)",
        )

    def test_rk4_power_decay(self):
        """RK4 with alpha>0 gives expected exponential power decay."""
        test_alpha = 10.0
        simu = make_nlse("CPU", n=S, alpha=test_alpha, n2=1e-10, Isat=1e20)

        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)
        E_out = simu.out_field(
            E_in, L, verbose=False, plot=False, normalize=False, method="RK4"
        )

        power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
        power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y

        expected_ratio = np.exp(-test_alpha * L)
        actual_ratio = power_out / power_in
        rel_error = np.abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_error < 0.01, f"RK4 power decay incorrect: {rel_error:.2%} error"

    def test_rk4_method_parameter(self):
        """method='RK4' produces same result as splitting='RK4' (backward compat)."""
        simu1 = make_nlse("CPU", n=S)
        simu2 = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_old = simu1.out_field(
            E_in.copy(), L, verbose=False, plot=False, splitting="RK4"
        )
        E_new = simu2.out_field(E_in.copy(), L, verbose=False, plot=False, method="RK4")

        np.testing.assert_allclose(
            E_new,
            E_old,
            rtol=1e-6,
            err_msg="method='RK4' does not match splitting='RK4'",
        )


class TestCNLSERK4:
    """Tests for CNLSE RK4 propagation scheme."""

    def test_cnlse_rk4_propagation(self):
        """CNLSE RK4 produces valid output."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, method="RK4")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all(), "CNLSE RK4 output contains NaN/Inf"

    def test_cnlse_rk4_with_potential(self):
        """CNLSE RK4 single step with potential exercises the V code path."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)
        A, _A_sq = simu._prepare_output_array(E_in, normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator_rk4(np.complex64)

        A_before = A.copy()
        simu.split_step_RK4(A, V, prop, plans, DZ_TEST)
        assert np.isfinite(A).all(), "CNLSE RK4 with V produced NaN/Inf"
        assert not np.allclose(A, A_before), "Field unchanged after RK4 step with V"

    def test_cnlse_rk4_nonlocal(self):
        """CNLSE RK4 with nl_length > 0."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, method="RK4")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()


class TestNLSEOutField:
    """Tests for out_field code paths."""

    def test_verbose_output(self, capsys):
        """Verbose mode prints timing info."""
        simu = make_nlse("CPU", n=S)
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

        simu = make_nlse("CPU", n=S)
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

        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        simu.out_field(
            E_in,
            L,
            verbose=False,
            plot=False,
            callback=[cb1, cb2],
            callback_args=[(0,), (1,)],
        )
        assert counts[0] > 0, "First callback was never called"
        assert counts[1] > 0, "Second callback was never called"

    def test_callback_invalid_raises(self):
        """Invalid callback type raises ValueError."""
        simu = make_nlse("CPU", n=S)
        E_in = np.ones((S, S), dtype=PRECISION_COMPLEX)

        with pytest.raises(ValueError, match="callbacks should be a callable"):
            simu.out_field(
                E_in,
                L,
                verbose=False,
                plot=False,
                callback="not_a_callable",
            )

    def test_normalize_false(self):
        """normalize=False skips power normalization."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, normalize=False)
        assert E_out.shape == (S, S)
        assert np.isfinite(E_out).all()

    def test_double_precision_out_field(self):
        """Double precision propagation produces valid result."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="strang")
        assert np.isfinite(E_out).all()

    def test_double_precision_with_potential(self):
        """Double precision propagation with V."""
        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="strang")
        assert np.isfinite(E_out).all()


class TestCNLSEExtended:
    """Extended CNLSE coverage tests on CPU."""

    def test_build_propagator(self):
        """CNLSE propagator has two components."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        assert prop.shape == (2, S, S)
        assert np.isfinite(prop).all()

    def test_propagator_caching(self):
        """CNLSE propagator is cached."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        prop1 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        prop2 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        assert prop1 is prop2

    def test_propagator_rk4(self):
        """CNLSE RK4 propagator."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        prop = simu._build_propagator_rk4(np.complex64)
        assert prop.shape == (2, S, S)
        # RK4 propagator should not be exponential (no delta_z)
        assert not np.allclose(np.abs(prop), 1.0)

    def test_take_components_cpu(self):
        """_take_components returns views on CPU."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
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
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="strang")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_single_precision_with_V(self):
        """CNLSE single precision with potential."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="lie")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_double_precision_with_V(self):
        """CNLSE double precision with potential."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="strang")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_rabi_coupling(self):
        """CNLSE with Rabi coupling (omega != None)."""
        n12_local = 0.5e-9
        omega = 1e3
        simu = make_cnlse("CPU", n=S, n12=n12_local, omega=omega)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="lie")
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
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        E_in = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(E_in, normalize=True)

        assert out.shape == (2, S, S)
        assert out_sq.shape == (2, S, S)
        # Check normalization for each component
        integral = (
            (out.real * out.real + out.imag * out.imag) * simu.delta_X * simu.delta_Y
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        np.testing.assert_allclose(
            integral,
            [simu.power, simu.power2],
            rtol=1e-4,
            err_msg="CNLSE normalization failed",
        )

    def test_nonlocal_propagation(self):
        """CNLSE with nl_length > 0."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local, nl_length=1e-3)
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
        simu = make_cnlse("CPU", n=S, n12=n12_local, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False, splitting="strang")
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_nonlocal_with_V(self):
        """CNLSE with nl_length > 0 and potential."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local, nl_length=1e-3)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        simu.V = 1e-4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2).astype(np.float32)
        E_in = np.zeros((2, S, S), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        E_out = simu.out_field(E_in, L, verbose=False, plot=False)
        assert E_out.shape == (2, S, S)
        assert np.isfinite(E_out).all()

    def test_cnlse_power_conservation(self):
        """CNLSE with alpha=0 conserves total power."""
        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, alpha=0, n12=n12_local)
        E_in = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        E_out = simu.out_field(E_in, L, verbose=False, plot=False)

        norm = np.sum(
            np.abs(E_out) ** 2 * simu.delta_X * simu.delta_Y,
            axis=simu._last_axes,
        )
        norm *= c * epsilon_0 / 2
        np.testing.assert_allclose(
            norm,
            [simu.power, simu.power2],
            rtol=1e-4,
            err_msg="CNLSE norm not conserved (alpha=0)",
        )


# Solvers that take a method argument, with a field carrying only its spatial
# axes -- which is exactly the shape the whole-step fused RK4 path claims.
RK4_SOLVERS = {
    "NLSE": (NLSE, (32, 32)),
    "NLSE_1d": (NLSE_1d, (32,)),
    "CNLSE": (CNLSE, (2, 16, 16)),
    "CNLSE_1d": (CNLSE_1d, (2, 32)),
}


@pytest.mark.parametrize("backend_name", [b for b in AVAILABLE_BACKENDS if b != "CPU"])
@pytest.mark.parametrize("solver_name", sorted(RK4_SOLVERS))
def test_rk4_agrees_with_the_cpu(solver_name, backend_name):
    """RK4 must integrate the same equation on every backend.

    Nothing compared RK4 across backends for the coupled 1d solver, and the
    fused-path tests skip MLX for RK4, so CNLSE_1d on MLX ran 5% away from the
    CPU without anything noticing. The cause was a shape test: the whole-step
    fused kernel is for a single component and asked `A.ndim == 2`, which a
    coupled 1d field (2, NX) also satisfies.

    Every solver here, not only that one, because the guard it got wrong is
    shared by all of them and each has a different field shape.

    Parameters
    ----------
    solver_name : str
        Which solver to integrate.
    backend_name : str
        Backend to compare against the CPU.
    """
    cls, shape = RK4_SOLVERS[solver_name]
    field = np.exp(-(np.linspace(-2, 2, int(np.prod(shape))) ** 2))
    field = field.reshape(shape).astype(PRECISION_COMPLEX)

    def run(backend):
        simu = make(cls, backend, n=shape[-1])
        return as_numpy(
            simu,
            simu.out_field(
                field.copy(),
                5e-4,
                verbose=False,
                plot=False,
                delta_z=5e-6,
                method="RK4",
            ),
        )

    got, expected = run(backend_name), run("CPU")
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-4,
        atol=1e-4 * float(np.max(np.abs(expected))),
        err_msg=f"{solver_name} RK4 on {backend_name} disagrees with the CPU",
    )


# Every solver that plots, which is every solver. NLSE_3d and GPE need their
# own construction -- one takes a window per axis and the other is written in
# atoms rather than watts -- so they cannot come through `make`, which is why
# they are easy to leave out of a sweep and were left out of this one.
PLOTTABLE = {
    "NLSE": (
        lambda b: make(NLSE, b, n=S),
        lambda simu: (simu.NY, simu.NX),
    ),
    "NLSE_1d": (
        lambda b: make(NLSE_1d, b, n=S),
        lambda simu: (simu.NX,),
    ),
    "CNLSE": (
        lambda b: make(CNLSE, b, n=S),
        lambda simu: (2, simu.NY, simu.NX),
    ),
    "CNLSE_1d": (
        lambda b: make(CNLSE_1d, b, n=S),
        lambda simu: (2, simu.NX),
    ),
    "NLSE_3d": (
        lambda b: NLSE_3d(
            alpha=0.0,
            energy=1e-6,
            window=np.array([4 * 2.23e-3, 8e-6]),
            n2=-1.6e-9,
            D0=1e-27,
            vg=2e8,
            V=None,
            L=L,
            NX=16,
            NY=16,
            NZ=16,
            Isat=1e5,
            backend=b,
        ),
        lambda simu: (simu.NX, simu.NY, simu.NZ),
    ),
    "GPE": (
        lambda b: GPE(
            gamma=0.0,
            N=1e6,
            window=4 * 2.23e-3,
            g=1e-3,
            V=None,
            m=1e-26,
            NX=S,
            NY=S,
            backend=b,
        ),
        lambda simu: (simu.NY, simu.NX),
    ),
}


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("solver_name", sorted(PLOTTABLE))
def test_plot_field_accepts_the_field_the_backend_holds(solver_name, backend_name):
    """plot_field is handed the running field, not a numpy copy of it.

    The tests above pass a numpy array and run on the CPU, so they never saw
    what plot_field is actually called with during a run: whatever array type
    the backend keeps. NLSE_1d was the one solver not routing its argument
    through _to_plot_array first, and called np.abs and np.angle straight on
    it -- fine on the CPU and on MLX, "setting an array element with a
    sequence" on CL.

    Parameters
    ----------
    solver_name : str
        Which solver to plot.
    backend_name : str
        Backend to run on.
    """
    import matplotlib.pyplot as plt

    build, shape_of = PLOTTABLE[solver_name]
    simu = build(backend_name)
    field = np.ones(shape_of(simu), dtype=PRECISION_COMPLEX)
    try:
        simu.plot_field(simu._backend.from_numpy(field), L)
    finally:
        plt.close("all")


class TestPlotField:
    """Tests for plot_field methods (uses Agg backend)."""

    def test_nlse_plot_field(self):
        """NLSE plot_field runs without error."""
        import matplotlib.pyplot as plt

        simu = make_nlse("CPU", n=S)
        A = np.ones((S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_nlse_plot_field_3d(self):
        """NLSE plot_field with >2D array drops dims."""
        import matplotlib.pyplot as plt

        simu = make_nlse("CPU", n=S)
        A = np.ones((3, S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_cnlse_plot_field(self):
        """CNLSE plot_field runs without error."""
        import matplotlib.pyplot as plt

        n12_local = 0.5e-9
        simu = make_cnlse("CPU", n=S, n12=n12_local)
        A = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        simu.plot_field(A, L)
        plt.close("all")

    def test_nlse_out_field_with_plot(self):
        """out_field with plot=True runs without error."""
        import matplotlib.pyplot as plt

        simu = make_nlse("CPU", n=S)
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)

        E_out = simu.out_field(E_in, L, verbose=False, plot=True)
        assert np.isfinite(E_out).all()
        plt.close("all")


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
class TestPropagatorMatchesFieldDtype:
    """The propagator must carry the same dtype as the field it multiplies.

    The GPU kernels select their single- or double-precision variant from the
    *field*, then index the propagator with that same variant. A complex128
    propagator against a complex64 field is therefore read as pairs of
    float32: CUPY returned NaN and it was written off as a driver limitation,
    and Apple OpenCL, which has no fp64 at all, could not even build the
    propagator.

    The cause was `splitting`, which meant two things at once: the split-step
    order (what the docstring documents) and the propagator's float width.
    Only the first is `splitting` now; the width follows the input field.
    """

    def test_propagator_follows_a_single_precision_field(self, backend):
        """A complex64 field must not produce a complex128 propagator."""
        simu = make_nlse(backend, n=64)
        E = np.ones((64, 64), dtype=np.complex64)
        simu.out_field(E, L, verbose=False, plot=False, splitting="strang")
        prop = simu.propagator
        prop = prop if isinstance(prop, np.ndarray) else simu._backend.to_numpy(prop)
        assert np.asarray(prop).dtype == np.complex64, (
            f"{backend}: splitting='strang' built a "
            f"{np.asarray(prop).dtype} propagator for a complex64 field. The "
            f"kernels read it at the field's width, so it comes back NaN."
        )

    def test_precision_double_needs_no_fp64(self, backend):
        """The double-order split step must run on a device without fp64.

        splitting="strang" is the splitting order, not float64. Requiring
        fp64 for it made every fp64-less backend skip tests it could run.
        """
        simu = make_nlse(backend, n=64)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / (2.23e-3) ** 2).astype(np.complex64)
        out = simu.out_field(E.copy(), L, verbose=False, plot=False, splitting="strang")
        out = out if isinstance(out, np.ndarray) else simu._backend.to_numpy(out)
        assert np.all(np.isfinite(np.asarray(out))), (
            f"{backend}: the double-order split step produced non-finite "
            f"values on a complex64 field"
        )

    def test_a_double_precision_field_gets_a_double_propagator(self, backend):
        """Real fp64: a complex128 field must get a complex128 propagator."""
        simu = make_nlse(backend, n=64)
        if not simu._backend.supports_double_precision():
            pytest.skip(f"{backend} has no fp64, so a complex128 field cannot run")
        E = np.ones((64, 64), dtype=np.complex128)
        simu.out_field(E, L, verbose=False, plot=False, splitting="strang")
        prop = simu.propagator
        prop = prop if isinstance(prop, np.ndarray) else simu._backend.to_numpy(prop)
        assert np.asarray(prop).dtype == np.complex128, (
            f"{backend}: a complex128 field got a "
            f"{np.asarray(prop).dtype} propagator, losing the precision the "
            f"caller asked for"
        )
