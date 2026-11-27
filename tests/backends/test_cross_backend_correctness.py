"""Cross-backend correctness tests.

CPU is the reference backend. All other backends are compared against it.
Also validates CPU backend internals (FFT plans, propagator, conservation laws).
"""

import matplotlib
from helpers import AVAILABLE_BACKENDS

matplotlib.use("Agg")  # non-interactive backend for plot tests

import numpy as np
import pytest
from helpers import as_numpy, make
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from scipy.constants import c, epsilon_0

GPU_BACKENDS = [b for b in AVAILABLE_BACKENDS if b != "CPU"]

PRECISION_COMPLEX = np.complex64

# Physical parameters matching reference
N = 128
waist = 2.23e-3
window = 4 * waist
L = 1e-3
n12 = 0.5e-9

# Step used wherever a test builds a propagator or takes a step by hand.
DZ_TEST = 1e-4
# Small grid for fast tests
S = 64
# The test potential: a Gaussian bump of this height (a.u., as V is documented
# at NLSE/solvers/cnlse.py:53) and 1/e radius (m), as the tests this file
# replaced built it (e60e2a4, this file:166).
V_HEIGHT = 1e-4
V_RADIUS = 2e-3
# Weak nonlinearities (m^2/W) and the step (m) at which CNLSE split step and RK4
# agree, from the test this file replaced (e60e2a4, this file:575-577).
N2_WEAK = -1e-10
N12_WEAK = 1e-10
DZ_FINE = 1e-5


def make_nlse(backend="CPU", n=N, **overrides):
    """Return an NLSE with this module's parameters."""
    return make(NLSE, backend, n=n, **overrides)


def make_cnlse(backend="CPU", n=N, **overrides):
    """Return a CNLSE with this module's parameters."""
    return make(CNLSE, backend, n=n, **{"n12": n12, **overrides})


def beam(simu):
    """Return the Gaussian test field on the solver's grid."""
    XX, YY = np.meshgrid(simu.X, simu.Y)
    return np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)


def beam_pair(simu):
    """Return the coupled test field: two Gaussians of different widths."""
    XX, YY = np.meshgrid(simu.X, simu.Y)
    E_in = np.zeros((2, *XX.shape), dtype=PRECISION_COMPLEX)
    E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
    E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)
    return E_in


def bump(n):
    """Return the test potential on an n x n grid spanning the window, float64."""
    x = np.linspace(-window / 2, window / 2, n)
    XX, YY = np.meshgrid(x, x)
    return V_HEIGHT * np.exp(-(XX**2 + YY**2) / V_RADIUS**2)


def grid_bump(simu):
    """Return the test potential on the solver's own grid, float32."""
    V = V_HEIGHT * np.exp(-(simu.XX**2 + simu.YY**2) / V_RADIUS**2)
    return V.astype(np.float32)


class TestCPUCorrectness:
    """Tests that validate CPU backend internals."""

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


def test_power_conservation(backend):
    """Propagation with alpha=0 conserves the norm."""
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


@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_power_decay(method):
    """Propagation with alpha>0 gives expected exponential power decay."""
    test_alpha = 10.0
    simu = make_nlse("CPU", n=S, alpha=test_alpha, n2=1e-10, Isat=1e20)
    E_in = beam(simu)
    E_out = simu.out_field(
        E_in, L, verbose=False, plot=False, normalize=False, method=method
    )
    power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
    power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y
    expected_ratio = np.exp(-test_alpha * L)
    rel_error = np.abs(power_out / power_in - expected_ratio) / expected_ratio
    assert rel_error < 0.01, f"{method} power decay incorrect: {rel_error:.2%} error"


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
@pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
def test_nlse_matches_the_cpu(backend, method, with_V):
    """NLSE on each backend matches the CPU reference."""
    V = bump(N) if with_V else None
    simu_ref = make_nlse("CPU", V=V)
    simu_test = make_nlse(backend, V=V)
    E_in = beam(simu_ref)
    E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False, method=method)
    E_test = simu_test.out_field(
        E_in.copy(), L, verbose=False, plot=False, method=method
    )
    np.testing.assert_allclose(
        E_test,
        E_ref,
        rtol=1.5e-2,
        atol=1e-4,
        err_msg=f"{method} {backend} does not match CPU reference (V: {with_V})",
    )


# Old test name -> (constructor overrides, out_field keywords, potential or not).
CNLSE_VS_CPU = {
    "coupled_propagation": ({}, {}, False),
    "cnlse_rk4": ({}, {"method": "RK4"}, False),
    "cnlse_rk4_with_potential": ({}, {"method": "RK4"}, True),
    "coupled_propagation_with_potential": ({}, {}, True),
    "coupled_propagation_with_omega": ({"omega": 1e3}, {}, False),
    "coupled_propagation_double_split_step": ({}, {"splitting": "strang"}, False),
}


@pytest.mark.parametrize("backend", GPU_BACKENDS)
@pytest.mark.parametrize("case", sorted(CNLSE_VS_CPU))
def test_cnlse_matches_the_cpu(backend, case):
    """CNLSE on each backend matches the CPU reference.

    A complex128 propagator under a complex64 field reads back NaN on the GPU
    kernels, hence the NaN check before the comparison.
    """
    overrides, run, with_V = CNLSE_VS_CPU[case]
    if with_V:
        overrides = {**overrides, "V": bump(S)}
    simu_ref = make_cnlse("CPU", n=S, **overrides)
    simu_test = make_cnlse(backend, n=S, **overrides)
    E_in = beam_pair(simu_ref)
    E_ref = simu_ref.out_field(E_in.copy(), L, verbose=False, plot=False, **run)
    E_test = simu_test.out_field(E_in.copy(), L, verbose=False, plot=False, **run)
    assert not np.any(np.isnan(E_test)), (
        f"{backend} returned NaN: the propagator dtype no longer matches "
        f"the field, so the kernels are reading it at the wrong width"
    )
    np.testing.assert_allclose(
        E_test,
        E_ref,
        rtol=1e-2,
        atol=1e-4,
        err_msg=f"CNLSE {backend} does not match CPU reference ({case})",
    )


@pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
def test_split_step_vs_rk4(backend, with_V):
    """Split-step and RK4 converge to same result."""
    V = bump(N) if with_V else None
    simu_ss = make_nlse(backend, V=V)
    simu_rk = make_nlse(backend, V=V)
    E_in = beam(simu_ss)
    E_ss = simu_ss.out_field(E_in.copy(), L, verbose=False, plot=False, splitting="lie")
    E_rk = simu_rk.out_field(E_in.copy(), L, verbose=False, plot=False, method="RK4")
    np.testing.assert_allclose(
        E_rk,
        E_ss,
        rtol=5e-2,
        atol=1e-3,
        err_msg=f"RK4 vs split_step mismatch on {backend} (V: {with_V})",
    )


@pytest.mark.parametrize("with_V", [False, True], ids=["no_V", "V"])
def test_cnlse_split_step_vs_rk4(backend, with_V):
    """CNLSE split-step and RK4 converge to same result, at weak nonlinearity."""
    overrides = {"n2": N2_WEAK, "n12": N12_WEAK}
    if with_V:
        overrides["V"] = bump(S)
    simu_ss = make_cnlse(backend, n=S, **overrides)
    simu_rk = make_cnlse(backend, n=S, **overrides)
    E_in = beam_pair(simu_ss)
    run = {"verbose": False, "plot": False, "delta_z": DZ_FINE}
    E_ss = simu_ss.out_field(E_in.copy(), L, splitting="lie", **run)
    E_rk = simu_rk.out_field(E_in.copy(), L, method="RK4", **run)
    np.testing.assert_allclose(
        E_rk,
        E_ss,
        rtol=5e-2,
        atol=1e-3,
        err_msg=f"CNLSE RK4 vs split_step mismatch on {backend} (V: {with_V})",
    )


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
        """Positive nl_length is kept, and its response keeps the total."""
        simu = make_nlse("CPU", n=S, nl_length=1e-3)
        assert simu.nl_length == 1e-3
        # R(K = 0) = 1: the smoothing redistributes the intensity only.
        assert simu._nonlocal_response()[0, 0] == 1.0


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


# Old test name -> (constructor overrides, potential or not, splitting).
NLSE_SPLIT_STEPS = {
    "double_precision_without_V": ({}, False, "strang"),
    "double_precision_with_V": ({}, True, "strang"),
    "single_precision_with_V": ({}, True, "lie"),
    "nonlocal_propagation": ({"nl_length": 1e-3}, False, "lie"),
    "nonlocal_with_V": ({"nl_length": 1e-3}, True, "lie"),
    "nonlocal_double_precision": ({"nl_length": 1e-3}, False, "strang"),
}


@pytest.mark.parametrize("case", sorted(NLSE_SPLIT_STEPS))
def test_nlse_split_step_is_finite(case):
    """One CPU split step on each code path leaves the field finite."""
    overrides, with_V, splitting = NLSE_SPLIT_STEPS[case]
    simu = make_nlse("CPU", n=S, **overrides)
    V = grid_bump(simu) if with_V else None
    A, A_sq = simu._prepare_output_array(beam(simu), normalize=True)
    plans = simu._build_fft_plan(A)
    prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
    simu.split_step(A, A_sq, V, prop, plans, DZ_TEST, splitting=splitting)
    assert np.isfinite(A).all(), f"split step produced NaN/Inf ({case})"


# Old test name -> (constructor overrides, potential or not, out_field keywords).
NLSE_RUNS = {
    "propagation_with_potential": ({}, True, {}),
    "rk4_propagation": ({}, False, {"splitting": "RK4"}),
    "rk4_nonlocal": ({"nl_length": 1e-3}, False, {"splitting": "RK4"}),
    "normalize_false": ({}, False, {"normalize": False}),
    "double_precision_out_field": ({}, False, {"splitting": "strang"}),
    "double_precision_with_potential": ({}, True, {"splitting": "strang"}),
}


@pytest.mark.parametrize("case", sorted(NLSE_RUNS))
def test_nlse_run_is_finite(case):
    """A CPU run on each code path returns a finite field of the input's shape."""
    overrides, with_V, run = NLSE_RUNS[case]
    simu = make_nlse("CPU", n=S, **overrides)
    if with_V:
        simu.V = grid_bump(simu)
    E_out = simu.out_field(beam(simu), L, verbose=False, plot=False, **run)
    assert E_out.shape == (S, S)
    assert np.isfinite(E_out).all(), f"output contains NaN/Inf ({case})"


# Old test name -> (constructor overrides, potential or not, out_field keywords).
CNLSE_RUNS = {
    "coupled_propagation": ({}, False, {}),
    "cnlse_rk4_propagation": ({}, False, {"method": "RK4"}),
    "cnlse_rk4_nonlocal": ({"nl_length": 1e-3}, False, {"method": "RK4"}),
    "double_precision_without_V": ({}, False, {"splitting": "strang"}),
    "single_precision_with_V": ({}, True, {"splitting": "lie"}),
    "double_precision_with_V": ({}, True, {"splitting": "strang"}),
    "rabi_coupling": ({"omega": 1e3}, False, {"splitting": "lie"}),
    "nonlocal_propagation": ({"nl_length": 1e-3}, False, {}),
    "nonlocal_double_precision": ({"nl_length": 1e-3}, False, {"splitting": "strang"}),
    "nonlocal_with_V": ({"nl_length": 1e-3}, True, {}),
}


@pytest.mark.parametrize("case", sorted(CNLSE_RUNS))
def test_cnlse_run_is_finite(case):
    """A CPU CNLSE run on each code path returns finite, non-empty components."""
    overrides, with_V, run = CNLSE_RUNS[case]
    simu = make_cnlse("CPU", n=S, **overrides)
    if with_V:
        simu.V = grid_bump(simu)
    E_out = simu.out_field(beam_pair(simu), L, verbose=False, plot=False, **run)
    assert E_out.shape == (2, S, S)
    assert np.isfinite(E_out).all(), f"CNLSE output contains NaN/Inf ({case})"
    assert np.sum(np.abs(E_out[0]) ** 2) > 0, "Component 0 has zero power"
    assert np.sum(np.abs(E_out[1]) ** 2) > 0, "Component 1 has zero power"


class TestNLSERK4:
    """Tests for RK4 propagation scheme."""

    def test_rk4_single_step_with_potential(self):
        """RK4 single step with potential exercises the V code path."""
        simu = make_nlse("CPU", n=S)
        A, _A_sq = simu._prepare_output_array(beam(simu), normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator_rk4(np.complex64)
        A_before = A.copy()
        simu.split_step_RK4(A, grid_bump(simu), prop, plans, DZ_TEST)
        assert not np.allclose(A, A_before)

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

    def test_rk4_method_parameter(self):
        """method='RK4' produces same result as splitting='RK4' (backward compat)."""
        simu1 = make_nlse("CPU", n=S)
        simu2 = make_nlse("CPU", n=S)
        E_in = beam(simu1)
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

    def test_cnlse_rk4_with_potential(self):
        """CNLSE RK4 single step with potential exercises the V code path."""
        simu = make_cnlse("CPU", n=S)
        A, _A_sq = simu._prepare_output_array(beam_pair(simu), normalize=True)
        plans = simu._build_fft_plan(A)
        prop = simu._build_propagator_rk4(np.complex64)
        A_before = A.copy()
        simu.split_step_RK4(A, grid_bump(simu), prop, plans, DZ_TEST)
        assert np.isfinite(A).all(), "CNLSE RK4 with V produced NaN/Inf"
        assert not np.allclose(A, A_before), "Field unchanged after RK4 step with V"


class TestNLSEOutField:
    """Tests for out_field code paths."""

    def test_verbose_output(self, capsys):
        """Verbose mode prints timing info."""
        simu = make_nlse("CPU", n=S)
        simu.out_field(beam(simu), L, verbose=True, plot=False)
        captured = capsys.readouterr()
        assert "Time spent to solve" in captured.out

    def test_callback_single(self):
        """Single callback is invoked during propagation."""
        call_count = [0]

        def my_callback(solver, A, z, i):
            call_count[0] += 1

        simu = make_nlse("CPU", n=S)
        simu.out_field(beam(simu), L, verbose=False, plot=False, callback=my_callback)
        assert call_count[0] > 0, "Callback was never called"

    def test_callback_list(self):
        """List of callbacks is invoked during propagation."""
        counts = [0, 0]

        def cb1(solver, A, z, i, idx):
            counts[idx] += 1

        def cb2(solver, A, z, i, idx):
            counts[idx] += 1

        simu = make_nlse("CPU", n=S)
        simu.out_field(
            beam(simu),
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


class TestCNLSEExtended:
    """Extended CNLSE coverage tests on CPU."""

    def test_build_propagator(self):
        """CNLSE propagator has two components."""
        simu = make_cnlse("CPU", n=S)
        prop = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        assert prop.shape == (2, S, S)
        assert np.isfinite(prop).all()

    def test_propagator_caching(self):
        """CNLSE propagator is cached."""
        simu = make_cnlse("CPU", n=S)
        prop1 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        prop2 = as_numpy(simu, simu._build_propagator(PRECISION_COMPLEX, DZ_TEST))
        assert prop1 is prop2

    def test_propagator_rk4(self):
        """CNLSE RK4 propagator."""
        simu = make_cnlse("CPU", n=S)
        prop = simu._build_propagator_rk4(np.complex64)
        assert prop.shape == (2, S, S)
        # RK4 propagator should not be exponential (no delta_z)
        assert not np.allclose(np.abs(prop), 1.0)

    def test_take_components_cpu(self):
        """_take_components returns views on CPU."""
        simu = make_cnlse("CPU", n=S)
        A = np.ones((2, S, S), dtype=PRECISION_COMPLEX)
        A[0] *= 2.0
        A[1] *= 3.0
        A1, A2 = simu._take_components(A)
        assert A1.shape == (S, S)
        assert A2.shape == (S, S)
        np.testing.assert_allclose(A1, 2.0)
        np.testing.assert_allclose(A2, 3.0)

    def test_cnlse_prepare_output_normalize(self):
        """CNLSE _prepare_output_array normalizes both components."""
        simu = make_cnlse("CPU", n=S)
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

    def test_cnlse_power_conservation(self):
        """CNLSE with alpha=0 conserves total power."""
        simu = make_cnlse("CPU", n=S, alpha=0)
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

    def test_nlse_plot_field_3d(self):
        """NLSE plot_field with >2D array drops dims."""
        import matplotlib.pyplot as plt

        simu = make_nlse("CPU", n=S)
        A = np.ones((3, S, S), dtype=PRECISION_COMPLEX)
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
