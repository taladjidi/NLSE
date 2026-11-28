"""Comprehensive benchmarks for NLSE kernels and solvers."""

import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend, list_available_backends
from NLSE.kernels.cpu import (
    nl_prop,
    nl_prop_c,
    nl_prop_without_V,
    nl_prop_without_V_c,
    rabi_coupling,
    square_mod,
)
from scipy.constants import atomic_mass, c

# Test parameters
N = 64
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Physical parameters for solvers
alpha = 10.0
power = 1.0
window = 5e-3
n2 = 1e-20
n12 = 1e-21
L = 1e-2
Isat = 1e10
waist = 1e-3
# GPE condensate: atoms, and an interaction in rad/s m^2 giving g n = 1e3
# rad/s at their mean density over the window, as examples/gpe.py does.
GPE_ATOMS = 1e5
GPE_G = 1e3 / (GPE_ATOMS / window**2)
# rad: the interaction turns the centre by 2.16 rad over the run against the
# linear one on every backend (measured); n2 as g, by under 1e-10.
GPE_MIN_PHASE = 1.0

# Kernel benchmark parameters
dz = 1e-5
alpha_k = 10.0
g_k = 1e-3
Isat_k = 1e4
omega_k = 1e3

# Steps per backend, so that propagation is >= 80% of each run (setup is
# ~2-3 ms, a step 0.02-0.4 ms). Changing them breaks comparison with older numbers.
BENCH_STEPS = {"CPU": 20, "CUPY": 200, "CL": 60, "MLX": 400}
DEFAULT_BENCH_STEPS = 100


def bench_steps(simu) -> int:
    """Return the step count this solver's backend is calibrated for."""
    return BENCH_STEPS.get(simu._backend.name, DEFAULT_BENCH_STEPS)


# Get available backends
BACKENDS = list_available_backends()


def _propagate(simu, E_in, z, **kwargs):
    """Propagate over z in exactly ``bench_steps(simu)`` steps.

    Parameters
    ----------
    simu : NLSE
        Solver to run.
    E_in : np.ndarray
        Input field.
    z : float
        Distance (or time) to propagate.
    **kwargs
        Passed through to out_field.

    Returns
    -------
    np.ndarray
        The propagated field.
    """
    dz = z / bench_steps(simu)
    out = simu.out_field(E_in, z, verbose=False, plot=False, **kwargs, delta_z=dz)
    assert simu._current_delta_z == dz, (
        f"the step limiter reduced delta_z from {dz:.3e} to "
        f"{simu._current_delta_z:.3e}, so this case runs more steps than the "
        f"rest and its timing is not comparable. Pick parameters inside the limit."
    )
    return out


def _random_field_2d(seed: int = 42) -> np.ndarray:
    """Create random 2D complex field."""
    rng = np.random.default_rng(seed)
    return (rng.random((N, N)) + 1j * rng.random((N, N))).astype(PRECISION_COMPLEX)


def _random_real_2d(seed: int = 42) -> np.ndarray:
    """Create random 2D real field."""
    rng = np.random.default_rng(seed)
    return rng.random((N, N)).astype(PRECISION_REAL)


def _gaussian_field_2d(waist: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create Gaussian field."""
    XX, YY = np.meshgrid(x, y)
    return np.exp(-(XX**2 + YY**2) / waist**2).astype(PRECISION_COMPLEX)


def assert_written(out, before, name):
    """Fail unless a kernel's output is finite and no longer its input.

    Parameters
    ----------
    out : np.ndarray
        The array the kernel wrote into.
    before : np.ndarray
        The same array before the kernel ran.
    name : str
        Kernel name, for the message.
    """
    assert np.all(np.isfinite(out)), f"{name} wrote non-finite values"
    assert not np.array_equal(out, before), f"{name} left its output as it was"


def _kernel_cases():
    """Return, per CPU kernel: the kernel, its arguments, the positions it writes."""
    A1, A2 = _random_field_2d(seed=42), _random_field_2d(seed=43)
    I1, I2 = _random_real_2d(seed=44), _random_real_2d(seed=45)
    V = _random_real_2d(seed=46)
    V_c = V.astype(PRECISION_COMPLEX)  # the coupled kernel is typed for a complex V
    g12, sat = g_k * 0.5, Isat_k
    return {
        "square_mod": (square_mod, (A1, I1), (1,)),
        "nl_prop": (nl_prop, (A1, I1, dz, alpha_k, V, g_k, sat), (0,)),
        "nl_prop_without_V": (
            nl_prop_without_V,
            (A1, I1, dz, alpha_k, g_k, sat),
            (0,),
        ),
        "nl_prop_c": (
            nl_prop_c,
            (A1, I1, I2, dz, alpha_k, V_c, g_k, g12, sat, sat),
            (0,),
        ),
        "nl_prop_without_V_c": (
            nl_prop_without_V_c,
            (A1, I1, I2, dz, alpha_k, g_k, g12, sat, sat),
            (0,),
        ),
        "rabi_coupling": (rabi_coupling, (A1, A2, dz, omega_k), (0, 1)),
    }


KERNEL_CASES = _kernel_cases()


def _fresh(args):
    """Copy the arrays among a kernel's arguments, in the form pedantic's setup takes."""
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args), {}


@pytest.mark.benchmark(group="kernels")
class TestKernelBenchmark:
    """Benchmark individual kernel functions (CPU backend only)."""

    @pytest.mark.parametrize("name", KERNEL_CASES)
    def test_kernel(self, benchmark, name):
        """Benchmark one kernel call, the copies of its arguments left out."""
        kernel, args, written = KERNEL_CASES[name]
        benchmark.pedantic(
            kernel, setup=lambda: _fresh(args), rounds=100, warmup_rounds=10
        )
        out, _ = _fresh(args)
        kernel(*out)
        for position in written:
            assert_written(out[position], args[position], name)


@pytest.mark.benchmark(group="solvers-single")
class TestSolverBenchmark:
    """Benchmark single-component solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d(self, benchmark, backend):
        """Benchmark NLSE 2D propagation."""
        simu = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = _gaussian_field_2d(waist, simu.X, simu.Y)

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_1d(self, benchmark, backend):
        """Benchmark NLSE_1d propagation."""
        simu = NLSE_1d(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N,)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_3d(self, benchmark, backend):
        """Benchmark NLSE_3d propagation."""
        simu = NLSE_3d(
            alpha=alpha,
            energy=1e-6,
            window=[window, window, 1e-12],
            n2=n2,
            D0=1e-26,
            vg=c,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=N,
            Isat=Isat,
            wvl=780e-9,
            backend=backend,
        )
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.zeros((N, N, N), dtype=PRECISION_COMPLEX)
        for i in range(N):
            E_in[:, :, i] = np.exp(-(XX**2 + YY**2) / waist**2)

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=5, warmup_rounds=1)
        assert result.shape == (N, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gpe(self, benchmark, backend):
        """Benchmark GPE propagation."""
        params = {
            "gamma": 0.1,
            "N": GPE_ATOMS,
            "window": window,
            "V": None,
            "m": 87 * atomic_mass,
            "NX": N,
            "NY": N,
            "sat": Isat,
            "backend": backend,
        }
        simu = GPE(g=GPE_G, **params)
        E_in = _gaussian_field_2d(waist, simu.X, simu.Y)

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))
        linear = GPE(g=0.0, **params)
        free = _propagate(linear, E_in, 1e-3, splitting="lie")
        centre = (N // 2, N // 2)
        ratio = (
            simu._backend.to_numpy(result)[centre]
            / linear._backend.to_numpy(free)[centre]
        )
        phase = np.angle(ratio)
        assert abs(phase) > GPE_MIN_PHASE, (
            f"{backend}: the interaction turned the centre by {phase:.2e} rad, "
            "so this benchmark times a linear run"
        )


@pytest.mark.benchmark(group="solvers-coupled")
class TestCoupledSolverBenchmark:
    """Benchmark coupled solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_2d(self, benchmark, backend):
        """Benchmark CNLSE 2D propagation."""
        simu = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N, N), dtype=PRECISION_COMPLEX)
        E_in[0] = _gaussian_field_2d(waist, simu.X, simu.Y)
        E_in[1] = _gaussian_field_2d(waist * 1.2, simu.X, simu.Y) * 0.5

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_1d(self, benchmark, backend):
        """Benchmark CNLSE_1d propagation."""
        simu = CNLSE_1d(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N), dtype=PRECISION_COMPLEX)
        E_in[0] = np.exp(-(simu.X**2) / waist**2)
        E_in[1] = np.exp(-(simu.X**2) / (waist * 1.2) ** 2) * 0.5

        def propagate():
            return _propagate(simu, E_in, 1e-3, splitting="lie")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N)
        assert np.all(np.isfinite(result))


# ── Coverage of the axes the benchmarks above leave out ──────────────────────
# One axis at a time (grid, method, precision, potential): a full cross
# product would be thousands of cases.


def _solver(backend, n, V=None, **overrides):
    """Build an NLSE at a given grid size, with an optional potential."""
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": V,
        "L": L,
        "NX": n,
        "NY": n,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(overrides)
    return NLSE(**params)


def _field(simu, n, count=None, dtype=PRECISION_COMPLEX):
    """Return a Gaussian input, optionally batched over `count` simulations."""
    env = _gaussian_field_2d(waist, simu.X, simu.Y).astype(dtype)
    if count is None:
        return env
    return np.broadcast_to(env, (count, n, n)).copy()


def _timed(benchmark, fn, rounds=10):
    """Run a benchmark and assert the result is usable."""
    result = benchmark.pedantic(fn, rounds=rounds, warmup_rounds=2)
    assert np.all(np.isfinite(np.asarray(result))), "benchmark produced NaN"
    return result


def _run_kernel(be, kernel, A, *args):
    """Run a kernel on A and wait for it; return the array holding the result.

    MLX's kernels return a new lazy array, the others write A and return
    nothing: without the result in hand MLX would time building a graph.
    """
    out = kernel(A, *args)
    result = A if out is None else out
    be.synchronize(result)
    return result


def _timed_kernel(benchmark, be, A, kernel, *args):
    """Benchmark one kernel call, the copy back to the host left out."""
    result = benchmark.pedantic(
        _run_kernel, args=(be, kernel, A, *args), rounds=20, warmup_rounds=2
    )
    assert np.all(np.isfinite(be.to_numpy(result))), "benchmark produced NaN"


@pytest.mark.benchmark(group="grid-scaling")
class TestGridScaling:
    """Cost against grid size."""

    @pytest.mark.parametrize("n", [64, 128, 256, 512])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d_scaling(self, benchmark, backend, n):
        """Benchmark NLSE 2D across grid sizes."""
        simu = _solver(backend, n)
        E_in = _field(simu, n)
        rounds = 10 if n <= 256 else 5
        _timed(
            benchmark,
            lambda: _propagate(simu, E_in, 1e-3),
            rounds=rounds,
        )


@pytest.mark.benchmark(group="methods")
class TestMethodAndOrder:
    """split_step against RK4, and the two split-step orders."""

    @pytest.mark.parametrize("splitting", ["lie", "strang"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_split_step_order(self, benchmark, backend, splitting):
        """Benchmark both split-step orders."""
        simu = _solver(backend, N)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: _propagate(simu, E_in, 1e-3, splitting=splitting),
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rk4(self, benchmark, backend):
        """Benchmark the RK4 integrator without a potential."""
        simu = _solver(backend, N)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: _propagate(simu, E_in, 1e-3, method="RK4"),
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rk4_with_a_potential(self, benchmark, backend):
        """Benchmark RK4 under a potential."""
        probe = _solver(backend, N)
        XX, YY = np.meshgrid(probe.X, probe.Y)
        V = (1e-6 * np.exp(-(XX**2 + YY**2) / (1e-3) ** 2)).astype(PRECISION_REAL)
        simu = _solver(backend, N, V=V)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: _propagate(simu, E_in, 1e-5, method="RK4"),
            rounds=5,
        )


@pytest.mark.benchmark(group="potential")
class TestPotential:
    """No potential against a real one against a complex one."""

    @pytest.mark.parametrize("kind", ["none", "real", "complex"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_potential_kind(self, benchmark, backend, kind):
        """Benchmark propagation under each kind of potential."""
        probe = _solver(backend, N)
        XX, YY = np.meshgrid(probe.X, probe.Y)
        ring = np.exp(-((np.sqrt(XX**2 + YY**2) - 1e-3) ** 2) / (2e-4) ** 2)
        V = {
            "none": None,
            "real": (1e-4 * ring).astype(PRECISION_REAL),
            "complex": (1e-4 * ring + 1j * 1e-5 * ring).astype(PRECISION_COMPLEX),
        }[kind]
        simu = _solver(backend, N, V=V)
        E_in = _field(simu, N)
        _timed(benchmark, lambda: _propagate(simu, E_in, 1e-3))


@pytest.mark.benchmark(group="batching")
class TestBatching:
    """Cost against batch size."""

    @pytest.mark.parametrize("count", [1, 2, 4, 8])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batched_n2_sweep(self, benchmark, backend, count):
        """Benchmark a batched parameter sweep."""
        n2_batch = np.linspace(n2, 2 * n2, count).reshape(count, 1, 1)
        simu = _solver(backend, N, n2=n2_batch)
        E_in = _field(simu, N, count=count)
        _timed(benchmark, lambda: _propagate(simu, E_in, 1e-3))


@pytest.mark.benchmark(group="float-width")
class TestFloatWidth:
    """complex64 against complex128 fields."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_double_precision_field(self, benchmark, backend):
        """Benchmark a genuine fp64 propagation."""
        simu = _solver(backend, N)
        if not simu._backend.supports_double_precision():
            pytest.skip(f"{backend} has no fp64")
        E_in = _field(simu, N, dtype=np.complex128)
        _timed(benchmark, lambda: _propagate(simu, E_in, 1e-3))


@pytest.mark.benchmark(group="kernels-backend")
class TestKernelsAcrossBackends:
    """The individual kernels, on every backend rather than only CPU."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_square_mod_nl_prop(self, benchmark, backend):
        """Benchmark the fused intensity + nonlinear step."""
        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())

        _timed_kernel(
            benchmark, be, A, be.kernels.square_mod_nl_prop, dz, alpha_k, g_k, Isat_k
        )

    @pytest.mark.parametrize("kind", ["real", "complex"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_square_mod_nl_prop_v(self, benchmark, backend, kind):
        """Benchmark the potential kernel, real against complex V."""
        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())
        V_host = _random_real_2d()
        V = be.from_numpy(
            V_host if kind == "real" else V_host.astype(PRECISION_COMPLEX)
        )

        _timed_kernel(
            benchmark,
            be,
            A,
            be.kernels.square_mod_nl_prop_v,
            V,
            dz,
            alpha_k,
            g_k,
            Isat_k,
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_apply_propagator(self, benchmark, backend):
        """Benchmark the propagator multiply, the bandwidth-bound case."""
        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())
        prop = be.from_numpy(_random_field_2d(seed=7))

        _timed_kernel(benchmark, be, A, be.kernels.apply_propagator, prop)
