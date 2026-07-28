"""Comprehensive benchmarks for NLSE kernels and solvers.

Benchmarks:
1. Individual kernel functions (CPU backend, N=64)
2. Single-component solver split_step methods (all backends)
3. Coupled solver split_step methods (all backends)
"""

import numpy as np
import pytest
from NLSE import CNLSE, DDGPE, GPE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from NLSE.backends import list_available_backends
from NLSE.kernels.cpu import (
    nl_prop,
    nl_prop_c,
    nl_prop_without_V,
    nl_prop_without_V_c,
    rabi_coupling,
    square_mod,
    vortex,
)
from scipy.constants import c

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

# Kernel benchmark parameters
dz = 1e-5
alpha_k = 10.0
g_k = 1e-3
Isat_k = 1e4
omega_k = 1e3

# Get available backends
BACKENDS = list_available_backends()


def skip_if_backend_unavailable(backend: str):
    """Skip benchmark if backend is not available."""
    if backend not in BACKENDS:
        pytest.skip(f"Backend {backend} not available")


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


@pytest.mark.benchmark(group="kernels")
class TestKernelBenchmark:
    """Benchmark individual kernel functions (CPU backend only)."""

    def test_square_mod(self, benchmark):
        """Benchmark square_mod kernel."""
        A = _random_field_2d()
        A_sq = _random_real_2d()

        def kernel():
            square_mod(A.copy(), A_sq.copy())

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A_sq))

    def test_nl_prop(self, benchmark):
        """Benchmark nl_prop kernel (with potential)."""
        A = _random_field_2d()
        A_sq = _random_real_2d()
        V = _random_real_2d(seed=43)

        def kernel():
            nl_prop(
                A.copy(),
                A_sq.copy(),
                dz,
                alpha_k,
                V.copy(),
                g_k,
                Isat_k,
            )

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A))

    def test_nl_prop_without_V(self, benchmark):
        """Benchmark nl_prop_without_V kernel (no potential)."""
        A = _random_field_2d()
        A_sq = _random_real_2d()

        def kernel():
            nl_prop_without_V(
                A.copy(),
                A_sq.copy(),
                dz,
                alpha_k,
                g_k,
                Isat_k,
            )

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A))

    def test_nl_prop_c(self, benchmark):
        """Benchmark nl_prop_c kernel (coupled with potential)."""
        A1 = _random_field_2d(seed=42)
        A_sq_1 = _random_real_2d(seed=44)
        A_sq_2 = _random_real_2d(seed=45)
        # V needs to be complex for numba typing
        V = _random_real_2d(seed=46).astype(PRECISION_COMPLEX)

        def kernel():
            nl_prop_c(
                A1.copy(),
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz,
                alpha_k,
                V.copy(),
                g_k,
                g_k * 0.5,
                Isat_k,
                Isat_k,
            )

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))

    def test_nl_prop_without_V_c(self, benchmark):
        """Benchmark nl_prop_without_V_c kernel (coupled, no potential)."""
        A1 = _random_field_2d(seed=42)
        A_sq_1 = _random_real_2d(seed=44)
        A_sq_2 = _random_real_2d(seed=45)

        def kernel():
            nl_prop_without_V_c(
                A1.copy(),
                A_sq_1.copy(),
                A_sq_2.copy(),
                dz,
                alpha_k,
                g_k,
                g_k * 0.5,
                Isat_k,
                Isat_k,
            )

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))

    def test_rabi_coupling(self, benchmark):
        """Benchmark rabi_coupling kernel."""
        A1 = _random_field_2d(seed=42)
        A2 = _random_field_2d(seed=43)

        def kernel():
            rabi_coupling(A1.copy(), A2.copy(), dz, omega_k)

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(A1))
        assert np.all(np.isfinite(A2))

    def test_vortex(self, benchmark):
        """Benchmark vortex kernel."""
        im = np.zeros((N, N), dtype=PRECISION_COMPLEX)
        coords = np.arange(N, dtype=np.int32)
        ii, jj = np.meshgrid(coords, coords)
        i0 = N // 2
        j0 = N // 2
        ll = 1

        def kernel():
            vortex(im.copy(), i0, j0, ii, jj, ll)

        benchmark.pedantic(kernel, rounds=100, warmup_rounds=10)
        assert np.all(np.isfinite(im))


@pytest.mark.benchmark(group="solvers-single")
class TestSolverBenchmark:
    """Benchmark single-component solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d(self, benchmark, backend):
        """Benchmark NLSE 2D propagation."""
        skip_if_backend_unavailable(backend)
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
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_1d(self, benchmark, backend):
        """Benchmark NLSE_1d propagation."""
        skip_if_backend_unavailable(backend)
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
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N,)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_3d(self, benchmark, backend):
        """Benchmark NLSE_3d propagation."""
        skip_if_backend_unavailable(backend)
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
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=5, warmup_rounds=1)
        assert result.shape == (N, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gpe(self, benchmark, backend):
        """Benchmark GPE propagation."""
        skip_if_backend_unavailable(backend)
        from scipy.constants import atomic_mass

        simu = GPE(
            gamma=0.1,
            N=1e5,
            window=window,
            g=n2,
            V=None,
            m=87 * atomic_mass,
            NX=N,
            NY=N,
            sat=Isat,
            backend=backend,
        )
        E_in = _gaussian_field_2d(waist, simu.X, simu.Y)

        def propagate():
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (N, N)
        assert np.all(np.isfinite(result))


@pytest.mark.benchmark(group="solvers-coupled")
class TestCoupledSolverBenchmark:
    """Benchmark coupled solver out_field methods."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_2d(self, benchmark, backend):
        """Benchmark CNLSE 2D propagation."""
        skip_if_backend_unavailable(backend)
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
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_cnlse_1d(self, benchmark, backend):
        """Benchmark CNLSE_1d propagation."""
        skip_if_backend_unavailable(backend)
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
            return simu.out_field(E_in, z=1e-3, verbose=False, precision="single")

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N)
        assert np.all(np.isfinite(result))

    @pytest.mark.skip(reason="DDGPE benchmark is slow and has callback complexity")
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_ddgpe(self, benchmark, backend):
        """Benchmark DDGPE propagation."""
        # DDGPE specific parameters
        gamma = 0.1
        omega = 1e3
        T = 1e-3
        omega_exc = 1.5e15
        omega_cav = 1.5e15
        detuning = 0.0
        k_z = 1e6

        simu = DDGPE(
            gamma=gamma,
            power=power,
            window=window,
            g=n2,
            g12=n12,
            omega=omega,
            T=T,
            omega_exc=omega_exc,
            omega_cav=omega_cav,
            detuning=detuning,
            k_z=k_z,
            V=None,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E_in = np.zeros((2, N, N), dtype=PRECISION_COMPLEX)
        E_in[0] = _gaussian_field_2d(waist, simu.X, simu.Y)
        E_in[1] = _gaussian_field_2d(waist * 1.2, simu.X, simu.Y) * 0.5

        # No-op laser excitation for benchmarking
        def no_laser(simu, A, z, i):
            pass

        def propagate():
            return simu.out_field(
                E_in,
                t=1e-3,
                laser_excitation=no_laser,
                verbose=False,
                callback=[],
                callback_args=[()],
            )

        result = benchmark.pedantic(propagate, rounds=10, warmup_rounds=2)
        assert result.shape == (2, N, N)
        assert np.all(np.isfinite(result))


# ── Coverage of the axes the benchmarks above leave out ──────────────────────
#
# Those cover one grid size, one method, one precision and no potential. The
# groups below sweep the axes that actually change the cost, so a regression
# in any of them shows up as a number rather than as a surprise in a user's
# run. They are deliberately one axis at a time: a full cross product is
# thousands of cases and nobody would run it.


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
    XX, YY = np.meshgrid(simu.X, simu.Y)
    env = np.exp(-(XX**2 + YY**2) / waist**2).astype(dtype)
    if count is None:
        return env
    return np.broadcast_to(env, (count, n, n)).copy()


def _timed(benchmark, fn, rounds=10):
    """Run a benchmark and assert the result is usable."""
    result = benchmark.pedantic(fn, rounds=rounds, warmup_rounds=2)
    assert np.all(np.isfinite(np.asarray(result))), "benchmark produced NaN"
    return result


@pytest.mark.benchmark(group="grid-scaling")
class TestGridScaling:
    """Cost against grid size.

    FFT cost is O(N^2 log N) in the number of points, so a backend that
    looks good at 64 can lose at 1024 and vice versa. One size hides that.
    """

    @pytest.mark.parametrize("n", [64, 128, 256, 512])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_nlse_2d_scaling(self, benchmark, backend, n):
        """Benchmark NLSE 2D across grid sizes."""
        skip_if_backend_unavailable(backend)
        simu = _solver(backend, n)
        E_in = _field(simu, n)
        rounds = 10 if n <= 256 else 5
        _timed(
            benchmark,
            lambda: simu.out_field(E_in, z=1e-3, verbose=False, plot=False),
            rounds=rounds,
        )


@pytest.mark.benchmark(group="methods")
class TestMethodAndOrder:
    """split_step against RK4, and the two split-step orders.

    precision="double" splits the nonlinear step around the linear one, so
    it costs a second nonlinear application per step; RK4 is four stages.
    Neither was benchmarked.
    """

    @pytest.mark.parametrize("precision", ["single", "double"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_split_step_order(self, benchmark, backend, precision):
        """Benchmark both split-step orders."""
        skip_if_backend_unavailable(backend)
        simu = _solver(backend, N)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: simu.out_field(
                E_in, z=1e-3, verbose=False, plot=False, precision=precision
            ),
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rk4(self, benchmark, backend):
        """Benchmark the RK4 integrator without a potential."""
        skip_if_backend_unavailable(backend)
        simu = _solver(backend, N)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: simu.out_field(
                E_in, z=1e-3, verbose=False, plot=False, method="RK4"
            ),
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_rk4_with_a_potential(self, benchmark, backend):
        """Benchmark RK4 under a potential.

        A potential dominates RK4's stability bound, so the step collapses
        and this is far slower than the same run without one. That gap is
        the point: it is what makes split_step the better choice whenever a
        potential is strong, and it should be visible rather than folklore.
        """
        skip_if_backend_unavailable(backend)
        probe = _solver(backend, N)
        XX, YY = np.meshgrid(probe.X, probe.Y)
        V = (1e-6 * np.exp(-(XX**2 + YY**2) / (1e-3) ** 2)).astype(PRECISION_REAL)
        simu = _solver(backend, N, V=V)
        E_in = _field(simu, N)
        _timed(
            benchmark,
            lambda: simu.out_field(
                E_in, z=1e-5, verbose=False, plot=False, method="RK4"
            ),
            rounds=5,
        )


@pytest.mark.benchmark(group="potential")
class TestPotential:
    """No potential against a real one against a complex one.

    A complex V takes a separate kernel that loads twice the data. The point
    of benchmarking all three is to confirm the real path is not paying for
    the complex one existing.
    """

    @pytest.mark.parametrize("kind", ["none", "real", "complex"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_potential_kind(self, benchmark, backend, kind):
        """Benchmark propagation under each kind of potential."""
        skip_if_backend_unavailable(backend)
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
        _timed(
            benchmark, lambda: simu.out_field(E_in, z=1e-3, verbose=False, plot=False)
        )


@pytest.mark.benchmark(group="batching")
class TestBatching:
    """Cost against batch size.

    Broadcasting exists to make a parameter sweep cheaper than N separate
    runs. That claim is worth measuring: CPU and CL loop over the batch per
    kernel launch, so their gain is only in the FFTs, while CUPY and MLX
    broadcast inside the kernels.
    """

    @pytest.mark.parametrize("count", [1, 2, 4, 8])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batched_n2_sweep(self, benchmark, backend, count):
        """Benchmark a batched parameter sweep."""
        skip_if_backend_unavailable(backend)
        n2_batch = np.linspace(n2, 2 * n2, count).reshape(count, 1, 1)
        simu = _solver(backend, N, n2=n2_batch)
        E_in = _field(simu, N, count=count)
        _timed(
            benchmark, lambda: simu.out_field(E_in, z=1e-3, verbose=False, plot=False)
        )


@pytest.mark.benchmark(group="float-width")
class TestFloatWidth:
    """complex64 against complex128 fields.

    The field's dtype now drives the propagator's and the potential's, so
    double precision is a real end-to-end fp64 run rather than a mixed one.
    Skipped where the device has no fp64.
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_double_precision_field(self, benchmark, backend):
        """Benchmark a genuine fp64 propagation."""
        skip_if_backend_unavailable(backend)
        simu = _solver(backend, N)
        if not simu._backend.supports_double_precision():
            pytest.skip(f"{backend} has no fp64")
        E_in = _field(simu, N, dtype=np.complex128)
        _timed(
            benchmark, lambda: simu.out_field(E_in, z=1e-3, verbose=False, plot=False)
        )


@pytest.mark.benchmark(group="kernels-backend")
class TestKernelsAcrossBackends:
    """The individual kernels, on every backend rather than only CPU.

    The kernel group above imports the CPU implementations directly, so a
    change to a GPU kernel is invisible to it. These go through the backend's
    own kernel object.
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_square_mod_nl_prop(self, benchmark, backend):
        """Benchmark the fused intensity + nonlinear step."""
        skip_if_backend_unavailable(backend)
        from NLSE.backends import get_backend

        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())

        def run():
            be.kernels.square_mod_nl_prop(A, dz, alpha_k, g_k, Isat_k)
            return be.to_numpy(A)

        _timed(benchmark, run, rounds=20)

    @pytest.mark.parametrize("kind", ["real", "complex"])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_square_mod_nl_prop_v(self, benchmark, backend, kind):
        """Benchmark the potential kernel, real against complex V."""
        skip_if_backend_unavailable(backend)
        from NLSE.backends import get_backend

        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())
        V_host = _random_real_2d()
        V = be.from_numpy(
            V_host if kind == "real" else V_host.astype(PRECISION_COMPLEX)
        )

        def run():
            be.kernels.square_mod_nl_prop_v(A, V, dz, alpha_k, g_k, Isat_k)
            return be.to_numpy(A)

        _timed(benchmark, run, rounds=20)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_apply_propagator(self, benchmark, backend):
        """Benchmark the propagator multiply, the bandwidth-bound case."""
        skip_if_backend_unavailable(backend)
        from NLSE.backends import get_backend

        be = get_backend(backend)
        A = be.from_numpy(_random_field_2d())
        prop = be.from_numpy(_random_field_2d(seed=7))

        def run():
            be.kernels.apply_propagator(A, prop)
            return be.to_numpy(A)

        _timed(benchmark, run, rounds=20)
