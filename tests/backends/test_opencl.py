"""Test optimized OpenCL kernels for correctness and performance."""

import numpy as np
import pytest
from NLSE.backends import list_available_backends

pytestmark = pytest.mark.skipif(
    "CL" not in list_available_backends(), reason="OpenCL not available"
)


class TestOptimizedKernelCorrectness:
    """Verify optimized kernels produce identical results to original."""

    def setup_method(self):
        """Set up OpenCL backend and test arrays."""
        from NLSE.backends.opencl import OpenCLBackend

        self.backend = OpenCLBackend()
        self.queue = self.backend.queue
        self.N = 256

        # Create test data
        rng = np.random.RandomState(42)
        self.A_host = (
            rng.randn(self.N, self.N) + 1j * rng.randn(self.N, self.N)
        ).astype(np.complex64)
        self.A_sq_host = (np.abs(self.A_host) ** 2).astype(np.float32)
        self.V_host = rng.randn(self.N, self.N).astype(np.float32)

        # Test parameters
        self.dz = 1e-4
        self.alpha = 20.0
        self.g = 1e-3
        self.Isat = 1e4

    def test_nl_prop_correctness(self):
        """Test nl_prop optimized kernel vs CPU reference."""
        from NLSE.kernels import cpu as cpu_kernels
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        # Run CPU reference
        A_cpu = self.A_host.copy()
        cpu_kernels.nl_prop(
            A_cpu, self.A_sq_host, self.dz, self.alpha, self.V_host, self.g, self.Isat
        )

        # Run optimized OpenCL kernel
        A_opt = cla.to_device(self.queue, self.A_host.copy())
        A_sq = cla.to_device(self.queue, self.A_sq_host)
        V = cla.to_device(self.queue, self.V_host)
        opt_kernels = OpenCLKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop(A_opt, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        result_opt = A_opt.get()

        # Compare results
        np.testing.assert_allclose(
            A_cpu,
            result_opt,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Optimized nl_prop differs from CPU reference",
        )

    def test_nl_prop_without_v_correctness(self):
        """Test nl_prop_without_V optimized kernel vs CPU reference."""
        from NLSE.kernels import cpu as cpu_kernels
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        # Run CPU reference
        A_cpu = self.A_host.copy()
        cpu_kernels.nl_prop_without_V(
            A_cpu, self.A_sq_host, self.dz, self.alpha, self.g, self.Isat
        )

        # Run optimized OpenCL kernel
        A_opt = cla.to_device(self.queue, self.A_host.copy())
        A_sq = cla.to_device(self.queue, self.A_sq_host)
        opt_kernels = OpenCLKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop_without_V(
            A_opt, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        result_opt = A_opt.get()

        # Compare results
        np.testing.assert_allclose(
            A_cpu,
            result_opt,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Optimized nl_prop_without_V differs from CPU reference",
        )

    def test_square_mod_correctness(self):
        """Test square_mod optimized kernel vs CPU reference."""
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        # Compute CPU reference
        A_sq_cpu = np.abs(self.A_host) ** 2

        # Run optimized OpenCL kernel
        A = cla.to_device(self.queue, self.A_host)
        A_sq_opt = cla.zeros(self.queue, (self.N, self.N), np.float32)
        opt_kernels = OpenCLKernels(self.backend.context, self.queue)

        opt_kernels.square_mod(A, A_sq_opt)
        result_opt = A_sq_opt.get()

        # Compare results
        np.testing.assert_allclose(
            A_sq_cpu,
            result_opt,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Optimized square_mod differs from CPU reference",
        )

    def test_nl_prop_c_correctness(self):
        """Test nl_prop_c optimized coupled kernel vs CPU reference."""
        from NLSE.kernels import cpu as cpu_kernels
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        # Create second component
        rng = np.random.RandomState(43)
        A2_host = (rng.randn(self.N, self.N) + 1j * rng.randn(self.N, self.N)).astype(
            np.complex64
        )
        A_sq_2_host = (np.abs(A2_host) ** 2).astype(np.float32)

        g11 = 1e-3
        g12 = 5e-4
        Isat1 = 1e4
        Isat2 = 2e4

        # Run CPU reference
        A1_cpu = self.A_host.copy()
        cpu_kernels.nl_prop_c(
            A1_cpu,
            self.A_sq_host,
            A_sq_2_host,
            self.dz,
            self.alpha,
            self.V_host,
            g11,
            g12,
            Isat1,
            Isat2,
        )

        # Run optimized OpenCL kernel
        A1_opt = cla.to_device(self.queue, self.A_host.copy())
        A_sq_1 = cla.to_device(self.queue, self.A_sq_host)
        A_sq_2 = cla.to_device(self.queue, A_sq_2_host)
        V = cla.to_device(self.queue, self.V_host)
        opt_kernels = OpenCLKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop_c(
            A1_opt, A_sq_1, A_sq_2, self.dz, self.alpha, V, g11, g12, Isat1, Isat2
        )
        result_opt = A1_opt.get()

        # Compare results
        np.testing.assert_allclose(
            A1_cpu,
            result_opt,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Optimized nl_prop_c differs from CPU reference",
        )

    def test_rabi_coupling_correctness(self):
        """Test native CL rabi_coupling kernel vs CPU reference."""
        from NLSE.kernels import cpu as cpu_kernels
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        # Create second component
        rng = np.random.RandomState(43)
        A2_host = (rng.randn(self.N, self.N) + 1j * rng.randn(self.N, self.N)).astype(
            np.complex64
        )

        dz = 1e-3
        omega = 5.0

        # Run CPU reference
        A1_cpu = self.A_host.copy()
        A2_cpu = A2_host.copy()
        cpu_kernels.rabi_coupling(A1_cpu, A2_cpu, dz, omega)

        # Run native OpenCL kernel
        A1_cl = cla.to_device(self.queue, self.A_host.copy())
        A2_cl = cla.to_device(self.queue, A2_host.copy())
        opt_kernels = OpenCLKernels(self.backend.context, self.queue)

        opt_kernels.rabi_coupling(A1_cl, A2_cl, dz, omega)

        # Compare results
        np.testing.assert_allclose(
            A1_cpu,
            A1_cl.get(),
            rtol=1e-5,
            atol=1e-7,
            err_msg="Native CL rabi_coupling A1 differs from CPU reference",
        )
        np.testing.assert_allclose(
            A2_cpu,
            A2_cl.get(),
            rtol=1e-5,
            atol=1e-7,
            err_msg="Native CL rabi_coupling A2 differs from CPU reference",
        )


class TestVortexBugFix:
    """Test that vortex_cp bug fix is correct."""

    def test_vortex_cp_uses_atan2(self):
        """Verify vortex_cp now uses atan2 instead of atan."""
        import numpy as np
        from NLSE.backends.opencl import OpenCLBackend
        from pyopencl import array as cla

        backend = OpenCLBackend()
        N = 128

        # Create coordinate grids
        x = np.arange(N) - N // 2
        ii, jj = np.meshgrid(x, x, indexing="ij")
        ii = ii.astype(np.float32)
        jj = jj.astype(np.float32)

        # Create vortex with charge 1 at coordinate origin (center of grid)
        im = cla.zeros(backend.queue, (N, N), np.float32)
        ii_gpu = cla.to_device(backend.queue, ii)
        jj_gpu = cla.to_device(backend.queue, jj)

        backend.kernels.vortex_cp(im, 0, 0, ii_gpu, jj_gpu, 1)
        phase = im.get()

        # Verify phase winding
        # For charge 1, phase should wind from -π to π
        assert phase.min() > -np.pi - 0.1, "Phase minimum incorrect"
        assert phase.max() < np.pi + 0.1, "Phase maximum incorrect"

        # Phase at vortex core should be near 0 (or any value, it's undefined)
        # Check phase makes a full 2π winding
        # Phase difference along any line through center should show winding
        phase_diff_vert = np.abs(np.diff(phase[:, N // 2]))
        assert np.any(phase_diff_vert > 3.0), (
            "No phase discontinuity found (not a vortex)"
        )


@pytest.mark.benchmark
class TestOptimizedKernelPerformance:
    """Benchmark optimized kernels vs original."""

    def test_nl_prop_speedup(self, benchmark):
        """Measure nl_prop speedup with optimized kernel."""
        import numpy as np
        from NLSE.backends.opencl import OpenCLBackend
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        backend = OpenCLBackend()
        N = 1024

        # Setup test data
        rng = np.random.RandomState(42)
        A_host = (rng.randn(N, N) + 1j * rng.randn(N, N)).astype(np.complex64)
        A = cla.to_device(backend.queue, A_host)
        A_sq = cla.to_device(backend.queue, (np.abs(A_host) ** 2).astype(np.float32))
        V = cla.to_device(backend.queue, rng.randn(N, N).astype(np.float32))

        opt_kernels = OpenCLKernels(backend.context, backend.queue)

        def run_optimized():
            opt_kernels.nl_prop(A, A_sq, 1e-4, 20.0, V, 1e-3, 1e4)
            backend.queue.finish()

        benchmark(run_optimized)


class TestSharedGridBroadcast:
    """A grid shared by a batch is addressed with the launch's global offset.

    The kernels index the field with a flat global id, so a batched run
    launches once per simulation and places each launch with global_offset.
    A shared potential or propagator is then read at
    ``idx - get_global_offset(0)``.

    The built-in is load-bearing: it is by definition the zero-based index
    within the launch, so the compiler keeps the wide loads it would use for
    a plain ``grid[idx]``. Passing the same number as an int argument defeats
    that and costs several times the runtime on the bandwidth-bound kernels,
    which shows up as a regression on unbatched propagations. These tests
    exist so that a future edit cannot quietly swap it back.
    """

    SHARED_GRID_KERNELS = [
        "apply_propagator",
        "nl_prop_fused",
        "square_mod_nl_prop_v_fused",
        "rk4_nl_rhs_v_fused",
        "square_mod_rk4_nl_rhs_v_fused",
    ]

    def test_shared_grid_reads_use_the_launch_offset(self):
        """Every kernel reading a shared grid must subtract the offset."""
        import re

        from NLSE.kernels.cl import _get_kernel_source

        source = _get_kernel_source("single")
        assert "{{" not in source, "unsubstituted placeholder left in the source"
        bodies = dict(re.findall(r"__kernel void (\w+)\((.*?)\n\}", source, re.DOTALL))
        for name in self.SHARED_GRID_KERNELS:
            assert "get_global_offset(0)" in bodies[name], (
                f"{name} indexes its shared grid without the launch offset, so "
                f"a batched run reads the wrong slice"
            )

    def test_no_kernel_takes_an_offset_argument(self):
        """The offset must come from the built-in, never from an argument.

        An int argument the compiler cannot see through costs the wide loads,
        which slows the unbatched path down substantially.
        """
        from NLSE.kernels.cl import _get_kernel_source

        source = _get_kernel_source("single")
        assert "grid_offset" not in source, (
            "a kernel takes the offset as an argument; use "
            "get_global_offset(0) so the index stays provably contiguous"
        )

    def test_a_shared_grid_broadcasts_across_the_batch(self):
        """The batched launches must reproduce a numpy broadcast exactly."""
        from NLSE.backends.opencl import OpenCLBackend
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        backend = OpenCLBackend()
        kernels = OpenCLKernels(backend.context, backend.queue)

        count, n = 3, 64
        rng = np.random.default_rng(0)
        field = (rng.random((count, n, n)) + 1j * rng.random((count, n, n))).astype(
            np.complex64
        )
        propagator = (rng.random((n, n)) + 1j * rng.random((n, n))).astype(np.complex64)

        A = cla.to_device(backend.queue, field.copy())
        kernels.apply_propagator(A, cla.to_device(backend.queue, propagator))

        np.testing.assert_allclose(
            A.get(),
            field * propagator,
            rtol=1e-6,
            err_msg="a shared propagator was not broadcast across the batch",
        )

    def test_an_unbatched_run_is_a_single_launch(self):
        """No batch axis means one launch at offset 0, as before broadcasting."""
        from NLSE.backends.opencl import OpenCLBackend
        from NLSE.kernels.cl import OpenCLKernels
        from pyopencl import array as cla

        backend = OpenCLBackend()
        kernels = OpenCLKernels(backend.context, backend.queue)
        n = 64
        A = cla.to_device(backend.queue, np.ones((n, n), dtype=np.complex64))
        grid = cla.to_device(backend.queue, np.ones((n, n), dtype=np.complex64))

        launches = list(kernels._launches(A, grid=grid))
        assert len(launches) == 1, (
            f"an unbatched field produced {len(launches)} launches; it must "
            f"stay a single whole-field launch"
        )
        offset, global_size, _, _ = launches[0]
        assert offset == 0
        assert global_size == (n * n,)
