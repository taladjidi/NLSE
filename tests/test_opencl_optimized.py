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
        """Setup OpenCL backend and test arrays."""
        from NLSE.backends.opencl import OpenCLBackend
        from pyopencl import array as cla

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
        """Test nl_prop optimized kernel vs original."""
        from pyopencl import array as cla
        from NLSE.kernels import cl as cl_kernels
        from NLSE.kernels.cl_optimized import OptimizedKernels

        # Run original kernel
        A_orig = cla.to_device(self.queue, self.A_host.copy())
        A_sq = cla.to_device(self.queue, self.A_sq_host)
        V = cla.to_device(self.queue, self.V_host)

        cl_kernels.nl_prop(A_orig, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        result_orig = A_orig.get()

        # Run optimized kernel
        A_opt = cla.to_device(self.queue, self.A_host.copy())
        opt_kernels = OptimizedKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop(A_opt, A_sq, self.dz, self.alpha, V, self.g, self.Isat)
        result_opt = A_opt.get()

        # Compare results
        np.testing.assert_allclose(
            result_orig, result_opt, rtol=1e-6, atol=1e-8,
            err_msg="Optimized nl_prop produces different results"
        )

    def test_nl_prop_without_v_correctness(self):
        """Test nl_prop_without_V optimized kernel vs original."""
        from pyopencl import array as cla
        from NLSE.kernels import cl as cl_kernels
        from NLSE.kernels.cl_optimized import OptimizedKernels

        # Run original kernel
        A_orig = cla.to_device(self.queue, self.A_host.copy())
        A_sq = cla.to_device(self.queue, self.A_sq_host)

        cl_kernels.nl_prop_without_V(
            A_orig, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        result_orig = A_orig.get()

        # Run optimized kernel
        A_opt = cla.to_device(self.queue, self.A_host.copy())
        opt_kernels = OptimizedKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop_without_V(
            A_opt, A_sq, self.dz, self.alpha, self.g, self.Isat
        )
        result_opt = A_opt.get()

        # Compare results
        np.testing.assert_allclose(
            result_orig, result_opt, rtol=1e-6, atol=1e-8,
            err_msg="Optimized nl_prop_without_V produces different results"
        )

    def test_square_mod_correctness(self):
        """Test square_mod optimized kernel vs original."""
        from pyopencl import array as cla
        from NLSE.kernels import cl as cl_kernels
        from NLSE.kernels.cl_optimized import OptimizedKernels

        # Run original kernel
        A = cla.to_device(self.queue, self.A_host)
        A_sq_orig = cla.zeros(self.queue, (self.N, self.N), np.float32)

        cl_kernels.square_mod(A, A_sq_orig)
        result_orig = A_sq_orig.get()

        # Run optimized kernel
        A_sq_opt = cla.zeros(self.queue, (self.N, self.N), np.float32)
        opt_kernels = OptimizedKernels(self.backend.context, self.queue)

        opt_kernels.square_mod(A, A_sq_opt)
        result_opt = A_sq_opt.get()

        # Compare results
        np.testing.assert_allclose(
            result_orig, result_opt, rtol=1e-6, atol=1e-8,
            err_msg="Optimized square_mod produces different results"
        )

    def test_nl_prop_c_correctness(self):
        """Test nl_prop_c optimized coupled kernel vs original."""
        from pyopencl import array as cla
        from NLSE.kernels import cl as cl_kernels
        from NLSE.kernels.cl_optimized import OptimizedKernels

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

        # Run original kernel
        A1_orig = cla.to_device(self.queue, self.A_host.copy())
        A_sq_1 = cla.to_device(self.queue, self.A_sq_host)
        A_sq_2 = cla.to_device(self.queue, A_sq_2_host)
        V = cla.to_device(self.queue, self.V_host)

        cl_kernels.nl_prop_c(
            A1_orig, A_sq_1, A_sq_2, self.dz, self.alpha, V, g11, g12, Isat1, Isat2
        )
        result_orig = A1_orig.get()

        # Run optimized kernel
        A1_opt = cla.to_device(self.queue, self.A_host.copy())
        opt_kernels = OptimizedKernels(self.backend.context, self.queue)

        opt_kernels.nl_prop_c(
            A1_opt, A_sq_1, A_sq_2, self.dz, self.alpha, V, g11, g12, Isat1, Isat2
        )
        result_opt = A1_opt.get()

        # Compare results
        np.testing.assert_allclose(
            result_orig, result_opt, rtol=1e-6, atol=1e-8,
            err_msg="Optimized nl_prop_c produces different results"
        )


class TestVortexBugFix:
    """Test that vortex_cp bug fix is correct."""

    def test_vortex_cp_uses_atan2(self):
        """Verify vortex_cp now uses atan2 instead of atan."""
        from NLSE.backends.opencl import OpenCLBackend
        from pyopencl import array as cla
        from NLSE.kernels import cl as cl_kernels
        import numpy as np

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

        cl_kernels.vortex_cp(im, 0, 0, ii_gpu, jj_gpu, 1)
        phase = im.get()

        # Verify phase winding
        # For charge 1, phase should wind from -π to π
        assert phase.min() > -np.pi - 0.1, "Phase minimum incorrect"
        assert phase.max() < np.pi + 0.1, "Phase maximum incorrect"

        # Phase at vortex core should be near 0 (or any value, it's undefined)
        # Check phase makes a full 2π winding
        # Phase difference along any line through center should show winding
        phase_diff_vert = np.abs(np.diff(phase[:, N // 2]))
        assert np.any(phase_diff_vert > 3.0), "No phase discontinuity found (not a vortex)"


@pytest.mark.benchmark
class TestOptimizedKernelPerformance:
    """Benchmark optimized kernels vs original."""

    def test_nl_prop_speedup(self, benchmark):
        """Measure nl_prop speedup with optimized kernel."""
        from NLSE.backends.opencl import OpenCLBackend
        from pyopencl import array as cla
        from NLSE.kernels.cl_optimized import OptimizedKernels
        import numpy as np

        backend = OpenCLBackend()
        N = 1024

        # Setup test data
        rng = np.random.RandomState(42)
        A_host = (rng.randn(N, N) + 1j * rng.randn(N, N)).astype(np.complex64)
        A = cla.to_device(backend.queue, A_host)
        A_sq = cla.to_device(backend.queue, (np.abs(A_host) ** 2).astype(np.float32))
        V = cla.to_device(backend.queue, rng.randn(N, N).astype(np.float32))

        opt_kernels = OptimizedKernels(backend.context, backend.queue)

        def run_optimized():
            opt_kernels.nl_prop(A, A_sq, 1e-4, 20.0, V, 1e-3, 1e4)
            backend.queue.finish()

        benchmark(run_optimized)
