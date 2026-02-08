#!/usr/bin/env python
"""Detailed profiling of Metal backend focusing on FFT and kernel operations."""

import time
import numpy as np
from line_profiler import LineProfiler

from NLSE import NLSE
from NLSE.kernels.metal import MetalArray, MetalFFTPlan
from NLSE.solvers.nlse import NLSE as NLSESolver

# Small workload for focused profiling
N = 64
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3

def profile_fft_operations():
    """Profile FFT operations in detail."""
    print("\n" + "="*80)
    print("FFT Operations Profiling")
    print("="*80 + "\n")

    # Create solver
    solver = NLSE(0, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="Metal")
    field = np.ones((N, N), dtype=np.complex64)

    # Build propagator and prepare arrays
    solver.propagator = solver._build_propagator(precision="single")
    A, A_sq = solver._prepare_output_array(field, normalize=True)
    solver._send_arrays_to_gpu()
    plans = solver._build_fft_plan(A)

    # Profile FFT operations
    print("Testing FFT plan...")

    # Time a single FFT
    A_test = MetalArray.from_numpy(np.ones((N, N), dtype=np.complex64))

    start = time.perf_counter()
    for _ in range(10):
        plans.fft(A_test, A_test)
    fft_time = (time.perf_counter() - start) / 10

    print(f"Single FFT time: {fft_time*1000:.3f} ms")

    # Compare to numpy FFT
    A_np = np.ones((N, N), dtype=np.complex64)
    start = time.perf_counter()
    for _ in range(10):
        _ = np.fft.fft2(A_np)
    numpy_fft_time = (time.perf_counter() - start) / 10

    print(f"NumPy FFT time: {numpy_fft_time*1000:.3f} ms")
    print(f"Ratio (Metal/NumPy): {fft_time/numpy_fft_time:.2f}x")

    # Check if data is being copied
    print("\nChecking for CPU-GPU transfers in FFT...")
    print("MetalFFTPlan.fft signature inspection:")
    import inspect
    print(inspect.getsource(plans.fft))


def profile_kernels():
    """Profile individual kernel operations."""
    print("\n" + "="*80)
    print("Kernel Operations Profiling")
    print("="*80 + "\n")

    solver = NLSE(0, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="Metal")
    field = np.ones((N, N), dtype=np.complex64)

    solver.propagator = solver._build_propagator(precision="single")
    A, A_sq = solver._prepare_output_array(field, normalize=True)
    solver._send_arrays_to_gpu()

    # Profile square_mod
    A_metal = MetalArray.from_numpy(A)
    A_sq_metal = MetalArray.zeros((N, N), np.float32)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        solver._kernels.square_mod(A_metal, A_sq_metal)
        times.append(time.perf_counter() - start)

    print(f"square_mod: {np.mean(times)*1000:.3f} ± {np.std(times)*1000:.3f} ms (n=100)")

    # Profile nl_prop_without_V
    times = []
    for _ in range(100):
        start = time.perf_counter()
        solver._kernels.nl_prop_without_V(
            A_metal, A_sq_metal,
            solver.delta_z, solver.alpha / 2,
            solver.k / 2 * solver.n2 * 3e8 * 8.854e-12,
            2 * solver.I_sat / (8.854e-12 * 3e8),
        )
        times.append(time.perf_counter() - start)

    print(f"nl_prop_without_V: {np.mean(times)*1000:.3f} ± {np.std(times)*1000:.3f} ms (n=100)")


def profile_linear_step():
    """Profile the linear step in detail."""
    print("\n" + "="*80)
    print("Linear Step Profiling (includes FFT)")
    print("="*80 + "\n")

    solver = NLSE(0, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="Metal")
    field = np.ones((N, N), dtype=np.complex64)

    # Setup line profiler
    lp = LineProfiler()
    lp.add_function(solver._linear_step)

    # Prepare solver
    solver.propagator = solver._build_propagator(precision="single")
    A, A_sq = solver._prepare_output_array(field, normalize=True)
    solver._send_arrays_to_gpu()

    # Profile
    lp_wrapper = lp(lambda: solver._linear_step(A, solver.propagator))
    for _ in range(35):  # Same number as in real run
        lp_wrapper()

    lp.print_stats()


def check_metal_fft_implementation():
    """Check the actual Metal FFT implementation."""
    print("\n" + "="*80)
    print("Metal FFT Implementation Analysis")
    print("="*80 + "\n")

    import inspect
    from NLSE.kernels import metal

    print("MetalFFTPlan class source:")
    print(inspect.getsource(metal.MetalFFTPlan))

    print("\nMetalFFTPlan methods:")
    for name, method in inspect.getmembers(metal.MetalFFTPlan, predicate=inspect.isfunction):
        if not name.startswith('_'):
            print(f"  - {name}")


def main():
    check_metal_fft_implementation()
    profile_fft_operations()
    profile_kernels()
    profile_linear_step()


if __name__ == "__main__":
    main()
