#!/usr/bin/env python
"""Profile NLSE backends to identify performance bottlenecks.

Usage:
    python profile_backends.py --backend Metal --solver NLSE --profile-type line
    python profile_backends.py --backend CPU --solver CNLSE --profile-type cprofile
    python profile_backends.py --all  # Profile all backends and solvers
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d

# Try to import line_profiler if available
try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    print("Warning: line_profiler not available. Install with: pip install line_profiler")

# Workload configurations
WORKLOADS = {
    "small": {"NX": 64, "NY": 64, "steps": 100},
    "medium": {"NX": 256, "NY": 256, "steps": 100},
    "large": {"NX": 512, "NY": 512, "steps": 50},
}

# Solver configurations
SOLVER_CONFIGS = {
    "NLSE": {
        "class": NLSE,
        "params": {
            "alpha": 0,
            "power": 1.05,
            "window": 4 * 2.23e-3,
            "n2": -1.6e-9,
            "V": None,
            "L": 1e-3,
            "Isat": 10e4,
        },
        "field_shape": lambda nx, ny: (nx, ny),
    },
    "NLSE_1d": {
        "class": NLSE_1d,
        "params": {
            "alpha": 0,
            "power": 1.05,
            "window": 4 * 2.23e-3,
            "n2": -1.6e-9,
            "V": None,
            "L": 1e-3,
            "Isat": 10e4,
        },
        "field_shape": lambda nx, ny: (nx,),
    },
    "CNLSE": {
        "class": CNLSE,
        "params": {
            "alpha": 0,
            "power": 1.05,
            "window": 4 * 2.23e-3,
            "n2": -1.6e-9,
            "n12": -1e-10,
            "V": None,
            "L": 1e-3,
            "Isat": 10e4,
        },
        "field_shape": lambda nx, ny: (2, nx, ny),
    },
    "CNLSE_1d": {
        "class": CNLSE_1d,
        "params": {
            "alpha": 0,
            "power": 1.05,
            "window": 4 * 2.23e-3,
            "n2": -1.6e-9,
            "n12": -1e-10,
            "V": None,
            "L": 1e-3,
            "Isat": 10e4,
        },
        "field_shape": lambda nx, ny: (2, nx),
    },
    "GPE": {
        "class": GPE,
        "params": {
            "gamma": 0,
            "N": 1e6,
            "window": 1e-3,
            "g": 1e3 / (1e6 / 1e-3**2),
            "V": None,
            "m": 87 * 1.66053906660e-27,  # 87 * atomic_mass
        },
        "field_shape": lambda nx, ny: (nx, ny),
    },
}


def create_solver(solver_name: str, backend: str, workload: dict[str, Any]) -> Any:
    """Create a solver instance with the given configuration."""
    config = SOLVER_CONFIGS[solver_name]
    params = config["params"].copy()
    params["NX"] = workload["NX"]
    if "NY" in workload:
        params["NY"] = workload["NY"]
    params["backend"] = backend

    return config["class"](**params)


def create_field(solver_name: str, workload: dict[str, Any]) -> np.ndarray:
    """Create an input field for the solver."""
    config = SOLVER_CONFIGS[solver_name]
    shape = config["field_shape"](workload["NX"], workload.get("NY", workload["NX"]))
    return np.ones(shape, dtype=np.complex64)


def run_workload(
    solver: Any,
    field: np.ndarray,
    L: float = 1e-3,
    precision: str = "single",
) -> tuple[np.ndarray, float]:
    """Run a solver workload and return result + elapsed time."""
    start = time.perf_counter()
    result = solver.out_field(field, L, verbose=False, plot=False, precision=precision)
    elapsed = time.perf_counter() - start
    return result, elapsed


def profile_cprofile(
    solver_name: str,
    backend: str,
    workload_name: str = "medium",
) -> None:
    """Profile using cProfile."""
    workload = WORKLOADS[workload_name]
    solver = create_solver(solver_name, backend, workload)
    field = create_field(solver_name, workload)
    L = SOLVER_CONFIGS[solver_name]["params"]["L"]

    print(f"\n{'='*80}")
    print(f"cProfile: {solver_name} on {backend} ({workload_name} workload)")
    print(f"Grid: {workload['NX']}x{workload.get('NY', workload['NX'])}")
    print(f"{'='*80}\n")

    profiler = cProfile.Profile()
    profiler.enable()

    result, elapsed = run_workload(solver, field, L)

    profiler.disable()

    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())

    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Throughput: {workload['NX'] * workload.get('NY', workload['NX']) / elapsed / 1e6:.2f} Mpixels/s")


def profile_line_profiler(
    solver_name: str,
    backend: str,
    workload_name: str = "medium",
) -> None:
    """Profile using line_profiler for detailed line-by-line analysis."""
    if not HAS_LINE_PROFILER:
        print("Error: line_profiler not installed")
        return

    workload = WORKLOADS[workload_name]
    solver = create_solver(solver_name, backend, workload)
    field = create_field(solver_name, workload)
    L = SOLVER_CONFIGS[solver_name]["params"]["L"]

    print(f"\n{'='*80}")
    print(f"Line Profiler: {solver_name} on {backend} ({workload_name} workload)")
    print(f"Grid: {workload['NX']}x{workload.get('NY', workload['NX'])}")
    print(f"{'='*80}\n")

    # Create line profiler and add methods to profile
    lp = LineProfiler()
    lp.add_function(solver.out_field)
    lp.add_function(solver.split_step)

    # Add backend-specific methods
    if backend == "Metal":
        lp.add_function(solver._send_arrays_to_gpu)
        if hasattr(solver, '_retrieve_arrays_from_gpu'):
            lp.add_function(solver._retrieve_arrays_from_gpu)

    # Run profiling
    lp_wrapper = lp(run_workload)
    result, elapsed = lp_wrapper(solver, field, L)

    # Print statistics
    lp.print_stats()

    print(f"\nTotal time: {elapsed:.3f}s")


def profile_metal_transfers(
    solver_name: str = "NLSE",
    workload_name: str = "medium",
) -> None:
    """Detailed profiling of Metal backend focusing on CPU↔GPU transfers."""
    if not NLSE.__METAL_AVAILABLE__:
        print("Metal backend not available")
        return

    workload = WORKLOADS[workload_name]
    solver = create_solver(solver_name, "Metal", workload)
    field = create_field(solver_name, workload)
    L = SOLVER_CONFIGS[solver_name]["params"]["L"]

    print(f"\n{'='*80}")
    print(f"Metal Transfer Analysis: {solver_name} ({workload_name} workload)")
    print(f"Grid: {workload['NX']}x{workload.get('NY', workload['NX'])}")
    print(f"{'='*80}\n")

    # Time individual operations
    times = {}

    # 1. Propagator building
    start = time.perf_counter()
    solver.propagator = solver._build_propagator(precision="single")
    times['build_propagator'] = time.perf_counter() - start

    # 2. Array preparation
    start = time.perf_counter()
    A, A_sq = solver._prepare_output_array(field, normalize=True)
    times['prepare_output'] = time.perf_counter() - start

    # 3. GPU transfer
    start = time.perf_counter()
    solver._send_arrays_to_gpu()
    times['send_to_gpu'] = time.perf_counter() - start

    # 4. FFT plan building
    start = time.perf_counter()
    plans = solver._build_fft_plan(A)
    times['build_fft_plan'] = time.perf_counter() - start

    # 5. One split-step iteration (excluding FFT)
    start = time.perf_counter()
    solver.split_step(A, A_sq, solver.V, solver.propagator, plans, "single")
    times['split_step'] = time.perf_counter() - start

    # 6. GPU retrieval
    start = time.perf_counter()
    if hasattr(solver, '_retrieve_arrays_from_gpu'):
        solver._retrieve_arrays_from_gpu()
    times['retrieve_from_gpu'] = time.perf_counter() - start

    # 7. Full propagation
    solver = create_solver(solver_name, "Metal", workload)
    field = create_field(solver_name, workload)
    start = time.perf_counter()
    result = solver.out_field(field, L, verbose=False, plot=False, precision="single")
    times['full_propagation'] = time.perf_counter() - start

    # Print breakdown
    print("Timing Breakdown:")
    print(f"  Build propagator:     {times['build_propagator']*1000:8.3f} ms")
    print(f"  Prepare output:       {times['prepare_output']*1000:8.3f} ms")
    print(f"  Send to GPU:          {times['send_to_gpu']*1000:8.3f} ms")
    print(f"  Build FFT plan:       {times['build_fft_plan']*1000:8.3f} ms")
    print(f"  Single split-step:    {times['split_step']*1000:8.3f} ms")
    print(f"  Retrieve from GPU:    {times['retrieve_from_gpu']*1000:8.3f} ms")
    print(f"  Full propagation:     {times['full_propagation']*1000:8.3f} ms")

    # Calculate estimated steps
    estimated_steps = int(L / solver.delta_z)
    transfer_overhead = times['send_to_gpu'] + times['retrieve_from_gpu']
    compute_time = times['full_propagation'] - transfer_overhead

    print(f"\nEstimated steps: {estimated_steps}")
    print(f"Transfer overhead: {transfer_overhead*1000:.3f} ms ({transfer_overhead/times['full_propagation']*100:.1f}%)")
    print(f"Compute time: {compute_time*1000:.3f} ms ({compute_time/times['full_propagation']*100:.1f}%)")
    print(f"Per-step time: {times['full_propagation']/estimated_steps*1000:.3f} ms")


def compare_backends(
    solver_name: str = "NLSE",
    workload_name: str = "medium",
) -> None:
    """Compare performance across all available backends."""
    workload = WORKLOADS[workload_name]
    L = SOLVER_CONFIGS[solver_name]["params"]["L"]

    backends = ["CPU"]
    if NLSE.__CUPY_AVAILABLE__:
        backends.append("CUPY")
    if NLSE.__METAL_AVAILABLE__:
        backends.append("Metal")

    print(f"\n{'='*80}")
    print(f"Backend Comparison: {solver_name} ({workload_name} workload)")
    print(f"Grid: {workload['NX']}x{workload.get('NY', workload['NX'])}")
    print(f"{'='*80}\n")

    results = {}
    for backend in backends:
        solver = create_solver(solver_name, backend, workload)
        field = create_field(solver_name, workload)

        # Warmup
        _ = solver.out_field(field, L, verbose=False, plot=False, precision="single")

        # Timed runs
        times = []
        for _ in range(3):
            solver = create_solver(solver_name, backend, workload)
            field = create_field(solver_name, workload)
            _, elapsed = run_workload(solver, field, L)
            times.append(elapsed)

        results[backend] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
        }

    # Print comparison table
    print(f"{'Backend':<10} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Speedup':<10}")
    print("-" * 66)

    baseline = results['CPU']['mean']
    for backend in backends:
        r = results[backend]
        speedup = baseline / r['mean']
        print(f"{backend:<10} {r['mean']*1000:>10.2f}   {r['std']*1000:>10.2f}   "
              f"{r['min']*1000:>10.2f}   {speedup:>8.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Profile NLSE backends")
    parser.add_argument(
        "--backend",
        choices=["CPU", "CUPY", "Metal"],
        default="Metal",
        help="Backend to profile",
    )
    parser.add_argument(
        "--solver",
        choices=list(SOLVER_CONFIGS.keys()),
        default="NLSE",
        help="Solver to profile",
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOADS.keys()),
        default="medium",
        help="Workload size",
    )
    parser.add_argument(
        "--profile-type",
        choices=["cprofile", "line", "metal-transfer", "compare"],
        default="metal-transfer",
        help="Type of profiling to perform",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run comprehensive profiling for all backends",
    )

    args = parser.parse_args()

    if args.all:
        # Comprehensive profiling
        for solver in ["NLSE", "CNLSE"]:
            compare_backends(solver, "medium")
            if NLSE.__METAL_AVAILABLE__:
                profile_metal_transfers(solver, "medium")
    else:
        if args.profile_type == "cprofile":
            profile_cprofile(args.solver, args.backend, args.workload)
        elif args.profile_type == "line":
            profile_line_profiler(args.solver, args.backend, args.workload)
        elif args.profile_type == "metal-transfer":
            if args.backend != "Metal":
                print("Warning: metal-transfer profiling only works with Metal backend")
            profile_metal_transfers(args.solver, args.workload)
        elif args.profile_type == "compare":
            compare_backends(args.solver, args.workload)


if __name__ == "__main__":
    main()
