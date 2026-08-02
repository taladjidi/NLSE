#!/usr/bin/env python3
"""
Profiling the CUDA backend
==========================

Runs under CuPy's profiler to attribute time inside a step.

NVIDIA profiling script for the CuPy backend.

Exercises the GPU hot path (split_step and RK4 propagation) under realistic
conditions with NVTX annotations and CUDA profiler API markers so that
nsys / ncu can capture only the interesting region.

Usage
-----
Standalone (prints timings, verifies GPU execution)::

    python examples/profile_cupy.py

Timeline profiling (nsys)::

    nsys profile -o profile_cupy_report \
      --cuda-memory-usage=true \
      --cudabacktrace=all \
      -c cudaProfilerApi \
      python examples/profile_cupy.py

Kernel-level profiling (ncu) — split_step kernels only::

    ncu --set full \
      --nvtx --nvtx-include "split_step/" \
      -o profile_cupy_kernels \
      python examples/profile_cupy.py

Open reports::

    nsys-ui profile_cupy_report.nsys-rep
    ncu-ui  profile_cupy_kernels.ncu-rep
"""

import sys
import time

import numpy as np
from NLSE import NLSE

try:
    import cupy as cp
except ImportError:
    print(
        "This example profiles the CUDA backend with nsys/ncu and needs cupy "
        "and an NVIDIA GPU. Nothing to profile here; see the module docstring "
        "for how to run it on a machine that has them."
    )
    sys.exit(0)

# Propagation step, passed to out_field and used by the plots below.
DELTA_Z = 0.5e-4

# ---------------------------------------------------------------------------
# Simulation parameters (realistic 2D, single precision)
# ---------------------------------------------------------------------------
N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # W/m^2
L = 10e-3
alpha = 20
N_STEPS = 50  # enough for stable stats without overwhelming ncu


def make_simu():
    """Create and return a configured NLSE simulation instance."""
    simu = NLSE(
        alpha,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="CUPY",
    )
    # Fix step size so we get exactly N_STEPS steps over distance z_run
    # Gaussian input field
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
    # Lens-like potential
    V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(np.complex64)
    simu.V = V
    z_run = abs(DELTA_Z) * N_STEPS
    return simu, E_0, z_run


def warmup(simu, E_0):
    """Run a few steps to trigger JIT / FFT planning outside profiled region."""
    z_warmup = abs(DELTA_Z) * 3
    simu.out_field(
        E_0.copy(),
        z_warmup,
        verbose=False,
        plot=False,
        splitting="lie",
        delta_z=DELTA_Z,
    )
    # Reset propagator so next call rebuilds cleanly
    simu.propagator = None
    simu.plans = None
    cp.cuda.Device().synchronize()


def profile_method(simu, E_0, z_run, method, splitting="lie"):
    """Profile one propagation method inside an NVTX range.

    Returns (cpu_seconds, gpu_milliseconds).
    """
    # Reset propagator for this method
    simu.propagator = None
    simu.plans = None
    cp.cuda.Device().synchronize()

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    cp.cuda.nvtx.RangePush(method)
    start_gpu.record()
    t0 = time.perf_counter()

    simu.out_field(
        E_0.copy(),
        z_run,
        verbose=False,
        plot=False,
        splitting=splitting,
        method=method,
        delta_z=DELTA_Z,
    )

    t_cpu = time.perf_counter() - t0
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    cp.cuda.nvtx.RangePop()

    return t_cpu, t_gpu


def main():
    print(f"Grid: {N}x{N}  |  Steps: {N_STEPS}  |  Backend: CUPY")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print()

    simu, E_0, z_run = make_simu()

    # --- Warm-up (outside profiler region) ---
    print("Warming up (JIT compile, FFT planning) ...")
    warmup(simu, E_0)
    print("Warm-up done.\n")

    # --- Start CUDA profiler (nsys -c cudaProfilerApi captures from here) ---
    cp.cuda.profiler.start()

    # --- Split-step profiling ---
    print("Profiling split_step ...")
    t_cpu_ss, t_gpu_ss = profile_method(simu, E_0, z_run, "split_step")
    print(f"  CPU wall: {t_cpu_ss:.4f} s  |  GPU time: {t_gpu_ss:.2f} ms")

    # --- RK4 profiling ---
    print("Profiling RK4 ...")
    t_cpu_rk4, t_gpu_rk4 = profile_method(simu, E_0, z_run, "RK4")
    print(f"  CPU wall: {t_cpu_rk4:.4f} s  |  GPU time: {t_gpu_rk4:.2f} ms")

    # --- Stop CUDA profiler ---
    cp.cuda.profiler.stop()

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"{'Method':<12} {'CPU (s)':>10} {'GPU (ms)':>10} {'Steps':>6}")
    print(f"{'split_step':<12} {t_cpu_ss:>10.4f} {t_gpu_ss:>10.2f} {N_STEPS:>6}")
    print(f"{'RK4':<12} {t_cpu_rk4:>10.4f} {t_gpu_rk4:>10.2f} {N_STEPS:>6}")
    print(
        f"\nRK4 / split_step ratio:  "
        f"CPU {t_cpu_rk4 / t_cpu_ss:.2f}x  "
        f"GPU {t_gpu_rk4 / t_gpu_ss:.2f}x"
    )


if __name__ == "__main__":
    main()
