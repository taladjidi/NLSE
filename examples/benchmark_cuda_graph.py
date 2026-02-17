#!/usr/bin/env python3
"""Benchmark: old NLSE vs new NLSE (with/without CUDA graph).

Runs the fig2_turbulence simulation on the CUPY backend three ways:
  1. Old NLSE package (~/Documents/LKB/NLSE)
  2. New NLSE package, cuda_graph=False
  3. New NLSE package, cuda_graph=True

Usage:
    python examples/benchmark_cuda_graph.py
"""

import sys

import cupy as cp
import numpy as np

# ── Simulation parameters (from fig2_turbulence.py) ──────────────────────────
n2 = -1.6e-9
window = 8e-3
power = 1.05
Isat = 10e4
L = 20e-2
alpha = 20
waist = 2e-3
waist_d = 1e-3
N_WARMUP = 1
N_RUNS = 3
GRID_SIZES = [256, 512, 1024, 2048]


def make_field_and_potential(simu, N):
    """Build E0 and V from a configured NLSE instance."""
    simu.delta_z = 1e-4
    simu.V = 1e-4 * np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist_d**2)
    kp = 2 * np.pi * 5e3
    E0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(np.complex64)
    E0[0 : N // 2, :] *= np.exp(1j * kp * simu.XX[0 : N // 2, :])
    E0[N // 2 :, :] *= np.exp(-1j * kp * simu.XX[N // 2 :, :])
    return E0


def time_run(run_fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Time a function: warmup, then measure n_runs, return (mean, std) in ms."""
    for _ in range(n_warmup):
        run_fn()
        cp.cuda.Device().synchronize()

    gpu_times = []
    for _ in range(n_runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        run_fn()
        end.record()
        end.synchronize()
        gpu_times.append(cp.cuda.get_elapsed_time(start, end))

    return np.mean(gpu_times), np.std(gpu_times)


def bench_old_nlse(N):
    """Benchmark the old NLSE package."""
    old_path = "/home/aladjidi/Documents/LKB/NLSE"
    sys.path.insert(0, old_path)
    for mod_name in list(sys.modules):
        if mod_name.startswith("NLSE"):
            del sys.modules[mod_name]

    from NLSE import NLSE as NLSE_old

    simu = NLSE_old(
        alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="GPU"
    )
    E0 = make_field_and_potential(simu, N)

    def run():
        simu.propagator = None
        simu.plans = None
        simu.out_field(E0.copy(), L, verbose=False, plot=False, precision="single")

    mean_ms, std_ms = time_run(run)

    sys.path.remove(old_path)
    for mod_name in list(sys.modules):
        if mod_name.startswith("NLSE"):
            del sys.modules[mod_name]

    return mean_ms, std_ms


def bench_new_nlse(N):
    """Benchmark the new NLSE package (CUDA graph is automatic)."""
    from NLSE import NLSE

    simu = NLSE(
        alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="CUPY"
    )
    E0 = make_field_and_potential(simu, N)

    def run():
        simu.propagator = None
        simu.plans = None
        simu.out_field(
            E0.copy(), L, verbose=False, plot=False, precision="single",
        )

    mean_ms, std_ms = time_run(run)

    for mod_name in list(sys.modules):
        if mod_name.startswith("NLSE"):
            del sys.modules[mod_name]

    return mean_ms, std_ms


def main():
    n_steps = int(L / 1e-4)  # = 2000
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Steps: ~{n_steps}  |  Warmup: {N_WARMUP}  |  Timed runs: {N_RUNS}")
    print()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = f"{'Grid':<10} {'Old (ms)':>12} {'New (ms)':>12} {'Speedup':>10}"
    print(hdr)
    print("-" * len(hdr))

    for N in GRID_SIZES:
        print(f"\n  [{N}x{N}] Running old NLSE ...", end="", flush=True)
        old_mean, old_std = bench_old_nlse(N)
        print(f" {old_mean:.0f} ms", end="", flush=True)

        print("  |  new ...", end="", flush=True)
        new_mean, new_std = bench_new_nlse(N)
        print(f" {new_mean:.0f} ms")

        print(
            f"  {N}x{N:<7} {old_mean:>8.1f}±{old_std:<4.1f}"
            f" {new_mean:>8.1f}±{new_std:<4.1f}"
            f" {old_mean / new_mean:>9.2f}x"
        )

    print()


if __name__ == "__main__":
    main()
