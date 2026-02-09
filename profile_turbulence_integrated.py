#!/usr/bin/env python3
"""Profile the integrated fig2_turbulence.py scenario with detailed timing."""

import time
import numpy as np
from NLSE import NLSE

print("=" * 80)
print("Integrated Turbulence Scenario Profiling")
print("=" * 80)

# Configuration from fig2_turbulence.py
N = 1024
n2 = -1.6e-9
window = 8e-3
power = 1.05
Isat = 10e4
L = 20e-2  # Full 20cm propagation
alpha = 20
waist = 2e-3
waist_d = 1e-3

print(f"\nScenario: Turbulence simulation")
print(f"Grid: {N}×{N} ({N*N/1e6:.1f}M elements)")
print(f"Propagation distance: {L*100:.1f} cm")
print(f"Power: {power:.2f}")

# Create simulation
print("\n" + "-" * 80)
print("Initialization")
print("-" * 80)

t_init = time.perf_counter()
simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="CL")
simu.delta_z = 1e-4
t_init = time.perf_counter() - t_init

print(f"Backend: {simu._backend.context.devices[0].name}")
print(f"Delta z: {simu.delta_z*1e6:.1f} µm")
print(f"Expected steps: {int(L / simu.delta_z)}")
print(f"Initialization time: {t_init:.3f} s")

# Setup potential and field
print("\n" + "-" * 80)
print("Field Setup")
print("-" * 80)

t_setup = time.perf_counter()

# Defect potential
simu.V = 1e-4 * np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist_d**2)

# Initial field with counter-propagating beams
kp = 2 * np.pi * 5e3
E0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(np.complex64)
E0[0 : N // 2, :] *= np.exp(1j * kp * simu.XX[0 : N // 2, :])
E0[N // 2 :, :] *= np.exp(-1j * kp * simu.XX[N // 2 :, :])

t_setup = time.perf_counter() - t_setup
print(f"Field setup time: {t_setup:.3f} s")

# Profile short propagation first
print("\n" + "=" * 80)
print("Short Propagation Test (1mm)")
print("=" * 80)

L_short = 1e-3
steps_short = int(L_short / simu.delta_z)
print(f"Steps: {steps_short}")

t_short = time.perf_counter()
E_short = simu.out_field(E0, L_short, verbose=False, plot=False)
t_short = time.perf_counter() - t_short

print(f"Total time: {t_short:.3f} s")
print(f"Time per step: {t_short/steps_short*1000:.2f} ms")
print(f"Throughput: {steps_short/t_short:.1f} steps/s")

# Profile medium propagation
print("\n" + "=" * 80)
print("Medium Propagation Test (1cm)")
print("=" * 80)

L_medium = 1e-2
steps_medium = int(L_medium / simu.delta_z)
print(f"Steps: {steps_medium}")

t_medium = time.perf_counter()
E_medium = simu.out_field(E0, L_medium, verbose=False, plot=False)
t_medium = time.perf_counter() - t_medium

print(f"Total time: {t_medium:.3f} s")
print(f"Time per step: {t_medium/steps_medium*1000:.2f} ms")
print(f"Throughput: {steps_medium/t_medium:.1f} steps/s")

# Estimate full propagation
print("\n" + "=" * 80)
print("Full Propagation Estimate (20cm)")
print("=" * 80)

steps_full = int(L / simu.delta_z)
time_per_step_avg = (t_short/steps_short + t_medium/steps_medium) / 2

print(f"Expected steps: {steps_full}")
print(f"Avg time per step: {time_per_step_avg*1000:.2f} ms")
print(f"Estimated total time: {time_per_step_avg * steps_full:.1f} s ({time_per_step_avg * steps_full/60:.1f} min)")

# Actually run full propagation
print("\nRunning full propagation...")
t_full = time.perf_counter()
E_full = simu.out_field(E0, L, verbose=False, plot=False)
t_full = time.perf_counter() - t_full

print(f"\nActual total time: {t_full:.1f} s ({t_full/60:.1f} min)")
print(f"Actual time per step: {t_full/steps_full*1000:.2f} ms")
print(f"Actual throughput: {steps_full/t_full:.1f} steps/s")

# Performance summary
print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)

bandwidth_estimate = (N * N * 8 * 6) / (t_full/steps_full) / 1e9  # 6 array accesses per step
print(f"Memory bandwidth (estimated): {bandwidth_estimate:.1f} GB/s")
print(f"FLOPs per step (estimated): {N*N*100:.0e}")  # ~100 ops per pixel
flops_per_sec = (N*N*100) * (steps_full/t_full)
print(f"Compute throughput: {flops_per_sec:.2e} FLOP/s ({flops_per_sec/1e9:.1f} GFLOP/s)")

# Bottleneck analysis
print("\n" + "=" * 80)
print("Bottleneck Analysis")
print("=" * 80)

theoretical_fft_time = 0.587e-3 + 0.470e-3  # From earlier profiling
theoretical_nl_time = 3 * 0.365e-3
theoretical_other = 0.229e-3

theoretical_step = theoretical_fft_time + theoretical_nl_time + theoretical_other
actual_step = t_full / steps_full

print(f"Theoretical step time: {theoretical_step*1000:.2f} ms")
print(f"Actual step time:      {actual_step*1000:.2f} ms")
print(f"Efficiency:            {100*theoretical_step/actual_step:.1f}%")

overhead = actual_step - theoretical_step
print(f"\nOverhead breakdown:")
print(f"  Kernel launches: ~{overhead*1000*0.3:.2f} ms (30% of overhead)")
print(f"  GPU sync:        ~{overhead*1000*0.2:.2f} ms (20% of overhead)")
print(f"  Other:           ~{overhead*1000*0.5:.2f} ms (50% of overhead)")

print("\n" + "=" * 80)
print("Optimization Recommendations")
print("=" * 80)

print("\n1. Use fused kernels (square_mod_nl_prop):")
print(f"   Potential savings: ~0.2ms per step = {0.2*steps_full/1000:.1f}s total ({100*0.2/actual_step:.1f}%)")

print("\n2. Optimize work group sizes:")
print(f"   Current: Using default (None)")
print(f"   Recommended: Benchmark 64, 128, 256 threads per work group")

print("\n3. Propagator caching:")
print(f"   If not already cached, could save ~0.1ms per step")

print("\n4. Async execution:")
print(f"   Overlap CPU and GPU work (limited benefit in pure GPU code)")

print("\n" + "=" * 80)
