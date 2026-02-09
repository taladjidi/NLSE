#!/usr/bin/env python3
"""Detailed OpenCL profiling with kernel-level timing."""

import time
import numpy as np
from NLSE import NLSE
import pyopencl as cl

# Test configuration
N = 1024
n2 = -1.6e-9
window = 8e-3
power = 1.05
Isat = 10e4
L = 20e-2
alpha = 20

print("=" * 80)
print("Detailed OpenCL Performance Profiling")
print("=" * 80)

# Setup simulation
simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="CL")
simu.delta_z = 1e-4

# Setup field
waist = 2e-3
E0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(np.complex64)

print(f"\nGrid: {N}×{N} ({N*N/1e6:.1f}M elements)")
print(f"Device: {simu._backend.context.devices[0].name}")
print(f"Double precision supported: {simu._backend.kernels._double_supported}")

# Enable profiling
simu._backend._queue = cl.CommandQueue(
    simu._backend._context,
    properties=cl.command_queue_properties.PROFILING_ENABLE
)

# Detailed split-step profiling
print("\n" + "=" * 80)
print("Split-Step Operation Breakdown")
print("=" * 80)

E_gpu = simu._backend.from_numpy(E0)
E_sq = simu._backend.allocate_real_field((N, N), np.float32)
plans = simu._backend.build_fft((N, N), axes=(-2, -1), dtype=np.complex64)

n_trials = 20

def profile_operation(name, func, *args):
    """Profile an operation with timing."""
    times = []
    for _ in range(n_trials):
        simu._backend._queue.finish()
        t0 = time.perf_counter()
        func(*args)
        simu._backend._queue.finish()
        times.append(time.perf_counter() - t0)
    
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    
    print(f"{name:30s} {mean_ms:7.3f} ms  (±{std_ms:5.3f})  [{min_ms:6.3f}, {max_ms:6.3f}]")
    return np.mean(times)

print("\nKernel operations:")
print("-" * 80)

# Profile each operation
t_square_mod = profile_operation(
    "square_mod()", 
    simu._backend.kernels.square_mod,
    E_gpu, E_sq
)

t_nl_prop = profile_operation(
    "nl_prop() [no potential]",
    simu._backend.kernels.nl_prop_without_V,
    E_gpu, E_sq, simu.delta_z, simu.alpha, simu.k * simu.n2, simu.I_sat
)

# With potential
V_gpu = simu._backend.from_numpy(np.zeros((N, N), dtype=np.float32))
t_nl_prop_v = profile_operation(
    "nl_prop() [with potential]",
    simu._backend.kernels.nl_prop,
    E_gpu, E_sq, simu.delta_z, simu.alpha, V_gpu, simu.k * simu.n2, simu.I_sat
)

t_fft = profile_operation(
    "FFT (forward)",
    simu._backend.fft,
    E_gpu, plans
)

t_ifft = profile_operation(
    "FFT (inverse)",
    simu._backend.ifft,
    E_gpu, plans
)

# Propagator multiplication (simulated)
propagator = simu._backend.from_numpy(np.ones((N, N), dtype=np.complex64))

def multiply_arrays(A, B):
    """Simulate propagator multiplication."""
    # This is what happens in split_step
    temp = A * B
    
times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    temp = E_gpu * propagator
    simu._backend._queue.finish()
    times.append(time.perf_counter() - t0)
t_multiply = np.mean(times)
print(f"{'Array multiplication':30s} {t_multiply*1000:7.3f} ms  (±{np.std(times)*1000:5.3f})  [{np.min(times)*1000:6.3f}, {np.max(times)*1000:6.3f}]")

# Memory transfer overhead
t_to_gpu = profile_operation(
    "CPU->GPU transfer",
    simu._backend.from_numpy,
    E0
)

t_from_gpu = profile_operation(
    "GPU->CPU transfer",
    simu._backend.to_numpy,
    E_gpu
)

# Analysis
print("\n" + "=" * 80)
print("Split-Step Time Analysis")
print("=" * 80)

# Standard split-step: square_mod + nl_prop + fft + nl_prop + ifft + nl_prop
step_time_theory = t_square_mod + 3*t_nl_prop + t_fft + t_ifft + 2*t_multiply
step_time_measured = None

# Measure actual split-step
print("\nMeasuring actual split_step() call...")
times = []
for _ in range(10):
    E_test = simu._backend.from_numpy(E0.copy())
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    
    # Simulate one split-step
    E_sq_temp = simu._backend.allocate_real_field((N, N), np.float32)
    simu._backend.kernels.square_mod(E_test, E_sq_temp)
    simu._backend.kernels.nl_prop_without_V(E_test, E_sq_temp, simu.delta_z/2, 
                                             simu.alpha, simu.k * simu.n2, simu.I_sat)
    simu._backend.fft(E_test, plans)
    # Would multiply by propagator here
    simu._backend.ifft(E_test, plans)
    simu._backend.kernels.nl_prop_without_V(E_test, E_sq_temp, simu.delta_z/2,
                                             simu.alpha, simu.k * simu.n2, simu.I_sat)
    
    simu._backend._queue.finish()
    times.append(time.perf_counter() - t0)

step_time_measured = np.mean(times)

print(f"\nTheoretical (sum of parts):  {step_time_theory*1000:7.3f} ms")
print(f"Measured (actual execution): {step_time_measured*1000:7.3f} ms")
print(f"Overhead:                    {(step_time_measured - step_time_theory)*1000:7.3f} ms ({100*(step_time_measured - step_time_theory)/step_time_measured:.1f}%)")

# Breakdown
print("\nOperation breakdown:")
print("-" * 80)
total = step_time_measured

def show_percent(name, time_val):
    print(f"  {name:35s} {time_val*1000:7.3f} ms  ({100*time_val/total:5.1f}%)")

show_percent("Square modulus", t_square_mod)
show_percent("NL propagation (3x)", 3*t_nl_prop)
show_percent("FFT + IFFT", t_fft + t_ifft)
show_percent("Array multiplications (2x)", 2*t_multiply)
show_percent("Overhead (launch, sync, etc.)", step_time_measured - step_time_theory)

# Optimization suggestions
print("\n" + "=" * 80)
print("Optimization Opportunities")
print("=" * 80)

print("\n1. Kernel Fusion Potential:")
fused_square_nl = t_square_mod + t_nl_prop
print(f"   square_mod + nl_prop:  {fused_square_nl*1000:.3f} ms → could save ~30% launch overhead")

print("\n2. Work Group Size Tuning:")
print(f"   Current: global_size only (no local_size specified)")
print(f"   Potential: Tune for {simu._backend.context.devices[0].name}")

print("\n3. Propagator Caching:")
print(f"   Current: Propagator likely recalculated each time")
print(f"   Potential: Pre-calculate and cache propagator array")

print("\n4. Memory Bandwidth:")
bytes_per_step = (N * N * 8) * 6  # 6 array accesses (read E, write E, read/write E_sq, etc.)
bandwidth_gbs = bytes_per_step / step_time_measured / 1e9
print(f"   Estimated bandwidth usage: {bandwidth_gbs:.1f} GB/s")
print(f"   (Actual device bandwidth likely much higher)")

print("\n" + "=" * 80)
