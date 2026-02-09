#!/usr/bin/env python3
"""Profile OpenCL backend performance using turbulence example."""

import time
import numpy as np
from NLSE import NLSE

# Test configuration based on fig2_turbulence.py
N = 1024
n2 = -1.6e-9
window = 8e-3
power = 1.05
Isat = 10e4
L = 20e-2
alpha = 20
waist = 2e-3
waist_d = 1e-3

print("=" * 70)
print("OpenCL Backend Performance Profiling")
print("=" * 70)

# Setup simulation with OpenCL
print(f"\nSetup: Grid {N}×{N}, L={L}m")
print("-" * 70)

t0 = time.perf_counter()
simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="CL")
simu.delta_z = 1e-4
setup_time = time.perf_counter() - t0
print(f"Backend initialization: {setup_time:.3f} s")

# Setup potential and initial field
t0 = time.perf_counter()
simu.V = 1e-4 * np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist_d**2)
kp = 2 * np.pi * 5e3
E0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(np.complex64)
E0[0 : N // 2, :] *= np.exp(1j * kp * simu.XX[0 : N // 2, :])
E0[N // 2 :, :] *= np.exp(-1j * kp * simu.XX[N // 2 :, :])
field_setup_time = time.perf_counter() - t0
print(f"Field initialization: {field_setup_time:.3f} s")

# Profile a single step to understand kernel timing
print("\n" + "=" * 70)
print("Single Step Profiling")
print("=" * 70)

# Import OpenCL for event profiling
import pyopencl as cl

# Enable profiling on the queue
simu._backend._queue = cl.CommandQueue(
    simu._backend._context,
    properties=cl.command_queue_properties.PROFILING_ENABLE
)

# Allocate field on GPU
E_gpu = simu._backend.from_numpy(E0)

# Profile FFT plan build
t0 = time.perf_counter()
plans = simu._backend.build_fft((N, N), axes=(-2, -1), dtype=np.complex64)
fft_plan_time = time.perf_counter() - t0
print(f"\nFFT plan build: {fft_plan_time:.3f} s")

# Profile individual operations
print("\nOperation timing (averaged over 10 iterations):")
print("-" * 70)

n_trials = 10

# 1. FFT forward
fft_times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    simu._backend.fft(E_gpu, plans)
    simu._backend._queue.finish()
    fft_times.append(time.perf_counter() - t0)
print(f"FFT (forward):     {np.mean(fft_times)*1000:6.2f} ms  (std: {np.std(fft_times)*1000:.2f} ms)")

# 2. FFT inverse
ifft_times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    simu._backend.ifft(E_gpu, plans)
    simu._backend._queue.finish()
    ifft_times.append(time.perf_counter() - t0)
print(f"FFT (inverse):     {np.mean(ifft_times)*1000:6.2f} ms  (std: {np.std(ifft_times)*1000:.2f} ms)")

# 3. Square modulus
from pyopencl import array as cla
A_sq = simu._backend.allocate_real_field((N, N), np.float32)

square_mod_times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    simu._backend.kernels.square_mod(E_gpu, A_sq)
    simu._backend._queue.finish()
    square_mod_times.append(time.perf_counter() - t0)
print(f"Square modulus:    {np.mean(square_mod_times)*1000:6.2f} ms  (std: {np.std(square_mod_times)*1000:.2f} ms)")

# 4. Nonlinear propagation (with potential)
V_gpu = simu._backend.from_numpy(simu.V.astype(np.float32))
nl_prop_times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    simu._backend.kernels.nl_prop(
        E_gpu, A_sq, simu.delta_z, simu.alpha, V_gpu,
        simu.k * simu.n2, simu.I_sat
    )
    simu._backend._queue.finish()
    nl_prop_times.append(time.perf_counter() - t0)
print(f"NL propagation:    {np.mean(nl_prop_times)*1000:6.2f} ms  (std: {np.std(nl_prop_times)*1000:.2f} ms)")

# 5. Data transfer (GPU -> CPU)
transfer_times = []
for _ in range(n_trials):
    simu._backend._queue.finish()
    t0 = time.perf_counter()
    _ = simu._backend.to_numpy(E_gpu)
    transfer_times.append(time.perf_counter() - t0)
print(f"GPU->CPU transfer: {np.mean(transfer_times)*1000:6.2f} ms  (std: {np.std(transfer_times)*1000:.2f} ms)")

# Estimate full step time
full_step_time = (
    np.mean(square_mod_times) +
    np.mean(nl_prop_times) +
    np.mean(fft_times) +
    np.mean(nl_prop_times) +  # Second half-step
    np.mean(ifft_times) +
    np.mean(nl_prop_times)    # Third half-step
)

print(f"\nEstimated full split-step time: {full_step_time*1000:.2f} ms")
print(f"Steps per second: {1/full_step_time:.1f}")

# Calculate percentage breakdown
total = full_step_time * 1000
print("\nTime breakdown:")
print("-" * 70)
print(f"  FFT operations:     {(np.mean(fft_times) + np.mean(ifft_times))*1000:6.2f} ms ({100*(np.mean(fft_times) + np.mean(ifft_times))/full_step_time:.1f}%)")
print(f"  NL propagation (3x): {3*np.mean(nl_prop_times)*1000:6.2f} ms ({100*3*np.mean(nl_prop_times)/full_step_time:.1f}%)")
print(f"  Square modulus:     {np.mean(square_mod_times)*1000:6.2f} ms ({100*np.mean(square_mod_times)/full_step_time:.1f}%)")

# Profile full propagation
print("\n" + "=" * 70)
print("Full Propagation Test")
print("=" * 70)

# Run short propagation
L_test = 1e-3  # 1mm propagation
steps = int(L_test / simu.delta_z)
print(f"\nPropagating {L_test*1000:.1f} mm ({steps} steps)")

t0 = time.perf_counter()
E_out = simu.out_field(E0, L_test, verbose=False, plot=False)
total_time = time.perf_counter() - t0

print(f"Total time: {total_time:.3f} s")
print(f"Time per step: {total_time/steps*1000:.2f} ms")
print(f"Throughput: {steps/total_time:.1f} steps/s")

# Memory usage estimate
bytes_per_array = N * N * 8  # complex64 = 8 bytes
n_arrays = 5  # E, E_sq, V, propagator, temp
total_memory = bytes_per_array * n_arrays / (1024**2)
print(f"\nEstimated GPU memory usage: {total_memory:.1f} MB")

print("\n" + "=" * 70)
print("Profiling Complete")
print("=" * 70)
