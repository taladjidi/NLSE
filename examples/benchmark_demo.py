#!/usr/bin/env python3
"""
FFT Auto-Benchmarking Demo

This script demonstrates the automatic FFT backend selection feature.
It shows how the system benchmarks available backends and selects the
fastest option for your hardware.
"""

import numpy as np
from NLSE import NLSE
from NLSE.backends import benchmark

print("=" * 60)
print("FFT Auto-Benchmarking Demo")
print("=" * 60)

# Step 1: Force a fresh benchmark (optional - normally cached)
print("\n[1] Invalidating any existing cache...")
benchmark.invalidate_cache()

# Step 2: Create NLSE with auto backend selection
print("\n[2] Creating NLSE with auto backend selection...")
print("    Grid size: 512 × 512")
print()

simu = NLSE(
    alpha=0,
    power=1,
    window=1e-3,
    n2=1e-20,
    V=None,
    L=1e-2,
    NX=512,
    NY=512,
    backend="auto",  # Automatic selection
)

print(f"\n✓ Selected backend: {simu.backend}")

# Step 3: Display benchmark results
print("\n" + "=" * 60)
print("Benchmark Results")
print("=" * 60)

cache = benchmark.load_benchmark_cache()
if cache:
    print(f"\nPlatform: {cache['platform']['system']} {cache['platform']['processor']}")
    print(f"Python: {cache['platform']['python_version']}")
    print(f"Grid size: {cache['grid_size'][0]} × {cache['grid_size'][1]}")
    print(f"Timestamp: {cache['timestamp'][:19]}")
    print(f"\nResults:")

    # Sort by time (fastest first)
    sorted_results = sorted(
        [(name, data) for name, data in cache["results"].items() if data["available"]],
        key=lambda x: x[1]["time_ms"],
    )

    for name, data in sorted_results:
        is_fastest = name == cache["fastest"]
        marker = " ⚡ (selected)" if is_fastest else ""
        print(
            f"  {name:6s}: {data['time_ms']:8.2f} ms  "
            f"(speedup: {data['speedup']:6.2f}×){marker}"
        )

# Step 4: Quick performance demonstration
print("\n" + "=" * 60)
print("Quick Performance Demo")
print("=" * 60)

print("\nRunning a quick simulation step...")

# Create simple initial field (Gaussian)
X, Y = np.meshgrid(simu.X, simu.Y)
E = np.exp(-(X**2 + Y**2) / (2 * (1e-4) ** 2)) + 0j

# Run one propagation step
import time

t0 = time.perf_counter()
E_out = simu.out_field(E, z=1e-3, verbose=False, plot=False)
elapsed = (time.perf_counter() - t0) * 1000

print(f"✓ Propagation completed in {elapsed:.2f} ms")
print(f"  Backend used: {simu.backend}")

# Step 5: Show cache location
print("\n" + "=" * 60)
print("Cache Information")
print("=" * 60)

from NLSE.utils import get_benchmark_cache_path

cache_path = get_benchmark_cache_path()
print(f"\nBenchmark results cached at:")
print(f"  {cache_path}")
print(f"\nCache is valid for 30 days or until manually invalidated.")
print(f"To force re-benchmark: benchmark.invalidate_cache()")

# Step 6: Tips
print("\n" + "=" * 60)
print("Tips")
print("=" * 60)
print("""
1. Use backend="auto" for optimal performance
2. First run benchmarks (2-5 sec), subsequent runs use cache (instant)
3. Cache is grid-size specific
4. View results: benchmark.load_benchmark_cache()
5. Force re-benchmark: benchmark.invalidate_cache()
6. Environment variable: export NLSE_BACKEND=CPU (override)

For more information, see BENCHMARKING.md
""")

print("=" * 60)
print("Demo Complete!")
print("=" * 60)
