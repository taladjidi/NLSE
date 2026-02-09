# FFT Auto-Benchmarking Guide

## Quick Start

The NLSE package now includes automatic FFT backend benchmarking to select the fastest option for your hardware.

### Using Auto-Selection (Recommended)

```python
from NLSE import NLSE

# Automatically selects the fastest backend for your hardware
simu = NLSE(
    alpha=0, power=1, window=1e-3, n2=1e-20,
    V=None, L=1e-2, NX=2048, NY=2048,
    backend="auto"  # ← Automatic selection
)
```

**What happens:**
1. First run: Benchmarks all available backends (~2-5 seconds)
2. Results cached to `~/.cache/nlse/fft_benchmark.json`
3. Subsequent runs: Uses cached results (instant)

### Manual Backend Selection

```python
# Force specific backend
simu = NLSE(..., backend="CPU")   # Force CPU
simu = NLSE(..., backend="CUPY")  # Force CUDA/CuPy
simu = NLSE(..., backend="CL")    # Force OpenCL
```

## Viewing Benchmark Results

```python
from NLSE.backends import benchmark
import json

# Load cached results
cache = benchmark.load_benchmark_cache()

if cache:
    print(f"Fastest backend: {cache['fastest']}")
    print(f"Grid size: {cache['grid_size']}")
    print(f"Platform: {cache['platform']['system']} {cache['platform']['processor']}")
    print("\nResults:")

    for name, data in cache['results'].items():
        if data['available']:
            print(f"  {name}: {data['time_ms']:.2f} ms (speedup: {data['speedup']:.2f}x)")
        else:
            print(f"  {name}: Not available")
```

## Advanced Usage

### Force Re-benchmark

```python
from NLSE.backends import benchmark

# Invalidate cache
benchmark.invalidate_cache()

# Next "auto" call will re-benchmark
simu = NLSE(..., backend="auto")

# Or directly:
fastest = benchmark.get_fastest_backend(grid_size=(2048, 2048), force_benchmark=True)
```

### Grid-Size Specific Benchmarking

The system benchmarks for your specific grid size:

```python
# These will use different cached results
simu_small = NLSE(..., NX=256, NY=256, backend="auto")   # Benchmarks 256×256
simu_large = NLSE(..., NX=4096, NY=4096, backend="auto") # Benchmarks 4096×4096
```

### Environment Variables

Control behavior without code changes:

```bash
# Force specific backend
export NLSE_BACKEND=CPU
python your_script.py  # Will use CPU even if backend="auto"

# Suppress auto-selection messages
export NLSE_QUIET=1
python your_script.py

# Force re-benchmark every run (for testing)
export NLSE_FORCE_BENCHMARK=1
python your_script.py
```

## Cache Location

Benchmarks are cached at:
- **Linux/macOS**: `~/.cache/nlse/fft_benchmark.json`
- **Windows**: `%LOCALAPPDATA%\nlse\fft_benchmark.json`

Cache is valid for **30 days** or until manually invalidated.

## Typical Results

### Apple Silicon (M1/M2/M3)
For 512×512 grid:
- **OpenCL**: 0.56 ms (1033× speedup) ⚡
- **CPU (FFTW)**: 578.83 ms (1.00× baseline)

### NVIDIA GPU
For 2048×2048 grid (typical):
- **CuPy**: 2-3 ms (500-1000× speedup) ⚡
- **CPU**: 1-2 seconds (baseline)

### CPU-only Systems
For 2048×2048 grid:
- **CPU**: 1-2 seconds (only option)

*Performance varies by hardware. Auto-benchmarking measures YOUR specific system.*

## API Reference

### Main Functions

#### `NLSE(..., backend="auto")`
Creates NLSE instance with automatically selected backend.

#### `benchmark.get_fastest_backend(grid_size=(2048, 2048), force_benchmark=False)`
Returns name of fastest backend for given grid size.

**Parameters:**
- `grid_size`: Tuple of (NX, NY) for benchmarking
- `force_benchmark`: Force re-benchmark even if cached

**Returns:** String ("CPU", "CUPY", or "CL")

#### `benchmark.load_benchmark_cache()`
Loads cached benchmark results.

**Returns:** Dictionary with results or None if cache invalid/missing

#### `benchmark.save_benchmark_cache(results)`
Saves benchmark results to cache.

#### `benchmark.invalidate_cache()`
Deletes cached results, forcing re-benchmark on next call.

### Environment Variables

- **`NLSE_BACKEND`**: Override backend selection ("CPU", "CUPY", "CL", or "auto")
- **`NLSE_QUIET`**: Set to "1" to suppress auto-selection messages
- **`NLSE_FORCE_BENCHMARK`**: Set to "1" to force re-benchmark every run

## Troubleshooting

### "Benchmarking failed" messages
This is normal if a backend isn't available on your system. The system will select from working backends.

### Cache seems stale
```python
from NLSE.backends import benchmark
benchmark.invalidate_cache()
```

### Want to see what's being selected
```python
import os
os.environ['NLSE_QUIET'] = '0'  # Ensure messages are shown
simu = NLSE(..., backend="auto")
```

### Grid size mismatch
If you change grid sizes frequently, each size will benchmark separately. The system will tell you:
```
Grid size mismatch (cached: (512, 512), requested: (2048, 2048)), re-benchmarking...
```

## Performance Tips

1. **Use "auto" for production**: Let the system choose the fastest option
2. **Use larger grids**: GPU backends shine with larger FFTs (>1024×1024)
3. **Smaller grids**: CPU may be competitive for <512×512 due to GPU overhead
4. **Cache warmup**: First run is slower due to benchmarking - run once before timing

## Implementation Details

### How Benchmarking Works

1. Allocates test array with your grid size
2. Builds FFT plan for each backend
3. Runs 3 warmup cycles (FFT + inverse FFT)
4. Times 5 trials and takes median (robust to outliers)
5. Computes speedup relative to CPU
6. Caches results with platform metadata

### When Does It Re-benchmark?

- Cache doesn't exist
- Cache is >30 days old
- Grid size differs from cached size
- `force_benchmark=True` specified
- `NLSE_FORCE_BENCHMARK=1` environment variable set

## Contributing

Found issues or have suggestions? Please report at:
https://github.com/Quantum-Optics-LKB/NLSE/issues
