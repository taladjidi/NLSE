# Backends

NLSE supports multiple compute backends to run on different hardware. Each backend provides array operations, FFT planning and execution, and numerical kernels.

## Available Backends

| Backend | Name | Library | Hardware | FFT |
|---------|------|---------|----------|-----|
| CPU | `"CPU"` | NumPy + Numba | Any CPU | `scipy.fft` (pocketfft) |
| CUPY | `"CUPY"` | CuPy | NVIDIA GPU (CUDA) | cuFFT |
| OpenCL | `"CL"` | PyOpenCL | Any GPU/CPU with OpenCL | VkFFT |
| MLX | `"MLX"` | Apple MLX | Apple Silicon | MLX FFT |

## Backend Selection

### Auto-detection

By default, NLSE picks the best available backend at import time:

1. **CUPY** if CuPy is installed and a CUDA GPU is available
2. **MLX** if MLX is installed (Apple Silicon only)
3. **CPU** (always available)

```python
from NLSE import NLSE

# Uses the auto-detected default backend
simu = NLSE(alpha=20, power=1.0, window=8e-3, n2=-1.6e-9, V=None, L=10e-3)
print(simu.backend)  # e.g., "CUPY"
```

### Explicit selection

Specify a backend at creation:

```python
simu = NLSE(..., backend="CPU")
simu = NLSE(..., backend="CUPY")
simu = NLSE(..., backend="CL")
simu = NLSE(..., backend="MLX")
```

### Auto-benchmarking

Use `backend="auto"` to benchmark all available backends and select the fastest for your grid size:

```python
simu = NLSE(..., backend="auto")
```

Benchmark results are cached to disk (in `NLSE/.cache/fft_benchmark.json`) so subsequent runs reuse them.

### Switching at runtime

You can change the backend after creating a solver:

```python
simu.backend = "CL"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NLSE_BACKEND` | Force a specific backend (e.g., `"CPU"`, `"CUPY"`, `"CL"`, `"MLX"`) |
| `NLSE_QUIET` | Suppress backend auto-selection messages when set |
| `NLSE_FORCE_BENCHMARK` | Re-run FFT benchmarks even if cache exists |
| `NLSE_FUSE_PROPAGATOR` | Set to `0` to stop CUPY applying the propagator from a cuFFT callback |

`NLSE_FUSE_PROPAGATOR` is an escape hatch rather than a tuning knob. On CUPY
the propagator multiply is normally folded into the forward transform, which
computes the same thing in one pass fewer; setting it to `0` puts the multiply
back in a kernel of its own. It is read when an FFT plan is built, so a session
that changes it also has to call `simu._backend.clear_fft_plans()`.

Example:

```bash
export NLSE_BACKEND=CPU
export NLSE_QUIET=1
python my_simulation.py
```

## Backend Capabilities

| Feature | CPU | CUPY | CL | MLX |
|---------|-----|------|----|-----|
| Single precision | Yes | Yes | Yes | Yes |
| Double precision | Yes | Yes | Device-dependent | No |
| Non-local interactions | Yes | Yes | Falls back | Yes |
| Broadcasting | Yes | Yes | Yes | Yes |
| Convolution | Yes | Yes | No | Yes |

Every backend broadcasts, but not in the same way: CUPY and MLX hand the
batched parameters to their kernels, while CPU and OpenCL take one
simulation's values per launch and loop, so their gain is in the transforms
rather than in the kernels. "Falls back" means the solver moves the run to a
backend that can serve it and says so, rather than refusing.

## Performance Tips

### Grid sizes

Use grid sizes that are powers of 2 (e.g., 256, 512, 1024, 2048) or have low prime factors. FFT performance drops significantly with large prime factors.

### The CPU transform

`scipy.fft` (pocketfft), multithreaded over every core with `workers=-1`. It
plans per call from a cache of its own, so there is no wisdom file, no warm-up
run and nothing to invalidate. It replaced PyFFTW, which is faster only where
FFTW is vectorized and on arm64 no prebuilt build is.

### CUPY

CuPy achieves best performance through kernel fusion (`cupy.fuse`) which reduces memory bandwidth overhead. It broadcasts a batch inside its kernels, so a parameter scan costs little more than one simulation.

### OpenCL

The OpenCL backend uses native C kernels (in `NLSE/kernels/cl_source/kernels.cl`) for maximum performance. These replace PyOpenCL array expressions to avoid implicit kernel launches and temporary buffer allocations. Built with `-cl-mad-enable` and nothing else: relaxed math lets the compiler
reassociate, and a kernel that reads a potential and its generated no-V twin
are not the same expression to reassociate, so a *zero* potential changed the
result on POCL.

### MLX

The MLX backend leverages Apple Silicon's unified memory architecture. It uses lazy evaluation -- computations are deferred and batched for efficiency.
