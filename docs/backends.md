# Backends

NLSE supports multiple compute backends to run on different hardware. Each backend provides array operations, FFT planning and execution, and numerical kernels.

## Available Backends

| Backend | Name | Library | Hardware | FFT |
|---------|------|---------|----------|-----|
| CPU | `"CPU"` | NumPy + Numba | Any CPU | `scipy.fft` |
| CUPY | `"CUPY"` | CuPy | NVIDIA GPU (CUDA) | cuFFT |
| OpenCL | `"CL"` | PyOpenCL | Any GPU/CPU with OpenCL | VkFFT |
| MLX | `"MLX"` | Apple MLX | Apple Silicon | MLX FFT |

## Backend Selection

### Auto-detection

By default, NLSE picks the best available backend at import time:

1. **CUPY** if CuPy imports (no device is probed)
2. **MLX** if `mlx.core` imports (Apple silicon)
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

Use `backend="auto"` to benchmark all available backends and select the fastest for your grid size. It times a pair of transforms, not a whole step:

```python
simu = NLSE(..., backend="auto")
```

Benchmark results are cached to disk, in `fft_benchmark.json`, so subsequent runs reuse them. It keeps one record per grid size, so alternating grids do not evict each other; a file of an older version is replaced. The file goes in `NLSE_CACHE_DIR` if it is set, and otherwise in the platform's cache directory: `~/Library/Caches/NLSE` on macOS, `$XDG_CACHE_HOME/NLSE` (by default `~/.cache/NLSE`) on Linux, `%LOCALAPPDATA%\NLSE\Cache` on Windows.

### Switching at runtime

You can change the backend after creating a solver:

```python
simu.backend = "CL"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NLSE_BACKEND` | Force a specific backend (e.g., `"CPU"`, `"CUPY"`, `"CL"`, `"MLX"`) |
| `NLSE_QUIET` | Suppress NLSE's backend, benchmark and tuning messages when set to `1` |
| `NLSE_FORCE_BENCHMARK` | Re-run FFT benchmarks even if cache exists |
| `NLSE_CACHE_DIR` | Directory for the backend benchmark and the CPU step's tuning, instead of the platform's cache directory |
| `NLSE_CPU_BLOCKING` | Set to `0` to never block the CPU split step, and never measure it |
| `NLSE_FORCE_TUNING` | Set to `1` to measure the CPU blocked step again, once per grid per process, whatever the cache holds |
| `NLSE_FUSE_PROPAGATOR` | Set to `0` to stop CUPY running the propagator and the nonlinear step from cuFFT callbacks |

`NLSE_FUSE_PROPAGATOR` is an escape hatch rather than a tuning knob. On CUPY
the propagator multiply is normally folded into the forward transform, and the
nonlinear step that follows the linear one into the inverse transform, which
computes the same thing in two passes fewer; setting it to `0` puts both back
in kernels of their own. It is read when an FFT plan is built, so a session
that changes it also has to call `simu._backend.clear_fft_plans()`.

`cpu_blocking.json`, beside `fft_benchmark.json`, is
`{"version": "1", "entries": {key: entry}}`. The key is
`system/machine/cpu/NYxNX/dtype/threads/scipy-x.y/numba-x.y`, for example
`Darwin/arm64/apple-m3/4096x4096/complex64/16/scipy-1.18/numba-0.66`, where
`cpu` is LLVM's name for the processor and `threads` the fewer of the CPUs
the process may use and `NUMBA_NUM_THREADS`. An entry is
`{"rows": int | null, "threads": int | null, "unblocked_ms": float,
"blocked_ms": float | null, "timestamp": ISO 8601}`; null rows and threads
mean the usual step. Entries older than 30 days, dated in the future, of
another version, or not of that form are measured again. The blocked step
must win by 10% to be chosen. `cpu_blocking.lock` exists while a process
measures and names it (host, process, thread); one older than 5 minutes is
taken over, and the process that made it then leaves it to the new owner. A
process that finds the lock held runs the usual step and looks again every
3 s.

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
| Non-local interactions | Yes | Yes | Yes | Yes |
| Broadcasting | Yes | Yes | Yes | Yes |

Every backend broadcasts, but not in the same way: CUPY and MLX hand the
batched parameters to their kernels, while CPU and OpenCL take one
simulation's values per launch and loop, so their gain is in the transforms
rather than in the kernels. A batched `CNLSE`, `CNLSE_1d` or `DDGPE` run is
the exception: CL and MLX cannot take one, so the solver moves it to the
fastest backend that can, with a warning, and raises `NotImplementedError` if
none is installed. A `complex128` field on a backend without double precision
(MLX, an OpenCL device without fp64) moves the same way. Either way the
solver stays on the backend it moved to for its later runs.

## Threads and processes on CPU

The CPU kernels run on numba's
[threading layer](https://numba.readthedocs.io/en/stable/user/threading-layer.html),
which numba picks when NLSE is imported: TBB, then OpenMP, then `workqueue`.
Two of them limit how runs share a process:

- `workqueue` is not thread-safe: numba ends the process when two Python
  threads run CPU kernels at once. It is the layer when neither TBB nor
  OpenMP is installed, and the one NLSE takes when `pyfftw` was imported
  before it.
- GNU OpenMP, the `omp` layer on Linux, is not fork-safe: numba ends a child
  forked from a process that has run a CPU kernel when the child runs one.
  Start processes with `multiprocessing`'s `spawn` method instead.

`numba.threading_layer()` names the layer; `NUMBA_THREADING_LAYER`, set before
`import NLSE`, chooses it.

## Performance Tips

### Grid sizes

Use grid sizes that are powers of 2 (e.g., 256, 512, 1024, 2048) or have low prime factors. FFT performance drops significantly with large prime factors. On CUPY the split step's cuFFT store callbacks make a 2D step 1.36-1.38x faster at 512 to 2048 points and 1.24-1.34x faster at 1280 to 3072 (RTX 3050, complex64; `docs/optimization-log.md`, CUDA).

### The CPU transform

`scipy.fft`, multithreaded over every core with `workers=-1`: pocketfft up to
scipy 1.17 and its successor ducc from
[1.18](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html#scipy-fft-improvements).
It plans per call from a cache of its own, so there is no wisdom file. The 2D
split step built on it is measured once per grid, as below. It replaced PyFFTW,
which is faster only where FFTW is vectorized and on arm64 no prebuilt build is.

### Tuning the CPU step

On CPU, a 2D split step can run in blocks sized to the processor's caches.
Whether that is faster, and at which block size, depends on the processor,
so it is measured: the first run on a grid times the usual step against
blocked ones (0.4-0.9 s at 1024², 12.1-12.5 s at 4096² complex128 on an M3
Max, `docs/optimization-log.md`) and keeps
the answer in `cpu_blocking.json` in the cache directory. While one process
measures, the others take the usual step and do not wait.

To measure ahead of time, for example before a batch of runs:

```python
from NLSE.backends import tune_cpu_step

for n in (1024, 2048, 4096):
    print(tune_cpu_step((n, n), "complex64"))
```

Each call returns the entry it keeps: `rows` and `threads` of the blocked
step, or `None` where the usual step was faster, with both timings; or
`None` while another process is measuring.

- After a hardware or library change, the key changes and the grid is
  measured again. To measure again anyway, call
  `tune_cpu_step(shape, dtype, force=True)` or set `NLSE_FORCE_TUNING=1`.
- To never block or measure, set `NLSE_CPU_BLOCKING=0`.
- To stop runs from measuring, set
  `NLSE.backends.cpu_tuning.TUNE_ON_FIRST_RUN = False` before the first
  run; unmeasured grids then take the usual step until `tune_cpu_step`
  measures them.
- `NUMBA_NUM_THREADS` caps the threads the blocked step uses, as it caps
  numba's.
- `python benchmarks/cpu_blocked_step.py` shows what the tuning gains end
  to end on this machine.

### CUPY

CuPy achieves best performance through kernel fusion (`cupy.fuse`) which reduces memory bandwidth overhead. It broadcasts a batch inside its kernels, so a parameter scan costs little more than one simulation.

### OpenCL

The OpenCL backend uses native C kernels (in `NLSE/kernels/cl_source/kernels.cl`) for maximum performance. These replace PyOpenCL array expressions to avoid implicit kernel launches and temporary buffer allocations. Built with `-cl-mad-enable` and nothing else: relaxed math lets the compiler
reassociate, and a kernel that reads a potential and its generated no-V twin
are not the same expression to reassociate, so a *zero* potential changed the
result on POCL.

### MLX

The MLX backend leverages Apple Silicon's unified memory architecture. It uses lazy evaluation -- computations are deferred and batched for efficiency.
