# Installation

## Basic Install

Clone the repository and install with pip:

```bash
git clone https://github.com/taladjidi/NLSE.git
cd NLSE
pip install .
```

For development (includes pytest, ruff, mypy, etc.):

```bash
pip install -e ".[dev]"
```

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `gpu` | `cupy` | the CUPY backend, on NVIDIA GPUs (see below) |
| `opencl` | `pyopencl`, `pyvkfft`, `mako` | the CL backend |
| `mlx` | `mlx` | the MLX backend, on Apple silicon |
| `dev` | `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, `mypy`, `pre-commit`, `hatchling` | the tests and the linters |
| `docs` | `sphinx`, `furo`, `myst-nb`, `sphinx-autoapi`, `sphinx-copybutton`, `sphinx-gallery` | building this documentation |

Combine them in one install: `pip install -e ".[dev,opencl]"`.

## Requirements

- **Python**: 3.10 or later
- **Platforms**: Linux, macOS, Windows

Core dependencies (`numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`) are installed automatically.

## GPU Backends

By default, NLSE uses the CPU backend (NumPy + Numba, with `scipy.fft` for the transform). For GPU acceleration, install one or more of the following:

### CUPY (NVIDIA CUDA)

For NVIDIA GPUs with CUDA support:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
# or
pip install cupy-cuda11x   # for CUDA 11.x
```

The `gpu` extra installs the `cupy` package instead, which PyPI ships as source
only, so it compiles against the CUDA toolkit on the machine. The prebuilt
wheels above are faster to install.

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.

### OpenCL (Cross-platform GPU)

For AMD, Intel, or NVIDIA GPUs via OpenCL:

```bash
pip install ".[opencl]"   # pyopencl, pyvkfft and mako
```

You also need an OpenCL runtime installed for your GPU. On macOS, OpenCL is available by default.

### MLX (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3/M4):

```bash
pip install ".[mlx]"
```

MLX provides native GPU acceleration on Apple hardware.

## Backend Selection

NLSE auto-detects the best available backend. The priority order is:

1. **CUPY** (if CuPy is installed and a CUDA GPU is available)
2. **MLX** (if MLX is installed on Apple Silicon)
3. **CPU** (always available)

You can override this with the `NLSE_BACKEND` environment variable:

```bash
export NLSE_BACKEND=CPU   # force CPU backend
```

See the [Backends](backends.md) page for more details.

## The CPU transform

The CPU backend transforms with `scipy.fft` (pocketfft), which needs no
planning, no wisdom file and no configuration. It uses every core by default.
To limit that, cap the thread pool scipy draws on:

```python
import scipy.fft

with scipy.fft.set_workers(4):
    E_out = simu.out_field(E_in, L)
```

## Running Tests

```bash
pytest tests/               # run all tests
pytest tests/ -v --tb=short  # verbose with short tracebacks
```
