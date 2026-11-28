# Installation

## Basic Install

Clone the repository and install with pip:

```bash
git clone https://github.com/taladjidi/NLSE.git
cd NLSE
pip install .
```

To work on the package itself, see [Contributing](contributing.md#development-setup).

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `gpu` | `cupy` | the CUPY backend, on NVIDIA GPUs (see below) |
| `opencl` | `pyopencl`, `pyvkfft`, `mako` | the CL backend |
| `mlx` | `mlx` | the MLX backend, on Apple silicon |
| `units` | `pint` | constructor arguments given as quantities with units ([Units](solvers_overview.md#units)) |

Combine them in one install: `pip install ".[opencl,mlx,units]"`, or in the
development environment `uv sync --extra opencl --extra mlx --extra units`.

## Requirements

- **Python**: 3.10 or later
- **Platforms**: Linux, macOS, Windows

Core dependencies (`numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`) are installed automatically.

## GPU Backends

Unless CuPy or MLX is installed, NLSE runs on the CPU backend (NumPy + Numba, with `scipy.fft` for the transform), even with the `opencl` extra: OpenCL is never the default. For GPU acceleration, install one or more of the following:

### CUPY (NVIDIA CUDA)

For NVIDIA GPUs with CUDA support:

```bash
pip install cupy-cuda13x   # for CUDA 13.x
# or
pip install cupy-cuda12x   # for CUDA 12.x
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

For Apple Silicon Macs:

```bash
pip install ".[mlx]"
```

MLX provides native GPU acceleration on Apple hardware.

## Backend Selection

NLSE picks its default backend when it is imported: CUPY if CuPy imports,
otherwise MLX if MLX imports, otherwise CPU. It checks only that the library
imports, not that it finds a device. Set `NLSE_BACKEND` to a backend name to
choose it for every solver built while it is set; it takes precedence over the
`backend=` argument. `NLSE_BACKEND=auto` is ignored, so each
solver keeps its own `backend=`:

```bash
export NLSE_BACKEND=CPU   # force CPU backend
```

A backend that is named but not installed is replaced by the fastest one that
is, with a warning. A name that is not a backend raises `ValueError`. See the
[Backends](backends.md) page for choosing one per solver and benchmarking them.
