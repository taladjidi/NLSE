# Installation

## Basic Install

Clone the repository and install with pip:

```bash
git clone https://github.com/Quantum-Optics-LKB/NLSE.git
cd NLSE
pip install .
```

For development (includes pytest, ruff, mypy, etc.):

```bash
pip install -e ".[dev]"
```

## Requirements

- **Python**: 3.10 or later
- **Platforms**: Linux, macOS, Windows

Core dependencies (`numpy`, `scipy`, `matplotlib`, `numba`, `pyfftw`, `tqdm`) are installed automatically.

## GPU Backends

By default, NLSE uses the CPU backend (NumPy + PyFFTW). For GPU acceleration, install one or more of the following:

### CUPY (NVIDIA CUDA)

For NVIDIA GPUs with CUDA support:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
# or
pip install cupy-cuda11x   # for CUDA 11.x
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.

### OpenCL (Cross-platform GPU)

For AMD, Intel, or NVIDIA GPUs via OpenCL:

```bash
pip install pyopencl
```

You also need an OpenCL runtime installed for your GPU. On macOS, OpenCL is available by default.

### MLX (Apple Silicon)

For Apple Silicon Macs (M1/M2/M3/M4):

```bash
pip install mlx
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

## PyFFTW Notes

The CPU backend uses [PyFFTW](https://pyfftw.readthedocs.io/en/latest/) for Fast Fourier Transforms. FFT planning uses the `FFTW_MEASURE` flag, which means the first run may be slower while FFTW measures optimal plans. These plans are cached (in `fft.wisdom`) so subsequent runs are faster.

PyFFTW uses all available CPU threads by default. To limit this:

```python
import pyfftw
pyfftw.config.NUM_THREADS = 4  # use 4 threads
```

## Running Tests

```bash
pytest tests/               # run all tests
pytest tests/ -v --tb=short  # verbose with short tracebacks
```
