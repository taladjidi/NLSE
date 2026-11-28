[![DOI](https://joss.theoj.org/papers/10.21105/joss.06607/status.svg)](https://doi.org/10.21105/joss.06607)

# NLSE

A package to easily simulate all sorts of non linear Schrödinger equations. It uses a [split-step spectral scheme](https://en.wikipedia.org/wiki/Split-step_method) to solve the equations.

## Where to find things

The documentation is published at **<https://taladjidi.github.io/NLSE/>**. It
is built from `docs/`, and the links below open those files, so they work in
this repository too.
[Installation](docs/installation.md) covers the backends and their extras, and
the [Quick Start](docs/quick_start.md) runs a first propagation.
[Solvers Overview](docs/solvers_overview.md) gives the equation each solver
integrates, [Numerical Methods](docs/numerical_methods.md) the splitting and the
step, and [Backends](docs/backends.md) which device runs it. The
[Examples Gallery](https://taladjidi.github.io/NLSE/auto_examples/) has working
scripts with the figures they drew.

To read the docs as a site rather than as files:

```bash
uv sync --group docs
uv run python -m sphinx -b html docs docs/_build/html -W --keep-going
uv run python -m http.server -d docs/_build/html      # then open http://localhost:8000
```

CI runs the same build on every push, and a push to `main` publishes it. The
first build also runs the gallery of `examples/`, which is slow:
[Contributing](docs/contributing.md#building-the-documentation) says what it
runs and what later builds skip.

## Installation

First clone the repository:

```bash
git clone https://github.com/taladjidi/NLSE.git
cd NLSE
```

Then install the package into a virtual environment. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install .
```

Activate the environment before running Python: `source .venv/bin/activate`,
or `.venv\Scripts\activate` on Windows.

`pip install .` works just as well if you prefer it.

Optional extras pull in the accelerated backends:

```bash
uv pip install ".[gpu]"      # CuPy built from source, for Nvidia GPUs
uv pip install ".[opencl]"   # PyOpenCL + pyvkfft, for OpenCL devices
uv pip install ".[mlx]"      # MLX, for Apple silicon
```

For CuPy, the prebuilt wheel for your CUDA version (`cupy-cuda13x` or
`cupy-cuda12x`) saves the build: see [Installation](docs/installation.md).

To work on the package itself, follow [Contributing](docs/contributing.md#development-setup).

## Basic usage

After installing `NLSE`, you can simply import one of the solvers and instantiate your problem as follows:

```python
import numpy as np
from NLSE import NLSE

N = 2048 # number of points in solver
n2 = -1.6e-9 # nonlinear index in m^2/W
waist = 2.23e-3 # initial beam waist in m
waist2 = 70e-6 # potential beam waist in m
window = 4*waist # total computational window size in m
power = 1.05 # input optical power in W
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3 # Length of the medium in m
alpha = 20 # linear losses coefficient in m^-1
backend = "auto" # or "CPU", "CUPY" (Nvidia), "CL" (OpenCL), "MLX" (Apple)

simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend)
# The field and the potential live on the solver's grid. The field must be
# complex: its width is what selects single or double precision.
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2)
E = simu.out_field(E_0, L, verbose=True, plot=True, splitting="lie")
```

The run warns that `delta_z` exceeds the split-step accuracy limit, which is
expected: the default step is chosen from the potential and the interaction,
while the limit also bounds the phase diffraction adds to the finest wave the
grid resolves, so the solver takes the shorter step.

## Requirements

Python 3.10 or later, on Linux, macOS or Windows: CI tests all three. The
dependencies are declared in [`pyproject.toml`](pyproject.toml): `numba`,
`numpy`, `scipy`, `matplotlib` and `tqdm`. The GPU libraries are optional
extras, so that installing on a machine without a GPU does not break.

### Troubleshooting a CUDA install

These are problems we have actually hit on a recent CUDA toolchain. They are
all in the surrounding libraries rather than in NLSE, so the symptoms can be
confusing.

**`pyvkfft` fails to build against CUDA 13.3 with a recent GCC.** `nvcc`
supports host compilers only up to GCC 15, so on a system defaulting to GCC 16
you have to point it at an older one:

```bash
NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-15' uv pip install pyvkfft
```

CUDA 13.2 and later also fix a separate clash between glibc's `rsqrt` and
`noexcept` ([report](https://forums.developer.nvidia.com/t/cuda-headers-in-crt-math-functions-h-still-broken-in-debian-13-repo/362940)),
so older toolkits may need more work than this.

**`cp.all` or `cp.sum` over a whole array raises `incomplete type
"__nv_fp8_e8m0"`.** `cupy-cuda13x` is the right wheel for CUDA 13, but some
builds ship bundled CCCL headers that do not compile against 13.3. Updating
usually resolves it:

```bash
uv pip install -U cupy-cuda13x
```

NLSE sums whole arrays too, when it sizes the step, so `out_field` can hit it
as well.

**`import NLSE` succeeds but `NLSE.__file__` is `None`.** Versions before 4.0.0
wrote their cache inside the installed package. Uninstalling then left an empty
`site-packages/NLSE/` directory behind, which Python imports as an empty
namespace package. Delete it by hand:

```bash
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/NLSE"
```

The cache now lives outside the package, so this cannot recur.

## Tests

The tests run with [`pytest`](https://docs.pytest.org/) from the root of the
repository, on every backend installed, and the full suite takes a while.
[Contributing](docs/contributing.md#development-setup) sets up the environment
they need and [lists the commands](docs/contributing.md#running-tests).

## Performance

One split step (complex64, no potential, Lie splitting), timed by
[`benchmarks/roofline.py`](benchmarks/roofline.py) at the usual grid sizes
against ceilings the same script measures on the same machine. A percentage is
the time the hardware would need for the step's compulsory traffic or work at
that ceiling, over the time the step took: 100% is running at the ceiling, and
the largest names the ceiling the step is closest to (two within 15% of each
other are both named). Bytes per point is the compulsory traffic of the path
the step took. Each figure is the median of 5 runs, after a first one that is
discarded.

**Apple M3 Max (40 GPU cores)**, 2026-11-28, commit [9b685c3](https://github.com/taladjidi/NLSE/commit/9b685c3):
macOS 27.2.1 (arm64), Python 3.13.18, 16 CPU threads; numpy 2.4.6, scipy
1.18.0, numba 0.66.0, pyopencl 2026.1.2, mlx 0.32.0. Ceilings: memory 351–366
GB/s (GPU), 243–247 GB/s (CPU); FP32 FMA 9.48–9.63 TFLOP/s (GPU), 0.62–0.63
TFLOP/s (CPU); sin+cos+exp 74.2–77.7 G/s (GPU), 3.0–3.3 G/s (CPU).

| Backend | Grid | Bytes/point | Step | Memory | FP32 flops | sin+cos+exp | Closest ceiling |
|---|---|---|---|---|---|---|---|
| CPU | 1024² | 104 | 1.714 ms | 26% | 21% | 19% | memory bandwidth |
| CPU | 2048² | 104 | 5.774 ms | 31% | 28% | 23% | memory bandwidth / FP32 flops |
| CPU | 4096² | 48 | 22.034 ms | 15% | 32% | 24% | FP32 flops |
| MLX | 1024² | 104 | 0.301 ms | 99% | 8% | 5% | memory bandwidth |
| MLX | 2048² | 104 | 1.821 ms | 67% | 6% | 3% | memory bandwidth |
| MLX | 4096² | 104 | 7.049 ms | 69% | 6% | 3% | memory bandwidth |

**NVIDIA GeForce RTX 3050 + AMD Ryzen 3 4100**, 2026-11-27, commit
[67d5b6e](https://github.com/taladjidi/NLSE/commit/67d5b6e): Linux 7.3.4 (Fedora 44, x86_64), Python 3.12.16, CUDA
13.2, driver 620.70.06, 8 CPU threads; numpy 2.5.5, scipy 1.18.1, numba
0.68.1, cupy-cuda13x 14.2.0, pyopencl 2026.1.5. Ceilings: memory 208 GB/s
(GPU), 32 GB/s (CPU); FP32 FMA 8.12–8.22 TFLOP/s (GPU), 0.46 TFLOP/s (CPU);
sin+cos+exp 84.7–86.0 G/s (GPU), 0.7–1.0 G/s (CPU).

| Backend | Grid | Bytes/point | Step | Memory | FP32 flops | sin+cos+exp | Closest ceiling |
|---|---|---|---|---|---|---|---|
| CUPY | 1024² | 64 | 0.390 ms | 83% | 7% | 3% | memory bandwidth |
| CUPY | 2048² | 64 | 1.598 ms | 81% | 8% | 3% | memory bandwidth |
| CUPY | 4096² | 64 | 8.832 ms | 59% | 6% | 2% | memory bandwidth |
| CL | 1024² | 104 | 0.601 ms | 87% | 5% | 2% | memory bandwidth |
| CL | 2048² | 104 | 3.337 ms | 63% | 4% | 1% | memory bandwidth |
| CL | 4096² | 104 | 18.736 ms | 45% | 3% | 1% | memory bandwidth |
| CPU | 1024² | 48 | 8.087 ms | 20% | 6% | 17% | memory bandwidth / sin+cos+exp |
| CPU | 2048² | 48 | 30.967 ms | 21% | 7% | 17% | memory bandwidth |
| CPU | 4096² | 48 | 139.606 ms | 18% | 7% | 15% | memory bandwidth |

Strang splitting was timed in the same runs and takes the same time to within
6%: its half steps merge.

Why. A split step does 220–260 flops per grid point: two FFTs, the propagator
and the nonlinear phase. What it must move depends on how it runs:
104 B per point when each stage is its own pass, about 2.3 flops per byte.
The GPUs can do 26.6 (M3 Max) and 39.1 (RTX 3050) flops per byte they move,
far more than the step asks, so CUPY, CL and MLX are memory-bound. CuPy moves
only 64 B per point: cuFFT store callbacks carry the propagator in the forward
transform and the nonlinear step in the inverse one. At 1024², MLX is at the
bandwidth ceiling (99%) and CL near it (87%): going faster there means moving
fewer bytes, or launching less, since the fixed cost of a step (the time of
one on a 64² grid) is 18% of the CL step at that size, and between 24% and 47%
of the MLX step over the five runs (median 27%). CuPy reaches 83%, 81% and 59%
of bandwidth at 1024², 2048² and 4096², CL 63% and 45% at 2048² and 4096²,
MLX 67% and 69%: they move more than the compulsory bytes or wait on
something else. The M3 Max
CPU's tuning picks its step per grid: the three-pass blocked step (48 B per
point, about 5 flops per byte) at 4096², the plain one (104 B, about 2.3) at
1024² and 2048², against a balance of 2.6 flops per byte, and is near none of
the three ceilings (largest 32%, FP32 flops, at 4096²). The Ryzen 3 4100 CPU
takes its three-pass blocked step at every size, 48 B per point, 4.6–5.4
flops per byte, against a balance of 14.4, and is near none of the three
ceilings (largest 21%, memory bandwidth). The
[optimization log](docs/optimization-log.md#what-the-machine-allows) has the
method and its history.

Earlier figure, June 2024 (RTX 4090 and Ryzen 7950X, the solver as it was
then): [`examples/benchmarks.py`](examples/benchmarks.py) against a plain
numpy split step, and the vortex precession of
[`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/blob/master/examples/2dvortexprecession.jl).
![benchmarks](img/benchmarks.png)

## Choosing a splitting

`out_field` alternates the linear step, applied in Fourier space, with the
real-space terms (potential, losses and interactions). Its `splitting` argument
picks how the two are composed:

| `splitting` | error | transform pairs | use it when |
|---|---|---|---|
| `"lie"` (default) | $\mathcal{O}(\delta z)$ | 1 | the field is `complex64` |
| `"strang"` | $\mathcal{O}(\delta z^2)$ | 1 | the field is `complex64` and you want the better constant |
| `"yoshida"` | $\mathcal{O}(\delta z^4)$ | 3 | the field is `complex128` |

The solver warns when the splitting does not suit the field's precision. In
`complex128`, on a self-focusing beam at $256^2$, `"yoshida"` reached 1.08e-09
in 31 ms where `"strang"` needed 1202 ms for the same accuracy
([measurements](docs/optimization-log.md)). [Numerical Methods](docs/numerical_methods.md)
explains each splitting and how the step is chosen.

## Contributing and issues

Contributions are welcome: [Contributing](docs/contributing.md) sets up the
environment and lists the checks a pull request has to pass. If you encounter
problems with this software, open an issue on this repository, or email me
(tangui.aladjidi[at]lkb.upmc.fr).

NLSE is released under the [GNU General Public License v3](LICENSE).
