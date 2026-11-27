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

The benchmarks can be run using [`examples/benchmarks.py`](examples/benchmarks.py) and compare a "naive" numpy implementation of the main solver loop to our solver.
We also compare for the example of the vortex precession presented in [`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/blob/master/examples/2dvortexprecession.jl) to our solver.
On a Nvidia RTX4090 GPU and Ryzen 7950X CPU, we test our solver to the following results:
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
