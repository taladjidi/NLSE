[![DOI](https://joss.theoj.org/papers/10.21105/joss.06607/status.svg)](https://doi.org/10.21105/joss.06607)

# NLSE

A package to easily simulate all sorts of non linear Schrödinger equations. It uses a [split-step spectral scheme](https://en.wikipedia.org/wiki/Split-step_method) to solve the equations.

## Where to find things

This file gets you running and explains the choices you have to make on the way
in. Everything below the halfway mark is a summary; the full treatment lives in
[`mkdocs-documentation/docs/`](mkdocs-documentation/docs/) and is published at
**<https://taladjidi.github.io/NLSE/>**. Each section here links to its
page rather than repeating it — the links point at the files, so they work in
this repository whether or not you have the site open.

| I want to… | Here | In the docs |
|---|---|---|
| install it | [Installation](#installation) | [Installation](mkdocs-documentation/docs/installation.md) |
| run something | [Basic usage](#basic-usage) | [Quick Start](mkdocs-documentation/docs/quick_start.md) |
| know what equation it solves | [Physical situation](#physical-situation) | [Physics Background](mkdocs-documentation/docs/physics_problem.md) |
| pick a solver class | [Inheritance](#inheritance) | [Solvers Overview](mkdocs-documentation/docs/solvers_overview.md) |
| pick a splitting, or RK4 | [Propagation](#propagation) | [Numerical Methods](mkdocs-documentation/docs/numerical_methods.md) |
| choose a step | [The propagation step](#the-propagation-step) | [Numerical Methods](mkdocs-documentation/docs/numerical_methods.md) |
| choose a backend, or debug one | [GPU computing](#gpu-computing) | [Backends](mkdocs-documentation/docs/backends.md) |
| watch or steer a run | [Callbacks](#callbacks) | [Callbacks](mkdocs-documentation/docs/callbacks.md) |
| copy a working script | [`examples/`](examples/) | [Examples Gallery](mkdocs-documentation/docs/examples.md) |
| look up a class or method | — | [API Reference](mkdocs-documentation/docs/reference/) |
| know why the code is shaped this way | — | [Optimization log](docs/optimization-log.md) |
| contribute | [Contributing](#contributing-and-issues) | [Contributing](mkdocs-documentation/docs/contributing.md) |

To read the docs as a site rather than as files:

```bash
uv pip install ".[docs]"
mkdocs serve -f mkdocs-documentation/mkdocs.yml
```

`mkdocs build --strict` is what checks the links, and CI runs it on every push;
a push to `main` publishes the result. The build leaves warnings about unclosed
`Div`s, which come from nbconvert rendering the tutorial notebook and are not
mkdocs's own.

## Installation

First clone the repository:

```bash
git clone https://github.com/taladjidi/NLSE.git
cd NLSE
```

Then install the package. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv pip install .
```

`pip install .` works just as well if you prefer it.

Optional extras pull in the accelerated backends and the development tooling:

```bash
uv pip install ".[gpu]"      # CuPy, for Nvidia GPUs
uv pip install ".[opencl]"   # PyOpenCL + pyvkfft, for OpenCL devices
uv pip install -e ".[dev]"   # pytest, ruff, mypy
uv pip install ".[mlx]"      # MLX, for Apple silicon
uv pip install ".[docs]"     # mkdocs, to build the documentation
```

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

# A solver with no potential, to get at its coordinate grids
simu = NLSE(
    alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=backend
)
# Define the input field and the potential on that grid. The field must be
# complex: its width is what selects single or double precision.
E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2)

simu = NLSE(alpha, power, window, n2, V, L, NX=N, NY=N, Isat=Isat, backend=backend)
E = simu.out_field(E_0, L, verbose=True, plot=True, splitting="lie")
```

<!-- TODO ADD IMAGE !!! -->

## Requirements

### Supported platforms

This code has been tested on the three main platforms: Linux, MacOs and Windows. The requirements are declared in [`pyproject.toml`](pyproject.toml) at the root of the repo.

### GPU computing

For optimal speed, this code uses your GPU (graphics card). For this, you need specific libraries. For Nvidia cards, you need a [CUDA](https://developer.nvidia.com/cuda-toolkit) install. For AMD cards, you need a [ROCm](https://rocmdocs.amd.com/en/latest/) install. Of course, you need to update your graphics driver to take full advantage of these. In any case we use [CuPy](https://cupy.dev) for the Python interface to these libraries.

**The `cupy` dependency is not a required dependency in order to not break installation on platforms that do not support it !** It ships as the optional `gpu` extra: `uv pip install ".[gpu]"`.

#### Choosing a backend

`backend=` takes `"CPU"`, `"CUPY"`, `"CL"`, `"MLX"` or `"auto"`. It is a
statement about *how* a run goes, not about what it computes, so naming one
that this machine cannot provide is answered rather than refused: the solver
moves to the fastest backend that is installed and tells you which and why.
The same happens when a backend is installed but cannot serve the run — MLX
and OpenCL have no convolution, so a non-local interaction lands on CPU or
CUPY. A name that is not a backend at all is still an error, because that is a
typo and guessing at it would hide the typo.

This is what lets a script written on one machine run on another. Pass the
backend explicitly if you would rather choose it yourself, and the warning
goes away.

### The CPU transform

If no GPU backend is available the solver falls back to the CPU, where the
transform is `scipy.fft` — pocketfft, multithreaded over every core
(`workers=-1`). It needs no planning, no wisdom on disk and no configuration,
so there is nothing to tune and nothing to invalidate.

This used to be [PyFFTW](https://pyfftw.readthedocs.io/en/latest/). FFTW is
only faster where it is vectorized, and on arm64 no prebuilt `pyfftw` is:
neither the PyPI wheel nor the conda-forge build ships NEON codelets, and only
an FFTW compiled from source with NEON on would. Measured on an M3 Max, a
2048x2048 complex64 pair took 37 ms through FFTW on 16 threads against 6.5 ms
through scipy, and 1.6 ms against 0.6 at 512x512. On x86, where the wheels
are vectorized, the gap should be smaller or reversed; the transform sits
behind the backend interface, so putting FFTW back for that case is a class,
not a rewrite.

### PyVkFFT

We found out that [PyVkFFT](https://github.com/vincefn/pyvkfft/tree/master) was outperforming CuFFT for our application so the GPU implementation uses this library for optimal performance.

Other than this, the code relies on these libraries :

- `numba` : compiles the CPU kernels, and runs them across cores
- `numpy`
- `scipy` : the CPU transform, and the non-local convolution
- `matplotlib`
- `tqdm`

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
`noexcept`, so older toolkits may need more work than this.

**`cp.all` or `cp.sum` over a whole array raises `incomplete type
"__nv_fp8_e8m0"`.** `cupy-cuda13x` is the right wheel for CUDA 13, but some
builds ship bundled CCCL headers that do not compile against 13.3. Updating
usually resolves it:

```bash
uv pip install -U cupy-cuda13x
```

NLSE's own reductions are all axis-wise and are unaffected either way, so this
only bites your own analysis code.

**`import NLSE` succeeds but `NLSE.__file__` is `None`.** Versions before 2.4
wrote their cache inside the installed package. Uninstalling then left an empty
`site-packages/NLSE/` directory behind, which Python imports as an empty
namespace package. Delete it by hand:

```bash
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/NLSE"
```

The cache now lives outside the package, so this cannot recur.

## Tests

Tests are included to check functionalities and benchmark performance.
You can run all tests by using [`pytest`](https://docs.pytest.org/en/8.2.x/) at the root of the repo.
It tests every backend it finds installed.
This can take some time !

Install the dev extra first, and re-run it after pulling: the build backend
moved to hatchling, and without it the packaging tests skip rather than fail,
so a green run may be hiding them.

```bash
uv pip install -e ".[dev]"
pre-commit install          # ruff + mypy on commit, tests on push
```

`pre-commit install` is optional but recommended: it runs the same checks CI
does, locally, before the code leaves your machine.

The benchmarks can be run using [`examples/benchmarks.py`](examples/benchmarks.py) and compare a "naive" numpy implementation of the main solver loop to our solver.
We also compare for the example of the vortex precession presented in [`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/blob/master/examples/2dvortexprecession.jl) to our solver.
On a Nvidia RTX4090 GPU and Ryzen 7950X CPU, we test our solver to the following results:
![benchmarks](img/benchmarks.png)

## How does it work ?

### Physical situation

The code offers to solve a typical [non linear Schrödinger](https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation) / [Gross-Pitaevskii](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) equation of the type :
$$i\partial_{t}\psi = -\frac{1}{2}\nabla^2\psi+V\psi+g|\psi|^2\psi$$

In this particular instance, we solve in the formalism of the propagation of a pulse of light in a non linear medium.
Within the [paraxial approximation](https://en.wikipedia.org/wiki/Paraxial_approximation), the propagation equation for the field $E$ in V/m solved is:

$$
i\partial_{z}E = -\frac{1}{2k_0}\nabla_{\perp}^2 E +
\frac{D_0}{2}\partial^2_t E
-\frac{k_0}{2}\delta n(r) E - n_2 \frac{k_0}{2n}c\epsilon_0|E|^2E
$$

Here, the constants are defined as followed :

- $k_0$ : is the electric field [wavenumber](https://en.wikipedia.org/wiki/Wavenumber) in $m^{-1}$
- $D_0$ : is the [group velocity dispersion](https://en.wikipedia.org/wiki/Group-velocity_dispersion) (GVD) in $s^2/m$
- $\delta n(\mathbf{r})$ : the "potential" i.e a local change in linear index of refraction. Dimensionless.
- $n_2$ : the [non linear index of refraction](https://en.wikipedia.org/wiki/Kerr_effect) in $m^2/W$.
- $n$ is the linear [index of refraction](https://en.wikipedia.org/wiki/Refractive_index). In our case 1.
- $c,\epsilon_0$ : the speed of light and electric permittivity of vacuum.

In all generality, the interaction term can be _non-local_ i.e $n_2=n_2(\mathbf{r})$.
This means usually that the response will be described as a convolution by some non-local kernel:

$$
n_2(\mathbf{r})|E|^2(\mathbf{r})=n_2\int_{\mathbb{R}^2}\mathrm{d}\mathbf{r}' K(\mathbf{r}-\mathbf{r}')|E|^2(\mathbf{r}'),
$$

where $K(\mathbf{r})$ is the non-local kernel, typically the Green function of some diffusion equation.

Please note that all of the code works with the **"God given" units** i.e **SI units** !

### The `NLSE` class

The `NLSE` class aims at providing a minimal yet functional toolbox to solve non-linear Schrödinger type equation in optics / atomic physics settings such as the propagation of light in a Kerr medium or solving the Gross Pitaevskii equation for the evolution of cold gases.

The propagation equation is:

$$
i\partial_{z}E = -\frac{1}{2k_0}\nabla_{\perp}^2 E +
-\frac{k_0}{2}\delta n(r) E - n_2 \frac{k_0}{2n}c\epsilon_0|E|^2E
$$

#### Initialization

The physical parameters listed above are defined at the instantiation of the `NLSE` class (`__init__` function).
A backend is chosen when the library is imported, but you can pick one per solver with the `backend` argument: `"CPU"`, `"CUPY"`, `"CL"`, `"MLX"`, or `"auto"` to benchmark what is installed and take the fastest. It can also be switched afterwards through the `backend` property.

#### Broadcasting

Since `numpy` / `cupy` allow for natural broadcasting of arrays of compatible size, one can leverage this in order to run parallel realizations. For instance, if we wish to propagate various initial state with the same physical parameters,
we simply have to initialize a _tensor_ of fields of dimensions `(N_real, Ny, Nx)` where `N_real` is the number of initial states we wish to propagate.

Similarly, one can broadcast over the physical parameters by setting some parameters to be tensors as well. If we wish for instance to study the effect of the variation of $n_2$, one can set the `n2` attribute to be a `(N_real, 1, 1)` tensor.
The field should then be initialized to a `(N_real, Ny, Nx)` tensor of identical fields and each slice over the first dimension will represent the same field propagated with different parameters.
Finally, one can combine broadcasting over several parameters at the same time: if we wish to do a grid search over $n_2$ and $\alpha$, one can instantiate `n2` to be a `(N_n2, 1, 1, 1)` array, `alpha` to be a `(1, N_alpha, 1, 1)` and the field
a `(N_n2, N_alpha, Ny, Nx)` array.

The take-home message is that the array shape should be compliant with `numpy` [broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Broadcasting works on every backend. CUPY and MLX pass batched parameters
into their kernels and broadcast there; CPU and OpenCL take one simulation's
values per launch and loop, so their gain over separate runs is in the FFTs
rather than in the kernels.

#### Numerical precision

The floating-point width follows the **input field**: pass a `complex64` array
for single precision, `complex128` for double, on a device that supports it.
The propagator and the potential are built to match, because the GPU kernels
read their precision from the field and then index those arrays with it.

The two are chosen separately, and the useful combinations are not the obvious
ones — `out_field`'s `splitting` argument picks how the linear and nonlinear
parts are composed, which is a different question from how wide the floats are.
The solver says so when the pair does not go together. See below.

#### Callbacks

The `out_field` functions support callbacks with the following signature `callback(self, A, z, i)` where `self` is the class instance, `A` is the field, `z` is the current position and `i` the main loop index.
For example if you want to print the step number every 100 steps, this is the callback you could write :

```python
def callback(nlse, A, z, i):
    if i % 100 == 0:
        print(i)
```

Notice that since the class instance is passed to the callback, you have access to all of the classes attributes.
Be mindful however that since the callback is running in the main solver loop, this function should not be called too often in order to not slow down the execution too much.
You can find several generic callbacks in the [`callbacks`](NLSE/callbacks.py) sublibrary.

A callback can also **change the step**, by returning a new one. Returning
nothing leaves it alone, which is what all the others do. The solver rebuilds
the linear propagator to match before taking the next step, so the two halves
of a split step always advance by the same distance:

```python
def refine_past_halfway(nlse, A, z, i, dz_fine):
    """Switch to a finer step once past the middle of the medium."""
    return dz_fine if z >= nlse.L / 2 else None
```

Make sure such a callback settles on a value rather than adjusting every step:
one that shrinks the step unconditionally will never reach the end of the
propagation. `adapt_delta_z` in the [`callbacks`](NLSE/callbacks.py)
sublibrary derives a step from the nonlinear refractive index change instead.

#### Propagation

The `out_field` method is the main function of the code that propagates the field for an arbitrary distance from an initial state `E_in` from z=0 (assumed to be the begining of the non linear medium) up to a specified distance z. This function simply works by iterating the spectral solver scheme i.e :

- (If the splitting is `"strang"`, apply the real space terms)
- Fourier transforming the field
- Applying the laplacian operator (multiplication by a constant matrix)
- Inverse Fourier transforming the field
- Applying all real space terms (potential, losses and interactions)

The `splitting` argument chooses how the two parts are composed. It is named
for the schemes rather than for a count, because there are three of them and
because "single" and "double" read as a floating-point width to everyone who
met them:

| `splitting` | error | transform pairs | use it when |
|---|---|---|---|
| `"lie"` (default) | $\mathcal{O}(\delta z)$ | 1 | the field is `complex64` |
| `"strang"` | $\mathcal{O}(\delta z^2)$ | 1 | the field is `complex64` and you want the better constant |
| `"yoshida"` | $\mathcal{O}(\delta z^4)$ | 3 | the field is `complex128` |

`"strang"` applies a half nonlinear step either side of the linear one. It
costs no extra transform in a run of them, because consecutive steps merge
their touching halves — that merge is exact only without loss and without an
absorbing potential, and the solver checks.

`"yoshida"` composes three Strang sub-steps with weights summing to one, the
middle one backwards. **In `complex64` it is a waste**: round-off accumulating
over steps sets the error there long before the splitting does, so the extra
order buys accuracy the arithmetic cannot hold. In `complex128` that floor is
gone and it dominates — on a self-focusing beam at $256^2$ it reached 1.08e-09
in 31 ms where `"strang"` needed 1202 ms for the same accuracy. The solver
warns if you ask for it in single precision, or ask for the others in double.

The backwards sub-step also means `"yoshida"` is wrong for a lossy medium,
where running a step backwards amplifies rather than decays. That warns too.

#### The propagation step

The step $\delta z$ is an argument to `out_field`, not a property of the solver: the same medium can be propagated at different steps, and a step chosen for one run should not silently apply to the next.

```python
E = simu.out_field(E_in, z)                 # step derived from the field
E = simu.out_field(E_in, z, delta_z=1e-5)   # step chosen by hand
```

Left to itself, the solver picks a step that imprints a fixed phase per step —
0.1 rad for split-step, 0.02 for RK4, whose truncation error is still falling
steeply where split-step's has flattened — measured against the energy the field
actually carries in each term: $\langle\psi|\hat{O}|\psi\rangle / \langle\psi|\psi\rangle$
for the kinetic, potential and interaction terms. That is the same quantity the
stability and accuracy limits are built from, so the default sits a fixed
distance inside them rather than at an arbitrary fraction of a length scale.

A step you pass is used as given, and lowered only if it would leave the
method's region of convergence: $\pi$ per step before split-step aliases,
$2\sqrt{2}$ before RK4 leaves its stability region. You get a warning when that
happens, naming the limit that bound.

Those are ceilings, not guarantees. They are written against the phase the
potential and the interaction imprint, on the reasoning that split-step applies
the linear part exactly — but the splitting error goes as the commutator of the
two parts, and a field carrying strong spatial frequencies of its own has a
large one at a phase per step that looks modest. If your field is turbulent or
sharply structured, check convergence against a finer step rather than trusting
the default; `adapt_delta_z_to_error` in the callbacks sublibrary will do it
from a measured error.

### Inheritance

In order to minimize duplication, all classes inherit from the main `NLSE` class according to the following graph:
![inheritance](img/inheritance_graph.png)

Each solver's propagation equation is written out in
[Solvers Overview](mkdocs-documentation/docs/solvers_overview.md#the-equation-each-solver-integrates),
alongside what it costs and what it supports:

| Class | Solves | Notes |
|---|---|---|
| [`NLSE`](mkdocs-documentation/docs/reference/nlse.md) | 2D paraxial propagation | the base every other class specialises |
| [`NLSE_1d`](mkdocs-documentation/docs/reference/nlse_1d.md) | the same in 1D | a specialization for speed, same features |
| [`NLSE_3d`](mkdocs-documentation/docs/reference/nlse_3d.md) | full 3D+1 with dispersion | space complexity goes as $N^3$; be careful |
| [`CNLSE`](mkdocs-documentation/docs/reference/cnlse.md) | two coupled fields | back-reaction of a fluid on a defect, two-component problems |
| [`CNLSE_1d`](mkdocs-documentation/docs/reference/cnlse_1d.md) | the same in 1D | |
| [`GPE`](mkdocs-documentation/docs/reference/gpe.md) | 2D Gross–Pitaevskii | atomic units: masses in kg, times in s |
| [`DDGPE`](mkdocs-documentation/docs/reference/ddgpe.md) | driven-dissipative coupled fields | built for exciton polaritons in microcavities |

Terms are turned off by leaving their parameter `None`, and the solver then
skips the corresponding evolution term rather than multiplying by zero.

## Contributing and issues

If you wish to contribute, do not hesitate to create a PR or email me (tangui.aladjidi[at]lkb.upmc.fr).
If you encounter problems with this software, you can create an issue directly on this repository.
