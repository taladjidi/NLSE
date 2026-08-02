# Examples Gallery

Example scripts are available in the [`examples/`](https://github.com/taladjidi/NLSE/tree/main/examples) directory of the repository.

## Basic Solvers

| Example | Description |
|---------|-------------|
| [`nlse.py`](https://github.com/taladjidi/NLSE/tree/main/examples/nlse.py) | 2D NLSE: Gaussian beam in a Kerr medium |
| [`nlse_1d.py`](https://github.com/taladjidi/NLSE/tree/main/examples/nlse_1d.py) | 1D NLSE propagation |
| [`nlse3d.py`](https://github.com/taladjidi/NLSE/tree/main/examples/nlse3d.py) | 3D NLSE with group velocity dispersion |

## Coupled Solvers

| Example | Description |
|---------|-------------|
| [`cnlse.py`](https://github.com/taladjidi/NLSE/tree/main/examples/cnlse.py) | 2D Coupled NLSE with two-component fields |
| [`cnlse_1d.py`](https://github.com/taladjidi/NLSE/tree/main/examples/cnlse_1d.py) | 1D Coupled NLSE |

## GPE and DDGPE

| Example | Description |
|---------|-------------|
| [`gpe.py`](https://github.com/taladjidi/NLSE/tree/main/examples/gpe.py) | Gross-Pitaevskii equation for a BEC |
| [`ddgpe.py`](https://github.com/taladjidi/NLSE/tree/main/examples/ddgpe.py) | Driven-dissipative GPE for polariton systems |

## Advanced Features

| Example | Description |
|---------|-------------|
| [`callbacks.py`](https://github.com/taladjidi/NLSE/tree/main/examples/callbacks.py) | Using callbacks for field sampling and adaptive step sizing |
| [`broadcasting.py`](https://github.com/taladjidi/NLSE/tree/main/examples/broadcasting.py) | Running parallel simulations via broadcasting |
| [`nonlocality.py`](https://github.com/taladjidi/NLSE/tree/main/examples/nonlocality.py) | Non-local interactions with a diffusive kernel |
| [`vortex_precession_animation.py`](https://github.com/taladjidi/NLSE/tree/main/examples/vortex_precession_animation.py) | Vortex dynamics in a GPE simulation, written out as an animation |
| [`fig2_turbulence.py`](https://github.com/taladjidi/NLSE/tree/main/examples/fig2_turbulence.py) | Counter-streaming beams going turbulent; the lossy problem the solved real-space step was measured on |

## Benchmarks and Comparisons

| Example | Description |
|---------|-------------|
| [`benchmarks.py`](https://github.com/taladjidi/NLSE/tree/main/examples/benchmarks.py) | Performance benchmarking across backends |
| [`juliaGPE_vs_NLSE.py`](https://github.com/taladjidi/NLSE/tree/main/examples/juliaGPE_vs_NLSE.py) | Comparison with FourierGPE.jl |
| [`profile_cupy.py`](https://github.com/taladjidi/NLSE/tree/main/examples/profile_cupy.py) | CuPy profiling for GPU performance analysis |
| [`fig1_benchmarks.py`](https://github.com/taladjidi/NLSE/tree/main/examples/fig1_benchmarks.py) | The backend comparison figure, drawn from a fresh run |
| [`vortex_precession_benchmark.py`](https://github.com/taladjidi/NLSE/tree/main/examples/vortex_precession_benchmark.py) | Vortex precession timed across backends |

## Interactive Tutorial

An interactive Jupyter notebook tutorial is available:

- [`nlse_tutorial.ipynb`](https://github.com/taladjidi/NLSE/tree/main/docs/nlse_tutorial.ipynb) -- step-by-step NLSE tutorial

This notebook is also rendered in the [Tutorial](nlse_tutorial) section of this documentation.
