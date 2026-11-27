# NLSE

:::{note}
This site is built from `main`: NLSE 4.1.0 plus the changes listed under
Unreleased in the [Changelog](changelog.md), some of which break existing
scripts. Check the version you have with `python -c "import NLSE;
print(NLSE.__version__)"`. Upgrading from 3.0.0 changes four things — see
[Migrating to 4.0.0](migration.md).
:::
**NLSE** is a Python package for solving the Nonlinear Schrödinger Equation and related equations using the Split-Step Fourier method. It is designed for simulating light propagation in nonlinear media and Bose-Einstein condensate dynamics.

## Key Features

- **Multi-backend**: CPU (NumPy/scipy.fft), CUDA (CuPy), OpenCL (PyOpenCL), Apple Silicon (MLX)
- **Multiple solvers**: NLSE in 1D/2D/3D, Coupled NLSE, Gross-Pitaevskii, Driven-Dissipative GPE
- **Four integrators**: split-step Fourier with Lie, Strang or Yoshida splitting, and RK4
- **GPU-accelerated**: automatic backend selection, benchmarking, and a fallback
  to the fastest backend that can serve a run when the one asked for cannot
- **Extensible**: callbacks for runtime monitoring and adaptive step sizing
- **Checked against physics**: the solvers are tested against closed-form
  solutions — diffraction, Beer's law, a bright soliton — and not only
  against each other. See [Physical Validation](physical_validation.md)

## Where to start

[Installation](installation.md) installs the package and the backend for your
hardware, and the [Quick Start](quick_start.md) runs a first propagation.
The user guide explains the [equations](physics_problem.md), how to
[choose a solver](solvers_overview.md) and the
[numerical methods](numerical_methods.md). The
[examples gallery](auto_examples/index) has working scripts with the figures
they drew, and the [API reference](api/index) documents every class and
function.

```{toctree}
:hidden:
:caption: Getting Started

installation
quick_start
```

```{toctree}
:hidden:
:caption: User Guide

physics_problem
solvers_overview
backends
callbacks
numerical_methods
physical_validation
auto_examples/index
```

```{toctree}
:hidden:
:caption: Tutorial

nlse_tutorial
```

```{toctree}
:hidden:
:caption: Reference

migration
changelog
contributing
```

```{toctree}
:hidden:
:caption: Development

optimization-log
literature-examples
```
