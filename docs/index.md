# NLSE

:::{note}
This site documents NLSE 4.1.0

It is built from `main`, so it describes the current code rather than the
version you have installed. Check with `python -c "import NLSE;
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

## Solvers

| Solver | Description |
|--------|-------------|
| [`NLSE`](api/NLSE/solvers/nlse/index) | 2D Nonlinear Schrödinger Equation (base class) |
| [`NLSE_1d`](api/NLSE/solvers/nlse_1d/index) | 1D specialization of NLSE |
| [`NLSE_3d`](api/NLSE/solvers/nlse_3d/index) | 3D+1 paraxial propagation (includes GVD) |
| [`CNLSE`](api/NLSE/solvers/cnlse/index) | 2D Coupled NLSE (two-component fields) |
| [`CNLSE_1d`](api/NLSE/solvers/cnlse_1d/index) | 1D Coupled NLSE |
| [`GPE`](api/NLSE/solvers/gpe/index) | 2D Gross-Pitaevskii Equation (atomic units) |
| [`DDGPE`](api/NLSE/solvers/ddgpe/index) | 2D Driven-Dissipative GPE (exciton-polaritons) |

## Quick Example

```python
import numpy as np
from NLSE import NLSE

# Physical parameters
N = 2048
n2 = -1.6e-9        # nonlinear index (m^2/W)
waist = 2.23e-3      # beam waist (m)
window = 4 * waist   # computational window (m)
power = 1.05         # optical power (W)
L = 10e-3            # propagation distance (m)

# Create solver and define input field
simu = NLSE(alpha=20, power=power, window=window, n2=n2, V=None, L=L,
            NX=N, NY=N, Isat=10e4)

# Complex: the field's width is what selects single or double precision.
E_in = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)

# Propagate
E_out = simu.out_field(E_in, L, splitting="lie")
```

## Getting Started

- [Installation](installation.md) -- install the package and optional GPU backends
- [Quick Start](quick_start.md) -- minimal working examples for each solver
- [User Guide](backends.md) -- backends, callbacks, numerical methods
- [API Reference](api/NLSE/solvers/nlse/index) -- full class and function documentation

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
