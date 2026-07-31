# NLSE

**NLSE** is a Python package for solving the Nonlinear Schrödinger Equation and related equations using the Split-Step Fourier method. It is designed for simulating light propagation in nonlinear media and Bose-Einstein condensate dynamics.

## Key Features

- **Multi-backend**: CPU (NumPy/PyFFTW), CUDA (CuPy), OpenCL (PyOpenCL), Apple Silicon (MLX)
- **Multiple solvers**: NLSE in 1D/2D/3D, Coupled NLSE, Gross-Pitaevskii, Driven-Dissipative GPE
- **Two numerical methods**: Split-step Fourier (single/double precision) and RK4
- **GPU-accelerated**: automatic backend selection and benchmarking
- **Extensible**: callbacks for runtime monitoring and adaptive step sizing

## Solvers

| Solver | Description |
|--------|-------------|
| [`NLSE`](reference/nlse.md) | 2D Nonlinear Schrödinger Equation (base class) |
| [`NLSE_1d`](reference/nlse_1d.md) | 1D specialization of NLSE |
| [`NLSE_3d`](reference/nlse_3d.md) | 3D+1 paraxial propagation (includes GVD) |
| [`CNLSE`](reference/cnlse.md) | 2D Coupled NLSE (two-component fields) |
| [`CNLSE_1d`](reference/cnlse_1d.md) | 1D Coupled NLSE |
| [`GPE`](reference/gpe.md) | 2D Gross-Pitaevskii Equation (atomic units) |
| [`DDGPE`](reference/ddgpe.md) | 2D Driven-Dissipative GPE (exciton-polaritons) |

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
- [API Reference](reference/nlse.md) -- full class and function documentation
