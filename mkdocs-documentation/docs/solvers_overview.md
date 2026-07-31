# Solvers Overview

All solvers share a common interface inherited from the base `NLSE` class. This page covers the solver hierarchy, features, and usage patterns.

## Inheritance

All solver classes inherit from the main `NLSE` class to minimize code duplication:

![inheritance](img/inheritance_graph.png)

```
NLSE (2D base class)
├── NLSE_1d          - 1D specialization
├── NLSE_3d          - 3D+1 paraxial propagation
├── CNLSE            - Coupled NLSE (two-component)
│   ├── CNLSE_1d     - 1D coupled
│   └── DDGPE        - Driven-dissipative GPE
└── GPE              - Gross-Pitaevskii equation
```

## Solver Comparison

| Solver | Dimensions | Components | Unit system | Non-locality | GVD |
|--------|-----------|------------|-------------|-------------|-----|
| `NLSE` | 2D | 1 | Optics (SI) | Yes | No |
| `NLSE_1d` | 1D | 1 | Optics (SI) | Yes | No |
| `NLSE_3d` | 3D | 1 | Optics (SI) | No | Yes |
| `CNLSE` | 2D | 2 | Optics (SI) | No | No |
| `CNLSE_1d` | 1D | 2 | Optics (SI) | No | No |
| `GPE` | 2D | 1 | Atomic (SI) | Yes | No |
| `DDGPE` | 2D | 2 | Polariton | No | No |

## Common Interface

All solvers are imported from the top-level package:

```python
from NLSE import NLSE, NLSE_1d, NLSE_3d, CNLSE, CNLSE_1d, GPE, DDGPE
```

### Creating a Solver

Each solver takes physical parameters at initialization:

```python
simu = NLSE(
    alpha=20,           # linear losses (m^-1)
    power=1.05,         # optical power (W)
    window=8e-3,        # computational window (m), can be (wx, wy) tuple
    n2=-1.6e-9,         # nonlinear index (m^2/W)
    V=None,             # potential array or None
    L=10e-3,            # propagation distance (m)
    NX=1024,            # grid points in x
    NY=1024,            # grid points in y
    Isat=np.inf,        # saturation intensity (W/m^2)
    nl_length=0,        # non-local length (m), 0 = local
    wvl=780e-9,         # wavelength (m)
    backend="CPU",      # backend name
)
```

### Propagation

The main entry point is `out_field()`:

```python
E_out = simu.out_field(
    E_in,                # input field (normalized 0-1)
    z,                   # propagation distance
    splitting="lie",  # "single" (O(dz)) or "double" (O(dz^3))
    method="split_step", # "split_step" or "RK4"
    callback=None,       # callback function(s)
    callback_args=(),    # additional callback arguments
    plot=False,          # plot results
    verbose=True,        # print progress
    normalize=True,      # normalize to total power
)
```

The returned field `E_out` is always a NumPy array on the CPU, in physical units (V/m for optical solvers).

### Coordinate Grids

After initialization, the solver provides coordinate arrays:

```python
# 2D solvers (NLSE, CNLSE, GPE, DDGPE)
simu.X       # 1D x-coordinate array
simu.Y       # 1D y-coordinate array
simu.XX      # 2D meshgrid (x)
simu.YY      # 2D meshgrid (y)
simu.delta_X # grid spacing in x
simu.delta_Y # grid spacing in y

# 1D solvers (NLSE_1d, CNLSE_1d)
simu.X       # 1D coordinate array
simu.delta_X # grid spacing

# Fourier space
simu.Kx      # x spatial frequencies
simu.Ky      # y spatial frequencies
simu.Kxx     # 2D meshgrid of frequencies (x)
simu.Kyy     # 2D meshgrid of frequencies (y)
```

### Setting the Potential

The potential can be set at initialization or later:

```python
# At initialization
simu = NLSE(..., V=my_potential)

# After initialization
simu.V = my_potential
```

For 2D solvers, the potential should have shape `(NY, NX)`. Set to `None` for free propagation.

#### Non-locality

`nl_length` models a diffusive non-locality, as a Bessel kernel convolved with
the intensity. The kernel spans `nl_length // delta_X` grid cells, so the grid
has to resolve the length you ask for: below one cell the kernel is a single
point, which is the identity. The solver warns and propagates locally in that
case rather than charging you for a convolution that does nothing.

Only CPU and CUPY have the convolution; OpenCL and MLX raise
`NotImplementedError` for a non-locality they cannot compute.

#### Complex potentials

`V` may be complex. Its real part shifts the phase, as a refractive index
change does; its **imaginary part is gain or loss**, entering the real part of
the exponent rather than the phase. That is how an absorbing boundary is built:

```python
import numpy as np
from NLSE import NLSE

waist = 2.23e-3
simu = NLSE(alpha=0, power=1.05, window=4 * waist, n2=-1.6e-9, V=None,
            L=10e-3, NX=256, NY=256, Isat=10e4)

# A ring of loss around the beam, to stop light wrapping around the window
r = np.hypot(simu.XX, simu.YY)
absorber = 1j * 2e2 * np.exp(-((r - 2e-3) ** 2) / (3e-4) ** 2)
simu.V = absorber.astype(np.complex64)

E_in = np.exp(-r**2 / waist**2).astype(np.complex64)
E_out = simu.out_field(E_in, 2e-3)
```

Positive imaginary part removes power, negative adds it. The potential is
transferred at the field's precision, so a `complex64` field carries a
`complex64` potential and a `complex128` one carries `complex128`.

### Step Size

`delta_z` is an argument to `out_field`. Left out, the solver derives one from
the field's energy; passed, it is used as given:

```python
E = simu.out_field(E_in, z, delta_z=1e-5)  # manual step size in meters
```

Either way the step is capped to the method's region of convergence before
propagation, with a warning if that binds (see [Numerical Methods](numerical_methods.md)).

## Coupled Solvers (CNLSE, CNLSE_1d, DDGPE)

Coupled solvers propagate two field components simultaneously. The input field has an extra leading dimension:

```python
import numpy as np
from NLSE import CNLSE

waist, L = 2.23e-3, 10e-3
simu = CNLSE(
    alpha=20, power=1.05, window=8e-3,
    n2=-1.6e-9, V=None, L=10e-3,
    NX=512, NY=512,
    n12=-0.5e-9,   # cross-interaction coefficient
    omega=1e3,     # Rabi coupling
)

# Input: shape (2, NY, NX)
E_in = np.zeros((2, 512, 512), dtype=np.complex64)
E_in[0] = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)  # component 1
E_in[1] = 0  # component 2 starts empty

E_out = simu.out_field(E_in, L, splitting="lie")
# E_out.shape == (2, 512, 512)
```

Setting coupling parameters to `None` disables the corresponding term for better performance.

## Broadcasting (Parallel Simulations)

You can run multiple simulations in parallel using NumPy broadcasting. This
works on **every backend**. CUPY and MLX hand the batched parameters to their
kernels and broadcast there; CPU and OpenCL take one simulation's values per
launch and loop over the batch, so their gain over separate runs is in the
FFTs rather than in the kernels.

```python
# Propagate 10 different initial states
E_in = np.random.randn(10, NY, NX) + 1j * np.random.randn(10, NY, NX)
E_out = simu.out_field(E_in, L)
# E_out.shape == (10, NY, NX)

# Scan over n2 values, same initial state for each
field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
simu.n2 = np.linspace(-2e-9, -1e-9, 5).reshape(5, 1, 1)
E_in = np.broadcast_to(field, (5, NY, NX)).copy()
E_out = simu.out_field(E_in, L)

# Grid search over two parameters
N_n2, N_alpha = 5, 3
simu.n2 = np.linspace(-2e-9, -1e-9, N_n2).reshape(N_n2, 1, 1, 1)
simu.alpha = np.linspace(0, 20, N_alpha).reshape(1, N_alpha, 1, 1)
E_in = np.broadcast_to(field, (N_n2, N_alpha, NY, NX)).copy()
E_out = simu.out_field(E_in, L)
```

Array shapes must follow NumPy [broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html).

!!! warning
    Broadcasting is only supported on the CUPY backend.

## Plotting

All solvers have a `plot_field` method for visualizing fields:

```python
simu.plot_field(E_out)
```
