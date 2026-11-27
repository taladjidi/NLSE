# Solvers Overview

All solvers share a common interface inherited from the base `NLSE` class. This page covers the solver hierarchy, features, and usage patterns.

## Inheritance

All solver classes inherit from the main `NLSE` class to minimize code duplication:

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
| `NLSE_3d` | 3D | 1 | Optics (SI) | Transverse | Yes |
| `CNLSE` | 2D | 2 | Optics (SI) | Yes | No |
| `CNLSE_1d` | 1D | 2 | Optics (SI) | Yes | No |
| `GPE` | 2D | 1 | Atomic (SI) | Yes | No |
| `DDGPE` | 2D | 2 | Polariton | Yes | No |

## The equation each solver integrates

`NLSE` itself is the 2D paraxial equation set out in
[Physics Background](physics_problem.md); the rest specialise or extend it.

- `NLSE_1d`: the same equation with $\partial_x^2$ in place of
  $\nabla_\perp^2$.
- `NLSE_3d`: the same equation with the dispersion term $D_0$. The field holds
  every time sample of every transverse point, so memory grows with
  `NX * NY * NZ`.
- `CNLSE`, and `CNLSE_1d` in 1D: two coupled components, in
  [Coupled NLSE](physics_problem.md#coupled-nlse-cnlse).
- `GPE`: the Gross-Pitaevskii equation, in
  [Gross-Pitaevskii Equation](physics_problem.md#gross-pitaevskii-equation-gpe).
- `DDGPE`: below.

### The `DDGPE` class

The DDGPE class allows to solve the temporal evolution of two coupled fields in a driven-dissipative context.

It was designed to study problems like the evolution of exciton polaritons in microcavities.

It works in rates (rad/s), in the frame turning at the pump frequency
$\omega_p$:

$$
\begin{split}
i\partial_t\psi_X &= \left(\omega_X - \omega_p + V + g|\psi_X|^2 s
  - g_{12}|\psi_C|^2 s - \frac{i\gamma}{2}s\right)\psi_X
  + \frac{\Omega}{2}\psi_C \\
i\partial_t\psi_C &= \left(\omega_C\sqrt{1 - \nabla^2/k_z^2} - \omega_p + V
  - g_{12}|\psi_X|^2 s - \frac{i\gamma}{2}s\right)\psi_C
  + \frac{\Omega}{2}\psi_X + F_p
\end{split}
$$

where

- $\psi_X$ and $\psi_C$ are the exciton and cavity fields
- $\omega_X$, $\omega_C$ are `omega_exc` and `omega_cav`; $k_z$ is `k_z`
- $\omega_p = \omega_{LP} + \delta$, with $\delta$ the `detuning` and
  $\omega_{LP} = \frac{\omega_X + \omega_C}{2}
  - \frac{1}{2}\sqrt{(\omega_X - \omega_C)^2 + \Omega^2}$ the lower polariton
- $\Omega$ is `omega`, the Rabi coupling
- $g$ and $g_{12}$ are `g` and `g12`; $\gamma$ is the loss: `gamma` the exciton's,
  `gamma2` the cavity's (an alias of `alpha2`), both starting at the constructor's `gamma`
- $V$ is the potential, on both fields
- $s = 1/(1 + |\psi_X|^2/I_\text{sat} + |\psi_C|^2/I_\text{sat})$, 1 by default
- $F_p$ is the pump, applied by the `laser_excitation` argument of `out_field`

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
    E_in,                # input field, rescaled to `power` when normalize=True
    z,                   # propagation distance
    delta_z=None,        # step; None derives one from the field
    splitting="lie",  # "lie" O(dz), "strang" O(dz^2), "yoshida" O(dz^4)
    method="split_step", # "split_step" or "RK4"
    callback=None,       # callback function(s)
    callback_args=(),    # additional callback arguments
    plot=False,          # plot results
    verbose=True,        # print progress
    normalize=True,      # normalize to total power
)
```

The returned field `E_out` is in physical units (V/m for optical solvers). It
is a NumPy array when `E_in` was one, and stays on the device when `E_in` was a
device array.

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

`nl_length` models a diffusive non-locality: the interaction reads the
intensity smoothed by `1 / (1 + nl_length**2 * K**2)` in Fourier space, the
transform of the diffusion equation's Green function. The product is taken on
the periodic grid, so the length is not rounded to whole cells and the two
axes keep their own spacing. Every backend runs it. In 3D only the transverse
wavenumbers enter: the index diffuses across the beam, not along the pulse.

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

In `NLSE`, a positive imaginary part removes power and a negative one adds it.
`GPE` and `DDGPE` take $V$ with the opposite sign, so there a positive
imaginary part is gain. The potential is
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

`omega=None`, the default, leaves the Rabi coupling out.

## Broadcasting (Parallel Simulations)

You can run multiple simulations in parallel using NumPy broadcasting, on every
backend; see [Backend Capabilities](backends.md#backend-capabilities) for how
each runs a batch.

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

:::{note}
The batch axes come first and the grid axes last, so a scan over two
parameters on a 2D grid is a four-dimensional array. Memory is the limit:
every simulation in the batch holds its own field.
:::

## Plotting

All solvers have a `plot_field` method for visualizing fields:

```python
simu.plot_field(E_out, z)
```
