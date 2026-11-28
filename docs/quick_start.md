# Quick Start

This page propagates a Gaussian beam through 1 cm of a Kerr medium in 2D, adds
a potential, samples the field along the way with a callback, then runs the
same beam in 1D and a condensate with the GPE. It assumes you have
[installed](installation.md) NLSE.

## 2D NLSE

Propagate a Gaussian beam through a Kerr medium:

```python
import numpy as np
from NLSE import NLSE

# Physical parameters
N = 2048                     # grid points per axis
n2 = -1.6e-9                 # nonlinear index (m^2/W)
waist = 2.23e-3              # beam waist (m)
window = 4 * waist           # computational window (m)
power = 1.05                 # optical power (W)
L = 10e-3                    # propagation distance (m)
alpha = 20                   # linear losses (m^-1)
Isat = 10e4                  # saturation intensity (W/m^2)

# Create solver
simu = NLSE(
    alpha=alpha,
    power=power,
    window=window,
    n2=n2,
    V=None,
    L=L,
    NX=N,
    NY=N,
    Isat=Isat,
)

# Define a Gaussian input field (values between 0 and 1, normalized internally)
# Complex: the field's width is what selects single or double precision.
E_in = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)

# Propagate and get the output field in V/m
E_out = simu.out_field(E_in, L, splitting="lie")
```

`E_out` is the field at the end of the medium, a `(2048, 2048)` `complex64`
array in V/m. The solver chose the step: it aims at a fixed phase per step
from the potential and the interaction, and takes a shorter one here, where the
limit on the phase diffraction adds to the finest wave the grid resolves binds.

## Adding a Potential

A potential is dimensionless and lives on the solver's grid. It is twice the
change of refractive index, `V = 2 * dn` (see
[the equation](physics_problem.md)). Assign it to `simu.V` and propagate the
same input again:

```python
# Define a potential landscape
waist_V = 70e-6
V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist_V**2)
simu.V = V

E_out = simu.out_field(E_in, L, splitting="lie")
```

`E_out` has the same shape.

## Using Callbacks

Monitor or modify the simulation during propagation:

```python
from NLSE import sample

# Pre-allocate storage for sampled fields
delta_z = L / 1000          # 1000 steps, numbered 0 to 999
save_every = 100            # keeps steps 0, 100, ..., 900
E_samples = np.zeros((1000 // save_every, N, N), dtype=np.complex64)

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    splitting="lie",
    callback=sample,
    callback_args=(save_every, E_samples),
)
```

`E_samples` now holds the field every 100 steps, a `(10, 2048, 2048)`
`complex64` array. This run does not warn: its `delta_z` is inside the limit.

See the [Callbacks](callbacks.md) guide for all built-in callbacks and how to write your own.

## 1D NLSE

```python
import numpy as np
from NLSE import NLSE_1d

N = 4096
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 376  # W/m, per unit length: gives the 2D beam above its peak intensity
L = 10e-3

simu = NLSE_1d(alpha=20, power=power, window=window, n2=n2, V=None, L=L, NX=N)

E_in = np.exp(-simu.X**2 / waist**2).astype(np.complex64)
E_out = simu.out_field(E_in, L, splitting="lie")
```

`E_out` is a `(4096,)` `complex64` array.

## GPE (Gross-Pitaevskii)

Solve the time-dependent GPE for a BEC in a harmonic trap. `V` and `g` are
energies divided by ħ, in rad/s and rad/s·m²:

```python
import numpy as np
from NLSE import GPE
from scipy.constants import atomic_mass, hbar

N_atoms = 1e6
g = 1e-11                    # interaction energy / hbar (rad/s*m^2)
m = 87 * atomic_mass          # Rubidium-87 mass
window = 100e-6               # computational window (m)
T = 1e-3                      # propagation time (s)

simu = GPE(gamma=0, N=N_atoms, window=window, g=g, V=None, m=m, NX=512, NY=512)

# Harmonic trap, as a potential energy divided by hbar (rad/s)
omega_trap = 2 * np.pi * 50   # trap frequency (rad/s)
V = 0.5 * m * omega_trap**2 * (simu.XX**2 + simu.YY**2) / hbar
simu.V = V

# Gaussian initial state
E_in = np.exp(-(simu.XX**2 + simu.YY**2) / (10e-6) ** 2).astype(np.complex64)
E_out = simu.out_field(E_in, T, splitting="lie")
```

`E_out` is the condensate after 1 ms, a `(512, 512)` `complex64` array.

## Next Steps

Each of these pages takes one part of this one further:

- [Solvers Overview](solvers_overview.md) -- detailed guide to each solver
- [Backends](backends.md) -- choosing a backend, and switching it after creation
- [Numerical Methods](numerical_methods.md) -- the splittings, RK4, and step size control
- [Examples Gallery](auto_examples/index) -- full example scripts
