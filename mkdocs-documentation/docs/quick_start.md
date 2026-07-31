# Quick Start

After [installing](installation.md) NLSE, import the solver you need and set up your simulation.

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
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, Isat=Isat)

# Define a Gaussian input field (values between 0 and 1, normalized internally)
# Complex: the field's width is what selects single or double precision.
E_in = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)

# Propagate and get the output field in V/m
E_out = simu.out_field(E_in, L, splitting="lie")
```

## 1D NLSE

```python
import numpy as np
from NLSE import NLSE_1d

N = 4096
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
L = 10e-3

simu = NLSE_1d(alpha=20, power=power, window=window, n2=n2, V=None, L=L, NX=N)

E_in = np.exp(-simu.X**2 / waist**2).astype(np.complex64)
E_out = simu.out_field(E_in, L, splitting="lie")
```

## GPE (Gross-Pitaevskii)

Solve the time-dependent GPE for a BEC in a harmonic trap:

```python
import numpy as np
from NLSE import GPE
from scipy.constants import atomic_mass

N_atoms = 1e6
g = 1e-2                     # interaction parameter (Hz*m^2)
m = 87 * atomic_mass          # Rubidium-87 mass
window = 100e-6               # computational window (m)
T = 1e-3                      # propagation time (s)

simu = GPE(gamma=0, N=N_atoms, window=window, g=g, V=None, m=m, NX=512, NY=512)

# Harmonic trap potential
omega_trap = 2 * np.pi * 50   # trap frequency (rad/s)
V = 0.5 * m * omega_trap**2 * (simu.XX**2 + simu.YY**2)
simu.V = V

# Gaussian initial state
E_in = np.exp(-(simu.XX**2 + simu.YY**2) / (10e-6) ** 2).astype(np.complex64)
E_out = simu.out_field(E_in, T, splitting="lie")
```

## Choosing a Backend

By default, NLSE selects the best available backend. You can specify one explicitly:

```python
# Use CPU backend
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, backend="CPU")

# Use CUPY (NVIDIA CUDA) backend
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, backend="CUPY")

# Use OpenCL backend
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, backend="CL")

# Use MLX backend (Apple Silicon)
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, backend="MLX")

# Auto-select fastest backend via benchmarking
simu = NLSE(alpha, power, window, n2, V=None, L=L, NX=N, NY=N, backend="auto")
```

You can also switch backend after creation:

```python
simu.backend = "CL"
```

## Choosing a splitting and a method

`splitting` says how the linear and nonlinear parts are composed. It is not the
floating-point width — that follows the dtype of the field you pass.

```python
# O(dz) error, one transform pair per step (default)
E_out = simu.out_field(E_in, L, splitting="lie")

# O(dz^2) error, and still one transform pair: consecutive steps merge
# their touching half-steps
E_out = simu.out_field(E_in, L, splitting="strang")

# O(dz^4), three transform pairs. Worth it only with a complex128 field
E_out = simu.out_field(E_in.astype(np.complex128), L, splitting="yoshida")

# RK4 instead of split-step
E_out = simu.out_field(E_in, L, method="RK4")
```

See [Numerical Methods](numerical_methods.md) for which to use when.

## Adding a Potential

```python
# Define a potential landscape
waist_V = 70e-6
V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist_V**2)
simu.V = V

E_out = simu.out_field(E_in, L, splitting="lie")
```

## Using Callbacks

Monitor or modify the simulation during propagation:

```python
from NLSE import sample

# Pre-allocate storage for sampled fields
delta_z = L / 1000          # divides L, so the run is exactly 1000 steps
n_steps = int(L / delta_z)
save_every = 100
E_samples = np.zeros((n_steps // save_every + 1, N, N), dtype=np.complex64)

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    splitting="lie",
    callback=sample,
    callback_args=(save_every, E_samples),
)
```

See the [Callbacks](callbacks.md) guide for all built-in callbacks and how to write your own.

## Next Steps

- [Solvers Overview](solvers_overview.md) -- detailed guide to each solver
- [Backends](backends.md) -- backend system and performance tuning
- [Numerical Methods](numerical_methods.md) -- algorithm details and step size control
- [Examples Gallery](examples.md) -- full example scripts
