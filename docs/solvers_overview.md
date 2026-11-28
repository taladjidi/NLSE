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
  $\nabla_\perp^2$. The beam is uniform along $y$, so `power` is per unit
  length, in W/m; `CNLSE_1d` takes it so too.
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

Its equations are Eq. (1.87)-(1.88) of K. Guerrero-Feuillet's thesis,
[*Rotational superradiance in a polariton quantum fluid*](https://theses.hal.science/tel-05677012)
(Sorbonne Université, 2026, [p. 42](https://theses.hal.science/tel-05677012/file/158335_GUERRERO_FEUILLET_2026_archivage.pdf#page=42)), with a cross term `g12`
and a saturation added. Its arguments are in the thesis's units:

- from [Table 2.1](https://theses.hal.science/tel-05677012/file/158335_GUERRERO_FEUILLET_2026_archivage.pdf#page=78): `T` in ps, `window` and `nl_length` in
  µm, `gamma` in 1/ps (ħγ = 0.07 meV) and `g` in rad·µm²/ps (ħg = 0.01
  meV·µm²);
- from [Table 3.1](https://theses.hal.science/tel-05677012/file/158335_GUERRERO_FEUILLET_2026_archivage.pdf#page=120): `omega`, `omega_exc`, `omega_cav` and
  `detuning` in rad/ps, the energies it gives in meV over ħ;
- from [Eq. (1.10)-(1.11)](https://theses.hal.science/tel-05677012/file/158335_GUERRERO_FEUILLET_2026_archivage.pdf#page=15): `k_z` in 1/µm;
- derived, as the thesis does not state them: `V` in rad/ps, a rate like the
  ω's; `g12`, not in the thesis, in the unit of `g`; and `Isat` in 1/µm²,
  since $g|\psi|^2$ is a rate and so $|\psi|^2$ a density in 1/µm².

It works in the frame turning at the pump frequency $\omega_p$:

$$
\begin{split}
i\partial_t\psi_X &= \left(\omega_X - \omega_p + V + g|\psi_X|^2 s
  - g_{12}|\psi_C|^2 s - \frac{i\gamma_X}{2}s\right)\psi_X
  + \frac{\Omega}{2}\psi_C \\
i\partial_t\psi_C &= \left(\omega_C\sqrt{1 - \nabla^2/k_z^2} - \omega_p + V
  - g_{12}|\psi_X|^2 s - \frac{i\gamma_C}{2}s\right)\psi_C
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
- $g$ and $g_{12}$ are `g` and `g12`; $\gamma_X$ and $\gamma_C$ are the losses:
  `gamma` the exciton's, `gamma2` the cavity's (an alias of `alpha2`), both
  starting at the constructor's `gamma`
- $V$ is the potential, on both fields
- $s = 1/(1 + |\psi_X|^2/I_\text{sat} + |\psi_C|^2/I_\text{sat})$, 1 by default
- $F_p$ is the pump, applied by the `laser_excitation` argument of `out_field`

## Common Interface

All solvers are imported from the top-level package:

```python
from NLSE import NLSE, NLSE_1d, NLSE_3d, CNLSE, CNLSE_1d, GPE, DDGPE
```

### Creating a Solver

Each solver takes physical parameters at initialization, by keyword only: a
positional argument raises `TypeError`.

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

### Units

A plain number is in the unit the solver's `UNITS` table gives the argument:
SI for the optical solvers, SI with rates in rad/s for `GPE`, and ps and µm,
with rates in rad/ps, for `DDGPE` (see [The `DDGPE` class](#the-ddgpe-class);
its `power` is never read and has no unit). With the `units` extra (`pip
install "NLSE[units]"`), every argument in `UNITS` also takes a
[pint](https://pint.readthedocs.io) quantity of that unit's dimension:

```python
import pint
from NLSE import NLSE

ureg = pint.UnitRegistry()
simu = NLSE(
    alpha=0.2 / ureg.cm,
    power=1.05 * ureg.W,
    window=8 * ureg.mm,
    n2=-1.6e-5 * ureg.cm**2 / ureg.W,
    V=None,
    L=1 * ureg.cm,
    Isat=10 * ureg.W / ureg.cm**2,
    wvl=780 * ureg.nm,
)
assert simu.alpha == 20.0  # in 1/m, as NLSE.UNITS["alpha"] says
```

Each constructor of the chain converts each quantity to its unit and strips
it before its body runs, a subclass's first, so the base class gets plain
numbers and every attribute holds one. A quantity of the wrong dimension
raises `pint.DimensionalityError` naming the argument and its unit, and one
for an argument without a unit (`NX`, `NY`, `NZ`, `backend`, `DDGPE`'s
`power`) raises `TypeError`. An attribute assigned after construction,
`simu.alpha = 30`, is not converted: it takes a plain number in that unit.

Hz is not refused. pint counts a radian as 1, so `5 * ureg.Hz` has the
dimension of rad/s and becomes 5 rad/s, not 10π rad/s: where `UNITS` says
rad/s or rad/ps, pass the angular frequency, `2 * np.pi * f`. Without the
extra nothing imports pint, and plain numbers work as before.

### Saving and Rebuilding

`simu.to_dict()` returns the constructor arguments as the solver now runs,
read back from its attributes (`Isat` from `I_sat`, `wvl` from `wl`), with
the attributes no argument sets (the second component's `n22`, `alpha2`,
`power2` and `I_sat2`, and `k2` once it was assigned; `DDGPE`'s `g2` and
`omega_pump`) and `k`, the wavenumber in force. `NLSE.from_dict(params)`
builds a solver that propagates identically, whose `k2` follows `k` exactly
when the original's did:

```python
params = simu.to_dict()
same = NLSE.from_dict(params)
```

The values are plain numbers in the units of `UNITS`, and arrays such as `V`
are the solver's own, not copies; the `window` list is a copy. The keys no
argument has are not in `UNITS`: each holds the attribute as the run reads
it, in the unit of the first component's attribute it mirrors (`n22` as
`n2`, `alpha2` as `alpha`, `power2` as `power`, `I_sat2` as `I_sat`, `k2` as
`k`), `g2` in that of `g` and `omega_pump` in rad/ps; `k` is in rad/m, and
in kg for `GPE`, where it holds `m`.

`wvl` is `2 pi / k`, and the constructor's `k = 2 pi / wvl` can miss `k`'s
last bit (`GPE`'s `m` likewise), so `from_dict` puts the saved `k` back when
it agrees with the rebuilt one to that round-off. When it does not, as after
editing `wvl` or `m` in the dict, it raises `ValueError`: remove the `k` key
to build from the edited argument alone.
`json.dumps(params)` works when no value is an array or a float32; an infinite
`Isat` is written `Infinity`, which Python's `json` reads back but a strict
JSON parser does not. A subclass's own constructor arguments come back if it
stores each as an attribute of the same name; its `*args` and `**kwargs` are
left out, so one that hands its base's arguments on through `**kwargs`
cannot be rebuilt from its dict: `from_dict` raises `TypeError` for the
missing ones.

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

`DDGPE.out_field(E_in, t, laser_excitation, ...)` differs. The time `t`
replaces `z`, and the pump is a required third argument (`None` for
`DDGPE.laser_excitation`). There is no `method` and no `normalize`: the field
is never rescaled. `callback_args` is a list with one tuple per callback, the
pump's first. `examples/ddgpe.py` is a complete run.

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
simu = NLSE(alpha=20, power=1.05, window=8e-3, n2=-1.6e-9, V=my_potential, L=10e-3)

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
propagation, with a warning if that binds on a step you passed (see
[Numerical Methods](numerical_methods.md)).

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
