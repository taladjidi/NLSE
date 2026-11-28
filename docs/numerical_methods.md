# Numerical Methods

NLSE implements two numerical methods for solving the propagation equation: the Split-Step Fourier method and the explicit Runge-Kutta 4 (RK4) method.

## Split-Step Fourier Method

The default method. It exploits the fact that the propagation equation has a linear part (diffraction/dispersion) and a nonlinear part (Kerr effect, potential, losses) that can be applied separately in Fourier and real space.

### Algorithm

Each propagation step of size $\delta z$:

1. **Fourier transform** the field $E$
2. **Apply the linear propagator** (multiplication by $e^{-i \frac{K_x^2 + K_y^2}{2k_0} \delta z}$ in Fourier space)
3. **Inverse Fourier transform**
4. **Apply nonlinear terms** in real space (potential $V$, Kerr effect $n_2|E|^2$, losses $\alpha$, saturation $I_\text{sat}$)

### Choosing a splitting

The `splitting` parameter says how the linear and nonlinear parts are composed.
It is *not* the floating-point width — that follows the dtype of the field you
pass, and the two are chosen separately.

**`splitting="lie"`** (default):

- Applies the nonlinear operator once per step
- Error: $\mathcal{O}(\delta z)$
- Cost: 1 transform pair per step

**`splitting="strang"`**:

- A half nonlinear step either side of the linear one
- Error: $\mathcal{O}(\delta z^2)$
- Cost: 1 transform pair per step, as Lie. Consecutive steps also merge their
  touching half steps into one real-space pass, except with a callback, a
  non-local interaction, a complex potential, or in the coupled solvers
  (`CNLSE`, `CNLSE_1d`, `DDGPE`)

**`splitting="yoshida"`**:

- Three Strang sub-steps composed, the middle one backwards
- Error: $\mathcal{O}(\delta z^4)$
- Cost: 3 transform pairs per step
- **Only worth it with a `complex128` field.** In `complex64` round-off
  accumulating over steps sets the error long before the splitting does, so
  the extra order buys accuracy the arithmetic cannot hold
- Valid with loss, and fourth order there, but its middle sub-step runs
  backwards and so *amplifies* a lossy field on the way through. The answer is
  right; the intermediate field is not physical and can overflow if the loss
  per step is large. The solver warns

### Loss and the order of a splitting

A splitting composes exponentials of the two parts of the equation, and its
order is only what its real-space step deserves. That step applies

$$\exp\left[\left(-\frac{\alpha}{2} s + i g |A|^2 s\right)\delta z\right], \qquad
s = \frac{1}{1 + |A|^2/I_\text{sat}}$$

with $\alpha$ the intensity loss of the constructor. Reading $|A|^2$ once, on
entry, is the exact solution only while the step leaves $|A|^2$ alone. Loss
does not: the amplitude decays inside the step while the interaction turns the
phase. Frozen like that, the step is first order over a run, and so is every
splitting built on it.

So the solver solves that step, from two exact facts: the phase over a step is
$\frac{g}{\alpha}\left(|A|^2_\text{in} - |A|^2_\text{out}\right)$ whatever the
saturation does in between, and $|A|^2_\text{out}$ follows from
$\ln y + y/I_\text{sat}$ falling by $\alpha\,\delta z$. Each splitting keeps
its order with loss. The coupled solvers and a complex potential still freeze
the step (see [Physical validation](physical_validation.md#what-this-does-not-cover)).

A step may take out only a few percent of the intensity: $2\alpha\,\delta z$
stays under `LOSS_PER_STEP_LIMIT` = 0.05. A coarser step is lowered, like
every other step limit here: with a warning if you passed it.

```python
# complex64 field: lie is fast, strang is the better constant
E_out = simu.out_field(E_in.astype(np.complex64), L, splitting="strang")

# complex128 field: yoshida reaches a given accuracy for far less work
E_out = simu.out_field(E_in.astype(np.complex128), L, splitting="yoshida")
```

The solver warns when the pair does not go together — `"yoshida"` on a
`complex64` field, or `"lie"`/`"strang"` on a `complex128` one.

## RK4 Method

An explicit 4th-order Runge-Kutta method that integrates the full equation (linear + nonlinear) together. Selected with `method="RK4"`.

```python
E_out = simu.out_field(E_in, L, method="RK4")
```

RK4 is the one method here that is not a splitting, which makes it the
independent check on one. Each step costs 4 transform pairs, against 1 for Lie
and Strang and 3 for Yoshida, and its step is bounded by the stability limit
below, dispersion included.

## Step Size

$\delta z$ is an argument to `out_field`, not a property of the solver: the same
medium can be propagated at different steps, and a step chosen for one run does
not silently carry into the next.

### Automatic computation

Left to itself, the solver chooses a step that imprints a fixed phase per step,
against the energy the field actually carries in each term:

$$
\delta z = \frac{\phi}{\sum_\text{terms} \langle\psi|\hat{O}|\psi\rangle / \langle\psi|\psi\rangle}
$$

The two methods aim at different phases:

- `DEFAULT_PHASE_PER_STEP` = 0.1 rad, for split-step.
- `RK4_PHASE_PER_STEP` = 0.02 rad, for RK4, whose truncation error is still
  falling steeply where split-step's has flattened.

Those expectation values are the same quantities the stability and accuracy
limits below are built from, rather than an arbitrary fraction of a length
scale. Split-step applies the linear part exactly in Fourier space, so only
the real-space terms enter its default, while RK4 approximates the whole
right-hand side and takes all of them. The split-step limit does count
dispersion, so a default can come out above it, and is then lowered like any
other step. A default run takes at least `DEFAULT_MIN_STEPS` = 10 steps.

These are defaults, not guarantees. They are written against the phase the
potential and the interaction imprint, on the reasoning that split-step applies
the linear part exactly — but the splitting error goes as the commutator of the
linear and nonlinear parts, and a field carrying strong spatial frequencies of
its own has a large one at a phase per step that looks modest. A turbulent or
sharply structured field can be badly under-resolved at the default; check it
against a finer step, or drive the step from a measured error with
[`adapt_delta_z_to_error`](callbacks.md).

Weighting by the field matters. A tall potential in a corner the beam never
reaches, or a high-$K$ corner with no spectral weight, would otherwise set the
step for a run it has no effect on.

### Manual override

Pass the step you want:

```python
E = simu.out_field(E_in, z, delta_z=1e-5)  # in meters (or seconds for GPE/DDGPE)
```

### Automatic step limit enforcement

Every step is checked against the method's limit, whether you passed it or
the solver chose it. Inside the limit it is used as given. Past it, it is
lowered to `CAPPED_STEP_FRACTION` = 0.9 of the limit. A step you passed is
lowered with a warning naming which limit bound; the default is lowered
silently, since nobody chose it. On a fine grid the dispersion limit sits
below the split-step default: the first run of the
[quick start](quick_start.md) is lowered that way.

**Split-step limit**: the smallest of three steps.

- The step at which the potential and the interaction, weighted by the field,
  imprint $\pi$ of phase. The potential counts by its magnitude: one of both
  signs whose mean over the beam vanishes still turns the phase where the
  beam is.
- The step at which the shortest wave on the grid turns by
  `SPLIT_STEP_DISPERSION_PHASE_LIMIT` = 1 rad. On a vortex dipole the
  density stops converging between 1.3 and 1.6 rad (4.1.0 in the
  [changelog](changelog.md)).
- The loss ceiling above.

**RK4 stability limit**: $\lambda\,\delta z$ stays within
`RK4_STABILITY_RADIUS` = 2.83, the reach of RK4's stability region along the
imaginary axis, for $\lambda$ the sum of every rate RK4 integrates. Dispersion
and the potential count at their largest on the grid, because past the radius
RK4 amplifies whatever a mode holds and round-off puts something in every
mode. The interaction counts at its peak as well, and against
`RK4_CUBIC_RADIUS` = 1.5 rather than the linear radius: on the cubic term
alone, $\psi' = -i|\psi|^2\psi$, RK4 diverges from a phase of 1.547 rad per
step. The loss counts weighted by the field.

## Propagator Caching

The linear propagator matrix $e^{-i \frac{K_x^2 + K_y^2}{2k_0} \delta z}$ is cached by `(grid_size, delta_z, k, precision)`, so calling `out_field` again with the same step reuses it.

Every step size gets its own entry, up to `PROPAGATOR_CACHE_SIZE` = 8, the
least recently used dropped first. When an adaptive callback changes the step, the
propagator is rebuilt to match before the next step, and a step still in the
cache costs nothing.

## Propagation distance

The loop takes whole steps of $\delta z$ and then covers whatever is left at
its own size, so a run lands on $z$ rather than up to a step past it. This is
not a rounding detail: the error an overshoot leaves is the phase the medium
imprints over the excess, and a step derived from the physics is an arbitrary
real number that rarely divides $z$.
