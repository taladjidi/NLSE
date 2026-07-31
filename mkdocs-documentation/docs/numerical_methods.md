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
- Cost: still 1 transform pair per step in a run of them, because consecutive
  steps merge their touching halves. The merge is exact only without loss and
  without an absorbing potential, and the solver checks

**`splitting="yoshida"`**:

- Three Strang sub-steps composed, the middle one backwards
- Error: $\mathcal{O}(\delta z^4)$
- Cost: 3 transform pairs per step
- **Only worth it with a `complex128` field.** In `complex64` round-off
  accumulating over steps sets the error long before the splitting does, so
  the extra order buys accuracy the arithmetic cannot hold. Not valid with
  loss either: the backwards sub-step amplifies

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

### When to use RK4

- When you need a well-understood global error estimate
- For stiff problems where split-step operator splitting introduces errors
- When comparing against reference solutions

### Trade-offs

RK4 requires 4 FFT evaluations per step (vs 1 or 2 for split-step) but may allow larger steps for smooth fields. It also has strict stability limits (see below).

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

The two methods want different phases and no longer share a number:

- `DEFAULT_PHASE_PER_STEP` = 0.1 rad, for split-step.
- `RK4_PHASE_PER_STEP` = 0.02 rad, for RK4, whose truncation error is still
  falling steeply where split-step's has flattened.

Those expectation values are the same quantities the stability and accuracy
limits below are built from, so the default sits a fixed distance inside them
rather than at an arbitrary fraction of a length scale. Terms that do not bind
are left out: split-step applies the linear part exactly in Fourier space, so
only the real-space terms enter its step, while RK4 approximates the whole
right-hand side and takes all of them.

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

A step you pass is used as given, and reduced only if it would leave the
method's region of convergence — to 90% of the limit, with a warning naming
which limit bound. A step the solver chose itself is already well inside.

**Split-step limit**: The nonlinear phase per step must stay below $\pi$ to avoid phase aliasing:

$$
\delta z_\text{max} = \frac{\pi}{g \cdot I_\text{peak} / (1 + I_\text{peak}/I_\text{sat})}
$$

where $g$ is the effective nonlinear coefficient and the intensity is the
field-weighted mean, not the peak: the limit follows the solution rather than
the grid.

**RK4 stability limit**: The purely imaginary eigenvalues of the dispersion operator must stay within the RK4 stability region (radius $\approx 2.83$):

$$
\delta z_\text{max} = \frac{2.83}{\sum_\text{terms} \langle\psi|\hat{O}|\psi\rangle / \langle\psi|\psi\rangle}
$$

Every term counts here, dispersion included, since RK4 approximates all of
them. Counting dispersion alone made this wrong by orders of magnitude under a
potential, which is scaled by $k_0/2$.

## Propagator Caching

The linear propagator matrix $e^{-i \frac{K_x^2 + K_y^2}{2k_0} \delta z}$ is cached by `(grid_size, delta_z, k, precision)`, so calling `out_field` again with the same step reuses it.

Every step size gets its own entry, which is what makes an adaptive callback cheap: when it changes the step, the propagator is rebuilt to match before the next step is taken, and a step seen before costs nothing.

## Propagation distance

The loop takes whole steps of $\delta z$ and then covers whatever is left at
its own size, so a run lands on $z$ rather than up to a step past it. This is
not a rounding detail: the error an overshoot leaves is the phase the medium
imprints over the excess, and a step derived from the physics is an arbitrary
real number that rarely divides $z$.

## Tips

- Start with `splitting="lie"` for quick iterations, switch to `"strang"` for
  final results, and to `"yoshida"` only with a `complex128` field
- Use powers-of-2 grid sizes for optimal FFT performance
- If the solver warns about step size reduction, consider using a finer initial grid or reducing the field power
- For convergence studies, run with decreasing $\delta z$ and check that results stabilize
