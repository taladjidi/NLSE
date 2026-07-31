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

### Single vs Double Precision

The `precision` parameter controls the splitting scheme:

**`splitting="lie"`** (default):

- Applies the nonlinear operator once per step
- Error: $\mathcal{O}(\delta z)$
- Cost: 1 FFT pair per step
- Best for: fast exploratory runs

**`splitting="strang"`**:

- Applies a half nonlinear step before and after the linear step (Strang splitting)
- Error: $\mathcal{O}(\delta z^3)$
- Cost: 2 FFT pairs per step (roughly doubles runtime)
- Best for: accurate results, convergence studies

```python
# Single precision (faster)
E_out = simu.out_field(E_in, L, splitting="lie")

# Double precision (more accurate)
E_out = simu.out_field(E_in, L, splitting="strang")
```

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

Left to itself, the solver chooses a step that imprints a fixed phase per step
— `DEFAULT_PHASE_PER_STEP`, 0.1 rad — against the energy the field actually
carries in each term:

$$
\delta z = \frac{0.1}{\sum_\text{terms} \langle\psi|\hat{O}|\psi\rangle / \langle\psi|\psi\rangle}
$$

Those expectation values are the same quantities the stability and accuracy
limits below are built from, so the default sits a fixed distance inside them
rather than at an arbitrary fraction of a length scale. Terms that do not bind
are left out: split-step applies the linear part exactly in Fourier space, so
only the real-space terms enter its step, while RK4 approximates the whole
right-hand side and takes all of them.

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

- Start with `splitting="lie"` for quick iterations, switch to `"double"` for final results
- Use powers-of-2 grid sizes for optimal FFT performance
- If the solver warns about step size reduction, consider using a finer initial grid or reducing the field power
- For convergence studies, run with decreasing $\delta z$ and check that results stabilize
