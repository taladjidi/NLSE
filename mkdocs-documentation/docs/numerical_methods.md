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

**`precision="single"`** (default):

- Applies the nonlinear operator once per step
- Error: $\mathcal{O}(\delta z)$
- Cost: 1 FFT pair per step
- Best for: fast exploratory runs

**`precision="double"`**:

- Applies a half nonlinear step before and after the linear step (Strang splitting)
- Error: $\mathcal{O}(\delta z^3)$
- Cost: 2 FFT pairs per step (roughly doubles runtime)
- Best for: accurate results, convergence studies

```python
# Single precision (faster)
E_out = simu.out_field(E_in, L, precision="single")

# Double precision (more accurate)
E_out = simu.out_field(E_in, L, precision="double")
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

### Automatic computation

The initial step size $\delta z$ is automatically computed from the nonlinear parameters:

$$
\delta z = 5 \times 10^{-3} \cdot z_\text{NL}, \quad z_\text{NL} = \frac{1}{k_0 |\Delta n|}
$$

where $\Delta n = n_2 P / w^2$ is the characteristic nonlinear index change.

### Manual override

You can set the step size manually:

```python
simu.delta_z = 1e-5  # in meters (or seconds for GPE/DDGPE)
```

### Automatic step limit enforcement

Before propagation starts, the solver checks that $\delta z$ does not exceed stability or accuracy limits. If it does, $\delta z$ is automatically reduced to 90% of the limit and a warning is issued.

**Split-step limit**: The nonlinear phase per step must stay below $\pi$ to avoid phase aliasing:

$$
\delta z_\text{max} = \frac{\pi}{g \cdot I_\text{peak} / (1 + I_\text{peak}/I_\text{sat})}
$$

where $g$ is the effective nonlinear coefficient and $I_\text{peak}$ is the peak field intensity.

**RK4 stability limit**: The purely imaginary eigenvalues of the dispersion operator must stay within the RK4 stability region (radius $\approx 2.83$):

$$
\delta z_\text{max} = \frac{2.83}{K_\text{max}^2 / (2k_0)}
$$

where $K_\text{max}$ is the largest spatial frequency on the grid.

## Propagator Caching

The linear propagator matrix $e^{-i \frac{K_x^2 + K_y^2}{2k_0} \delta z}$ is cached by `(grid_size, delta_z, k, precision)`. If you call `out_field` multiple times with the same parameters, the propagator is reused without recomputation.

The cache is invalidated when any of these parameters change (e.g., when `adapt_delta_z` modifies the step size, or when switching precision).

## Tips

- Start with `precision="single"` for quick iterations, switch to `"double"` for final results
- Use powers-of-2 grid sizes for optimal FFT performance
- If the solver warns about step size reduction, consider using a finer initial grid or reducing the field power
- For convergence studies, run with decreasing $\delta z$ and check that results stabilize
