# Physics Background

The NLSE package solves several variants of the nonlinear Schrödinger equation. This page describes the equations implemented by each solver.

## Nonlinear Schrödinger Equation (NLSE)

The code solves a typical [nonlinear Schrödinger](https://en.wikipedia.org/wiki/Nonlinear_Schr%C3%B6dinger_equation) / [Gross-Pitaevskii](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) equation of the form:

$$i\partial_{t}\psi = -\frac{1}{2}\nabla^2\psi+V\psi+g|\psi|^2\psi$$

In NLSE, we solve in the formalism of the propagation of a pulse of light in a nonlinear medium.
Within the [paraxial approximation](https://en.wikipedia.org/wiki/Paraxial_approximation), the propagation equation for the field $E$ in V/m is:

$$
i\partial_{z}E = -\frac{1}{2k_0}\nabla_{\perp}^2 E +
\frac{D_0}{2}\partial^2_t E
-\frac{k_0}{2}\delta n(r) E - n_2 \frac{k_0}{2n}c\epsilon_0|E|^2E
$$

The constants are:

- $k_0$: electric field [wavenumber](https://en.wikipedia.org/wiki/Wavenumber) in $m^{-1}$
- $D_0$: [group velocity dispersion](https://en.wikipedia.org/wiki/Group-velocity_dispersion) (GVD) in $s^2/m$
- $\delta n(\mathbf{r})$: local change in linear index of refraction (the "potential"), dimensionless
- $n_2$: [nonlinear index of refraction](https://en.wikipedia.org/wiki/Kerr_effect) in $m^2/W$
- $n$: linear [index of refraction](https://en.wikipedia.org/wiki/Refractive_index) (taken as 1)
- $c, \epsilon_0$: speed of light and electric permittivity of vacuum

The `NLSE` class solves the 2D transverse equation ($\nabla_\perp^2 = \partial_x^2 + \partial_y^2$), while `NLSE_1d` solves the 1D version and `NLSE_3d` includes the GVD term $D_0$.

All NLSE solvers use **SI units** throughout.

### Non-local interactions

The interaction term can be _non-local_, meaning $n_2 = n_2(\mathbf{r})$. The response is described as a convolution with a non-local kernel:

$$
n_2(\mathbf{r})|E|^2(\mathbf{r})=n_2\int_{\mathbb{R}^2}\mathrm{d}\mathbf{r}' K(\mathbf{r}-\mathbf{r}')|E|^2(\mathbf{r}'),
$$

where $K(\mathbf{r})$ is a non-local kernel, typically the Green function of a diffusion equation. The solvers use that one, of length $\ell$ set by the `nl_length` parameter: $K_0(r/\ell)/(2\pi\ell^2)$ in 2D and $e^{-|x|/\ell}/(2\ell)$ in 1D. The convolution is applied in Fourier space on the periodic grid, as a multiplication by $1/(1+\ell^2 k_\perp^2)$, and runs on every backend. In 3D it acts across the beam only.
## Coupled NLSE (CNLSE)

The `CNLSE` class solves a system of two coupled nonlinear Schrödinger equations:

$$
\begin{split}
i\frac{\partial\psi_1}{\partial z} &= -\frac{1}{2k_1}\nabla^2\psi_1 -\frac{1}{2}n_2^{11} k_1 c\epsilon_0|\psi_1|^2\psi_1 + k_1 n_2^{12}c\epsilon_0|\psi_2|^2\psi_1-\frac{i\alpha_1}{2}\psi_1 + \frac{\Omega}{2} \psi_2  \\
i\frac{\partial\psi_2}{\partial z} &= -\frac{1}{2k_2}\nabla^2\psi_2 -\frac{1}{2}n_2^{22} k_2 c\epsilon_0|\psi_2|^2\psi_2 + k_2 n_2^{12}c\epsilon_0|\psi_1|^2\psi_2-\frac{i\alpha_2}{2}\psi_2 + \frac{\Omega}{2} \psi_1
\end{split}
$$

This describes the coupling between two field components -- two polarizations,
two frequencies, or a fluid and a defect beam -- with:

- Self-interaction ($n_2^{11}$, $n_2^{22}$) and cross-interaction
  ($n_2^{12}$) terms, which are `n2`, `n22` and `n12` in the constructor
- Independent loss coefficients ($\alpha_1$, $\alpha_2$)
- Rabi coupling ($\Omega$)

Setting parameters to `None` disables the corresponding term for optimal performance. The `CNLSE_1d` class is the 1D specialization.

## Gross-Pitaevskii Equation (GPE)

The `GPE` class solves the 2D Gross-Pitaevskii equation for the temporal evolution of a Bosonic field:

$$
i\hbar\partial_{t}\psi = -\frac{\hbar^2}{2m}\nabla^2\psi+\hbar V\psi+\hbar g|\psi|^2\psi
$$

The GPE uses SI units, with the potential and the interaction given as energies
divided by $\hbar$:

- $m$: atom mass in kg (defaults to Rubidium-87)
- $g$: interaction energy divided by $\hbar$, in rad/s$\cdot$m$^2$; $g > 0$ is repulsive
- $V$: potential energy divided by $\hbar$, in rad/s; a harmonic trap is
  $m\omega^2 r^2 / (2\hbar)$
- $\gamma$: atom loss rate in s$^{-1}$, so that $N(t) = N(0)\,e^{-\gamma t}$
- `sat`: saturation density in atoms/m$^2$; the interaction and the loss are
  divided by $1 + |\psi|^2/\mathrm{sat}$
- $N$: total atom number

A uniform condensate of density $n$ therefore turns its phase at
$-(V + g n)$ rad/s.

The solver interface is identical to NLSE, but the physical interpretation changes: the propagation coordinate is time (not space), and the field is normalized to atom number (not optical power).

## Driven-Dissipative GPE (DDGPE)

The `DDGPE` class extends CNLSE to solve the driven-dissipative Gross-Pitaevskii equation for exciton-polariton systems. It describes two coupled fields (exciton and cavity photon) with:

- Exciton-photon Rabi coupling ($\Omega$)
- Cavity and exciton dispersion relations ($\omega_\text{cav}(\mathbf{k})$, $\omega_\text{exc}$)
- Polariton-polariton interactions ($g$)
- Cavity losses ($\gamma$)
- External laser driving (pump)
- Detuning from the lower polariton ($\delta$)

The DDGPE evolves in time and uses a rotating frame defined by the pump frequency.
