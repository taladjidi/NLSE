# Examples from the literature

The gallery shows what the solvers *do*. These pages are about what they are
*for*: the experiments the LKB quantum optics group runs on paraxial fluids of
light, one example per result, each one a page in the gallery with the figure
the script drew.

Four of the five are in the gallery; the analogue horizon (§2) is not. Take
the parameters from the papers rather than reconstructing them from the
physics.

The **dimensionless form of the equation** — which is how these problems are
posed in the group, and the natural way to write an example that is about a
regime rather than about one vapour cell — is set out in the Jones-Roberts
soliton paper below.

The framework these all sit in is the review, [Paraxial fluids of
light](https://arxiv.org/abs/2504.06262) (Glorieux, Piekarski, Schibler,
Aladjidi, Baker-Rasooli), which sets out the mapping from the paraxial NLSE to
a 2D+1 Gross-Pitaevskii equation.

## The budget every one of these has to fit

A gallery example is executed when the documentation builds, so it has to be
worth seconds rather than minutes. Two relations make that predictable rather
than a matter of trying:

- **Steps ≈ 10 × $L/z_{nl}$.** The nonlinear length $z_{nl} = 1/(k\Delta n)$ is
  by definition one radian of nonlinear phase, and it is the same quantity as
  $\xi/c_s$, since $\xi = 1/(kc_s)$ and $c_s^2 = \Delta n$. The solver aims at
  0.1 rad per step, so a run of 20 $z_{nl}$ is about 200 steps whatever the
  medium — the two scales are the same number, which is what makes the cost of
  one of these examples answerable before it is written.
  Sizing $z$ against the window instead made one attempt ~30× too long.
- **The window has to hold the wake.** These are periodic grids. A wake that
  wraps round and re-enters the obstacle is not the physics being modelled.
- **A phase ramp has to close on the grid.** `exp(i k_x x)` needs
  $k_x = 2\pi m / W$ on a window $W$; any other $k_x$ leaves a phase jump at
  the wrap, which seeds vortices along the seam: 85 000 phase singularities
  on a 512² grid, none of them physical.

## 1. Swimming against the flow — `CNLSE`

[Swimming against a superfluid flow: self-propulsion via vortex-antivortex
shedding in a quantum fluid of light](https://arxiv.org/abs/2512.09028) —
Baker-Rasooli, Aladjidi, Ferreira, Bramati, Albert, Larré, Glorieux.

A mobile impurity in a flowing 2D superfluid of light sheds vortex-antivortex
pairs once it is above the critical velocity, and the recoil from them drives
it *upstream*.

The impurity is the second component, not a potential. That distinction is the
whole result: a static `V` cannot recoil, so it cannot swim. A static Gaussian
potential tried in its place is an obstacle — the density at its centre falls
to 0.006 of the mean at Mach 0.3 and fills back to 0.52 at Mach 2.0 — but it
sheds no clean pairs: too short and nothing sheds, longer and the whole box
goes turbulent. So

- **component 1** is the fluid: broad, defocusing, carrying a transverse
  velocity from a phase ramp;
- **component 2** is the defect: a small, **self-focusing** beam, which holds
  itself together and therefore has a finite mass;
- `n12` couples them, which is what makes the defect an obstacle at all.

The example scans the Mach number $\beta$ and images the exit: the fluid
density with the shed pairs circled, the pair count and the drag against
$\beta$, and the impurity's velocity in the fluid's frame, read against the
$-\beta$ of an impurity that does not feel the fluid.

Built as `examples/vortex_shedding.py`.

## 2. An analogue horizon — `DDGPE`

[Polariton fluids as quantum field theory simulators on tailored curved
spacetimes](https://physics.aps.org/articles/v18/s92) — Falque, Delhom,
Glorieux, Giacobino, Bramati, Jacquet, Phys. Rev. Lett. **135**, 023401 (2025).

A polariton fluid crossing from subsonic to supersonic flow carries an
acoustic horizon, and shaping the drive shapes the spacetime it stands in.
`DDGPE` is the solver for this: it is driven and dissipative by construction.

What the figure should show: $v(x)$ and $c_s(x)$ crossing, with the horizon
marked where they meet.

One caveat belongs with this one. The coupled solvers still apply their lossy
real-space step frozen rather than solved, so they are first order in a lossy
medium — see *Open* in the [optimization log](optimization-log.md). DDGPE is
where that matters most, and an example that leans on it is also a reason to
fix it.

## 3. Spin and density modes — `CNLSE`

Spin and density modes in a binary fluid of light — Piekarski, Cherroret,
Aladjidi, Glorieux, Phys. Rev. Lett. **134**, 223403 (2025).

A two-component fluid has two Bogoliubov branches: the components moving
together, and against each other, at different speeds set by $g \pm g_{12}$.

Perturb in phase and out of phase, and measure how fast each spreads. This is
the most *quantitative* of the four: the two speeds have closed forms.

Built as `examples/spin_and_density_modes.py`.

## 4. Rayleigh-Taylor fingers — `CNLSE`

Two immiscible components — $g_{12}^2 > g_{11}g_{22}$ — with the heavier
pushed into the lighter. The interface is unstable and breaks into fingers.

Built as `examples/rayleigh_taylor.py`, without a flat interface. A flat
interface driven by an index ramp cannot be made to finger in an affordable
window: keeping the density positive caps the drive at $g<\mu/X$,
and a free beam dilutes as it expands, which together leave under two
e-foldings however large the beam. The example instead lets a harmonic trap
supply the gravity — the dense component in an outer shell falling inward
through a light core — which makes the acceleration $Cr_0$ and hands the mode
number to the circumference. Seven fingers against a predicted 8.2, growing at
11 m⁻¹ against 16.

## 5. Jones-Roberts solitons — `NLSE`

[Observation of Jones-Roberts solitons in a paraxial quantum fluid of
light](https://arxiv.org/abs/2501.08383) — Baker-Rasooli, Aladjidi, Krause,
Bradley, Glorieux, Phys. Rev. Lett. **134**, 233401 (2025).

Single component, and the easiest to start with: the initial condition is a
known ansatz and the result is that it propagates without changing shape,
which is a sharp thing for a figure to show and a sharp thing to assert.

Built as `examples/jones_roberts_solitons.py`.
