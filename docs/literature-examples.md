# Planned: examples from the literature

The gallery shows what the solvers *do*. It does not yet show what they are
*for*, which is the experiments the LKB quantum optics group runs on paraxial
fluids of light. This is the plan for closing that, one example per result,
each one a page in the gallery with the figure the script drew.

Nothing here is implemented. It is written down because the parameter regimes
are the hard part and they are not guessable — see *What went wrong on the
first attempt* at the end, which is a worked example of exactly that.

The framework these all sit in is the review, [Paraxial fluids of
light](https://arxiv.org/abs/2504.06262) (Glorieux, Piekarski, Schibler,
Aladjidi, Baker-Rasooli), which sets out the mapping from the paraxial NLSE to
a 2D+1 Gross-Pitaevskii equation.

## The budget every one of these has to fit

A gallery example is executed when the documentation builds, so it has to be
worth seconds rather than minutes. Two relations make that predictable rather
than a matter of trying:

- **Steps ≈ healing times × 10.** The solver aims at 0.1 rad of nonlinear
  phase per step, and one healing time $\xi/c_s$ *is* one radian of it, because
  $\xi/c_s \cdot k\Delta n = \Delta n/c_s^2 = 1$. So a run of 20 healing times
  is about 200 steps whatever the medium — the two scales are the same number.
- **The window has to hold the wake.** These are periodic grids. A wake that
  wraps round and re-enters the obstacle is not the physics being modelled,
  and it is what turned a first attempt into noise.

## 1. Swimming against the flow — `CNLSE`

[Swimming against a superfluid flow: self-propulsion via vortex-antivortex
shedding in a quantum fluid of light](https://arxiv.org/abs/2512.09028) —
Baker-Rasooli, Aladjidi, Ferreira, Bramati, Albert, Larré, Glorieux.

A mobile impurity in a flowing 2D superfluid of light sheds vortex-antivortex
pairs once it is above the critical velocity, and the recoil from them drives
it *upstream*.

The impurity is the second component, not a potential. That distinction is the
whole result: a static `V` cannot recoil, so it cannot swim. So

- **component 1** is the fluid: broad, defocusing, carrying a transverse
  velocity from a phase ramp;
- **component 2** is the defect: a small, **self-focusing** beam, which holds
  itself together and therefore has a finite mass;
- `n12` couples them, which is what makes the defect an obstacle at all.

What the figure should show: the vortex pairs downstream, and the defect's
position against $z$ going the other way.

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
the most *quantitative* of the four — the two speeds have closed forms, so it
belongs in `tests/physics/` as well as in the gallery, next to the checks that
are already there.

## 4. Rayleigh-Taylor fingers — `CNLSE`

Clara Piekarski's thesis, chapter 6 §2, *Towards hydrodynamical instabilities*.

Two immiscible components — $g_{12}^2 > g_{11}g_{22}$ — with the heavier
pushed into the lighter. The interface is unstable and breaks into fingers.

What the figure should show: the fingers growing along $z$, which needs the
interface seeded with noise and a window wide enough for several wavelengths
of the fastest-growing mode.

## 5. Jones-Roberts solitons — `NLSE`

[Observation of Jones-Roberts solitons in a paraxial quantum fluid of
light](https://arxiv.org/abs/2501.08383) — Baker-Rasooli, Aladjidi, Krause,
Bradley, Glorieux, Phys. Rev. Lett. **134**, 233401 (2025).

Single component, and the easiest to start with: the initial condition is a
known ansatz and the result is that it propagates without changing shape,
which is a sharp thing for a figure to show and a sharp thing to assert. Like
§3, it could be a physics test as much as an example.

## What went wrong on the first attempt

Recorded because it is the cost of guessing at a regime, and two of the three
mistakes are not specific to that example.

1. **A static Gaussian potential was used as the obstacle.** It cannot recoil,
   so the physics being reproduced was absent by construction.
2. **The phase ramp did not close on the periodic grid.** `exp(i k_x x)` with
   $k_x$ not a multiple of $2\pi/W$ leaves a phase discontinuity at the wrap,
   which seeds vortices along the whole seam: 85 000 phase singularities on a
   512² grid, none of them physical. $k_x$ has to be $2\pi m / W$.
3. **The propagation was ~30× too long**, from sizing $z$ against the window
   rather than against $\xi/c_s$. See the budget above.

With 1 and 2 fixed the obstacle behaves — the density at its centre falls to
0.006 of the mean at Mach 0.3 and fills back to 0.52 at Mach 2.0 — but no
clean shedding appeared: too short and nothing sheds, longer and the whole box
goes turbulent. That is where a dedicated session starts, and with the real
experimental parameters rather than reverse-engineered ones.
