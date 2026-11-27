# Changelog

## Unreleased

The fixes from a full audit of 4.1.0. Several change what a run returns or
what the API accepts, so the next release is a major one. The breaking changes
come first.

### Changed (breaking)

- **`GPE` takes `V` and `g` as rates, in rad/s.** They went through the
  optical conversion, with a wavenumber `GPE` only passes to satisfy the base
  constructor: `g` reached the kernels 2e-28 times too small and with the
  optical sign. `GPE` integrates
  `i ħ ∂ψ/∂t = −ħ²/(2m) ∇²ψ + ħ V ψ + ħ g |ψ|² ψ`, so `V` is in rad/s and `g`
  in rad/s·m², the unit the kinetic term and `gamma` already used; `V > 0`
  repels and `g > 0` is repulsive. Runs written for the old scaling need their
  `V` and `g` divided by ħ. A run that only passed `g` saw almost no
  interaction before and sees it now.
- **`NLSE_3d` applies the dispersion with the documented sign.** The operator
  was `−i (D0/2) Ω²`, the opposite of `i ∂E/∂z = (D0/2) ∂²E/∂t²`. Every run
  with `D0 ≠ 0` changes the sign of its chirp; the pulse width does not move.
- **`DDGPE`'s potential advances the phase by `−V t`,** as its documented
  equation asks; it gave `+V t`. The imaginary part flips with it: a positive
  imaginary part is now gain.
- **The non-local response is a product in Fourier space,**
  `1/(1 + nl_length² K²)`, on every backend and every solver, split step and
  RK4. It was a real-space convolution with a truncated, renormalised K0
  kernel, on CPU and CuPy only. The response is periodic, like the
  propagator, where the old kernel was zero-padded, so results with
  `nl_length > 0` change. `Backend.convolution` and the `nl_profile`
  attribute are gone. 100 steps at 512² on CPU: 0.40–0.45 s → 0.16 s.
- **The medium acts over exactly `L`.** The in-medium step count was rounded
  up, so the nonlinearity and the loss acted past `L` by up to one step. The
  run is now split at `L` and each part ends on a partial step. Results
  change for every run whose `L` is not a whole number of steps.
- **A complex `delta_z` raises `TypeError`, and `z < 0` raises `ValueError`.**
  Imaginary time was never implemented; its branches and the claims about it
  are removed. A backward run is a negative real `delta_z`. A zero or
  non-finite `delta_z`, also one a callback returns, and a non-finite `z`
  raise `ValueError`.
- **`evaluate_delta_n` returns the saturated Kerr index,**
  `n2 I / (1 + I / I_sat)` with `I = c ε0 |A|² / 2`. It divided `|A|²`
  (V²/m²) by `I_sat` (W/m²). Its values change wherever `I_sat` is finite.
- **Kernels nothing called are removed:** `NLSE.kernels.cpu.vortex`,
  `backend.kernels.vortex_cp` on CuPy and OpenCL, and from
  `NLSE.kernels.cupy` (alias `NLSE.kernels_gpu`) `rabi_coupling`,
  `vortex_cp`, `apply_propagator`, `square_mod`, `rk4_axpy`, `rk4_accumulate`
  and their `_fused` twins.
- **`DDGPE` drive and noise.** `F_pump_t` and `F_probe_t` hold one sample per
  asked step, taken at the step's midpoint, and a profile shorter than the
  run raises `ValueError`. The noise is a Wiener increment, `√|dt|` per step,
  so the intensity it adds per unit time no longer depends on the step.

### Fixed

- **The Rabi coupling is applied by every scheme.** `CNLSE`, `CNLSE_1d` and
  `DDGPE` applied it in the `lie` step only: `strang`, `yoshida` and RK4 ran
  as if `omega` were zero.
- **A `DDGPE` run with a potential takes the step asked.** The step estimate
  read `k V / 2` with an optical `k` and capped the step near 1e-30, so the
  run never finished.
- **`DDGPE`'s cavity saturates on its own density first,** `(I_sat2, I_sat)`;
  the pair was swapped. Results change when the two differ.
- **A backward run stays backward.** The partial last step, the step cap and
  the adaptive callbacks each turned a negative `delta_z` forward.
- **Callbacks see the step just taken,** also when it was cut to land on `L`
  or `z`, so a `DDGPE` pump integrates over the right length.
  `adapt_delta_z_to_error` scales the step the loop takes next, not a cut one.
- **RK4 bounds its step by the grid's fastest linear rate.** The limit
  averaged the dispersion and the potential over the beam, and RK4 amplifies
  round-off in any mode it is unstable for: a run left to choose its step on
  a fine grid returned NaN. Such a run is now lowered to the bound, with the
  usual warning. In the coupled solvers the bound takes the faster of the two
  potentials and adds the Rabi coupling, `omega / 2`, which no bound counted.
- **The default step follows the potential of the run.** The rate estimate
  kept the first potential's grids, so a second run on the same solver took
  the step of the first one's `V`.
- **`NLSE_3d.plot_field` draws the x–t panels from a row of the field.** It
  took a column: the wrong plane, and an `IndexError` for `NX < NY`.
- **CuPy: the captured loop waits for the work queued before it.** Without
  that a Strang bracket could read a field still being written; one test
  failed 11 isolated runs of 12 on an RTX 3050.
- **CUDA kernels are built without `--use_fast_math`.** Its `sincos` is
  accurate on [−π, π] only: the error of `exp(i V)` reached 1.4e-4 for
  `|V| ≤ 1e3` rad and is 9.3e-8 now, at no cost in step time.
- **Changing a solver's backend drops what the old one built** (propagator,
  plans, RK4 buffers, caches), and what a run precomputes is cleared when it
  ends. A propagator left behind made a later `split_step` use the previous
  run's step.
- **`adapt_delta_z` reads the solver's own nonlinear rate.** For a `GPE` it
  returned 4e23 s instead of `1/(12 g n)`.
- **The benchmark cache cannot break backend selection.** A cache of another
  shape, with a wrong-typed field, naming a backend this install lacks, or in
  a directory that cannot be made, counts as no cache. It is written beside
  itself and renamed, so a failed write keeps the previous one.
- **`out_field` raises `ValueError` for a wrong shape and `TypeError` for a
  wrong dtype;** they were asserts. A callback list without `callback_args`
  gets empty arguments for each. `DDGPE.out_field` no longer changes the
  caller's callback list or, on CPU, the input field.

### Changed

- `adapt_delta_z_to_error` is exported from the package.
- **The CPU complex64 kernels compute in float32.** They ran in double: numba
  widened the float32 field next to `dz`, `alpha`, `g` and `Isat`, which the
  solver passes as Python floats. The nonlinear kernels are 1.4x faster on an
  M3 Max and 1.6 to 2.2x on an AVX2 machine; a whole split step gains 7 to 12%
  and 3 to 32%. complex128 results are bit-identical. complex64 results move
  at the 2^-24 level; a lossless step keeps the rounding it had up to the
  default 0.1 rad per step. Figures in `docs/optimization-log.md`.
- **CUPY runs the nonlinear step inside the inverse transform.** A 2D local
  `NLSE` or `GPE` split step applied it in a pass of its own; a cuFFT store
  callback now applies it as the transform writes each element, as the
  propagator already was. A step moves 72 bytes per grid point instead of 88
  and is 1.17 to 1.19x faster at 512² to 3072² in complex64 (RTX 3050; 1.12x
  at 2560² and 3000², 0.89x at 256², unchanged in complex128). Results are
  bit-identical wherever cuFFT plans the same transform with a callback as
  without; `NLSE_FUSE_PROPAGATOR=0` now switches both callbacks off. Figures in `docs/optimization-log.md`.
- One-step `out_field` at 2048² (M3 Max): CPU 73 → 47 ms, MLX 70 → 41 ms; the
  dispersion eigenvalues are kept and the normalization writes in place. The
  per-step time is unchanged.
- The test suite keeps its benchmark cache in each test's own directory and
  no longer touches the user's.

## 4.1.0 — 2026-08-05

30 commits since 4.0.0. Nothing in the API moved, but two things change the
number a run comes back with: the step is now bounded for stability as well as
accuracy, and a coupled run whose two components have different wavelengths
was using the wrong constant for one of them.

### Fixed

- **Each coupled component now carries its own wavenumber in the cross term.**
  `CNLSE` scaled both components' cross-coupling by `k`, when the index one
  component writes becomes a phase rate for the *other* at that other's
  wavenumber. Right only while the two beams share a wavelength, which is the
  case `k2 = k` the constructor defaults to — so the suite never saw it, and
  neither did cross-backend comparison, all four backends sharing the same
  wrong constant. A two-line experiment, 780 and 795 nm say, was off by 1.9%
  in that term. Small, and not always: it moved a quench-ring amplitude by
  38%.
- **The split-step step is bounded by dispersion, for stability.** This is the
  change most likely to be noticed, because **existing runs on fine grids will
  get slower** — the bound goes as `1/dx^2`, and a run that now emits
  `exceeds split-step accuracy limit ... Reducing to` was previously at or
  near the unstable regime.

  The limit used to leave the kinetic term out, reasoning that split-step
  applies the linear part exactly in Fourier space so a purely linear problem
  is solved exactly at any step. True, and not enough: on a finite-amplitude
  background the real-space step couples modes, and the linear phase at the
  shortest resolved wave resonates with that coupling — the conditional
  instability of Weideman and Herbst, SIAM J. Numer. Anal. **23**, 485 (1986).

  It does not present as a numerical fault. The background fills with density
  fluctuations of order the density itself and thousands of phase
  singularities, which reads as the fluid going turbulent. It was found by a
  gallery example destroying its own background twice.

  The bound is 1 rad of linear phase per step at the grid's maximum, measured
  rather than assumed: on a vortex dipole in a uniform defocusing background,
  the density fluctuation after 40 nonlinear lengths runs 0.99, 0.97, 0.93,
  0.87 at 3.95, 2.63, 1.97, 1.58 rad and then 0.044 at 1.32 and below, where
  it stops moving. Note the threshold is well under pi — raising the existing
  pi ceiling would not have helped.
- **The tutorial's equations render.** `nlse_tutorial.ipynb` is rendered by
  mkdocs-jupyter through nbconvert, not through the mkdocs markdown pipeline,
  so its maths never reached `pymdownx.arithmatex`: it arrived in the page as
  literal `$...$` inside `class="jp-MarkdownCell"`. The MathJax configuration
  processed the `arithmatex` class alone and declared only the `\(...\)`
  delimiters, so it walked straight past all of it — six display and
  twenty-two inline formulas were published as their own LaTeX source, for as
  long as the page has existed and including in 4.0.0.

  `tests/test_docs.py` now pins, for each renderer the docs use, that the
  configuration processes the class that renderer emits and accepts the
  delimiters it leaves behind; the docs CI job checks the built HTML as well,
  because a source-level test cannot see a renderer changing its markup.
- **The tutorial's stored outputs were from an old version.** They still
  carried `No FFT wisdom found, starting over ...`, a message pyFFTW printed
  and which no longer exists in the package, and the pre-4.0.0 wording of the
  backend fallback. The notebook has been re-run against 4.0.0.

### Added

- **Every backend can do non-local interactions.** MLX and OpenCL had no
  convolution, and `nl_length > 0` is gated on one, so both handed those runs
  to another backend. Both now have an FFT convolution matching
  `scipy.signal.oaconvolve`'s signature, agreeing with it to 2e-5 and 3.7e-7
  respectively over 1D, 2D and 3D, full and same, square grids and not. The
  capability table reads Yes across the row.

  OpenCL needed more than the transform: PyOpenCL will neither copy nor assign
  into a non-contiguous slice, which is what zero-padding a grid into a larger
  one is, so the padding and the crop go through `clEnqueueCopyBufferRect`.
  Transform sizes are rounded to products of small radices — a 1024 grid's
  exact support is 2047, which is 23 x 89.
- **Three examples from the literature**, each reproducing published numbers
  rather than illustrating an API: Jones-Roberts solitons, where a vortex pair
  annihilates into a rarefaction pulse and recovers; spin and density modes in
  a binary fluid, where the two Bogoliubov branches are measured to 0.2% and
  cross because saturation gives them different powers of its denominator; and
  a mobile impurity shedding vortices, read in the fluid's frame rather than
  the lab's.

### Changed

- **The documentation is built with Sphinx**, not MkDocs. MkDocs 2.0 removes
  the plugin system that every plugin this site used depends on. The pages
  stayed markdown (MyST), the API reference is generated from the source by
  sphinx-autoapi, and maths now goes through one renderer for markdown and
  notebooks alike -- which is the bug above deleted rather than fixed.
- **The examples are a gallery.** Every script runs when the docs are built
  and the figure on its page is the one it produced, so a broken example fails
  the build instead of rotting quietly. Two benchmark sweeps are listed
  without being executed.

## 4.0.0 — 2026-08-01

174 commits since 3.0.0. The major number is not optional: `out_field` no
longer takes the argument it used to take, the step is chosen differently, and
a lossy run returns a different — correct — answer.

Note that 3.0.0 and every release before it reported themselves as `2.3.0`:
the version lived in two files and neither was bumped. Both are now, and a
test fails if they ever disagree again.

### Breaking

- **`out_field(precision=...)` is now `out_field(splitting=...)`**, and its
  values are `"lie"`, `"strang"` and `"yoshida"` rather than `"single"` and
  `"double"`. The old names counted nonlinear applications per step, and
  readers took them for the float width — which the *field's* dtype decides,
  separately and at the same time. There is no compatibility shim: passing
  `precision=` raises `TypeError`.
- **The step is passed to `out_field`, not set on the solver.** `delta_z` is
  a parameter now; assigning `simu.delta_z` before a run no longer does
  anything.
- **The default step changed.** Given no `delta_z`, the solver derives one
  from the field's own energy, aiming at a fixed phase per step rather than
  running just under a stability ceiling. Runs that relied on the old default
  will take a different number of steps.
- **A lossy run gives a different answer**, because the previous one was
  wrong. See below.

### Fixed

- **The lossy real-space step is solved rather than frozen.** It applied
  `exp(-alpha*s*dz + i*g*|A|^2*s*dz)` with `|A|^2` read once on entry, which
  is exact only while the step preserves `|A|^2` — true of a pure rotation,
  false with loss. Every splitting came out **first order** on a lossy
  problem, Strang and Yoshida included. On the turbulence example at 256²,
  best error at matched cost improved **69x for Strang and 7000x for
  Yoshida**. A lossless run is unchanged to the bit.
- **Constants follow the field's width, not the splitting's name.** A
  complex64 Strang or Yoshida run scaled the potential by a float64 scalar and
  handed it to a kernel reading `float*`: NaN from the first step above 128²,
  and a field 30% wrong below it.
- **`NLSE_3d` can be non-local at all.** It has accepted `nl_length` since it
  existed and could never run with one: the kernel was built transverse and
  the solver convolves over three axes. The non-locality is transverse — the
  index diffuses across the beam, not along the pulse.
- **Batched runs work with `nl_length > 0`.** Both convolutions require equal
  rank of their arguments, and a batch carries axes the shared kernel does
  not, so any such run raised out of the convolution.
- **Absorption stops at the end of the medium.** Leaving `L` used to zero the
  nonlinear coupling alone, so a beam past the medium stopped self-phase
  modulating and went on being absorbed by something it was no longer in.
  `alpha` (and `alpha2`) now stop with `n2`. GPE passes `L=0` and DDGPE sets
  `L=T`, so neither reaches the cutoff.
- DDGPE on OpenCL: the laser no longer destroys the cavity field, and
  `add_noise` adds noise.
- The built-in callbacks (`sample`, `norm`, `evaluate_delta_n`) work on every
  backend, not only the CPU one.
- A batched coupled run applies its real-space step.
- OpenCL no longer reassociates the two halves of a kernel pair apart under
  `-cl-fast-relaxed-math`, and no longer asks for a division it is allowed to
  get wrong.
- MLX runs with a potential no longer fail on numpy dtype introspection.

### Added

- **`splitting="yoshida"`**, a fourth-order composition, for fields wide
  enough to use the order. It warns on complex64, where round-off sets the
  error long before the splitting does, and on a lossy medium, where its
  backward middle sub-step amplifies.
- **An adaptive step.** `adapt_delta_z` and `adapt_delta_z_to_error` move the
  step during a run; the solver holds it inside the convergence region of the
  method and of the loss iteration.
- **Closed-form physics tests** (`tests/physics/`): Gaussian diffraction, the
  Gouy phase, the Talbot distance, Beer's law, saturated self-phase
  modulation, the identity the lossy step is derived from, and a bright
  soliton. Until now nothing compared a returned number against anything from
  outside the package.
- `benchmarks/profile_backends.py --alpha`, which measures the lossy step
  rather than only the lossless one the tool built by default.

### Performance

- **CUDA: the propagator is folded into the transform that writes the field**,
  through a cuFFT store callback. **1.19x** per step at 512², 1.15x at 2048²,
  and it composes with the CUDA graph rather than excluding it.
- **Metal: a lossless run stopped paying for the loss.** MLX evaluates both
  arms of an `mx.where`, so the solved step cost a *lossless* MLX run 1.45x
  even though its answer was unchanged. The graph is now chosen on the host.
- **Metal: the solved lossy step is a hand-written kernel.** `mx.compile` does
  not fuse the iteration — each op materialized a full array — so it went to
  `mx.fast.metal_kernel`: **0.69x** at 512², bringing a lossy step back within
  noise of a lossless one.
- **CPU: the transform is `scipy.fft`.** pyFFTW is faster only where FFTW is
  vectorized, which on arm64 no prebuilt wheel is. The nonlinear kernels use
  polynomial `sincos` and `exp` and now run at the machine's transcendental
  ceiling.
- The solved lossy step costs nothing measurable on CUDA, OpenCL or CPU.

### Documentation

- `docs/optimization-log.md` records what was tried on each backend, what was
  kept, what was rejected, and the numbers behind each verdict — including the
  rejections, so they can be challenged.
- The docs are served with their own MathJax rather than from a third party.
