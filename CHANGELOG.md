# Changelog

## Unreleased

### Fixed

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
