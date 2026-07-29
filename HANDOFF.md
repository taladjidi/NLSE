# NLSE — session handoff

Working notes. Uncommitted by design: delete it once the remaining work
below is done or recorded elsewhere.

## Current state

**`main` is at `2d688b4`**, pushed. Branch `ci/run-on-every-branch` is 3
commits ahead, local only.

715 tests: 703 passing and 12 skipped on macOS (CPU, CL, MLX), all green on
Linux+RTX (CPU, CUPY, CL); ruff, format and mypy clean.

## This session's changes

### 1. The no-potential kernels are generated, not written out again

Each V-reading kernel had a hand-written twin taking no potential and
otherwise identical line for line: 8 such pairs in the OpenCL template, 6 in
the CUDA one. Nothing compared them, because the solvers pick between them on
whether V is None — an ordinary run exercises one or the other, never both.

The marker/macro generation written for the `_cv` twins now emits three
variants instead of two: `<name>` with no potential, `<name>_v` with a real
one, `<name>_cv` with a complex one. `V_ARG` carries the whole parameter so it
can vanish from the signature; `V_RE` became `V_PHASE`, an additive term like
`V_LOSS`, so the no-V expansion leaves nothing rather than an added zero.

The generator moved to `kernels/templating.py`, parameterised on dialect. It
imports nothing, which is what makes the CUDA half testable here.

**How the CUDA half was checked without CUDA**: preprocess both templates with
`clang -E` and diff every generated kernel against the hand-written one it
replaces. All 23 match, up to redundant parentheses where a macro vanished,
plus one re-association in `rk4_nl_rhs_c` — its no-V twin now accumulates
`p + (a*b - c*d)` rather than `(p + a*b) - c*d`, which is the form its with-V
twin always used. Since confirmed by a real compile on the RTX machine.

Kernel dictionaries are built from the template rather than listed by hand, so
host and device cannot disagree about what exists.

### 2. delta_z comes from the field's energy

It came from a constructor heuristic (`5e-3 * z_nl`, and `0.5e-2 * z_nl` in
NLSE_3d) built on `power/window^2` as a stand-in for intensity — wrong by the
ratio of window area to beam area. The energy rates added for the step limits
are that quantity done properly, but they only capped the guess.

`out_field` now derives the step from those rates unless the caller set one:
a target of `DEFAULT_PHASE_PER_STEP` (0.1 rad) per step, against the same rate
the limit for that method uses. An assigned `delta_z` is still capped to the
region of convergence.

The target is measured. Sweeping step size against a fine reference: RK4 is at
its accuracy floor (1.5e-5 relative) by 0.15 rad/step and gains nothing below;
split-step's discretisation error stays under the complex64 round-off floor
across three decades, so only the aliasing ceiling binds it. For a strong
nonlinearity that is a **7x longer step at indistinguishable error**.

It also fixed a discontinuity at `n2 = 0`: exactly zero took a fallback and
gave L/200, while 1e-20 — the same physics — used an unbounded nonlinear
length and gave 4.7e6 m, a step 5e8 times the cell.

### 3. Benchmarks take a fixed number of steps

They ran at whatever `delta_z` the solver chose, so the workload followed the
default and moved with it. Every solver benchmark now goes through
`_propagate`, which pins the step to `BENCH_STEPS` (10) steps whatever the
solver, backend or distance, and asserts the limiter did not lower it.

**Numbers from before `617e2fb` are not comparable to ones after.** These
cases were running a *single* step: `n2` is `1e-20`, so the old heuristic gave
`delta_z = 1.55e6 m` and the accuracy ceiling was too large to clamp it,
leaving `ceil(1e-3 / 1.55e6) = 1`. Every timing was dominated by fixed cost
with the stepping work invisible underneath.

Two conclusions recorded from those 1-step tables were artifacts of it:

- **CUPY leads at every grid size**, 64 included (0.965 vs CPU 1.269 vs CL
  2.018 ms). "CPU is fastest at small grids" was measuring launch overhead.
- **fp64 on CUPY costs +39%**, not the +6% recorded before (1.343 vs 0.965
  ms) — what a consumer RTX at 1/32 fp64 throughput should show.

### RTX baseline at `617e2fb`, for comparison later

| case | CPU | CUPY | CL |
|---|---|---|---|
| `nlse_2d` @64 | 1.269 | **0.965** | 2.018 ms |
| @512 | 44.98 | **12.85** | 16.70 ms |
| scaling 64→512 | 35.5x | 12.8x | 8.4x |
| `square_mod_nl_prop` kernel | 27.05 | 33.60 | 74.58 us |
| complex V over real V | +2.1% | +1.3% | -0.1% |
| batch 1→8 | 6.4x | **1.7x** | 3.4x |

The kernel row is the direct check on the generated twins: no-V, real-V and
complex-V land within 1.3% of each other on CUPY and CL, 3.4% on CPU, so the
expansion costs nothing.

### 4. Packaging tests fail instead of skipping

All 17 skipped without hatchling or uv, which reads as a pass. hatchling is a
dev dependency, so it now fails with instructions. uv is no longer required at
all: without it the backend builds the sdist and pip builds the wheel from it.

## Plan of record

Agreed order, so that later steps benefit from the earlier ones:

1. ~~**Docs**~~ — done, `4837a8f`, and `tests/test_docs.py` now keeps them honest.
2. **Ugliness ledger** — in progress: 14 of 17 cleared (one added).
   Left: the mypy override (deferred to last, by request) and the two
   in-place GPU transfer helpers.
3. **Reassess simplification.** Only after 1 and 2: clearing the ledger will
   change what is worth simplifying, so the survey is worth redoing rather
   than planning now.
4. **Performance** — the MLX regression last, since it predates this session
   and is independent of everything above.

## Remaining work

| # | Item |
|---|---|
| — | **MLX is 35% slower per step at n=512 than at release 3.0.0** (125.5 -> 170.0 us/step), and it happened *before* this session: 3.0.0 -> `5697a3f`. This session added ~4% more at n=512 and ~21% at n=256. CPU and CL are flat-to-better over the same span, so it is MLX-specific. Not investigated |

### 5. delta_z is an argument to out_field (branch `refactor/delta-z-in-out-field`)

`out_field(E_in, z, delta_z=None, ...)`, threaded as an ordinary argument
through the propagator builders, both step methods and the loop. `simu.delta_z`
is gone. **Breaking**, and it touched 41 files.

It closed a real bug: `adapt_delta_z` assigned `simu.delta_z` mid-run, the
nonlinear step picked it up but the propagator did not, so every step after
the first advanced its linear half by the wrong distance. Callbacks now
*return* a new step and the loop rebuilds the propagator. Confirmed by
reintroducing the bug against the new test.

Callbacks still need to see the step — they only receive
`(simu, A, z, i, *args)` — so the loop publishes `_current_delta_z` as
read-only run state. DDGPE's noise and laser-excitation callbacks scale by it.

Two tests were wrong in ways the tiny old default hid, most notably
`test_ddgpe::test_build_propagator`, which asserted the exciton formula for
the cavity propagator too.

### 6. Test slimming

Solver construction was 42% of the cross-backend file and 24-37% of seven
others. Each file now has a `make_solver` (or `make_nlse`/`make_cnlse`)
defaulting to its module's parameters, so a call states only what makes that
test different. **Suite 8778 -> 7643 lines**, same 573 tests; the big file went
2451 -> 1499.

Done with an `ast` rewriter, in `scratchpad/factor.py` if more is wanted. Its
one important rule: **never drop an argument whose name the enclosing function
rebinds.** Without that guard it silently gutted the broadcasting tests, which
build batched `alpha`/`n2`/`Isat` locally and pass them under those names — and
it would not have shown up locally, since those bodies sit behind
`if __CUPY_AVAILABLE__`.

Still spelled out: `test_nonlocality.py` (five different solver classes, one
construction each — a factory would not pay), and the benchmarks, where the
explicit parameters are arguably the point.

### 7. README, and the position callbacks are given (merged)

Every code block in the README was broken: `backend="GPU"` is not a backend
name and raises, the headline field was float64 where `out_field` asserts
complex, and the callback example read the removed `simu.delta_z`. All three
now run, checked by executing them out of the file. Three stale claims went
too — broadcasting being GPU-only, precision being "hardcoded at the top of
nlse.py", and a `"GPU"` backend to switch to.

**A real bug fell out of writing the adaptive-callback example.** Callbacks
were handed the *total* propagation distance, the same number every step,
while every callback docstring and the README call it "the current propagation
distance". Nothing in-tree read it — the built-ins all key off the step index —
so it had gone unnoticed since the callback API was added. The loop now
advances before dispatching. Two regression tests.

### 8. GPE and DDGPE name their own parameters (task #9, merged)

They kept a second copy of every value under its other name (`self.m = self.k`,
`self.g = self.n2`), taken once in `__init__`. A `Parameter` descriptor replaces
the copies, so each solver declares the mapping — `m = Parameter("k", ...)`,
`g = Parameter("n2", ..., scale=-1)` — and there is one storage. `scale` covers
both the sign convention (the two parametrisations write the interaction term
opposite ways) and GPE's saturation, which is stored converted.

Two bugs closed:

- Assigning the documented name did nothing. `simu.gamma = 0.5` moved the copy
  and left the solver on `alpha`.
- **DDGPE's step was clamped to 1.9e-26 for any field of order one — 1e23
  steps, a run that never returns.** CNLSE's precompute scales the couplings by
  `k/2 * c * epsilon_0`, and DDGPE's `k` comes from `wvl=1e-30`, a wavelength
  supplied only to satisfy the base constructor. The interaction rate came out
  1.7e26 instead of 0.02. Existing tests missed it because their initial field
  is near zero, so the rate vanished with it; the new test uses `np.ones`.

`DDGPE.g` also reported the negative of what its caller passed, since the copy
was taken from storage that holds the kernel's sign.

### 9. Propagation lands on z (merged)

The loop took `ceil(z / delta_z)` whole steps, so unless the step divided `z`
the run went past it, and the error left behind is the phase the medium
imprints over the excess.

Found while checking performance. Scanning step counts around the default
turned up a **285x error spike at exactly 237 steps** — 1.08e-1 where 236 and
238 both gave 3.8e-4. `z / delta_z` for 237 steps is `237.00000000000003`, so
`ceil` asked for 238 and the run went 0.42% too far; over 23.6 rad of
accumulated nonlinear phase that is 0.1 rad, and `|exp(0.1i) - 1|` is 0.1. So
it is not only steps that fail to divide `z` — floating point makes ones that
should divide it fail unpredictably, and a step derived from the physics is an
arbitrary real number.

The loop now takes whole steps and covers the leftover at its own size, with
the shorter propagator swapped in and back out.

## Performance, measured against release 3.0.0

Same case, default step, converged complex128 reference, CPU:

| | steps | time | error |
|---|---|---|---|
| 3.0.0, strong | 1701 | 988 ms | 8.2e-4 |
| **now, strong** | **236** | **144 ms** | **3.8e-4** |
| 3.0.0, typical | 170 | 102 ms | 1.4e-2 |
| **now, typical** | **24** | **15 ms** | **2.0e-5** |

**Faster and more accurate**: 3.0.0 was losing more to the overshoot than it
gained from its much smaller step.

Per-step cost over the same span, min of repeats: CPU 552 -> 532 us/step at
n=64 and 2152 -> 2143 at n=512; CL 472 -> 411 and 569 -> 551. MLX is the
exception — see the remaining-work table.

Within this session (`617e2fb` -> `04674de`, 83 benchmark cases, 3 paired
rounds, minimums): median ratio 0.999, two cases >10% either way and both MLX
noise. The `delta_z` threading and the de-aliasing cost nothing.

### 10. Documentation matches the API, and is tested (merged into the branch)

Five of six copy-pasteable examples did not run: three built a float64 field
where `out_field` asserts complex, one used an undefined name, and the README's
headline block had no import. Six blocks still used `simu.delta_z`. The prose
was staler than the code — the old `5e-3 * z_nl` formula, an RK4 limit
described as counting dispersion alone, and broadcasting called CUPY-only on
two pages. Complex potentials were never documented at all.

`tests/test_docs.py` runs every self-contained example and rejects removed API
in any block. It asserts it found the docs first: the checker I wrote while
doing this globbed a path that did not exist and reported "0 problems" having
read nothing.

## Ugliness ledger (task #10)

Fourteen cleared, three left. Highest value first:

- **mypy checks 15% of the package.** `pyproject.toml` has an
  `ignore_errors` override for `NLSE.solvers.*` and `NLSE.kernels.*`, added in
  `9158272` as "multi-backend complexity" — 16 of 26 files, 9787 of 11518
  lines. Measured with the override lifted and the same settings: **109 errors
  in 10 files**, of three kinds.

  - ~19 `prange has no attribute __iter__` — mypy does not understand numba.
    Noise; a per-loop ignore, or exclude `kernels/cpu.py` alone.
  - ~22 attribute errors on `simu` (`_current_delta_z`, `_random`, `NX`,
    `delta_X`, `gamma`, `gamma2`). **A real annotation bug**: DDGPE's
    `add_noise` and `laser_excitation` declare `simu: object`, so every
    attribute access on it is an error. Annotating them `DDGPE` fixes all of
    them at once.
  - The rest is genuine backend polymorphism (`queue`, `data`), which is what
    the comment was actually about.

  Fix the first two, see how far 109 drops, and narrow the override to
  whatever resists. Worth doing: `attr-defined` is the class of bug that bit
  twice this week, and the checking is off exactly where the bugs have been.
- **Organise the private methods, and separate them from the public surface.**
  `NLSE` has 43 methods of which 35 are private, in no particular order and
  interleaved with the 6 public ones (`backend`, `split_step`,
  `split_step_RK4`, `out_field`, `plot_field`). Reading the class gives no
  sense of what a caller is meant to touch. Minimum: order them public-first
  and group the private ones by phase — setup, step limits, transfers, steps,
  loop — with section comments. Name mangling (`__x`) is the wrong tool here:
  subclasses override `_dispersion_operator`, `_energy_rates`,
  `_take_components` and others, so mangling would break the hierarchy. If
  stronger separation is wanted, move the stateless helpers to module-level
  functions, which are genuinely not part of the class surface.
- `_send_arrays_to_gpu` / `_retrieve_arrays_from_gpu` mutate `self` in place;
  this is what let `_propagator_fft` go stale.
- ~~`kernels/cupy.py` vs `kernels/cupy_kernels.py`~~ — not duplication. The raw
  CUDA kernels take scalar parameters and one flat index; `cupy.py` is the
  `cp.fuse` path `_needs_broadcast` falls back to for a batched parameter or a
  shared grid. `cupy.py` now says so. Drift between them is what
  `test_broadcasting.py` checks: on CUPY the batched run takes one path and
  the individual runs the other.

### 11. nl_length falls back to local, and the docs are tested

Below one grid cell the non-local kernel is a single point — the identity — so
the run was local but still paid for a convolution every step. It warns and
drops to the local path now, which also lets OpenCL and MLX accept such a call:
they refuse non-locality for want of a convolution kernel, and there is none
left to do.

**The docs build clean in `--strict` mode**, and did before. But the tutorial
notebook was the stalest thing in the repository — `puiss`, renamed to `power`
long ago, in eleven places across two copies that had also drifted apart from
each other. `tests/test_docs.py` covers notebooks now. `pyproject` gained a
`docs` extra; building the site needed four packages nothing named.

### 12. CI on every branch, and plot_field shared

`ci.yml` held the linting, the type check and the twelve-way matrix but fired
only on `main`/`develop`; `tests.yml` fired on every push and ran pytest alone
on one configuration. A working branch was checked by neither ruff nor mypy —
which is why every branch this week was verified by hand. `ci.yml` now runs
everywhere, with the matrix scoped: one configuration on a branch, all twelve
on a PR or the trunk. `tests.yml` is deleted, being a cell of that matrix.

**The action version bumps are unverified** — `checkout@v5`, `setup-python@v6`,
`codecov@v5` (whose input is `files`, not `file`). Worth a glance at the first
run.

`plot_field` was ~50 lines of matplotlib in six solvers, differing only in
presentation. Those differences are class attributes now and GPE/DDGPE inherit
the method: 349 lines to 243. The tests only checked it ran, so
`tests/solvers/test_plot_labels.py` reads the figure back.

## Environment notes

Moved into the README: see **Troubleshooting a CUDA install** under
Requirements, and the dev-extra note under Tests.

## Conventions used

Check the relevant tests before touching code; write the failing test first
where the bug is reproducible. Verify claims by running them rather than
reasoning from source — several conclusions were wrong until measured,
including the cause of the stale-wheel bug, the completeness of the first CUPY
broadcast fix, a first attempt at the split-step step limit that bounded the
wrong quantity, and the first two attempts at measuring error against step
size (an unnormalised field, then a reference too coarse to be one).

Commit before mutation testing. `git checkout -- <file>` after a mutation
reverts to HEAD, which throws away uncommitted work in the same file, and
untracked files are not restored at all.

Keep measurements out of docstrings and comments, and keep comments short.
Measurements belong in commit messages, where they are dated and attached to
the change that produced them.

### 13. Step constants have one definition, and examples stop writing to the cwd

`2dd0d37` closes the `getattr(self, "_g", <recompute>)` ledger item. The
fallback at each of ~50 read sites restated the physics, so a subclass that
scaled a coupling differently was heard only by the precompute — which is how
DDGPE's couplings came to be converted as an optical `n2`. `_step_constants()`
is the single table; `_constant()` reads the attribute once it exists and the
table until then. `_V_scaled`, `_V2_scaled` and `_propagator_fft` are per-run
state, not derived constants, so they became class-level `None` defaults.

Guarded by `TestStepConstantTable` and `test_ddgpe_couplings_are_not_scaled_by
_the_optical_constant` in `tests/solvers/test_solver_state.py`. Four mutations
tried, four caught: dropping DDGPE's override (the 1e26 bug), restating `_g11`
with `epsilon_0` missing, a precompute that skips a declared name, and a
`_constant` that ignores the table.

Examples now write through `examples/_output.output_path` into
`examples/output/`. This let six `.gitignore` patterns go, including a
repo-wide `*.npy` that would have hidden real data. `tests/test_examples.py`
holds the line; three mutations tried, three caught.

### 14. Coupled solvers broadcast (needs the NVIDIA box)

`CNLSE` and `CNLSE_1d` could not run a batch at all. Three places read the
component axis as literal 0 and 1 — `_prepare_output_array`, the write-backs
after each nonlinear step, and `_energy_rates` — so a leading batch axis made
the solver take simulation 0 for component 0. Nothing caught it: every
broadcasting test built `NLSE` through `make_solver`, so all of them ran on a
single-component field. The CUPY kernels have carried broadcast fallbacks for
`nl_prop_c` and `rk4_nl_rhs_c` all along, unreachable from above.

Components go through `_component(i)`, derived from `_last_axes`, with
`_set_components` opposite `_take_components`. That retires `CNLSE_1d`'s copy
of `_take_components`, and a `_norm_target` hook retires `CNLSE`'s and
`NLSE_3d`'s copies of `_prepare_output_array`.

**Verified on CPU only.** CUPY takes the same generic path and its kernels
already have the fallbacks, so it should work — that is what the NVIDIA run
needs to confirm, `tests/integration/test_broadcasting.py`. CL and MLX use a
fused coupled kernel that assumes one field of the coupled rank; MLX silently
returned `(2, 3, 2, N, N)` for a batch of three. Both now refuse through
`_check_batch_support`. **Making them broadcast is kernel work, still open.**

Four mutations tried on the first pass, two caught. The two that survived —
a shared normalization target, hidden because `power2` defaults to `power`,
and a dropped `_set_components`, invisible on CPU where the components are
views — now have tests of their own (`b22e93d`).
