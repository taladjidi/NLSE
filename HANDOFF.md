# NLSE — session handoff

Tracked, so it moves between machines with the code. Sections are appended as
work lands; the plan below is the only part that needs reading first.

## Start here

**`main` is at `9a700c4`. Branch `perf/profiling` is 13 commits ahead and is
where all the recent work lives.** 811 tests pass on macOS (CPU, CL, MLX);
ruff, format and mypy clean. The branch has not been verified on CUPY.

### The plan, in order

**0. Verify and merge `perf/profiling`.** Thirteen commits: the profiling
tools, the FFTW wisdom fix, the `alpha == 0` kernel specialisation, the
backend-capability conversion. Run the suite on the NVIDIA box, then merge.
Everything below assumes it is in.

**1. Fused coupled kernels for CUPY.** ~~The measured payoff, and the largest
single item left.~~ **Done — see section 23.** It absorbed item 3 as well:
the copies were counted twice, once as kernels and once as memcpys.

**2. ~~Let CPU skip the inverse FFT's normalisation.~~** Done — section 28.
The flag was wired only to `kernels.linear_step`, which the CPU has no reason
to implement, so `ifft` took a `normalize` argument instead. 5.6% per step.

**3. ~~Find the 11.3% of GPU time in memory operations.~~** Done, with item 1,
and it was not allocation zeroing. It was `k[:] = A_in` at the head of each
RK4 stage and `_set_components` writing the components back: both are
device-to-device memcpys rather than kernels, which is why they appeared in a
separate table and looked like a separate problem. Section 23.

**4. mypy coverage.** The last ugliness-ledger item, deferred by request
until the rest was done. `pyproject.toml` has an `ignore_errors` override for
`NLSE.solvers.*` and `NLSE.kernels.*`, hiding 109 errors in 10 files. About 19
are numba `prange` noise and about 22 come from DDGPE's `add_noise` and
`laser_excitation` declaring `simu: object`; the rest is genuine backend
polymorphism. Fix the first two, see how far it drops, narrow the override to
what resists.

### The tools

    benchmarks/profile_backends.py    per-step cost, --baseline <rev> to compare
    benchmarks/profile_kernels.py     per-kernel roofline against bandwidth
    benchmarks/trace_solvers.py       phase breakdown of a real run
    benchmarks/nsys_summary.py        folds an nsys report into the same phases

The command that produced the trustworthy CUPY profile:

    nsys profile -o steady --force-overwrite true --trace cuda \
        python benchmarks/trace_solvers.py --backends CUPY \
            --solvers NLSE CNLSE --methods split_step RK4 \
            --sizes 1024 --plain 2000 --no-cuda-graph
    python benchmarks/nsys_summary.py steady.nsys-rep --top 20

### Environment

The macOS machine needed `pyfftw` from conda-forge rather than PyPI: the
arm64 wheel bundles an FFTW with no NEON, four times slower (section 19). The
CPU backend now times itself against scipy and warns if it is in that state,
so any machine will say so rather than being quietly slow.

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
2. ~~**Ugliness ledger**~~ — 16 of 17 cleared. Only the mypy override is
   left, deferred by request until last.
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

Sixteen cleared, one left — the mypy override, deferred by request
until the rest of the work is done.


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
- ~~`_send_arrays_to_gpu` / `_retrieve_arrays_from_gpu` mutate `self`~~ — see
  section 15. They still mutate, but the mutation is now scoped and undone on
  every exit.
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

### 13. Step constants have one definition

The precompute wrote fixed-precision constants onto the solver for the
kernels, and every other reader took them with a default that recomputed the
same physics -- `getattr(self, "_g", self.k / 2 * self.n2 * c * epsilon_0)` at
some fifty sites. A subclass that scaled a coupling differently was heard only
by the precompute, which is how DDGPE's couplings came to be converted as an
optical `n2`, putting its interaction rate 1e26 too high and its step limit at
1e-26 m. `_step_constants()` is now the one table; `_constant()` reads the
attribute once it exists and the table until then.

Examples write through `examples/_output.output_path` into `examples/output/`,
which let six `.gitignore` patterns go, including a repo-wide `*.npy` that
would have hidden real data.

### 14. Coupled solvers broadcast

`CNLSE` and `CNLSE_1d` could not run a batch at all: three places read the
component axis as literal 0 and 1, so a leading batch axis made the solver
take simulation 0 for component 0. Nothing caught it because every
broadcasting test built `NLSE`. Components go through `_component(i)` derived
from `_last_axes`, with `_set_components` opposite `_take_components`.

Two CUPY-only regressions followed and were fixed: a per-component
normalization target left on the host (CuPy refuses a numpy operand against a
device array), and batched constants keeping the component axis, which CPU
tolerated and CuPy broadcast into `(count, count, NY, NX)`. Both are pinned by
CPU tests now.

CL and MLX refuse a batched coupled run through `_check_batch_support` rather
than returning a wrong-shaped array, which MLX had been doing silently.

### 15. The device transfers are paired

`out_field` moved V, `nl_profile`, the propagator and any batched parameter
onto the device at the top and brought them back eighty lines later with
nothing between. Anything raising in between left them there, and the run that
broke was the *next* one: on CL that surfaced as an unrelated `ValueError`
from which the solver never recovered. `_arrays_on_device` is a context
manager restoring in a `finally`.

### 16. Simplification, reassessed

Measured rather than eyeballed; most of what the ledger implied was left had
already been taken by the ledger work. Comparing every override against the
method it overrides, only three exceed 0.55 similarity, and the two that
matter carry real physics differences. No dead code: `vortex`/`vortex_cp`
looked dead but is a user-facing utility with its own tests. The two dispatch
mechanisms were already one. 43% of the package is docstrings, so the 11.6k
line count overstates it -- the code is ~5.6k across four backends.

The one real finding was fourteen `self._backend.name == "..."` branches in
the solvers, coexisting with the capability flags meant to replace them.

### 17. Capabilities replace the backend name checks

All fourteen are gone. `Backend.convolution` returns the overlap-add
convolution or None, which retires the `oaconvolve` choice (written out twice
verbatim) and the `nl_length` check too: a backend with no convolution is
exactly one that cannot do non-locality. `Backend.synchronize()` and
`Backend.timed()` replace ~25 lines of CUDA events, `queue.finish` and
`mx.eval` inside `out_field`, plus a function-local `import mlx.core`.
`normalizes_on_host` covers the CL/MLX normalization route.

**`out_field`: 114 lines and 15 branches -> 91 and 8.**

Two corrections the tests forced: the MLX flag was first called
`arrays_are_immutable`, which is false, and then a mutation showed the renamed
flag made no measurable difference at all, so it and both its branches were
deleted. Fourteen name checks became four capabilities and two deletions.

### 18. Performance, measured: there is no regression

**No MLX regression, and none on any backend.** Against 3.0.0 every cell of an
18-cell matrix lands between 0.99x and 1.06x.

The 35% figure this session had been carrying was an artifact of timing a
whole `out_field` and dividing by the step count, which charges the
once-per-run setup to the steps. At NX=NY=64 the setup is ~2.2 ms on MLX
against 0.022 ms per step, so a ten-step run was 91% constructor. My own first
harness had the same flaw and reported a 1.6x MLX regression that reversed
sign when run again.

`profile_backends.py` takes the slope between two step counts, so setup
subtracts out. `--baseline <rev>` compares against a git worktree of any
revision and refuses to report if it imported the wrong tree.

Benchmark step counts are calibrated per backend (`BENCH_STEPS`) so
propagation is >=80% of each run, and the CI `alert-threshold` drops from 200%
to 140% against a measured spread of 2.9% median, 8.5% worst.

### 19. Why pyfftw was slow: the arm64 wheel has no NEON

The PyPI wheel vendors its own FFTW, and that build has no SIMD:
`pyfftw.simd_alignment == 4` (a vectorised build reports 16 or 32) and `nm`
finds zero neon/simd/avx symbols in a 677 KB library where a SIMD build is
several MB. Single-threaded pyfftw is 236 ms against scipy's 38 at 2048x2048,
so not threading; in-place and out-of-place are the same, so not transposes.

Unnoticed because x86 wheels are vectorised, because the gap only bites at
large N (1.5 ms at 512, 35 at 2048, 159 at 4096), and because planning, wisdom
and the stale-wisdom check all behave normally -- they plan scalar codelets
well.

`conda install -c conda-forge pyfftw` fixed it, 3.9x. It needed
`pip uninstall pyfftw` first, since conda could not see the pip one and
silently no-op'd. **conda-forge's build also reports `simd_alignment == 4`**,
so that value does not discriminate, which is why the CPU backend times itself
against scipy instead and warns.

**A bigger win was behind it: stale FFTW wisdom**, cached on disk and
outliving the library swap -- 34 ms against 8.8 ms for the same transform, and
the old guard allowed 400 ms so never fired. The scipy comparison now decides
whether to discard wisdom.

**CPU split step end to end: 1024 7.5 -> 2.8 ms, 2048 39.8 -> 11.4 ms.**

### 20. Why -fveclib bought nothing, and where CPU kernel time goes

**`-fveclib=Accelerate` does fire** -- the build emits `_vexpf`, `_vsinf`,
`_vcosf`; an earlier claim that it did not was a bad grep. Both loops
vectorise, width 4. It bought nothing anyway. Lossless rotation at 2048x2048,
all threads:

| variant | time | GB/s |
|---|---|---|
| no maths at all -- the streaming floor | 0.492 ms | 170 |
| inlined polynomial, ideal SIMD maths | 0.772 ms | 109 |
| libm cosf/sinf, what C ships | 0.862 ms | 97 |
| numba, what we ship | 1.725 ms | 49 |

The floor agrees with `apply_propagator` at 181 GB/s. The distance between a
scalar libm call and perfectly inlined SIMD maths is ~10%: at the libm point
the loop is already 57% memory-bound.

**Small arrays are slower, not faster**: 46 GB/s at 512x512 (cache-resident)
against 99 at 2048x2048, because the OpenMP parallel region costs ~50-100 us
to set up.

numba's gap is general codegen, not a missing vector maths library: 2x above
the same loop in C with the same layout, threads and libm. A C rewrite is
worth ~2x on this kernel, 8-10% of a CPU step, against a compiler,
per-platform wheels and an OpenMP runtime. Not taken.

The cheap win was taken instead: **`alpha == 0` skips the exponential**, exact
because exp(0) is 1. 1.60x on `nl_prop_without_V`, 1.52x on
`square_mod_nl_prop`. Only the three kernels without a potential -- the V
variants add `1j * V`, and a complex V puts a real term back in the exponent.

### 21. Where the time goes inside a step (`benchmarks/trace_solvers.py`)

Wraps the backend's kernels and transforms and attributes each phase.
`--nvtx` pushes the same names as NVTX ranges; `--per-call` lists kernels
individually; `--plain N --no-cuda-graph` runs one long unwrapped propagation,
which is the only workload whose nsys profile describes the solver.

At 1024x1024: NLSE CPU split 2.98 ms/step (transform 76%); NLSE CPU RK4 13.2
ms (transform 67%); NLSE CL split 0.83 ms (one fused kernel, 85%); NLSE CL RK4
3.72 ms (rhs 56%, **stage 38%**, 9 launches); NLSE MLX RK4 1.89 ms (**one
fused kernel, 89%**, 1 launch).

**RK4 costs 4.4x a split step because it runs 8 transforms against 2.**

`--per-call` also shows the CPU inverse transform costing 28% more than the
forward, because pyfftw normalises by 1/N. See plan item 2.

### 22. CUPY, profiled properly: where a step actually goes

One long propagation, nothing wrapped, every launch itemised. 29157 ms of GPU
kernel time across four cases at 1024x1024.

| phase | share | launches |
|---|---|---|
| transform | 36.5% | 80080 |
| RK4 stage | 19.2% | 32032 |
| **array copies** | **13.1%** | **56064** |
| linear | 12.5% | 20020 |
| RK4 rhs | 12.2% | 24024 |
| nonlinear | 6.4% | 16016 |

Per step: NLSE split 0.598 ms, NLSE RK4 3.750, CNLSE split 1.798, CNLSE RK4
10.567.

**GPU kernel time / CUDA API time is 0.87.** The GPU is busy; there is no
launch-overhead problem. Earlier readings of this ratio (0.18, then 0.01) were
artifacts of profiling a harness or a traced run rather than a propagation.

**The copies are `_take_components`, confirmed exactly**: counting the call
sites predicts 18.0 complex64 and 10.0 float32 copies per step across the four
cases, and nsys measured 18.02 and 10.01. Entirely in the coupled solvers --
NLSE never calls it -- where CNLSE RK4 makes 24 copies per step, ~192 MB of
pure copying at this size.

CL and MLX pay none of it, because their fused coupled kernels never split the
field. That is plan item 1.

**Tooling caveats learned the hard way**, all handled by the scripts now:
profiling `profile_backends.py` measures its own warmups and transfers;
profiling a traced run measures the tracing; and with CUDA graphs active nsys
itemises only the captured launches, so a 7000-launch workload reported 90.

### 23. CUPY stops copying, and RK4 stops writing what it reads back

Plan items 1 and 3, which turned out to be one defect counted twice. **Per-step
at 1024x1024: CNLSE RK4 10.626 -> 6.417 ms (-40%), CNLSE split 1.832 -> 1.208
(-34%), NLSE RK4 3.784 -> 3.248 (-14%), NLSE split unchanged.** GPU kernel time
over the four-case workload 29150 -> 22078 ms, memory operations 3774 -> 173.

**The copies were 22.5% of GPU busy time, not 13.1%.** Section 22 read the
kernel table alone, where `cupy_copy__*` accounted for 18.02 complex and 10.01
float copies per step and matched `_take_components` exactly. It matched
because the other half was in a table nobody had opened: 36036 device-to-device
memcpys, 3596 ms, 369 GB. CuPy takes an elementwise-copy *kernel* for
`.copy()` and a *memcpy* for a slice assignment, so the same defect landed in
two tables and read as two problems. The full per-step accounting, which now
closes exactly:

| operation | per step | form |
|---|---|---|
| `_take_components`, complex | 18.0 | copy kernel |
| `_take_components`, float32 | 10.0 | copy kernel |
| `_set_components` write-back | 10.0 | DtoD memcpy |
| `k[:] = A_in` at each RK4 stage | 8.0 | DtoD memcpy |

CUPY declared none of the six fused capabilities. The backend docstring
argued why: CUDA graphs remove the launch overhead fusion exists to amortize.
That is right about launches and irrelevant here — **a graph replays a copy as
faithfully as it replays arithmetic**. What the interleaved kernels save is
traffic, not launches, and the ratio of GPU kernel time to CUDA API time was
already 0.87, so there was no launch problem to solve.

Four capabilities now declared, modelled on the OpenCL originals:
`coupled_nl_prop_c`, `coupled_rk4_nl_rhs_c` and `rabi_coupling_interleaved`
read both components from the one `(2, ...)` array; `rk4_rhs_fused` and
`rk4_rhs_coupled_fused` let the transform move `A_in` into `k` instead of a
copy doing it first, cuFFT plans being out-of-place already; and
`rk4_set_and_axpy`/`rk4_acc_and_axpy` halve the stage-update launches. A fifth,
`rk4_final_update`, closes the step as `A += h/6 * (acc + k4)` rather than
writing `acc` only to read it straight back — four passes over memory where
there were six.

**CL and MLX refuse a batched coupled run, so their fused gate never needed a
batch check. CUPY serves one**, and its raw kernels take scalars and one flat
index, so `_can_fuse_components` now gates on the batch as well as on
`nl_length`.

**Every remaining kernel is at the card's bandwidth ceiling** — 203-208 GB/s
across all eight on an RTX 3050. Nothing left is inefficient; what remains is
passes over memory. `apply_propagator` alone is 16.5% and moves the minimum a
separate kernel can, so removing it means folding the propagator multiply into
the transform's store phase with cuFFT callbacks, which CuPy's plan API does
not expose. Fusing each stage's RHS with its stage update would save perhaps
8% more and needs a kernel per (stage kind x potential kind x coupled or not).

**A round-off difference is not automatically round-off.** The first fused
coupled RK4 differed from the generic path by 8.7e-5, which looked like
float32 noise; it came from writing the accumulate as one expression where the
kernel it replaces uses temporaries, and `--use_fast_math` contracted the two
differently. The same re-association section 1 recorded. With that matched, one
stage agrees to 7e-8 with component 2 bitwise identical, and a whole run agrees
to 1e-8..4e-7 on a Gaussian beam. It still shows 6.6e-5 on a random-noise field
under a violent nonlinearity, which is that system amplifying a last bit, not
the kernels: **an adversarial initial condition measures the chaos, not the
code.**

`nsys_summary.py` now names the kernels that fall into `other` rather than
letting them read as a residual: `rk4_final_update` silently became 4.4% of it.

**Found, not fixed: complex128 coupled RK4 is broken on both CPU and CUPY**,
and was before this work. CUPY returns all-NaN, CPU raises `Invalid output
dtype` from pyfftw. `cnlse.py` allocates `_rk4_A_sq_c` as `np.float32`
unconditionally where `nlse.py` uses `E_in.real.dtype`, and the CPU coupled
RK4 path uses a plan built for complex64. No test covers the combination.

### 24. An RK4 stage stops writing the slope it immediately reads back

A stage computed its slope into `k`, and the stage update read `k` straight
back: eight accesses per element where six will do. `square_mod_rk4_stage` and
`coupled_rk4_stage_c` take the linear part as the transform left it, finish the
slope in registers and spend it on the accumulator and the next stage's
argument without it reaching memory.

**One kernel serves all four stages**, with the stage kind as a `mode`
parameter. It is uniform across the grid, so the branches cost nothing and each
stage still touches only what it needs: stage 1 *sets* the accumulator rather
than reading it, and stage 4 does not write it. Writing six kernels instead --
three stage kinds by two solver families -- would have meant eighteen compiled
variants after the V expansion, against six.

**Bitwise identical to the path it replaces**, on every solver, every kind of
potential and both precisions. That is the standard these should be held to:
the arithmetic is unchanged, so any difference at all would be a
re-association, and a tolerance would hide it.

In stages 2-4 the argument and the output are the same buffer. Each thread
reads and writes only its own index, so this is safe -- and worth knowing,
because it also means the write hits a line the read has already brought in,
which is why the kernel appears to exceed the card's bandwidth.

**NLSE RK4 3.229 -> 2.867 ms/step, CNLSE RK4 6.411 -> 5.689** (-11% each).

### 25. Every complex128 RK4 run was broken, on every backend

Found while validating the fused kernels, and present without them. CUPY
returned all-NaN and CPU raised `Invalid output dtype` from pyfftw. Three
places pinned a width to single while the run's precision comes from the field:

- `_allocate_rk4_buffers` allocated `complex64` scratch, so a double run wrote
  the field into a stage buffer half its width and the FFT plan -- built from
  the field -- met an array of the wrong dtype;
- `CNLSE._allocate_rk4_buffers` did the same with `float32` for the intensity;
- `_compute_propagator_rk4` ended in `.astype(np.complex64)`, in all six
  implementations.

The last is the one worth remembering: the GPU kernels choose single or double
**from the field** and then index the operator with the same flat id, so an
operator of the other width is read as the wrong type. `_field_dtype`'s
docstring already described this exact failure in the other direction. The cast
now happens once, in `_build_propagator_rk4`, which is the only place that
knows what the field is -- computing the operator narrow and widening it
afterwards would have carried single-precision values in a double array. Width
is part of the cache key, or the cache hands back the other one.

`split_step` was unaffected, which is why 809 tests passed over it: **nothing
ran RK4 in double**. `tests/solvers/test_double_precision.py` does now, over
five solver classes and every backend, and four mutations reinstating each
pinned width all fail it.

`_build_propagator_rk4` takes the dtype as a required argument rather than
defaulting to complex64: a default is what let this go unnoticed, and there is
no width a caller should get without asking.

### 26. cuFFT callbacks: measured, works, not taken

The last structural item, since `apply_propagator` is 18.3% of GPU kernel time
and already at bandwidth -- the only way past it is to stop it being a separate
pass. A cuFFT **store callback** on the forward transform applies the
propagator as the transform writes its output.

**It works on this machine and it is faster: 0.485 -> 0.402 ms** for
fft/propagator/ifft at 1024x1024, bitwise identical, and it survives CUDA graph
capture. The saving is exactly two passes over the field -- 16.8 MB at 203 GB/s
is 83 us, and 83 us is what was measured. Extrapolated: ~2430 ms of the 19905,
or **12%**.

Not taken, and the reasons are about what it would cost everyone else:

- **The legacy callback ABI is gone in CUDA 13.** `nvprune` fails on
  `libcufft_static.a`. Only `cb_ver="jit"` works, so this is a hard dependency
  on cuFFT JIT LTO.
- **It needs headers the wheels do not ship.** The build wants `cufft.h`;
  it is not in the pip `nvidia-*` packages, only in a system CUDA install. It
  failed twice here before `CPATH` was set, on a machine with a full toolkit.
- **The plan becomes tied to the propagator's address**, passed as callback
  data at plan creation. The propagator is rebuilt whenever `delta_z` changes,
  which `adapt_delta_z` does mid-run, so it would need a fixed buffer copied
  into and a plan cache keyed on it. Two things currently independent become
  entangled.
- Plan construction is ~0.5 s.

It would need a runtime probe and a fallback for every machine where the build
fails. Worth revisiting if cuFFT JIT LTO becomes something the wheels carry.
The probe is `scratchpad/cb_probe2.py` in the session directory if it is.

### 27. What the algorithm has left, measured

`benchmarks/work_precision.py` answers the question a per-step timing cannot:
what each method costs to reach a given accuracy. Same problem, same distance,
one converged reference, the step swept as phase per step.

**Split-step is the right method and it is already at the floor.** Every error
curve has a minimum and rises again as the step shrinks -- on CPU, Lie is
1.80e-4 at 0.1 rad and 1.59e-3 at 0.005, because 2512 steps accumulate more
complex64 round-off than 126. So in the lossless regime the accuracy floor is
the number representation, not the splitting order, and a 4th-order
composition (Yoshida, Blanes-Moan) would pay 3x per step to reduce an error
that is not what binds. Split-step is also at 2 transforms per step, the
minimum for an operator split.

**RK4 is dominated except below ~3e-5**, where it is the only option: it
reaches 3.6e-6, an order better than split-step can, at ~40x the cost, and
diverges above 0.2 rad per step.

**Strang dominates Lie everywhere.** Both cost 2 transforms; Strang is 2-4x
more accurate at equal step, so it buys a much larger one. CPU: Lie at 0.1 rad
is 126 steps and 1.80e-4; Strang at 0.8 rad is 16 steps and 3.96e-5 -- 4.3x
faster and 4.5x more accurate. The same on CUPY and CL.

**But not in a lossy regime.** With `alpha=20` the error grows with the step
instead: Strang at 0.8 rad is 1.21e-2 against the default's 2.88e-3. There the
splitting error dominates and never reaches the round-off floor. A larger
fixed default would quietly spoil absorbing-medium runs, which is why the gain
belongs in an error-driven step (section 29) rather than a new constant.

### 28. Three things taken from it

**The CPU skips the inverse transform's 1/N** (plan item 2). FFTW's backward
transform is unnormalized and pyfftw divides in a pass of its own, which costs
a fifth of the transform -- an inverse runs 25% longer than a forward at 512,
1024 and 2048. The flag was wired only to `kernels.linear_step`, which the CPU
has no reason to implement; `ifft` takes a `normalize` argument instead.
1024x1024: split 14.670 -> 13.843 ms/step, RK4 53.478 -> 50.633.

**Consecutive Strang steps share their half steps.** A run of them is
`N(h/2) [L(h) N(h)] ... N(-h/2)`, and the bracketed body is exactly a Lie
step, so the loop body is the one `precision="single"` already runs and there
is no third stepper. CPU 1.62x, CL 1.33x, CUPY 1.04x. Exact only where
`N(a) N(b) == N(a+b)`, which needs `|A|` to survive `N`: no loss, no absorbing
potential, no Rabi coupling, and DDGPE opts out. Forcing it on a lossy run
doubles the error, so the guard is pinned by a test that measures the damage.

**The step can be chosen by measuring** -- section 29.

### 29. Why an error-driven step has to be capped, and floored

`adapt_delta_z_to_error` takes the same distance whole and in halves and moves
the step by the usual controller. 30 steps for 3.8e-5 where the fixed default
takes 125 for 7.8e-5. Two failures found by running it, both instructive
beyond this callback:

- **The local error estimate has a round-off floor.** Below ~0.8 rad per step
  the difference between one step and two halves is complex64 noise, flat at
  ~3e-7 from 0.1 to 0.4 rad. Every tolerance above it reads as "no error" and
  asks for a bigger step whatever the step is. Uncapped, the controller
  doubles each time it fires and the answer is unrecognisable within three
  adjustments -- 0.38 relative. **In complex64 a tolerance below ~1e-5 is not
  resolvable at all**, and the physics cap, not the tolerance, is what sets
  the step there.
- **An unmeetable tolerance must stop shrinking.** Without a floor the
  controller halves for ever: nothing raises, nothing diverges, the run just
  never returns. The floor has to be absolute -- one computed as a fraction of
  the step in force shrinks with it and never binds, which was the first
  attempt.

Both are tested by driving the controller directly. A test through a
propagation could not reach the ceiling to exercise the cap, and could only
ever time out on the floor.

### 30. The propagator is built where the field is

`_build_propagator` is `exp(theta * dz)` for the operator RK4 already
integrates, computed by `Backend.exp` on the backend's own arrays. The seven
`_compute_propagator` implementations are gone with it.

**A rebuild at 512x512: CUPY 9.68 -> 0.064 ms, CL 9.72 -> 0.254, CPU 9.60 ->
7.61.** 151x and 38x; the CPU keeps the exponential either way and saves only
rebuilding `Kxx**2 + Kyy**2` in complex128. It matters wherever the step
changes during a run.

**The first attempt looked like a hang and was a type change.** Returning
backend-native arrays broke every test that compares a propagator against
numpy, and the two device backends fail that differently: CuPy raises, while
PyOpenCL's `Array` has no `__array__` at all, so numpy reads it one element at
a time -- **866 ms for a single 256x256 array**. A suite full of those is not
recognisable as a type error. The conversion is explicit at the test boundary
now.

Three things that had to move with it:

- `_send_arrays_to_gpu` handed `from_numpy` whatever it found. PyOpenCL does
  not reject an `Array`; it tries to enqueue a write *from* it.
- The propagator cache is bounded and least-recently-used. It held a field per
  distinct step, which is nothing while the step is fixed and unbounded once
  it is chosen from a measured error.
- On a device backend the host copy of the operator is not kept. Peak RSS went
  4.57 GB -> 2.28, its baseline.

**NLSE_3d was integrating a different equation in each method.** Its split step
applied the temporal dispersion with no step at all -- `exp(-1j D0/2 Omega^2)`
-- while RK4 treated the same expression as a rate. `D0` is documented in
s^2/m, so `D0/2 Omega^2` is a phase per metre and the exponent is not
dimensionless without a length. Writing the propagator as `exp(theta*dz)`
fixes it by construction, and **3D results change**.

The test is that a linear run cannot depend on its step count: with no
nonlinearity the linear operator is applied exactly and has nothing to fail to
commute with. Four steps against sixteen differed by 2.6e-4 and now agree to
9.1e-7. **It needs `alpha=0` as well as `n2=0`** -- loss is saturable, so with
a finite `Isat` it depends on the intensity, does not commute either, and
leaves 1.2e-4 of its own, which looks exactly like the bug.

**DDGPE's dispersion operator paired a scalar with a grid.** The exciton branch
has no k dependence, and `np.array([scalar, grid])` is an inhomogeneous array,
which numpy has raised on since 1.24. Unreachable while only RK4 read it and
DDGPE exposes no `method` argument; load-bearing now that the split step is
built from it.

### 31. What is left, and where the adaptive stepper's cost went

**`_energy_rates` is now the whole of it.** It is still `to_numpy(A)` and numpy
reductions -- 9.16 ms at 512x512 -- and the step cap calls it at every
adaptation. Measured at 512x512 with the same step count either way, the
error-driven step costs 2.77x the fast loop on CUPY, and about 27 ms of that
32 ms is `_energy_rates`. It is the same fix as the propagator: reduce where
the field is. `Backend.norm` already exists for it.

End to end, where the step is free to grow, it already pays for itself: 0.93x
the fixed-step fast loop on CL and 0.80x on CPU, against 1.90x on CUPY, whose
steps are too cheap to amortize three trials.

**`Backend.norm` on OpenCL shipped raising.** pyopencl builds its reductions
from a mako template and does not depend on mako, so without it every
reduction raises the first time one is asked for -- inside a callback, partway
through a propagation. It falls back now. Nothing had checked that a backend's
norm agrees with numpy, because the tests that exercise it run on CPU; and
with mako installed the fallback is unreachable, so the test has to simulate
the failure or it passes either way.

## Conventions used

## Conventions used

- Commit before mutation testing. `git checkout -- <file>` during a mutation
  run has destroyed uncommitted work three times in this project, and once a
  commit went out describing a change that was no longer in it.
- The same applies to bulk edits of this file: truncating it from a search
  match destroyed sections 16-22 once, and they had to be rewritten from the
  conversation. It is tracked now, so git has a copy.
- A new test is not trusted until a mutation makes it fail.
- **Test data symmetric in the thing under test cannot see an error in it.**
  Both components of a coupled field built from the same Gaussian make
  `|A1|^2 == |A2|^2`, and `CNLSE.__init__` sets `alpha2 = alpha`,
  `n22 = n2`, `I_sat2 = I_sat`, `k2 = k`. Between them, seven mutations
  swapping one component for the other survived the whole suite. A test that
  reads `simu.k2` for its own expectation agrees with a propagator that
  ignores it.
- **A test of one fast path must switch off every faster one.** The solver
  takes the most fused path a run qualifies for, so turning off the flag whose
  name matches the kernel under test can leave both sides of the comparison on
  the same path. Half a file went vacuous that way the moment a new level was
  added above it, and nothing failed.
- Measure repeatability before believing a benchmark. Three separate
  measurement harnesses in this project produced confident wrong answers that
  only fell over when run twice.
- Verify a bulk edit of the tests against the collected test IDs, not the
  diff. `pytest --collect-only -q` before and after must match exactly.
