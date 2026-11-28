# Handoff

## State (2026-11-27)

A full audit of the repo was done at commit `fa7aa59` (v4.1.0 plus six
commits). Every finding that was to be fixed is fixed and merged into `main`
with `--no-ff`, one topical branch each. This file was shrunk on 2026-11-27
to what is still open: the record of the merged work (the ranked findings,
the measurements, the verification runs) is `git show 5dc5c6a:HANDOFF.md`,
and the figures that matter are in `docs/optimization-log.md` and
`CHANGELOG.md`.

Specs, patches, scripts and logs are outside the repo, in
`~/Documents/LKB/NLSE_tangui_fix_specs/`.

`CHANGELOG.md` has an `## Unreleased` section with every change, the breaking
ones first. The next release is a major one (5.0.0 is the suggestion); the
version is not bumped and nothing is published.

`main` is pushed after each reviewed merge (owner, 2026-11-27: "No push also
to main as we go along !"). The auto-mode classifier refused `git push origin
main` once: when it does, ask the owner to run `! git push origin main`, and
do not look for another way.

### Merged

`git log --merges --oneline fa7aa59..main` is the full list.

| Work | Merge |
|---|---|
| Audit fixes: p7 docs, packaging, CI; p6 closed-form tests and cache isolation; p5 kernel cleanup; p3 GPE units; p1b run state; p1a stepping loop; p4 non-local response; p2 coupled solvers | `ed00008`, `3f3b6dd`, `36d952d`, `7a5fd6a`, `bcda6d8`, `41f39a4`, `4eef782`, `36bf9e1` |
| Post-merge test fixes; RK4 step bounded by the grid's fastest linear rate | `b76f9e5`, `8bce83b` |
| CPU complex64 kernels in float32 | `71db255` |
| CuPy: nonlinear step in the inverse cuFFT store callback; separable propagator in the forward one; the 4096² column pass (closed, log only) | `5113580`, `ce290d8`, `6fbb522` |
| uv, ruff and ty | `62bd2e8` |
| Batches over several parameter axes: CPU, OpenCL, CuPy without callbacks | `496cd96`, `f26b694`, `083a2d9` |
| Transform wording and the library re-measure | `c04afd7` |
| Tests that bite (mutation audit) | `9487346` |
| CPU blocked step; its scratch from a private mapping; its two tests that numba's threading layer decides | `7518025`, `4bb3432`, `e60e2a4` |
| Debloat: repo, tests, package, `pyproject.toml` | `5faeb6e`, `fd0e6af`, `f29b585`, `f8d224e` |
| Benchmark waits on what lazy transforms return (MLX was timed doing nothing) | `3dfc28f` |
| Docs: gallery cached in CI; page review along Diataxis | `18f4cc7`, `3bd8af5` |

Last full suites: Mac (CPU, OpenCL, MLX) 2340 passed, 221 skipped at
`0ba9aaa`; box, combined environment (CuPy and OpenCL), 2327 passed, 104
skipped, six times and once with `NLSE_FUSE_PROPAGATOR=0`, at `5af91cd`
(`~/Documents/nlse_main_5af91cd/run*.txt` there). Each box run also reports
22 errors, all of `tests/test_packaging.py`: `nlse_env_both` has no
hatchling. At `b6fa684` the box ran only the potential-sign, coupled and
DDGPE tests (121 passed). Nothing merged since `b6fa684` has run on CuPy.

### Resume here (written 2026-11-28)

`main` is pushed; its last code change is `fix/examples-benchmarks-review`
(`cea5ea2`), the two merges after it are tests.
Orders from the owner for the night of
2026-11-27: "run independently tonight : debloating, docs, and review.
Iterate over this. Also when reviewing ignore the previous context to be
impartial. Look at all aspects of the code, notably complexity and code
architecture, testability etc ..." and "don't ask me questions. Make
decisions and worst case scenario we'll revert them tomorrow morning". So
reviewers get the code and no conclusions, and do not read this file; what
was decided alone is under "Decided alone" below.

1. **Review, then iterate** (debloat, docs, review). Done and merged: the
   solvers (two fix rounds), the backends, the docs pages (round 1), the
   kernels (rounds 1 and 2), the test suite's four MAJOR gaps, `examples/`
   and `benchmarks/` (round 1). Nothing is in flight. Next rounds, none
   started: a second docs round, a review of what round 1 of the examples
   did not open. The examples' debloat candidates are dismissed (see that
   round's entry).
2. **README headline performance** (docs): done, branch
   `docs/readme-performance`. The README has `## Performance` and the
   optimization log's Mac and box tables are these runs. Spec, revision 2:
   `docs/readme_perf.md` in the specs directory; the run files are beside it
   in `docs/readme_perf_runs/` (`box2_run{1..6}.txt` at `67d5b6e`,
   2026-11-27; `mac_run{1..6}.txt` at `9b685c3`, 2026-11-28; the first run
   of a set is discarded; `summarize_runs.py <prefix>` prints the medians).
   - Mac runs: on mains, no agent running; Firefox was using about 19% of a
     core when they started. The CPU path is mixed there (plain at 1024² and
     2048², blocked at 4096²), blocked at every size on the box.
   - Decided alone: the June 2024 figure moves into the new section,
     labelled (O1); Lie rows only, Strang measured as a check (O2); measured
     ceilings only, no rated figures (O3); a ceiling whose min and max print
     the same is written once; the box CPU row at 1024² reads "memory
     bandwidth / sin+cos+exp" (17% against 20%, exactly the 15% rule).
   - Fresh review: every number matched the runs; three MAJORs, all text,
     fixed (the log's July ceilings table and its sentences now carry these
     runs; the two "open" levers in "Lowering the floor" say they are built;
     the README no longer says moving fewer bytes is the only way at 1024²).
     The fixes were read by me, not by a second reviewer.
   - Left: Strang is 5% faster than Lie on MLX at 2048² (1.723 against
     1.821 ms, ranges apart), unexplained; the box CPU's sin+cos+exp probe
     reads 0.7–1.0 G/s where July's read 1.3, unexplained; CuPy's drop to
     59% of bandwidth at 4096², unexplained; the log's dispatch figures (CL
     0.44 ms, MLX 0.05 ms, CuPy 0.011 ms) and its "dual-channel DDR" clause
     are July's, not re-measured; double precision and RK4 were not re-run;
     the run files record no commit (the trees were clean at the named
     ones); the two machines ran tool commits 13 lines apart (a refusal of
     disturbed timings, no change to a measurement).
   - The tool's own changes (`13345a4`, `67d5b6e`, `9b685c3`) had no review
     on record, so a fresh one ran before the push. Byte counts, paths,
     waits and probes: nothing found (CuPy not run). One MAJOR, fixed
     (`fix/roofline-zero-ceiling`): a probe whose slope was not positive
     read 0 and the shares divided by it; `measure_ceilings` now gives no
     ceiling unless all three are positive. Left, all MINOR:
     - `tests/test_benchmarks.py` lets these mutants live: the slope divided
       by `2 * steps`, `max` for `min` in `fastest_run`, CUPY's
       `len(ours) == 2` as `>= 1`, the "left the path" check removed,
       `WARMUP_STEPS = 0`, `STEP_DZ = 1e-5`. No test reaches the CUPY branch
       of `step_path` or `main`; the real timing is only asserted finite.
     - On CUPY `_fused` holds the callbacks ever built, not the ones the
       timed step used: a step that declined them would still read
       "callbacks" (plausible, not run).
     - `main` exits 0 when every row failed.
     - `PROPAGATOR_FLOPS` and `NONLINEAR_FLOPS` (6 and 14) have no source
       but "as this tool has always charged them"; bare numbers in
       `step_cost`, `time_step` and `step_path`; `main` is 35 statements.
     - The 64x64 timing test raises "disturbed" on a loaded machine (25
       repeats on each backend did not).
3. **Afterwards, the owner's earlier order:** the GPU P4 patches (fix their
   spec first), thread counts per machine in the tuner, the increment-form
   linear step for complex64 (recommend only), the fewer-passes lessons on
   MLX and OpenCL, the Mac profiles, a closing review pass and the final
   report. Version 5.0.0 is neither bumped nor published.

#### Night of 2026-11-27: landed, decided, left

- **Coupled Strang runs were wrong in 4.0.0 and 4.1.0** (`fix/cnlse-strang-merge`,
  found by the solver reviewer, reproduced on both tags): `CNLSE` and
  `CNLSE_1d` with `splitting="strang"` and no callback bracketed the run with
  the scalar `_nonlinear_step` (no `n12`, no `alpha2`). Fixed by opting the
  coupled solvers out of the merge (`_lie_step_is_strang_body = False`), the
  smallest correct fix. Reviewed (SHIP). Left:
  - A coupled `_nonlinear_step` would let lossless coupled runs merge again.
    The cost of not merging is not measured: one real-space pass per step.
  - `_merges_strang_halves` still declines on `omega`, which now only
    `NLSE_3d` trips, by a name clash (its `omega` is the frequency axis): it
    would merge correctly (3.5e-15 in complex128, the reviewer's run). No
    test covers that branch since the Rabi test went.
  - `docs/optimization-log.md:210-212` ("Already built") still reads as if
    the coupled solvers merged.
  - Not run: CuPy and OpenCL (the box was timing); DDGPE read only.
- **Solver review, round 1** (`fix/solver-review-round1`, eight findings of
  a fresh-context review, then two fix rounds, each on a fresh review).
  Landed: every
  loss read goes through `_loss_constants` / `_peak_loss_half` (step
  ceiling, RK4 bound, Yoshida warning); `adapt_delta_z` returns `None` at a
  zero rate, is capped by `_split_step_max_dz` and counts `g12`
  (`_max_interaction_rate`); RK4 gets neither splitting advice nor the
  Strang bracket (in 4.1.0 too; rel. diff 1.8 at `n2` = -1e-9 m²/W, 1 W on
  a 2 mm window, `L` = 5 cm, 64², default step, and 1.6e-6 at the test's
  parameters, hence its `array_equal`); GPE keeps the
  caller's `sat`; DDGPE's `gamma2` is `Parameter("alpha2")`, so the cavity
  loss ends at `T`; `_estimated_rates` deleted. `ty-baseline.json` 99 -> 87.
  Decided alone: the `gamma2`/`alpha2` alias (public, in the changelog).
  Left, all MINOR:
  - `out_field` still cuts an RK4 run's step by `YOSHIDA_SPAN` when given
    `splitting="yoshida"`, and `adapt_delta_z` caps an RK4 run by the
    split-step limit, not `_rk4_max_dz`.
  - `adapt_delta_z` on a batched `CNLSE` called outside a run takes the max
    over cross pairs of simulations (11599.7 where 5799.9 is due): the
    fallback constants have shape (B,1,1,1). Right in a run; safe side.
  - `adapt_delta_z` costs one transform a call now: 1.84 -> 5.91 ms (NLSE)
    and 3.86 -> 15.54 ms (CNLSE) at 1024², CPU, the reviewer's timing.
  - Past the medium the step in force differs from the medium's last by
    6e-8 relative (a float32 round trip somewhere in the restore), unexplained.
  - `_loss_max_dz` turns `alpha / 2` back into `alpha` through a bare 2.
  - No test of `adapt_delta_z` on DDGPE; `alpha2` is not in
    `CNLSE._gpu_param_attrs`, so a batched `alpha2` on a device is unverified.
  - The RK4-ignores-splitting test runs `NLSE` only; the second reviewer
    also ran `NLSE_1d` by hand, nobody ran `NLSE_3d` or `GPE`.
  - `method` is never validated: `"rk4"` runs split-step without a word.
  - Not run: CuPy and OpenCL for this series.
- **Backend review, round 1** (`fix/backend-review-round1`, seven findings
  B1-B7 of a fresh-context review; the architect's six verified patches,
  write-up in `~/Documents/LKB/NLSE_tangui_fix_specs/review_round1/backends.md`;
  second fresh review: SHIP). Landed: `NLSE_BACKEND` is read once, in
  `get_backend` (`_installed_backend` builds without override or fallback);
  the benchmark cache is `{"version": "4", "grids": {...}}`, one record per
  grid; `_check_precision_support` moves a complex128 run off a backend
  without float64; the tuning lock holds `host pid thread` and
  `_release_lock` removes only its own; `MLXBackend.ifft(normalize=False)`
  raises; `_collapse_singletons` hands a one-value array over as its value.
  Decided alone: B3 moves the run with a warning, as the batched coupled run
  already does, where the review suggested `ValueError`.
  Step time unchanged, scalar `alpha`, complex64, CPU, `NLSE_CPU_BLOCKING=0`,
  100 steps, three alternated processes: 415-416 against 415-417 ms at
  1024². At 256² compare in one environment: this worktree's interpreter
  (3.13.18) is 14% slower than the main tree's (3.13.13) on the same code.
  Left, all MINOR:
  - The two "moved to another backend" warnings point inside the library
    (`stacklevel=3`), not at the caller's `out_field` line.
  - The move is for the solver's life: a later complex64 run stays on CPU.
  - `save_benchmark_cache` reads, adds and writes without a lock: two
    processes benchmarking different grids at once can lose one record.
  - `_read_cache_file` says "predates a change to what it timed" for a file
    with no `version` at all; `_backend_builders` builds its dict per call.
  - `_collapse_singletons` runs per kernel call (2-4% at 64², the reviewer's
    timing; nothing at 1024²).
  - `test_a_one_value_array_runs_the_field_in_one_call` counts calls on a
    stand-in kernel, not the real ones.
  - Not checked: CuPy, an OpenCL device without fp64, Windows, a cache
    directory on a shared network home (the lock's host name).

- **Solver review, round 1: what was not taken.** Lines are the review's,
  stale since; find by name.
  - `_dispersion_max_dz` binds on a linear problem (`n2 = 0`, 64²: 0.1 m cut
    to 7.17e-4 m with a warning) though split-step is exact there (its M6).
  - `NLSE_3d`: the time axis is `linspace(..., endpoint=True)` on a periodic
    grid, so `delta_T` and the power are off by `NZ / (NZ - 1)`; `vg` is
    stored and never read.
  - `NLSE_1d` builds the 2D grids at the default `NY = 1024` (about 100 MB
    for `NX = 4096`) and benchmarks the backend at a 2D size.
  - `callback=[]` raises `IndexError`; a later callback's returned step
    silently overrides an earlier one. The step warning's `stacklevel=2`
    points into `nlse.py`.
  - `_norm_grid_factor` and `_norm_constant` are float32: a complex128 run's
    power carries 3e-8. `_nonlocal_intensity` allocates `A * conj(A)` a step.
  - `_can_rate_on_device` builds a tuple whose first entry it never reads;
    `adapt_delta_z_to_error` has a bare `1e5` and `order = 2`, and with
    `L = 0` (GPE) its floor is 0; a NaN error reaches `_check_step`.
  - `DDGPE._random` is chosen once by backend name and uses numpy's global
    state; `DDGPE.out_field` breaks `NLSE.out_field`'s positional order;
    `wvl=1e-30` (DDGPE) and `wvl=2*pi/m` (GPE) fake a wavelength.
  - `evaluate_delta_n` is optical only; `norm` is an unscaled sum; the
    callbacks reach about 12 private members; `_make_step` nests three
    closures (the no-nested-defs rule) and `_run_propagation` takes 12
    parameters.
  - Its five structural suggestions, none started: the nonlinear step owned
    by the class with the physics; one `_k_squared` hook for the override
    quartet (about 120 lines); a tuple for the 14 coupled constants; a
    `propagator_for(dz)` context for the three swap-and-restore copies;
    `nlse.py` (2800 lines) split into run state, caches and loops.
- **Docs review, round 1** (`docs/review-round1`, `966508a`). Fixed: the
  callback pages' sample counts (`n_steps // save_every`), MLX's place in
  the detection order, what `NLSE_QUIET` silences. Lint and format now cover
  `benchmarks/` and `docs/`; the formatter skips `*.md`. Left:
  - `docs/nlse_tutorial.ipynb` cell 23 sizes its samples `N_samples + 1`:
    the notebook stores its outputs, so it is the owner's to re-run.
  - `CLAUDE.md` is the owner's to edit. Stale there: line 7 (CuPy transforms
    with cuFFT, not VkFFT), 34-35 (lint paths), 57 (Strang is "still 1
    pair/step" for the scalar solvers only; a complex128 field moves off
    MLX), 87 (`D101` is not ignored), and `CACHE_VERSION` is 4.
- **Test-suite review** (mutation, about 65 mutants). Its four MAJOR gaps
  are item 1 above. MINOR, not taken: `cpu_tuning._cache_key` on a
  non-square grid; `plot_field`'s Y extent and intensity panel (every test
  has a square window); the default cache directory per platform (every
  test sets `NLSE_CACHE_DIR`); the loss cap's lower bound; type-only or
  self-computed assertions (`test_auto_backend.py` near 142,
  `test_benchmark.py` near 125-160 and 449); helpers copied across about 70
  test files; `test_packaging.py` needs the network.
- **Kernel review, round 1** (fresh context, at `3a144d2`; fix on
  `fix/coupled-direct-potential-width`). Read: `kernels/cpu.py`, `cl.py`
  with `kernels.cl`, `mlx_kernels.py`, and how the solvers call them. **Not
  read**: `cpu_blocked.py`, `cupy.py`, `cupy_kernels.py`, `cuda_source/`,
  `templating.py`; no mutants run. That is the next round.
  - Nothing found in: the scalar kernels against a DOP853 reference
    (complex128 on CPU, within 2.5e-6, the three-pass tolerance); CPU, CL and
    MLX in complex64 against CPU complex128 (within 2e-5 over `alpha` in
    {0, ±0.8, 3}, `dz` in {±0.05, 0.4}, `Isat` in {inf, 2}, no, real and
    complex `V`); the coupled kernels across the three backends (1e-8); the
    Rabi sign; the wrapped exports' argument order; the dispatch.
  - Its one MAJOR is known and documented, and was not taken: the coupled
    real-space step freezes the loss, local error O(dz²) (5.9e-4, 1.5e-4,
    3.76e-5, 9.4e-6 at `dz` = 0.04 to 0.005, `alpha1` = 1, `Isat` = (2, 3),
    the same on the three backends). `docs/numerical_methods.md`,
    `docs/physical_validation.md` and the log's *Open* section say so, and
    the log says why the one-channel closed form does not carry over. It is
    a feature, and the first one worth the owner's time: DDGPE lives there.
  - Fixed: a coupled `split_step` called directly scaled `V` by a float32
    `k / 2` (2.75e-9 rad on 0.1 rad in complex128, now 2.8e-17); kernel
    comments where "single/double" meant Lie/Strang. Its review said ship;
    left: on MLX that fallback copies `V` to the host on every direct call
    (`np.iscomplexobj` on an MLX array), and it was not run on CuPy, with
    `CNLSE_1d` or with a batched `V`.
  - Checked, not a bug: a batched parameter on an unbatched field under RK4
    raises `ValueError` on CPU and on MLX alike (MLX's message, "Cannot
    broadcast array of shape (3,64,64) into shape (64,64)", is the poorer);
    a batched field on MLX agrees with CPU to 1.3e-6.
  - MINOR, left: the loss limit 0.1 and the three passes are written once
    per dialect (held together by a text test since round 2), and the CL
    and MLX complex64 lossy step is
    pinned only through the cross-backend comparison, which needs float64
    and is skipped here: a complex64 test against the ODE would close both.
    The CPU kernels with a potential have no lossless branch (the ones
    without have a rotation): a lossless step with `V` runs the three passes
    at `u = 0`. Exact, cost unmeasured.
- **Test-suite gaps** (`test/review-gaps`, the architect's five commits,
  write-up `review_round1/tests.md` in the specs directory; fresh review:
  SHIP, 37 mutants, 33 killed, and 12 of 13 survive the old tests). New
  tests: the frozen-loss step against its closed form on CPU, CL and MLX;
  the cadence of `adapt_delta_z_to_error`; the propagator cache key; the
  benchmark's choice of the fastest backend; the loss cap on the step. No
  library bug found. The commits carry `Co-Authored-By: Claude Opus 5.5`,
  the model that wrote them. Left, all MINOR:
  - `tests/solvers/test_nlse.py` near 90: the dtype case of the propagator
    key never asserts the dtype (a propagator always built at complex64
    passes).
  - `test_lossy_substep.py`: `CAPPED_STEP_FRACTION` is imported, so 0.09 for
    0.9 passes; the cap is not tried at its boundary (`> 2 * max_dz`
    passes) nor on a backward step (a lost `copysign` passes); one test's
    name says "frozen" though its lossless cases take the solved path.
  - Not covered: skipped cadence steps still appending to `delta_z`; the
    `scale == 0` branch of `adapt_delta_z_to_error`; the NX/NY terms of the
    cache key (the cache is per instance).
- **Kernel review, round 2** (fresh context, at `b6fa684`: `cpu_blocked.py`,
  `cupy.py`, `cupy_kernels.py`, `cuda_source/`, `templating.py`; tests on
  `test/kernel-review-2`). No arithmetic bug. 23 mutants on
  `cpu_blocked.py`, 18 killed; the five left: two equivalent `min()` clips,
  the tuning lock removed (its test is skipped on macOS,
  `test_cpu_blocked_step.py` near 879), an over-sized scratch, the copy-back
  branch no test reaches. Its two MAJORs, both answered with tests:
  - No test ran the CUDA source off a device. Now
    `tests/backends/test_cuda_source_on_host.py` builds the step and the
    store callbacks as C++ behind a shim and holds them to the CPU kernels
    (1.6 eps worst; Apple clang 21 and GCC 16.2.1 on the box). Skipped
    without `c++` and on Windows. It shares the CPU's algorithm, so a wrong
    shared formula passes; the shim's `float2`/`double2` are not aligned as
    nvcc's (8 and 16), and the out-of-line call is probably inlined.
  - The loss limit and pass count are written in five dialects.
    `test_each_dialect_carries_the_cpu_loss_model` reads the four copies as
    text against `cpu.py`. Decided alone: a drift test, not one templated
    source; the polynomial coefficients are not held, and `kernels.cl` is
    still never compiled without a device.
  Left, all MINOR: solver-level blocked parity runs only at a square 32
  (the reviewer's 37x53 sweep agreed to 3.5-6 eps); a negative `dz` in the
  blocked step is run only near -1e-9; `_takes_blocked_step` checks no
  dtype; `cpu_tuning.py` (911 lines) tunes with no potential, no loss and
  Kerr only; `_wait` and a partly failed submit, and `KeyboardInterrupt`,
  are untested; the CUDA kernels index with int32 (over 2^31 elements);
  `blocked_step()` costs about 1.8 us a call. Not checked by it: a real
  CuPy/cuFFT run, OpenCL and MLX execution, the tuning's quality.
- **Examples and benchmarks review, round 1**
  (`fix/examples-benchmarks-review`; second fresh review: one MAJOR, fixed).
  Landed:
  - `examples/ddgpe.py` ran with no pump and no loss since 2024 and plotted
    zeros. Decided alone: the pump (3) and decay (0.07 meV) it was written
    with are back; 53 s on CPU, the total still rising 3% over the last
    20 ps.
  - `benchmarks/profile_kernels.py` passed one cross constant to the coupled
    fused kernels, which take two (a `TypeError` on both rows).
  - `examples/vortex_precession_benchmark.py` and `vortex_precession.jl`
    leave the first run of a size out of the timings. The Julia line was
    parsed, not run (FourierGPE.jl at 8192² is hours).
  - New: `test_a_uniform_pump_reaches_the_linear_steady_state` pins the
    driven DDGPE (drive sign, `omega / 2`, `gamma / 2`, the detuning's sign)
    against the two-mode closed form; no such test existed. First order in
    the step: 8.0e-3 at 1600 steps, tolerance 1.5e-2, so a 1% error passes;
    Lie only, uniform, `g = 0`.
  Left, all MINOR:
  - `profile_backends.py` and `propagator_fusion.py` label a row by the
    backend asked for; `get_backend` falls back with a warning when it is
    not installed, so the row can be another backend's.
  - `work_precision.py`: the float32 floor, and a `ceil` near 218.
  - `examples/benchmarks.py` and `fig1_benchmarks.py`: `E_cpu` is unbound
    with no CPU row; the check runs only at `i == 0`; `N_AVG = 2`; the two
    files are near copies.
  - `examples/ddgpe.py`: `gamma` and `F_pump` are set inside `main`, and
    `F_pump = 3` has no source; `plot_field` labels its axes in mm, which
    this example's units are not.
  - `jones_roberts_solitons.py`: the docstring's numbers are not verified.
    `tests/test_examples.py` and `tests/test_benchmarks.py` parse, never
    run. Nested functions in examples nobody touched.
  - Debloat candidates, looked at and dismissed (decided alone):
    `fig1_benchmarks.py` into `benchmarks.py` (it is the paper's figure 1,
    backends by method, and its own gallery page; 125 lines);
    `vortex_precession_animation.py` into `vortex_precession_benchmark.py`
    (189 of their 257 lines differ: an animation and a timing, two gallery
    pages); `vortex_shedding.py` split (moves 810 lines, removes none, and a
    gallery example is one file).
  - Dismissed: a MAJOR on `rayleigh_taylor.py` (false, V is minus a
    potential energy there); `NOISE = 0.0` is deliberate.
  - Not opened: `trace_solvers.py`, `nsys_summary.py`,
    `tests/benchmarks/test_benchmark_profile.py`, the docs pages that quote
    the examples, any CuPy path.
- **Decided alone** (revert what does not suit): the DDGPE example's pump
  and decay; a drift test for the loss model rather than one source; fixes
  of three files or more made inline where each file's change was a line or
  two (no architect, no implementer), each branch then given to a fresh
  reviewer; coupled solvers opted out
  of the Strang merge rather than given their own half step; RK4 ignores
  `splitting`; `gamma2` is `alpha2`; a complex128 run leaves MLX with a
  warning; architects' verified patch series were applied directly, no
  implementer; reviewers ran as background agents, no workflow; `CLAUDE.md`
  and the tutorial notebook untouched; the 55 local branches merged into
  `main` are kept (`git branch --merged main`).

The final report has to say, beyond what "Review findings left as they are"
and "Still open" list:

- Not reviewed: the `pyproject.toml` cleanup (T6, B8), the docs review's fix
  round (no second reviewer), my own edits to architect assets and the
  re-cut of entry 6, the scratch fix written inline.
- Not run or not seen: the Windows mmap branch; the fork test outside CI on
  macOS; the deployed docs site by eye; CuPy and OpenCL claims of the docs
  pages (read, not run); the benchmark reviewer could not run CuPy or OpenCL;
  `examples/` edits covered by ruff and grep only; implementers ran ruff on
  their own files only.
- Inferred, not measured: the page-colouring mechanism behind the blocked
  step's scatter; the batched-`n2` Strang merge was not traced.
- Loose ends: an unreproduced one-off P4 failure; the zero-length-axis MINOR;
  four unsourced numbers in the guide pages; the docs equations are not
  pinned by a test; the pins brief counted 88 against 93; the CI watcher was
  restarted once; gallery cache entries accumulate, one per run, and a stale
  one is a trade-off the reviewer noted; the box's verification commits; the
  architect's 64² smoke run may have left an entry in the default NLSE cache;
  an architect deleted a scratch directory named `r2` by mistake (only its
  own transcript names it).
- The owner's to edit: `CLAUDE.md` lines 86-87 (stale ruff ignores), line
  7 (CuPy uses cuFFT, not VkFFT), lines 34-35 (ruff now also takes
  `benchmarks/ docs/`) and line 57 (Strang is one pair per step with or
  without the merge, and only the scalar solvers merge); the debloat and
  docs decisions left open.

## Open work

### Packages not built

- Thread counts per machine, not built (next package, through the
  pipeline): on the M3 Max (12 P and 4 E cores) numba's `prange` kernels
  are 21-34% faster on 12 threads than on 16 at 1024², 2048² and 4096² in
  both dtypes, and the whole unblocked step on 12 against 16 threads is
  1.18 against 1.54 ms at 1024² complex64, 2.12 against 2.40 at complex128,
  5.85 against 5.58 at 2048² complex64, 27.8 against 26.0 at 4096²
  complex64, 43.3 against 45.9 at 4096² complex128 (lossless, best of the
  interleaved rounds). `fft2` alone prefers 12 at 1024² (0.404 against
  0.533 ms complex64) and 16 at 2048² and 4096² complex64 (9.6 against
  11.4 ms). So kernel threads and transform workers want separate counts,
  measured by the tuner and cached with the block; the blocked step's
  candidates (`THREAD_DIVISORS`: 16 and 8 here) should include the P-core
  count too.
- GPU fixed costs and the coupled runs that leave the fused step:
  `NLSE_tangui_fix_specs/gpu_p4_spec.md` with `gpu_p4_T1..T4.patch` (base
  `f26b694`), written and measured by an architect, nothing applied. Spec
  review: the patches apply stacked on `main` and on the blocked step with
  line offsets only, and pass the Mac suite (1171 passed, 137 skipped in
  `tests/backends` and `tests/solvers`), but fix the spec first: every
  expected count was measured at `f26b694` and is stale (re-measure on
  `main`, on the box too, where `main`'s newer Rabi and DDGPE tests have
  never run against T1; T4 has no OpenCL baseline); the box commands lack
  directory and environment; no task writes the CHANGELOG and the log, and
  `ab.py` and `calls.py` exist only in `~/Documents/nlse_scratch_scripts/p4/`
  on the box. Minor: DDGPE takes the fused kernels for the first time and
  the spec does not say so; the Rabi angle now comes from the double `dz`,
  so complex64 equality to the bit is observed, not guaranteed; bare `1`
  and `2` for the mode in `kernels.cu`; nothing fails if T4's flag is
  reverted. The reviewer's stage-by-stage local counts are in the session
  transcript; re-measure rather than trust them. T1 Rabi coupling in the fused step (CuPy CNLSE 2048² strang
  8.43 to 4.55 ms, RK4 43.9 to 20.0), T2 `_energy_rates` once a call, T3
  the propagator stays on the GPU, T4 a fused RK4 stage on OpenCL (8 to
  13%). Left to the owner, with options in the spec: the CNLSE field
  download and the potential grids rebuilt every call, and VkFFT
  `tune_config` on OpenCL.
- **complex64 accuracy is set by the transforms**, not the kernels: a round
  trip loses 2.8 units of 2^-24 of power, every step. Writing the linear step
  as `A + ifft((P - 1) fft(A))` cuts the field error 40 to 390 times for 7 to
  26% of a CPU step (prototype, optimization log, *Open*). The user has not
  decided: recommend only.
- Per-call costs from the CUDA-box profile, below: the propagator download,
  `_energy_rates` twice, the potential grids.
- MLX and OpenCL: the fewer-passes lessons are not ported.
- The Mac profiles (CPU and MLX): see "Performance profiles".

### Left open by merged work

- **Debloat.** Specs in `debloat/` (`repo.md`, `package.md`, `tests.md`).
  The package reviewer's minor notes, not fixed: three shortened comments
  lost their reason (`kernels.cu:287` why `A_orig` may alias `out`;
  `ddgpe.py` near 296 why one profile at a time; `cpu.py` near 20 "not
  `fastmath=True`").
  - Left for the owner, not done: the tutorial notebook's 8 MB of outputs
    (re-render smaller), `paper/` and `img/output.png` (the JOSS record),
    `examples/juliaGPE_vs_NLSE.py` with `vortex_precession.jl`, merging
    `examples/benchmarks.py` and `fig1_benchmarks.py`, the `kernels_cpu/gpu/cl`
    aliases, the `splitting="RK4"` mapping, the `speedup` field of the
    benchmark cache, `tests/benchmarks/` with the CI benchmark job, and one
    module for the per-solver legacy tests. `examples/profile_cupy.py`
    (overlaps `benchmarks/trace_solvers.py`) is decided in the docs pass.
- **Docs gallery** (`18f4cc7`). The Documentation job keeps
  `docs/auto_examples` in the Actions cache and sphinx-gallery skips an
  example whose source is unchanged; a weekly `schedule` run and "Run
  workflow" execute them all. Accepted trade-off: a change to `NLSE/` that
  breaks an example passes CI until that run. Design and what to watch:
  `docs/gallery.md`, section 3. Not checked yet: the cache on GitHub
  itself. For the owner: the weekly run also runs the test matrix and the
  benchmark job; gate them on the event if unwanted.
- **Docs page review** (`3bd8af5`). Specs, with the decisions left to the
  owner: `docs/pages_entry.md` (decisions 1-7) and `docs/pages_guide.md`
  (D1-D5).
    Not pinned by a test: the equations themselves (LaTeX against the
    kernels), so a sign flipped on a page is caught by review only.
    Left as they are, for the owner: the README still carries the install
    steps and one example (entry decision 1) and its relative links do not
    resolve on PyPI; the changelog entry does not mention the CuPy wheel
    correction; `contributing.md` lists the dev group without pytest-cov
    and pytest-benchmark; no page gives the tests' layout any more;
    `literature-examples.md` keeps four numbers no script reproduces (Mach
    0.3 and 2.0 densities, 85 000 singularities, ~30x, 7%); `backends.md`
    says `NUMBA_THREADING_LAYER` picks the layer, which
    `NLSE/kernels/cpu.py:14-16` overrides when pyfftw is imported first;
    the DDGPE equation leaves out `g2`. The CuPy and OpenCL claims of the
    pages were checked by reading the code, not by a run.
- **CuPy two-axis batch without the callbacks** (`083a2d9`).
  - Reviewer MINORs left: a non-contiguous field would be reshaped to a copy
    and the multiply lost (no caller passes one; the old code was wrong there
    too); the branch does not check that the propagator is the field's
    trailing block; no two-axis case for RK4 or a coupled field in that file
    (`tests/integration/test_broadcasting.py` covers RK4 with the callbacks
    off only when run with `NLSE_FUSE_PROPAGATOR=0`); one launch per member.
  - After a change to the CuPy kernels, run the box suite once more with
    `NLSE_FUSE_PROPAGATOR=0`: the default never reaches the unfused kernels
    on a batch, and CI has no GPU.
- **CPU blocked step** (`7518025`, `4bb3432`). Spec:
  `cpu_blocked_step_spec.md`; logs and scripts: `cpu_blocked_step/ryzen/`.
  `MARGIN` and measuring on first use (`TUNE_ON_FIRST_RUN`: 0.4-0.9 s at
  1024², 12.1-12.5 s at 4096² complex128 inside the first `out_field`) are
  the owner's to confirm.
  At `MARGIN = 0.10` the 1.07x at 2048² complex128 on the Mac is not taken.
  The plain step's `fft2` at 1024² complex128 has two speeds in a long-lived
  process; why is not established. That the page order is the mechanism of
  the scratch fix is inferred (no counter read). Huge pages for the scratch
  were tried and rejected. The Windows branch of `_private_mapping` has run
  nowhere; a grid with a zero-length axis now raises from `mmap`.
  Rule from the owner (2026-11-27): hot working buffers come from "a fixed
  memory stack", a region the library maps and keeps, not from malloc. The
  factors and the solver's field were not moved there: no cell needs it
  (measured after another grid ran in the process). My reading, to confirm
  with the owner, is that the rule asks for it only where a measurement
  shows a cell moving.
  Left as they are (MINOR): a declined hook call costs 1.8-3.9 us a step;
  the cache key leaves out numba's threading layer; timestamps are naive
  local time; with an unwritable cache directory every process tunes; a
  process that misses the cache just as the lock holder finishes tunes a
  grid twice, and two threads of one process see each other as "another
  process"; an interrupt inside `_wait` releases the step's lock while
  shares still write; all candidates' scratch stays allocated while
  tuning; the autouse fixture runs every other test unblocked. Mutants the
  suite lets through: the row factor without its 1/N, the time not divided
  by `STEPS`, `DUEL_ROUNDS = 1`, `THREAD_DIVISORS`, the warm-up removed.
  Open cost: where the blocked step is taken (and on CUPY) the first step
  of a solver builds the factors, about 0.17 s and 530 MiB transient at
  4096², repaid in about 14 steps.
- **CPU float32 kernels** (`71db255`). Left as they are: the float32 angle
  reduction is exact only below 8192 rad and unbounded past about 3e7;
  `benchmarks/roofline.py` still measures the double `_sincos` and `_exp` as
  its transcendental ceiling, which is no longer what a complex64 run calls.
- **CuPy nonlinear callback** (`5113580`). Scripts: `cuda_fewer_passes/`,
  copies on the box in `~/Documents/nlse_scratch_scripts/`.
  - Reviewer, two rounds: no critical or major finding left. Minor, not
    fixed: no test runs the callback above 64² or at a non-power-of-two size
    (checked by hand at 13 sizes); `ifft_nonlinear_ready`'s refusals
    (non-contiguous field, `V` of another shape or width) have no test; the
    Strang half step handed to the callback is pinned only by the box's
    bitwise tests; `nonlinear_words` raises `OverflowError` on a finite
    complex64 scalar above the float32 maximum (direct kernel calls only);
    a nested `def refuse` in `tests/backends/test_fft_propagator_fusion.py`;
    bare `2` and `(1,)` in `nlse.py` and `cupy_backend.py` lines the change
    sits next to; `NLSE_FUSE_PROPAGATOR` now names two callbacks; a cuFFT
    without LTO callbacks pays three failed builds per plan.
- **CuPy separable propagator** (`ce290d8`). complex128 runs at 0.97 to
  0.99x: kept under the one-path decision, for the user to accept or not.
  - Reviewer: pass, minor findings only, not fixed. With
    `NLSE_FUSE_PROPAGATOR=0` the solver still builds the factors the plan
    then refuses (0.1 to 0.2 s and 0.5 GB of host memory at 4096², the
    reviewer's figures). The axes come from `_compute_propagator_rk4`, so a
    subclass overriding only `_dispersion_on_device` would get factors that
    do not describe its propagator (no such subclass exists).
    `_take_partial_step` swaps the linear state before its `try`, as before
    the change. `FUSED_TRANSFORM_AXES` is reused as an ndim.
    `_forget_device_state` does not clear `_propagator_fft` (only
    `_forget_run_state` does). The fallback `if` in `linear_nonlinear_step`
    has no `else`. Seven returns in `cupy_backend.py:126`; bare `1j` and `2`
    in the two test files. A 32-bit divide in the callback, which the
    reviewer suggested, was measured and changes nothing.
- **Serial numba kernels**, learned on the blocked-step prototype:
  - A serial numba loop keeps Python's division check and does not
    vectorize (15 against 2.5 ns a point on the box): the serial kernels
    need `error_model="numpy"`. Two dispatchers made from one Python function
    with `cache=True` share one cache entry on disk, whatever their options:
    the second loads the first's machine code (reproduced). Serial twins of
    the parallel kernels cannot be cached that way.
- **Tests that bite** (`9487346`). 5 mutants are left without a test
  (`tests_bite_spec.md` says why); `test_audit/crosscheck.py --survivors` is
  the regression check. A batch and its members run alone differ by 5e-4
  when the step is left to the solver (the batch takes the strictest
  member's step): a test comparing the two has to pass `delta_z`. Left by
  the reviewer, minor, not fixed:
  - `tests/solvers/test_plot_labels.py` draws a field of unit modulus, so
    `|A|` for `|A|**2` in the plotted intensity passes. Modulus 2 and
    `4 * W_PER_CM2` fix it. The `CNLSE` panels have no value check.
  - `test_a_capped_step_lands_inside_the_limit`
    (`tests/solvers/test_solver_state.py`) asserts `0 < capped < bound`
    only: `CAPPED_STEP_FRACTION = 0.3` passes. Compare with the constant.
  - `WAVENUMBER_CHANGE` is defined in `test_dispersion_built_once.py` and in
    `test_closed_forms.py`; `MIXED_N2` restates a literal of the same file.
  - `tests/solvers/test_cnlse.py` relies on `helpers.make(symmetric=False)`
    giving the second component other constants and does not assert it.
  - The docstring in `tests/test_doc_claims.py` says the orders and costs
    are measured; only `yoshida` is tied to the code.
  - Older than this work, seen while merging it: `tests/test_doc_claims.py`
    globs `docs/**/*.md`, so a local Sphinx build adds
    `docs/_build/html/_static/mathjax/VENDORED.md` and five cases (1611
    passed in this checkout, 1606 in a clean one); and
    `test_a_wrong_typed_field_is_no_cache` (`tests/backends/test_benchmark.py`)
    carries a timestamp taken at import in its test id.
- **Packaging and tooling** (`62bd2e8`). ty 0.0.112 checks `NLSE/` against
  `ty-baseline.json`; a change that moves a recorded diagnostic needs
  `uv run ty check --update-baseline`, after checking that the diagnostics
  are the recorded ones.
  - Not done, for the user to decide: the licence expression
    (`GPL-3.0-only` or `GPL-3.0-or-later`, with `license-files` and
    hatchling 1.27), dropping Python 3.10, ruff 0.17.6, ty on `tests/`
    (456 diagnostics), pinning the actions by hash and bumping their
    majors, an SBOM artifact and a commit-message hook.
  - Not checked: the CuPy and OpenCL extras were not synced from the lock
    on the box.
- Defects the audit found, the first two for the owner (a third,
  `_loss_max_dz` reading `alpha` only, was fixed on 2026-11-28):
  - `_loss_max_dz` (`NLSE/solvers/step_size.py`) divides by `2 * alpha` while
    the kernels receive `alpha / 2`: the kernel's `u` at the cap is 0.025,
    and the comments there and in `NLSE/kernels/cpu.py` say 0.05. Safe side.
    Change the code or the comments.
  - `DDGPE` flips the sign of `g` but not of `g12` nor `g2`
    (`NLSE/solvers/ddgpe.py`); `tests/solvers/test_parameter_names.py` pins
    the current sign.
  - `benchmarks/nsys_summary.py`: a "Total Time"-only table parses to an
    empty list, and `parse()` reads on into the next table.
  - The comments "Lossless, or too lossy to solve" in `NLSE/kernels/cpu.py`
    are wrong: a lossless step takes the solved loop.
- OpenCL batches, left by the reviewer of `fix/cl-multi-axis-batch`:
  - A potential that differs between simulations whose parameters differ is
    refused (`_launch_count`, `NLSE/kernels/cl.py`); it was silently wrong.
    Running it needs one sub-buffer of `V` per launch
    (`Buffer.get_sub_region`), not a kernel argument: the comment above
    `KERNEL_DECL` records that an `int` shift costs about 3x.
  - A parameter varying over the grid axes (`n2` of shape `(N, N)`) is not
    refused: it runs one launch per grid point, right but very slow. CPU
    raises a numba `TypingError` there.
  - `_launches` reads only `A.shape` and `A.size` (PYS003); nine callers.
  - No test on OpenCL for a batch over three axes, for 1D, or for `NLSE_3d`
    (the reviewer's 3D probe hung, cause not found).
- The probes behind the log's transform-library entry are not in the repo:
  `NLSE_tangui_fix_specs/cpu_fft/` (`vdsp_rows.py`, `vdsp_scale.py`,
  `results_*.txt`) and, on the box, `~/Documents/nlse_scratch_scripts/fft/fft_libs.py`
  with the venv `fftw_venv`. Move them to `benchmarks/` if the entry is
  challenged.
- Left by the reviewer of the batch fix, all minor, none fixed. They belong
  with the tests that bite:
  - `_pick_field` (`NLSE/kernels/cpu.py`) does not peel a leading axis of
    size 1: `V` of shape `(1, 4, N, N)` with `n2` of shape `(3, 1, 1, 1)` and
    a `(3, 4, N, N)` field raises `IndexError`. `value[b if value.shape[0] > 1
    else 0]` fixes it.
  - `_shared_grid_batch_len` does not see `V` of shape `(1, N, N)` under a
    `(3, N, N)` field with scalar parameters as shared: the kernel is called
    once and its flat index runs past `V`. Older than the fix; values not
    checked.
  - A batched `omega` of shape `(B, 1, 1, 1)` fails under RK4 in
    `_add_rabi_slope` (`NLSE/solvers/cnlse.py`). Older than the fix.
  - The error for a parameter that does not broadcast names the shape of the
    sliced member, not of the field the caller passed.
  - The two-axis test is 3 by 3: give `ALPHA_VALUES` a fourth value so a
    swapped axis cannot pass.
  - The `wrapper` inside `_broadcast_batch` now holds the loop and the
    recursion; a module-level `_run_batch(kernel, positions, n_outputs,
    args)` would leave the factory pure.
  - Not run by that review: OpenCL, CuPy, DDGPE, complex128, `GPE` with `V`,
    `NLSE_3d` over two axes, strang and yoshida against per-member runs.
- Scratch to delete. Box: `~/Documents/nlse_cuda_sp2c` (= `main` after the
  merge of 2b, less the docs), `nlse_cpu32`, `nlse_cuda_fp` (= 5113580, the
  tree the CPU prototypes import). Mac: `git worktree list` (this session's
  scratchpad, `~/Documents/LKB/NLSE_wt/`).

### Review findings left as they are (MINOR)

- `tests/backends/test_cpu_blocked_step.py`, the skip on `workqueue`: no
  comment says `run()` must come first so that numba has chosen its layer
  (the helper's docstring does), and its reason cites no page (numba's
  page does not state the limit; its abort message does). The fork test
  now runs in CI on macOS only.
- p1a: a callback's shape error surfaces after a step; bare numbers in
  `tests/solvers/test_backward_run.py:151,170` and
  `tests/solvers/test_out_field_arguments.py:40`; a CNLSE strang run with a
  cut step in complex64 is 1.4e-3 from the fine run.
- p1b: no test fails when `_forget_device_state` is dropped from
  `_require_backend`; `_forget_device_state` names the RK4 buffers by hand and
  misses `_rk4_A_sq_c`; `_propagator_rk4_cache_key()` omits the window;
  `ddgpe.py` binds `_random` at construction; `_energy_rates_on_device` moves
  `V` to the host on every call, also on a cache hit.
- p2: RK4 with a batched `omega` of shape `(B,)` or `(B, 1, 1, 1)` raises on
  CPU (`_add_rabi_slope` in `cnlse.py`); `_default_delta_z` leaves
  `_drive_step` set when called directly; no test for `omega` with a
  step-changing callback or a batched `omega`; `docs/solvers_overview.md:263`
  says a positive imaginary potential removes power, which is the NLSE
  convention and now the opposite of DDGPE's.
- p3: the floor of `adapt_delta_z_to_error`, `L/1e5`, is 0 for a GPE; the
  default GPE step ignores the kinetic term; no test with `g < 0`, `V < 0` or
  complex128.
- p4: `nl_length=np.inf` is accepted and gives NaN; the smoothed intensity can
  dip to about -4e-7 of its maximum; comments at `nlse.py:855`,
  `cnlse.py:369`, `examples/nonlocality.py:5` and two test names in
  `tests/integration/test_broadcasting.py` still say "convolve"; no non-local
  test for GPE and DDGPE; `_multiply_in_fourier_space` takes a positional
  bool.
- p5: a batched `omega` on a CNLSE raises TypeError on CuPy.
- p6: the saturated Kerr index is written twice in `callbacks.py`;
  `evaluate_delta_n` is optical only (wrong for GPE and DDGPE, and it reads
  component 1 on a coupled solver); the cache's temporary file is named by PID
  alone; `invalidate_cache` raises if the cache path is a directory;
  `docs/backends.md:51` still gives the old cache path.
- p7: `examples/benchmarks.py:187`, the self-check cannot see a wrong
  diffraction sign; the CI link check only warns; `docs/literature-examples.md`
  still opens with "None should be written from scratch".
- RK4 (review of `36bf9e1..10bf834`): the bound takes the interaction rate
  the field weighs, not its peak, and the first component's loss only, not
  `alpha2`; an RK4 run left to choose its step warns that the step was
  lowered; `_capped_delta_z` computes `_energy_rates` twice per run;
  `CUT_RUNS` in `tests/solvers/test_ddgpe_drive.py:46` writes its integrals
  as numbers instead of deriving them from `CUT_DT`; `cnlse.py:143` nests a
  function (code this work did not touch).
- RK4 (re-review of `10bf834..4ac2bb2`, verdict pass): no test fails when
  the bound reads `|Re V|` for `|V|`, so the absorption of a complex
  potential is unpinned; with `alpha=0`, `alpha2=1e5` and a step of 1e-2 a
  CNLSE RK4 run reaches 2.4e5 (the `alpha2` gap above, run); for a Gaussian
  the weighted interaction rate is half the peak one; `0.1 * peak` is a
  bare number in `tests/solvers/test_solver_state.py`;
  `benchmarks/profile_backends.py --solver CNLSE_1d` raises TypeError
  because `build()` passes `NY` (older than this work).

### Design decisions

- GPE integrates `i hbar dt psi = -hbar^2/2m lap psi + V psi + g|psi|^2 psi`
  with `V` and `g|psi|^2` in rad/s. Breaking.
- A complex `delta_z` raises TypeError; imaginary time is not offered.
- The non-local response is `1/(1 + nl_length^2 (Kx^2 + Ky^2))`, exact on the
  periodic grid, on every backend. A sub-cell `nl_length` is kept; one longer
  than the window is not refused.
- The medium is the first `L` of travel, also on a backward run (no mirror).
  `z < 0` raises ValueError; a negative `delta_z` runs backward to the end,
  also when a callback hands back a step.
- Callbacks read `_current_delta_z` as the step just taken, also on a cut
  step. `_step_in_force` is the step the loop takes next, the one the
  propagator is built from and the one `adapt_delta_z_to_error` scales.
- DDGPE: noise std goes as `sqrt(|dt|)`. `F_pump_t[j]` is the drive during
  `[j dt, (j+1) dt)` with `dt` the step asked of `out_field`; a profile that is
  too short raises ValueError. The potential gets the documented sign
  (`POTENTIAL_SCALE = -1.0`). Breaking.
- NLSE_3d: the temporal dispersion gets the documented sign
  (`dz E = +i D0/2 Omega^2 E`). Breaking.
- The potential's grids on the device are dropped whenever `V` is rescaled
  (`_forget_potential_grids`), rather than keyed by the potential's content.
- RK4's bound counts dispersion and potential by their grid maximum, and the
  interaction and the loss by the field.
- Not done on purpose: skipping the energy-rate estimate when `delta_z` is
  given (the cap on a caller's step needs it).

### Still open, not fixed

- Numba's threading layer limits the CPU backend, and nothing in the library
  or the docs says so: on `workqueue` (no TBB, no OpenMP: the macOS CI
  runner) two `out_field` calls in two threads abort the process, and on GNU
  OpenMP (Linux without TBB) a forked child that runs the CPU backend is
  killed. Neither comes from the blocked step. For the owner: document it
  (docs pass), or hold a lock around the CPU run on `workqueue`.
- `NLSE_3d` never merges its Strang half steps: it stores its frequency
  grid as `self.omega` (`nlse_3d.py:104`), and `_merges_strang_halves`
  (`nlse.py:2330`) reads any `omega` that is not None as a Rabi coupling. A
  3D Strang run pays two real-space passes a step where one would do. Found
  by the docs spec review; not fixed, the docs say it.
- A run with a callback never merges them either (`_loop_with_callbacks`
  does not ask): 20 nonlinear calls for 10 Strang steps against 12
  without, counted on CPU at 64x64. The methods page says so. A batched
  `n2` counted 20 as well, with `_merges_strang_halves` true; not traced.
- The default step exceeds the split-step cap on a default 2D run:
  `_default_delta_z` aims at 0.1 rad of potential and interaction only,
  while the cap also takes the dispersion bound, so the quick start and
  the README example warn "delta_z=... exceeds split-step accuracy limit"
  and run at the cap (2.12e-4 m asked, 1.55e-5 m run, at 2048² on a
  window of four waists). A docstring in `step_size.py`
  (`_capped_delta_z`) says the default is well inside the limit. RK4's
  default was inside its limit on the same run (2.28e-5 m, no warning). The
  1D quick-start block takes 157,241 steps (54 s on CPU) without a warning.
  Found by the docs audit; for the owner.
- Found by the page review, not fixed: `_loss_max_dz` (`step_size.py`)
  bounds `2*alpha*dz` by `LOSS_PER_STEP_LIMIT`, so 2.5% of the intensity a
  step where the constant's comment says 5% (the page quotes the code);
  `ddgpe.py:165-166` passes `n2=-g` but `n12=g12`; `callbacks.py:314`
  floors the error controller's step at `simu.L / 1e5`, which is 0 for a
  GPE (`L=0`); `nlse.py:520` names a `_merged_strang_bracket` that does not
  exist; the solver docstrings offer a backend `"GPU"`;
  `docs/make_validation_figure.py` prints 12 step-cap warnings and 2
  Strang-on-complex128 ones a run.
- `tests/test_doc_claims.py` globs `docs/**/*.md`, build output included
  (`docs/_build/html/_static/mathjax/VENDORED.md`), so it runs 93 tests
  with a built site and 88 without.
- `evaluate_delta_n` convention: `k/2` or `k`.
- Setting `simu.wl` does not update `k`.
- `get_fastest_backend` crashes on a wrong-typed `grid_size`.
- `_normalization_factor` upcasts to float64.
- `tests/benchmarks/test_benchmark_profile.py` runs GPE with `g=n2`.
- `omega=0` (not None) still pays two identity rotations in strang.
- `DDGPE(omega=None)` raises TypeError in the constructor, so its
  `omega is None` branches never run.

### Smaller audit findings, left out of the report

Status on `main` not re-checked.

- `simu.V` becomes a float32 copy after a device run (`nlse.py:1700-1765`).
- Real-valued input fails on a bare `assert` (`nlse.py:357-363`).
- `adapt_delta_z_to_error` is not exported from `NLSE/__init__.py`.
- *(reviewer, not re-checked)*: `examples/benchmarks.py` numpy baseline
  exponentiates the propagator twice; `examples/juliaGPE_vs_NLSE.py` loads
  files nothing writes; README and `docs/numerical_methods.md` disagree on
  yoshida with loss; `docs/installation.md` omits the extras;
  `docs/backends.md:55` cache path; no Unreleased changelog section;
  classifiers stop at 3.12; `--use_fast_math` at
  `NLSE/kernels/cupy_kernels.py:124`; benchmark cache written non-atomically;
  7.8 MB `docs/nlse_tutorial.ipynb` and unused images tracked.

### Decisions of the audit

- Kernels: no wrong-result bug found. CPU, OpenCL and MLX agree to 2e-7
  (float32) over about 2000 cases; CUDA source was read, not run.
- The "empty parameter set" skips in `tests/integration/test_nonlocality.py`
  are benign here: no backend without convolution is installed.

## Performance work (2026-11-26 and 27)

Asked, in these words: "Really check at the lowermost level possible, so we
can see where memory bandwidth is spent. From this point on, we need to
transfer less data."; "Then we'll take the teachings of these optis for the
other platforms. Let's make sure we look at the CPU backend and make sure
everything emits vectorized instructions"; on the CPU FFT, "Revisit these
claims, probably there are newer libraries / updates ? Then also how about
writing our own implementation of threading for vDSP ?"; then "continue alone
through the night. Let's try to squeeze the most out of the hardware. At the
end, do make a thorough review pass, including arch, testing, and docs. Follow
the good practices in Downloads." (`~/Downloads/Good practices.md`: every
timing against the hardware peak, with its configuration; docs along Diátaxis;
facts sourced.)

Open in the task list of the closing review pass (asked on 2026-11-27):

- [ ] **Headline performance in the README** (asked on 2026-11-27: "add the
  headline perf numbers in the README of the repo, in % of max flops/s /
  memory bandwidth (highlighting whether we are compute / or bandwidth
  limited and why)"). Per backend and machine, at 1024², 2048² and 4096²,
  both dtypes: the step time, its share of the measured peak (memory
  bandwidth and flop/s), which of the two limits it, and why. The figures and
  their configurations are in `docs/optimization-log.md`; write this once the
  CPU blocked step has landed, so the CPU row is final.
- [ ] Review of the architecture, the tests and the docs (docs page by page
  against Diátaxis), fix what is critical or major, then push.

Decisions taken by the user:

- CuPy store callbacks: one path at every grid size and dtype ("No no, let's
  not split paths, let's keep the fastest one"). The prototype lost 20 to 26%
  at 1280, 1536, 2560, 3000 and 3072: "Document that gap for non power of 2
  sizes, 20-26% slower is bearable. The most used config is power of 2 !" The
  shipped callback has no such gap (it was the stack spill); what is left is
  256², 3 µs a step slower, which the one-path decision keeps.
- "Typical sizes are 1024, 2048 and 4096": a change is accepted or rejected
  on those three grids. Other sizes are side information. 4096² needs 12
  rounds or more on the RTX 3050 (8 rounds gave 0.88x and 0.99x for a 1.05x).

### Performance profiles (started 2026-11-26, not finished)

Asked, in these words: "I want deep profiles, make suggestions as to how to
improve perf"; "do profiles over several steps to get some statistics. Like 50
steps"; "we need the breakdown for each kernel within the step, both CPU / GPU
and memory".

What to deliver, per backend (CPU and MLX on the Mac, CuPy and OpenCL on the
CUDA box):

- One row per kernel the step calls: each transform, the propagator multiply,
  `square_mod`, each `nl_prop*`, `square_mod_nl_prop[_v]`, the non-local
  multiply and its transforms, `rabi_coupling`, the RK4 stages, normalisation,
  callbacks. The rows must add up to the untraced step, or the gap is named.
- Per row: host CPU time, device time, and memory. "Memory" was read as both
  traffic (against the `benchmarks/roofline.py` ceiling) and footprint
  (allocations per step, peak device and host memory); the user did not say
  which.
- 50 steps or more per cell, median and spread.
- Cells: every solver and method; at least NLSE and CNLSE, lie split step and
  RK4, at 512² and 2048², with and without `V`. Then the options (`alpha`,
  `nl_length`, `omega`, complex128, a batch), small grids, and the cost of a
  call as the intercept of time against step count.
- Then suggestions, ranked: what, `file:line`, measured share, expected gain,
  size, risk. Suggestions, not an implementation. Read
  `docs/optimization-log.md` first: nothing it lists as deployed or rejected
  comes back.

Tools, asked 2026-11-26 in these words: "for profiling use CUDA tools (ncu,
nsys etc ...), whatever metal alternative exists and on the CPU I want
extensive profiling, ideally not sampling, but whatever is the most accurate."

- CUDA (CuPy): `nsys` for the timeline and `ncu` per kernel, not wall-clock
  timers around a synchronize; timers only as a cross-check. Both are
  installed on the box (`/usr/local/cuda/bin`, nsys 2026.1.3). The repo
  already has `benchmarks/trace_solvers.py --nvtx` and
  `benchmarks/nsys_summary.py`.
- OpenCL on that box: not checked whether `nsys` or `ncu` see its kernels.
  If they do not, say so and use OpenCL's own event profiling.
- Metal (MLX): the Mac has `xctrace` 27.0 with the templates "Metal System
  Trace", "CPU Counters", "Allocations", "System Trace" and "Time Profiler",
  and `powermetrics`. None was tried on MLX here: find out which gives GPU
  time per kernel, and say what it cannot give.
- CPU: instrumented rather than sampled, and whichever is the more accurate.
  Installed: `xctrace` "CPU Counters", `dtrace` and `sample` on the Mac;
  `perf` on the box (`perf_event_paranoid` is 2), no `valgrind`. My reading,
  not the user's words: a counter around every kernel call, with hardware
  counters for cache misses and traffic, the cost of the instrumentation
  measured against the untraced step, and a sampler only to cross-check.
- Every table names its tool and its command line.

State:

- CUDA box (CuPy and OpenCL): done, except `ncu`. The report is
  `~/Documents/LKB/NLSE_tangui_fix_specs/perf_profile_cuda_box.md`: per-kernel
  tables from `nsys` (CuPy) and from OpenCL profiling events, fixed cost per
  call, the options that leave the fast path, four prototypes, twelve ranked
  candidates, the scripts. Its first section says what was re-run by hand:
  the three prototype figures reproduce; the step figures agree with
  `profile_backends.py`, but one cell moved 19% between rounds where the
  report says under 1%. Not shown to the user as suggestions yet.
- `ncu` is refused on the box (`ERR_NVGPUCTRPERM`,
  <https://developer.nvidia.com/ERR_NVGPUCTRPERM>): the counters are open to
  root only, and there is no password-less sudo. The user said the
  permission is to be given. The report lists the two ways; the script for
  the first is on the box, to be run as root. Then run `ncu` per kernel on
  the must-have cells. `nsys` and `ncu` do not see OpenCL kernels.
- Mac (CPU and MLX): not started. Run it with the machine idle (a push runs
  the suite).
- Nothing is in `docs/optimization-log.md` from the profiles yet, and no
  candidate is implemented.

Top of the ranking, CuPy at 2048², one-step call (measured, prototypes only):

- The propagator is downloaded after every call (`_gpu_array_attrs`,
  `NLSE/solvers/nlse.py:1668`): 40.6 -> 28.5 ms for NLSE split step without
  it. Who reads `simu.propagator` on the host afterwards is not traced.
- The coupled `_energy_rates` runs twice per RK4 call and downloads the field
  (`NLSE/solvers/cnlse.py:127`): 282 -> 211 ms with one run, CNLSE RK4 with V.
- The potential grids are rebuilt on every call: 17-25 ms with V.
- Per step, nothing to gain on the fused CuPy paths (four of five kernels at
  the bandwidth ceiling). The gains are where a run leaves them: CNLSE RK4
  with `omega` (2.23x), CNLSE strang with `omega` (2.2x), `nl_length`,
  callbacks, a batched RK4, DDGPE. On OpenCL: VkFFT's `tune_config` (-23% on a
  2048² transform, not run in the solver) and an unfused RK4 stage (19%).

Known before the profiles:

- The solver lowers a step past its limits, so a run can take more steps than
  it was asked for. Count the steps taken. `profile_backends.step_for` is now
  under both limits; `benchmarks/trace_solvers.py` and
  `benchmarks/propagator_fusion.py` still ask 1e-4 m at every size, which
  split-step lowers at 1024² and above (1.79 steps per step asked).
- A call costs many steps. One `out_field` call of one step on CuPy at 2048²,
  solver already run: 37.7 ms for NLSE split step (a step is about 1.95 ms),
  54.4 ms for RK4 (9.9 ms a step), 109.8 and 181.3 ms for CNLSE. Table in the
  optimization log; the script and raw output are `fixed_cost.py` and
  `fixed_cost_cupy_raw.log` in `NLSE_tangui_fix_specs/`. Not broken down.
- From reading `NLSE/solvers/step_size.py`, not measured: one call computes
  `_energy_rates` up to four times (`_default_delta_z`, `_capped_delta_z`,
  `_rk4_max_dz`, Yoshida's extra `_split_step_max_dz`), each with a transform
  of a copy of the field, and `_energy_rates_on_device` copies `V` to the host
  every time.
- Open in the optimization log already: a plan callback in 1D, an
  `alpha == 0` branch in `_nl_prop`, the linear step as an increment.

## Environment

- The project `.venv` has CPU and MLX only. CuPy and OpenCL run on the CUDA
  box in one environment, `~/Documents/nlse_env_both`: a clone of the conda
  OpenCL environment plus the `cupy-cuda13x` wheel, with the conda
  `cuda-nvrtc` removed. That package (NVRTC 13.5, pulled in by `pyvkfft`)
  made cuFFT 12.3 of the system's CUDA 13.3 refuse the store callbacks with
  status 17, `CUFFT_MISSING_DEPENDENCY`, which CuPy reports as `KeyError: 17`.
  The two older environments still exist. OpenCL no longer runs on this Mac
  (see "Verification of the final tree").
- Sphinx comes with `uv sync --group docs` (repeat the `--extra` flags of the
  backends in use). `paper/` was not read.
- Running `docs/make_validation_figure.py` with any argument rewrites
  `docs/img/validation.png`.
