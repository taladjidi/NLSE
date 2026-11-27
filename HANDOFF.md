# Handoff

## State (2026-11-27)

A full audit of the repo was done at commit `fa7aa59` (v4.1.0 plus six
commits); its findings are listed under "Findings, ranked" below. All of them
that were to be fixed are fixed, in eight packages plus two follow-ups, each on
its own branch and merged into `main` with `--no-ff`:

| Package | Branch | Merge | Covers |
|---|---|---|---|
| p7 | `docs/packaging-ci-cleanup` | `ed00008` | README, docs, sdist, `uv.lock`, CI, examples, unused files |
| p6 | `test/closed-forms-and-cache` | `3f3b6dd` | `evaluate_delta_n`; benchmark cache isolation; tests that could not fail; NLSE_3d dispersion sign |
| p5 | `refactor/kernels-cleanup` | `36d952d` | unused CuPy and vortex kernels removed; MLX scalar cache bounded; CUDA built without `--use_fast_math` |
| p3 | `fix/gpe-units` | `7a5fd6a` | GPE takes `V` and `g` as rates in rad/s with the GPE sign; `adapt_delta_z` on GPE |
| p1b | `fix/solver-run-state` | `bcda6d8` | state left after a run or a backend switch; asserts become ValueError/TypeError; stale potential grids; NLSE_3d xt panels |
| p1a | `fix/stepping-loop` | `41f39a4` | medium ends exactly at `L`; backward runs; complex, zero and non-finite `delta_z` raise; CuPy stream race |
| p4 | `fix/nonlocal-response` | `4eef782` | non-local response applied in Fourier space; `Backend.convolution` and `nl_profile` removed |
| p2 | `fix/coupled-solvers` | `36bf9e1` | Rabi coupling in strang, yoshida, RK4; DDGPE noise, pump sampling, potential sign |
| follow-up | `fix/rk4-stability-bound` | last merge on `main` | post-merge test fixes (`6bd77e7`); RK4 step bounded by the grid's fastest linear rate (`10bf834`), in the coupled solvers too (`a53a470`); the timing tool asks a step no run lowers (`4ac2bb2`, then under the split-step limit too) |

The specs and patches of the eight packages are outside the repo, in
`~/Documents/LKB/NLSE_tangui_fix_specs/`. The worktrees `../NLSE_wt/pN` are no
longer needed.

`CHANGELOG.md` has an `## Unreleased` section with every change, the breaking
ones first. The next release is a major one (5.0.0 is the suggestion); the
version is not bumped and nothing is published.

### In flight (2026-11-27, night session)

`main` is pushed after each reviewed merge (owner, 2026-11-27: "No push also
to main as we go along !"). The auto-mode classifier refused `git push origin
main` once: when it does, ask the owner to run `! git push origin main`, and
do not look for another way. Merged on 2026-11-27: the uv, ruff and ty switch (`62bd2e8`, the
untracked `CLAUDE.md` and the git hooks follow it), the CPU batch fix
over several parameter axes (`496cd96`), the transform wording with the
log entry of the library re-measure (`c04afd7`) and the same batch fix for
the OpenCL loop (`fix/cl-multi-axis-batch`), the tests that bite
(`test/tests-that-bite`), the CPU blocked step (`7518025`), the fix of its
scratch (`fix/cpu-blocked-scratch`) and the two tests of it that numba's
threading layer decides (`fix/ci-threading-layer-tests`). Still open:

- **Order set by the owner (2026-11-27, evening): "work alone tonight on
  this, first debloat then docs."** It replaces the earlier order: no `/clear`
  first, and the GPU patches and the thread counts per machine (both
  described below) come after the docs. Merge each package to `main` as it
  lands, and push.
- **Debloat**, in three packages, specs in
  `~/Documents/LKB/NLSE_tangui_fix_specs/debloat/` (`repo.md`, `package.md`,
  `tests.md`, with patches and scripts beside them):
  - `chore/debloat-repo`: done. Dead code in the examples and
    `benchmarks/roofline.py`, a duplicate image, dead CI and Sphinx settings,
    `.gitignore`. Reviewer: pass. `pyproject.toml` (task T6 of `repo.md`)
    waits for the two packages below, since two of its ruff ignores depend
    on them.
  - `test/debloat-tests`: done. `tests/` goes from 20558 to 17429 lines:
    duplicated cross-backend cases merged, 32 unread constants deleted,
    `AVAILABLE_BACKENDS` and the `backend` fixture defined once
    (`tests/helpers.py`, `tests/conftest.py`), the CuPy send and retrieve
    pairs merged into one test that skips, docstrings cut to their summary
    (80 rewritten). Reviewer: pass; its mutants of what the deleted
    duplicates covered are killed by the survivors. Suites, branch against
    `main`: Mac 1723 passed, 160 skipped; box CuPy 1796 passed, 68 skipped
    (1806 and 68); box OpenCL 1700 passed, 153 skipped (1714 and 149). The
    10 fewer are the merged and deleted tests; without CuPy the four merged
    ones skip where a half of each pair used to pass on the host.
  - `refactor/debloat-package`: done. `NLSE/` loses 357 lines net, mostly
    comments over four lines; their measurements that were in no log are in
    the commit messages. Code: `benchmark._synchronize_backend` and
    `_assumed_rank` are gone (the benchmark calls `Backend.synchronize`, so
    a CUPY timing no longer includes a copy to the host, and
    `CACHE_VERSION` is 3), as are `CNLSE.propagator1/2`, a second
    `self.V = V` in `GPE` and `Parameter.__set_name__`. Reviewer: pass; the
    comment-only files have the same AST, the kernel sources the same code
    with comments stripped. Suites unchanged: Mac 1723 passed, 160 skipped;
    box CuPy 1796 and 68; box OpenCL 1700 and 153 (the box ran 2416eb3, one
    docstring before the head). Step time, complex64, before and after in
    turn, ms per step: Mac CPU 1.94 and 1.91 (1024²), 7.31 and 7.44
    (2048²), 25.8 and 24.2 (4096²); MLX 0.283 and 0.282, 1.777 and 1.780,
    7.40 and 7.42; box CUPY 0.388 and 0.386, 1.569 and 1.559, 8.63 and
    8.58; box CL 0.595 and 0.595, 3.408 and 3.410, 18.56 and 18.57. No
    difference above the floor. The reviewer's minor notes, not fixed:
    three shortened comments lost their reason (`kernels.cu:287` why
    `A_orig` may alias `out`; `ddgpe.py` near 296 why one profile at a
    time; `cpu.py` near 20 "not `fastmath=True`").
  - `fix/benchmark-times-lazy-backends`: done, found by the package review.
    `benchmark_backend` dropped what `fft` and `ifft` return and waited on
    its input; MLX's transforms are lazy and return a new array, so MLX was
    timed doing nothing (0.001 to 0.005 ms at 2048² on the M3 Max, 1.4 ms
    once fixed; CPU 4.4 ms) and `backend="auto"` ranked it first on that
    figure. A stand-in backend in `tests/backends/test_benchmark.py` fails
    on the old code and if a trial stops waiting. Reviewer: pass. Box:
    the benchmark tests pass under CuPy and OpenCL (CUPY 1.82 ms, CL
    2.6 ms at 2048²).
  - Left for the owner, not done: the tutorial notebook's 8 MB of outputs
    (re-render smaller), `paper/` and `img/output.png` (the JOSS record),
    `examples/juliaGPE_vs_NLSE.py` with `vortex_precession.jl`, merging
    `examples/benchmarks.py` and `fig1_benchmarks.py`, the `kernels_cpu/gpu/cl`
    aliases, the `splitting="RK4"` mapping, the `speedup` field of the
    benchmark cache, `tests/benchmarks/` with the CI benchmark job, and one
    module for the per-solver legacy tests. `examples/profile_cupy.py`
    (overlaps `benchmarks/trace_solvers.py`) is decided in the docs pass.
- **Docs**, after the debloat.
  - The gallery: done (`ci/cache-docs-gallery`). The CI Documentation job
    took 58 minutes, 57 of them sphinx-gallery running all 19 examples on
    each push. Owner: "If these runs are so expensive maybe it's better to
    store the final states rather than redo them". The job now keeps
    `docs/auto_examples` in the Actions cache (key: hash of `uv.lock` and
    `examples/_output.py`, then the run id), and sphinx-gallery skips an
    example whose source is unchanged. A weekly `schedule` run and "Run
    workflow" pass `run_stale_examples=True` and execute them all. Accepted
    trade-off: a change to `NLSE/` that breaks an example passes CI until
    that run. Design and what to watch on the first runs:
    `~/Documents/LKB/NLSE_tangui_fix_specs/docs/gallery.md`, section 3.
    Nothing is cached until one run on `main` finishes cold (about an
    hour): a push to `main` meanwhile cancels it. Not checked yet: the
    cache on GitHub itself. For the owner: the weekly run also runs the
    test matrix and the benchmark job; gate them on the event if unwanted.
  - The page review: done (`docs/page-review`), 13 pages against the code
    along Diataxis, one commit a page. Specs, with the decisions left to the
    owner: `docs/pages_entry.md` (decisions 1-7) and `docs/pages_guide.md`
    (D1-D5), in the same directory. Two reviewers checked every equation
    and sign term by term: no CRITICAL. Fixed after review: the methods
    page said a default step is always inside its limit (it is not, see
    "Still open"); the README's `gpu` extra and docs build disagreed with
    the pages it links; `tests/test_doc_claims.py` let 10 of 18 mutants of
    the pages through and now measures the transform pairs per step from a
    run and reads a quoted constant across a line break (13 of 13 mutants
    killed, list in the commit).
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
  - Next: the README's headline performance.
- CI on `e60e2a4`: all green, the four macOS jobs on a second try (GitHub
  cancelled them the first time for lack of a runner).
- The CPU blocked step, merged (`7518025`): three passes in
  `NLSE/kernels/cpu_blocked.py`, chosen per machine and grid by
  `NLSE/backends/cpu_tuning.py` and cached in `cpu_blocking.json`; spec in
  `NLSE_tangui_fix_specs/cpu_blocked_step_spec.md`. Reviewed twice (second
  verdict: ship). Changed after the spec, by hand and not through an
  implementer:
  - The tuner ranks the blocked candidates (`_fastest`), then times the
    first against the unblocked step window for window, taking turns
    (`_duel`, `DUEL_ROUNDS = 5`). The best of ~30 windows against the best
    of 3 had read 1.13x on a loaded Mac where blocking loses.
  - `MARGIN = 0.10`. At 0.05 the tuner chose 64 rows x 16 at 1024²
    complex128 inside the benchmark process (read 1.08x); fresh processes
    lose 10-18% there (lossless, 2.55-2.57 against 3.01-3.04 ms a step).
    The plain step's `fft2` at that size has two speeds in a long-lived
    process (0.84-0.87 or 1.03-1.09 ms a transform; it flips after a tuning
    at another size, only when interleaved with the numba kernels, and
    stays). Why is not established. The price of 0.10: the 1.07x at 2048²
    complex128 on the Mac is not taken.
  - The hook's `factors` is a zero-argument callable, called only by a step
    that is taken; a grid another process is tuning is asked for again
    every `BUSY_RETRY` (3 s) instead of staying unblocked for the life of
    the process; `_wait` lets every share finish before it raises. These
    were the first reviewer's two MAJORs and one MINOR.
  `MARGIN` and measuring on first use (`TUNE_ON_FIRST_RUN`: 0.4-0.9 s at
  1024², 12.1-12.5 s at 4096² complex128 inside the first `out_field`) are
  the owner's to confirm.
  Measured, M3 Max, 16 threads, Spotlight holding one core, median ms a
  step, `main` against the branch in fresh processes (NLSE lie with loss
  and saturation): 4096² complex64 30.4-39.1 (two speeds) to 25.4-25.8,
  complex128 52.3-52.5 to 40.6 (1.29x); 2048² complex128 11.66-11.80 to
  10.92-11.04 (1.07x, not taken at `MARGIN = 0.10`); 2048² complex64 and
  both 1024² stay unblocked. `benchmarks/cpu_blocked_step.py`, interleaved
  in one process: 4096² 1.17x complex64 and 1.25x complex128. With a
  second core held by a video player the gains read higher (1.10x and
  1.27x at 2048², 1.45x and 1.53x at 4096²): the blocked step is the
  steadier of the two under load.
  Ryzen 3 4100 (8 threads, the benchmark, one process): complex64 1024²
  8.16 to 7.16 (1.14x), 2048² 35.4 to 27.0 (1.31x), 4096² 145.1 to 109.6
  (1.32x, rounds 4.4% apart; 106.3-106.5 in fresh processes, 1.37x);
  complex128 14.9 to 11.2 (1.34x), 60.8 to 45.3 (1.34x), 283.7 to 193.8
  (1.46x), with the scratch fix below; fresh processes agree. A tuning there takes
  1.2-2.4 s at 1024², 5-9 s at 2048², 21 s and 40 s at 4096². CuPy suite on
  the branch: 1802 passed, 68 skipped; the CuPy step did not move (6, 6 and
  12 alternating pairs at 1024², 2048², 4096² complex64). The box ran the
  tree at `MARGIN = 0.05`; every cell it blocked read over 1.22x, so the
  verdicts are those of 0.10.
  Fixed (`fix/cpu-blocked-scratch`): the Ryzen's 2048² complex64 step at
  the tuned 32 rows (512 KiB, the L2) ran at 27.0 to 40.0 ms from one fresh
  process to the next, 35.3-35.8 unblocked. The cause was the scratch, not
  the block: `np.empty` served it from heap pages recycled in the order
  earlier arrays left them (glibc's mmap threshold rises to 32 MiB), where
  the tuner measured on fresh ones. `_scratch` in
  `NLSE/kernels/cpu_blocked.py` now cuts every thread's block from one
  fresh private anonymous mapping: 27.0-27.9 ms five times of five, `main`
  27.1-36.3. The same bug reached 4096² complex64 (16 rows, 512 KiB) after
  another grid ran in the process (`main` 106.4-115.8 and once 145.8, the
  fix 106.6-108.2), and made one 1024² complex64 tuning in eight decline to
  block (now 1.21-1.25x, eight of eight). No other cell, tuning time, first
  call or peak moved; M3 Max unchanged but 2048² complex128, which sits on
  the 10% margin (1.07-1.11x) and blocked once in two tunings. Huge pages
  for the scratch were tried and rejected: no gain for any layout the tuner
  picks, and blocks above the L2 2.3x slower, 5 s more per 4096² tuning.
  That the page order is the mechanism is inferred (no counter read). The
  Windows branch of `_private_mapping` has run nowhere. Reviewed twice
  (ship); left as it is: a grid with a zero-length axis now raises from
  `mmap`. Written by hand, not through an implementer. Logs and scripts:
  `NLSE_tangui_fix_specs/cpu_blocked_step/ryzen/`.
  Rule from the owner (2026-11-27): hot working buffers come from "a fixed
  memory stack", a region the library maps and keeps, not from malloc. The
  factors and the solver's field were not moved there: no cell needs it
  (measured after another grid ran in the process). My reading, to confirm
  with the owner, is that the rule asks for it only where a measurement
  shows a cell moving.
  The kernel split costs nothing (both trees' `square_mod_nl_prop`
  interleaved, 0.996-1.004). After 20 steps the blocked field is within
  6e-6 of the peak of the unblocked one in complex64 and 1e-14 in
  complex128, and the branch's unblocked field equals `main`'s bit for bit
  (arm64).
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
- Tests that bite, merged. Mutation audit: 234 mutants of the package, 52
  survived the suite; 47 are now killed (`test_audit/crosscheck.py
  --survivors`, 34 by their own test files, 13 by the suite), 5 are left
  without a test and the spec `tests_bite_spec.md` says why. The reviewer
  found three more blind tests (frozen step, `sample`, `norm`); they are
  rewritten and the mutants it named now fail them. Left by that reviewer, minor,
  not fixed:
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
- Defects the audit found, none fixed, the first three for the owner:
  - `_loss_max_dz` (`NLSE/solvers/step_size.py`) divides by `2 * alpha` while
    the kernels receive `alpha / 2`: the kernel's `u` at the cap is 0.025,
    and the comments there and in `NLSE/kernels/cpu.py` say 0.05. Safe side.
    Change the code or the comments.
  - `DDGPE` flips the sign of `g` but not of `g12` nor `g2`
    (`NLSE/solvers/ddgpe.py`); `tests/solvers/test_parameter_names.py` pins
    the current sign.
  - `_loss_max_dz` reads `alpha` only; `CNLSE` and `DDGPE` hand the kernels
    `alpha2` too, so a larger `alpha2` or `gamma2` can pass the limit the
    kernel assumes. Read from the code, not run.
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

### Verification of the final tree

- CI (2026-11-27). The first push with the blocked step (`c503edb`) went red
  on two of its tests, for what
  [numba's threading layer](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
  allows, not for the step. Ubuntu runners get GNU OpenMP, and numba ends a
  forked child that enters it (`test_a_child_forked_mid_step_steps_blocked`,
  exit -15, numba's message in the log). The macOS runner aborted in
  `test_two_threads_step_as_they_would_one_after_the_other` (exit 134), as
  `workqueue` does when two threads enter it (4 runs in 6 on the box, with
  numba's message); that the runner is on `workqueue` is inferred, its log
  does not show the message. Each test now skips on that one layer.
  Reproduced and checked with `NUMBA_THREADING_LAYER`: Mac `workqueue`
  1736 passed, 157 skipped (`omp`, vendor Intel as numba names it, is the
  Mac's default: 1737, 156); box `omp` (GNU) 1805 passed, 69 skipped (`tbb` is its default:
  1806, 68). The failed log is
  `~/Documents/LKB/NLSE_tangui_fix_specs/cpu_blocked_step/ryzen/ci_failed_c503edb.log`.
- Scratch fix (`fix/cpu-blocked-scratch`, 2026-11-27), code as at `3b8113f`
  (the commit after it changes a comment): Mac 1737 passed, 156 skipped;
  `ruff check`, `ruff format --check` and `ty check` clean. On the box:
  CuPy 1806 passed, 68 skipped; OpenCL 1714 passed, 149 skipped. Mutants
  that fail the two new tests: malloc scratch, a shared mapping.
- At `7518025` (2026-11-27), Mac (M3 Max, CPU and MLX): `uv run --locked
  pytest tests/ -q -m "not benchmark"` gives 1735 passed, 156 skipped;
  `ruff check`, `ruff format --check` and `ty check` are clean. CuPy on the
  box, on the blocked-step tree before its last edits (`MARGIN` back to
  0.10, four tests of the tuner): 1802 passed, 68 skipped, one run. OpenCL
  was not run on this tree; its last full run is the one below. The lines
  below are from the audit, at `4ac2bb2`.
- Mac (M3 Max, CPU and MLX, `.venv`), at `4ac2bb2`: `pytest tests/ -q -m "not
  benchmark" --ignore=tests/test_packaging.py` gives 1404 passed, 126 skipped;
  `ruff check`, `ruff format --check` and `mypy NLSE/` are clean. Five of
  those tests exist only while `docs/_build` does (`tests/test_doc_claims.py`
  reads a file in it), so a clean checkout gives 1399.
- Sphinx, `-W --keep-going`, in a scratch venv with the `docs` extra: exit 0
  and no warning at `10bf834` (full build, 25 min) and at `4ac2bb2`
  (incremental).
- CuPy on the CUDA box (RTX 3050), at `4ac2bb2`: 1468 passed, 67 skipped, in
  6 runs of 6.
- OpenCL on the CUDA box (NVIDIA OpenCL, CPU and CL available), at
  `4ac2bb2`: 1405 passed, 120 skipped.
- After `4ac2bb2` only the docs and `benchmarks/profile_backends.py` changed.
- Step time against 4.1.0 (`38a8704`): no cell slower by more than its own
  scatter on CuPy (RTX 3050), MLX and CPU (M3 Max), NLSE and CNLSE, split step
  and RK4, 1024² and 2048². Numbers and configuration in
  `docs/optimization-log.md`, "The audit fixes cost no step time".
- OpenCL cannot be run on this Mac any more (macOS 27.2). Apple's driver
  segfaults inside `pyvkfft`'s `VkFFTApp` for any transform, a bare 64x64 one
  outside NLSE included, and 4.1.0 crashes the same way. The conda env
  `poclrepro` (POCL) cannot link its kernels: 96 failures on 4.1.0 too.

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
- p3: `_estimated_rates` uses an optical intensity for GPE when no field is
  given; the floor of `adapt_delta_z_to_error`, `L/1e5`, is 0 for a GPE; the
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

## Findings, ranked

Each was reproduced by a script unless marked *(read)* or *(reviewer)*.

### Wrong results, no error

1. **Rabi coupling is skipped for `strang` and `yoshida`.**
   `NLSE/solvers/cnlse.py:810`, `NLSE/solvers/ddgpe.py:508` test
   `splitting == "lie"`. CNLSE, `omega=200`, `z=1e-2`: lie gives
   P2/P1 = 2.4255 = tan²(ωz/2); strang and yoshida give 0.
2. **GPE `g` and `V` do nothing in the documented units.** GPE inherits
   `_g = k/2·n2·c·ε0` and `_k_half = k/2` with `k = m`
   (`NLSE/solvers/nlse.py:1524-1529`). A run with g·n ≈ 64 kHz equals the
   `g=0` run exactly; a 100 Hz trap `0.5·m·ω²·r²` equals no trap; uniform
   `V0=1000` gives 7e-26 rad in 1 ms, `V0=2000/m` gives 1.0 rad. Same formula
   in commit `9d94782` (2025-06), so not a regression. Units (Hz or rad/s) and
   sign are a decision to take with the fix.
3. **The medium runs past `L`.** `NLSE/solvers/nlse.py:2254-2256` rounds the
   in-medium step count up. `L=1e-2`, `z=2e-2`: phase is 1.20× `k·n2·I·L` for
   `dz=3e-3`, 1.148× for `4e-3`, 1.000× for `1e-3`.
4. **Non-local kernel.** `NLSE/solvers/nlse.py:233-244`: `nl_length` is
   floored to whole pixels (3.0·dx and 3.99·dx give the same kernel), `delta_X`
   is used for both axes (NX=64, NY=128: 0.78 mm × 0.39 mm), and the
   convolution is zero-padded on a periodic grid (a plane wave gets 57% of the
   centre phase at the edge, 32% at the corner). Candidate fix: apply
   `1/(1 + nl_length²·K²)` in Fourier space; cost not measured.
5. **`evaluate_delta_n` with a finite `Isat`.** `NLSE/callbacks.py:116`
   divides `|A|²` (V²/m²) by `I_sat` (W/m²) *(read)*. At I = 3·I_sat the
   result is 0.0018 of the true value.
6. **Backward runs.** A negative `delta_z` takes its last partial step forward
   (`NLSE/solvers/nlse.py:2200-2209`): round-trip error 5e-15 at `z=1e-3`,
   6.2e-4 at `z=1.05e-3`. `out_field(z<0)` returns the input, no error.
7. **DDGPE noise scales with `dt`, not `√dt`.** `NLSE/solvers/ddgpe.py:174-179`.
   Mean |ψ|² at T=0.2: 1.32e-6 (dt=2e-3), 6.60e-7 (1e-3), 3.31e-7 (5e-4).
   Confirm the intended convention before changing it.
8. **`plot_field` is upside down.** `imshow` with `extent` and the default
   origin: a spot at y=+0.5 mm is drawn at −0.52 mm
   (`NLSE/solvers/nlse.py:1041-1084`, same in `cnlse.py`, `nlse_3d.py`).

### Crashes and blind spots

9. **Imaginary time (complex `delta_z`) is documented and crashes** at
   `NLSE/solvers/step_size.py:583`, then `NLSE/solvers/nlse.py:1156`, then in
   the numba kernels. Docs: `NLSE/solvers/nlse.py:290,307`.
10. **Switching backend after a run** keeps the old propagator:
    MLX→CPU raises at `NLSE/kernels/cpu.py:1176`. The setter
    (`NLSE/solvers/nlse.py:267-270`) clears no cache.
11. **Public `split_step` ignores its `V` after any `out_field`**, and keeps
    `_g = 0` after a run past `L` (`NLSE/solvers/nlse.py:484-486`).
12. **Callback lists.** DDGPE `out_field` inserts into the caller's list
    (`NLSE/solvers/ddgpe.py:562-569`), so the second call raises `TypeError`;
    `callback=[f, g]` with default `callback_args` raises at
    `NLSE/solvers/nlse.py:2225`.
13. **DDGPE pump profile is indexed by step number**
    (`NLSE/solvers/ddgpe.py:246`): a capped step gives `IndexError`.
14. **Tests cannot see a scale error.** `"_k_half": self.k` (every `V`
    doubled) passes 544 tests in solvers, integration, physics and
    cross-backend. *(reviewer)*: the same holds for three RK4 kernel
    mutations, the non-local width, 3D dispersion ×2, the `k` entry of the
    propagator cache key, and the callback return values.
15. **Tests that cannot fail** *(reviewer)*: unitarity tests on a uniform
    field (`tests/solvers/test_nlse.py:165-205`, `test_nlse_1d.py:63`,
    `test_cnlse_1d.py:75`), and env tests in
    `tests/backends/test_auto_backend.py:60-72,104-176`.
16. **Benchmark-cache tests use the real cache** (`tests/backends/test_benchmark.py:131,209,231,260`);
    the audit's own test runs rewrote `~/Library/Caches/NLSE/fft_benchmark.json`
    for a 64×64 grid. A cache file holding `null` raises `TypeError`
    (`NLSE/backends/benchmark.py:258`).
17. **Docs contradict the code.** README builds docs with mkdocs
    (`README.md:35-41,67`); the docs are Sphinx. `CLAUDE.md:74` and
    *(reviewer)* `README.md:122-124`, `docs/physics_problem.md:43-46`,
    `docs/solvers_overview.md:234` say MLX has no convolution; it has.
18. **Packaging.** `uv.lock` records nlse 2.3.0 with pyfftw and mkdocs.
    The sdist ships `tests/` without the files they read
    (`pyproject.toml:82`). CI ignores failures at
    `.github/workflows/ci.yml:176,286`.

### Speed and size

19. **Fixed cost per `out_field` call**, 2048² complex64: CPU 57–62 ms
    (about 5 steps at 10.9 ms/step), MLX 49–53 ms (about 20 steps at
    2.49 ms/step). `_capped_delta_z` is 35 ms even when `delta_z` is given;
    `_dispersion_operator` is built twice on the host, 14–17 ms.
20. **Dead and duplicated code.** About 130 unreferenced lines in
    `NLSE/kernels/cupy.py` (`rabi_coupling`, `apply_propagator`, `rk4_axpy`,
    `rk4_accumulate`, their `_fused` twins, `vortex_cp`); `vortex` kernels used
    only by two tests; DDGPE `split_step` (`NLSE/solvers/ddgpe.py:354-512`)
    repeats CNLSE's, about 110 lines; no-op blocks at `nlse.py:32-33`,
    `cnlse.py:10-14`, `ddgpe.py:270-274,346-347`; `_SCALAR_CACHE` in
    `NLSE/kernels/mlx_kernels.py:12-25` never shrinks.

### Smaller, left out of the report

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

## Decisions

- Kernels: no wrong-result bug found. CPU, OpenCL and MLX agree to 2e-7
  (float32) over about 2000 cases; CUDA source was read, not run.
- The "empty parameter set" skips in `tests/integration/test_nonlocality.py`
  are benign here: no backend without convolution is installed.

## Open items

Resume here. Everything above and the performance work below are merged
into `main` and pushed. Open: the performance profiles, and the
release (5.0.0 suggested; the version is not bumped and nothing is published).

### Performance work (2026-11-26 and 27)

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

Done:

- **CPU complex64 kernels in float32** (`perf/cpu-float32-kernels`). numba
  promoted them to double. Kernels 1.4x (M3 Max) and 1.6 to 2.2x (box), a
  whole lie step 1.07 to 1.12x and 1.03 to 1.32x; complex128 bit-identical.
  The lossless rotation is a sum (`_rotate32`), so it rounds as double did up
  to 0.1 rad a step. `_rabi_coupling` stays in double on purpose. Test:
  `tests/backends/test_cpu_kernel_precision.py`. Entry in the optimization
  log, CPU section. Scripts: `~/Documents/LKB/NLSE_tangui_fix_specs/cpu_float32/`.
  Reviewer: no critical or major finding. Left as they are: the float32
  angle reduction is exact only below 8192 rad and unbounded past about 3e7;
  `benchmarks/roofline.py` still measures the double `_sincos` and `_exp` as
  its transcendental ceiling, which is no longer what a complex64 run calls.

- **CuPy: the nonlinear step inside the inverse transform**
  (`perf/cuda-fewer-passes`, spec 1). A cuFFT store callback applies
  `square_mod_nl_prop` as the inverse transform writes; 2D local `NLSE` and
  `GPE`, every splitting. RTX 3050, complex64, lie: 88 → 72 bytes a point
  (`ncu`), a step 1.17 to 1.19x faster than `main` at 512² to 3072² (1.12x at
  2560² and 3000²), 1.34 to 1.38x faster than no callback at all; 0.89x at
  256²; complex128 unchanged. Still at 87 to 92% of the 208 GB/s ceiling.
  Entry in the optimization log, CUDA section, with the tables.
  - The callback calls the step out of line (`__noinline__`,
    `NLSE/kernels/cuda_source/fft_callbacks.cu`). Inlined, cuFFT's kernel
    spilled to the stack and the path ran at 0.84x of `main`.
    `tests/backends/test_kernel_templating.py` pins it.
  - Box: suite 1556 passed; callback tests six times; 36 runs bit-identical
    to `main` with callbacks on and off. At 2560, 3000, 3072 and 3600 points a
    callback plan is another cuFFT plan and differs from the unfused path by
    rounding. That was already true of the propagator callback in `main`.
  - Tests: `mutants_spec_1.py` (in `NLSE_tangui_fix_specs/`) kills 14 of 14,
    0 of 14 before the fix round.
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
  - Scripts: `~/Documents/LKB/NLSE_tangui_fix_specs/cuda_fewer_passes/`
    (`fp_step.py`, `fp_three.py`, `fp_ncu*.py`, `fp_bitwise.py`,
    `fp_sizes_diff.py`, `fp_variants.py`), copies on the box in
    `~/Documents/nlse_scratch_scripts/`, `ncu` reports in its `rep/`.

Task list for the closing review pass (asked on 2026-11-27):

- [x] **Tests must bite.** "review thoroughly tests to make sure they
  actually bite and test": for each test file, break the behaviour it claims
  to pin (a mutation of the code under test) and see the test fail. Report
  every test that still passes, that asserts nothing, or whose tolerance is
  wider than the effect it guards; fix or delete those.
  - Done: four mutation audits, one per slice of `tests/`, in
    `~/Documents/LKB/NLSE_tangui_fix_specs/test_audit/` (`mutants_s1.py`
    solver state, `s2` solver physics, `s3` backends, `s4` integration and
    docs): 234 mutants, 106 not killed by their own slice's files.
    `crosscheck.py TREE s1 s2 s3 s4` runs each mutant against its slice and,
    if it passes, against the whole suite; a mutant is a gap only if the
    whole suite lets it through. Mac output: `crosscheck_mac_s134.txt` and
    `crosscheck_mac_s2.txt` there. **52 mutants survive the whole Mac suite**
    (s1 15, s2 12, s3 10, s4 15), none by timeout.
  - The 36 of them that do not depend on the backend
    (`survivors_for_box.txt`) were run against the CuPy and OpenCL tests on
    the box (tree 8493fe3, `crosscheck.py ../tree s1 s2 s3 s4 -j 2
    --survivors survivors_for_box.txt -k "CUPY or CL or cupy or opencl or
    cuda or cufft"`): **35 survive there too** (`crosscheck_box.txt`). The
    one killed, "s1: split-step limit: pi rad -> 2 pi rad", dies only on
    `test_propagator_fft_refresh.py::...::test_refreshed_when_step_limiter_clamps_delta_z[CUPY]`,
    so it still needs a test that runs on every backend.
  - On the batch fix's branch the two batch mutants of `s4` still survive
    (`crosscheck_mac_batch2.txt`): `Isat` dropped from the batched positions
    of `nl_prop` on CPU, and the MLX lossy kernel indexing a shared `V` with
    the batch's flat id.
  - Survivors by kind: step-size rates and limits, three callbacks, the
    CNLSE second component (`n22` at `k` not `k2`, its `I_sat`, its `alpha`),
    four DDGPE terms, CPU kernel signs and saturation on the lossless and
    non-local paths, MLX batch and loss modes, the benchmark cache (grid and
    processor not compared), the docs capability table, the `nsys` parser,
    the wheel's module list. Kill recipes: a uniform field against
    `k*n2*I*L`, `k*n2*I/(1+I/Isat)*L`, GPE `-g|psi|²t` and `-Vt`,
    `exp(-gamma t)`, `exp(-alpha_i z)` per component.
  - Found on the way and fixed (`fix/cpu-two-axis-batch`): the two-parameter
    batch of `README.md:285` raised on CPU, and a one-value array parameter
    moved only the first field of a batch.
  - A batch and its members run alone differ by 5e-4 when the step is left
    to the solver: the batch takes the strictest member's step. Not a bug;
    a test comparing the two has to pass `delta_z`.
  - Done on `test/tests-that-bite`, from the architect's
    `NLSE_tangui_fix_specs/tests_bite_spec.md` and `tests_bite_1..14.patch`
    after its spec review and a review of the diff; the touched files pass
    on the box under CuPy and under OpenCL. `crosscheck.py --survivors` is
    the regression check (every listed mutant has to die). What the
    reviewer left is under "In flight".
- [x] **Packaging and tooling** ("I also would like you to check packaging
  and tooling in general. I want something modern based on uv ruff and ty",
  2026-11-27), which carries the mypy to ty switch ("Let's switch also from
  mypy to ty at the end"). Done on `build/uv-ruff-ty` from the architect's
  `NLSE_tangui_fix_specs/tooling_uv_ruff_ty_spec.md` (revision 2, after spec
  review). What it settles:
  - dev and docs are dependency groups, the backends stay extras, hatchling
    stays. Set up with `uv sync --extra opencl --extra mlx`; every later
    `uv sync` needs the same `--extra` flags, or it removes the backends.
  - ty 0.0.112, pinned, checks all of `NLSE/` against `ty-baseline.json`
    (99 diagnostics at the switch). A change that moves a recorded
    diagnostic needs `uv run ty check --update-baseline`, and only after
    checking that the diagnostics are the recorded ones.
  - ruff gains UP and PGH. The hooks and CI run every tool through
    `uv run --locked`; CI installs uv 0.13.9 with setup-uv v10.4.0. A plain
    `git commit` or `git push` no longer needs the venv on PATH, but it
    needs uv.
  - Not done, for the user to decide: the licence expression
    (`GPL-3.0-only` or `GPL-3.0-or-later`, with `license-files` and
    hatchling 1.27), dropping Python 3.10, ruff 0.17.6, ty on `tests/`
    (456 diagnostics), pinning the actions by hash and bumping their
    majors, an SBOM artifact and a commit-message hook.
  - The workflow has run on GitHub since the push of `c503edb`: see the CI
    entry under "Verification of the final tree". Not checked: the CuPy and
    OpenCL extras were not synced from the lock on the box.
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

In progress, in this order:

- **CuPy, a separable propagator** (`perf/cuda-separable-propagator`, spec
  2b: `~/Documents/LKB/NLSE_tangui_fix_specs/cuda_fewer_passes_spec_2b.{md,patch}`).
  Merged. The forward store callback multiplies
  `rows[iy] * columns[ix]` instead of reading the propagator. RTX 3050,
  complex64, lie, 12 interleaved rounds: 73 → 65 bytes a point (`ncu`), a step
  1.04x faster at 1024², 1.05x at 2048², 1.04x at 4096²; 1.10x at 512², 0.90x
  at 256²; complex128 0.97 to 0.99x (that step is at 18% of the memory peak:
  arithmetic binds it). Kept under the one-path decision; the complex128 cost
  is for the user to accept or not. Entry in the optimization log, CUDA
  section.
  - The tie between the array the kernel would read and the factors it
    gets: `_build_propagator` caches `(array, dtype, delta_z)`; `_built_step`
    finds an array by identity; `_update_propagator_fft` records
    `_propagator_fft_origin`; `_propagator_fft_factors` returns None unless
    the origin is that very array and its cache key still holds. Every swap
    of the linear state goes through `_linear_state` /
    `_restore_linear_state`. Any doubt gives None and the kernel reads the
    full array.
  - Mac: suite 1498 passed, 155 skipped; `mutants_spec_2.py` kills 17 of 17.
    Box (CuPy 14.2.0): suite 1596 passed, 67 skipped; the fusion file 41
    passed three times; the five CUDA-only mutants killed
    (`run_cuda_mutants.py`, same directory as the spec).
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
  - Scripts on the box: `fp_sep.py ROUNDS N,N,... [DTYPE]`, `fp_ncu.py`,
    `fp_ncu_table.py` in `~/Documents/nlse_scratch_scripts/`; reports
    `rep/fp_sep_2c*.txt` and `rep/ncu_2c_*`.
- **CuPy at 4096² is at 62% of the memory peak**, against 85% at 1024² and
  2048², and the gap is inside cuFFT: the pass down the columns costs 0.097
  ns a point for columns of 2048 and 0.166 for columns of 4096, whatever the
  stride; rows cost 0.077 at every length. Padding the rows (0.250 to 0.263
  against 0.240 ns a point) and a transposed layout (the copy costs 0.157 to
  save 0.089) are measured and rejected. Closed; in the optimization log's
  CUDA section. Box scripts `fft_axes.py`, `fft_stride.py`,
  `fft_transpose.py`.
- Scratch to delete. Box: `~/Documents/nlse_cuda_sp2c` (= `main` after the
  merge of 2b, less the docs), `nlse_cpu32`, `nlse_cuda_fp` (= 5113580, the
  tree the CPU prototypes import). Mac: `git worktree list` (this session's
  scratchpad, `~/Documents/LKB/NLSE_wt/`).
- **CPU: a step that owns its passes.** Measured as a prototype, not
  designed or implemented. Three passes a step, each over blocks that fit a
  core's cache, the field back in real space after each: transform the rows
  | gather a block of columns, transform, multiply by the `y` factor,
  transform back, scatter times the `x` factor | transform the rows back and
  apply the nonlinear kernel. Python threads over static shares; every call
  inside releases the GIL (scipy.fft 1D with `workers=1`, serial `nogil`
  numba kernels). It fits the hook spec 2b adds
  (`linear_nonlinear_step(..., factors=...)`), so it comes after it.
  - Decided by the user on 2026-11-27: "For this type of optimizations, this
    should be done by a utility, since this will vary machine from machine /
    chip to chip", then "Basically the library should keep a cache of best
    chunk sizes per grid size. Very much like the old API used to cache FFT
    plans." So no constant and no rule from the cache sizes: the block size,
    or "do not block", is measured on the machine and kept in a cache the
    library owns, keyed by grid shape, dtype and thread count, next to the
    backend benchmark cache (`NLSE/backends/benchmark.py`, `NLSE_CACHE_DIR`).
    My reading, not confirmed: the first run on an untuned grid measures and
    stores, and the same routine can be called ahead of time.
  - Box, 8 rounds (Ryzen 3 4100, 4 threads, lossless Kerr, ms a step,
    today's step against the best block): complex64 1024² 7.22 / 6.43
    (1.12x, 32 rows), 2048² 35.14 / 24.75 (1.42x, 16 rows), 4096² 144.66 /
    98.83 (1.46x, 16 rows; 8 rows 1.26x, 32 rows 0.56x); complex128 13.67 /
    10.75 (1.27x, 32 rows), 59.45 / 41.00 (1.45x, 16 rows), 277.76 / 178.73
    (1.55x, 8 rows; 16 rows 0.90x). The best block is 256 to 512 KiB (a
    core's L2 is 512 KiB) and 1 MiB loses. Reports on the box:
    `rep/cpu_fused3_bytes8_complex{64,128}.txt`.
  - The buffer must fit the core's L2 (512 KiB on the box): 1 MiB gives
    0.58x at 4096² complex64. 128 KiB gives 1.01 to 1.15x. Eight logical
    threads lose to four (1.17 to 1.30x): a pair shares one L2.
  - Mac (M3 Max, 12 to 16 threads, machine not idle, the chunk rule not yet
    run there): complex64 0.77x at 1024², 0.98x at 2048², 1.41x at 4096²;
    complex128 0.92x, 1.21x, 1.46x. The field fits the 16 MiB L2 at 1024²
    and three thread dispatches weigh on a 1.6 ms step. The best block there is
    1 MiB. Whether to block at all on a grid is what the cache records.
  - The same passes threaded by scipy and numba alone (no Python threads)
    do not win: 0.79 to 1.03x on the box in complex64. The gain is each
    thread keeping its block in its own cache across the transform and the
    kernel.
  - A serial numba loop keeps Python's division check and does not
    vectorize (15 against 2.5 ns a point on the box): the serial kernels
    need `error_model="numpy"`. Two dispatchers made from one Python function
    with `cache=True` share one cache entry on disk, whatever their options:
    the second loads the first's machine code (reproduced). Serial twins of
    the parallel kernels cannot be cached that way.
  - ducc0 is 5 to 8% faster than scipy's transforms here: not taken, too
    little for a second transform dependency. Its licence is GPLv2 or
    later (PyPI classifier), which this GPLv3 project could take.
  - Scripts and reports: `~/Documents/LKB/NLSE_tangui_fix_specs/cpu_fft/`
    (`fused_proto.py`, `nl_serial_check.py`, `results_*.txt`), and on the box
    `~/Documents/nlse_scratch_scripts/rep/cpu_fused*.txt`. The box venv is
    `~/Documents/nlse_scratch_scripts/fftw_venv`.
- **The CPU transform libraries**, re-checked on 2026-11-27: nothing taken.
  The figures and the decision are in `docs/optimization-log.md`, CPU
  section ("The transform libraries, measured again"). The docs no longer
  call the transform pocketfft (scipy 1.18 ships ducc), and the log gives
  the box 4 MiB of L3.
- **complex64 accuracy is set by the transforms**, not the kernels: a round
  trip loses 2.8 units of 2^-24 of power, every step. Writing the linear step
  as `A + ifft((P - 1) fft(A))` cuts the field error 40 to 390 times for 7 to
  26% of a CPU step (prototype, optimization log, *Open*). The user has not
  decided.
- Per-call costs from the CUDA-box profile, below: the propagator download,
  `_energy_rates` twice, the potential grids.
- MLX and OpenCL: the fewer-passes lessons are not ported.

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

Environment:

- The project `.venv` has CPU and MLX only. CuPy and OpenCL run on the CUDA
  box, each in its own environment; OpenCL no longer runs on this Mac (see
  "Verification of the final tree").
- Sphinx comes with `uv sync --group docs` (repeat the `--extra` flags of the
  backends in use). `paper/` was not read.
- Running `docs/make_validation_figure.py` with any argument rewrites
  `docs/img/validation.png`.
