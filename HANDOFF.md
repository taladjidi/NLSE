# Handoff

## State (2026-11-26)

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

### Verification of the final tree

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

- `evaluate_delta_n` convention: `k/2` or `k`.
- Setting `simu.wl` does not update `k`.
- `get_fastest_backend` crashes on a wrong-typed `grid_size`.
- `_normalization_factor` upcasts to float64.
- `tests/benchmarks/test_benchmark_profile.py` runs GPE with `g=n2`.
- `omega=0` (not None) still pays two identity rotations in strang.
- `DDGPE(omega=None)` raises TypeError in the constructor, so its
  `omega is None` branches never run.
- The `yaml-parses` pre-commit hook needs PyYAML, which the `dev` extra does
  not list.
- `ci.yml:162` caches the backend benchmarks between runs; the tests now keep
  theirs in each test's own directory, so check the step still saves anything.

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

Resume here. Everything above is merged into `main` and pushed. Open: the
performance profiles, and the release (5.0.0 suggested; the version is not
bumped and nothing is published).

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

State:

- CUDA box: a profiling run was under way when this was written. Its tables,
  ranked candidates and the names of its scripts are in
  `~/Documents/LKB/NLSE_tangui_fix_specs/perf_profile_cuda_box.md`, partial if
  the run was cut short. The file did not exist yet when `main` was pushed
  (`8bce83b`): if it is still missing, the run was lost and starts again from
  the list above. Not vetted: check its top numbers before using them.
- Mac: not started. Run it with the machine idle (a push runs the suite).
- Nothing is in `docs/optimization-log.md` from the profiles yet.

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
- Open in the optimization log already: the separable propagator, the
  nonlinear step as an inverse-transform store callback, a plan callback in
  1D, an `alpha == 0` branch in `_nl_prop`.

Environment:

- The project `.venv` has CPU and MLX only. CuPy and OpenCL run on the CUDA
  box, each in its own environment; OpenCL no longer runs on this Mac (see
  "Verification of the final tree").
- Sphinx is in no project environment: build the docs from a scratch venv
  with `uv pip install -e ".[docs]"`. `paper/` was not read.
- Running `docs/make_validation_figure.py` with any argument rewrites
  `docs/img/validation.png`.
