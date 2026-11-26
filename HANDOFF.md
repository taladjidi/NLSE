# Handoff

## State (2026-11-26)

A full audit of the repo was done at commit `fa7aa59` (v4.1.0 plus six commits).
The fixes are specified in eight packages, each applied in its own git
worktree. Every worktree holds its whole patch. p7 and p6 are reviewed and
merged into `main` (not pushed); p4 and p2 are reviewed and committed on their
branches; p1a, p1b, p3 and p5 are applied and uncommitted, with a code review
running on each.

Baseline at `38a8704`, Apple M3 Max, arm64, Python 3.13.13, CPU and MLX
backends: `pytest tests/ -q -m "not benchmark"` gives 974 passed, 123 skipped,
62 deselected; `ruff check`, `ruff format --check` and `mypy NLSE/` are clean.
In a plain copy of the tree `tests/test_packaging.py` errors 13 times (no
hatchling there): deselect it, which leaves 956 passed, 123 skipped.

## Fix work in progress

Each package has an architect spec, reviewed once and revised, and the
architect's verified result as a patch against `38a8704`. Both are outside the
repo: `~/Documents/LKB/NLSE_tangui_fix_specs/pN.md` and `pN.patch` (p7 also has
`p7_claude.patch` for the untracked local `CLAUDE.md`). Every patch passes
`git apply` on a fresh `git archive 38a8704` tree.

The worktrees are `../NLSE_wt/pN`, cut from `38a8704`, each with an untracked
`.venv` link to this checkout's venv (delete the link before committing, and
never `git add -A` there). Haiku implementers apply one brief each, in the
order the spec gives.

**How to tell where a package stands:** apply `pN.patch` to a fresh
`git archive 38a8704` copy and `diff -rq` it against the worktree (leave out
`.git`, `.venv` and the caches). No difference means every brief landed. A
difference means the chain stopped at a brief: read that brief in the spec and
finish it, or apply the patch.

| Package | Branch | Covers | State when this was written |
|---|---|---|---|
| p1a | `fix/stepping-loop` | medium ends exactly at `L`; backward runs; complex `delta_z` raises; callback lists; CuPy stream race | applied, equal to the patch; code review running |
| p1b | `fix/solver-run-state` | state left after a run or a backend switch; caller's arrays restored; asserts become ValueError/TypeError; dispersion built once; `origin="lower"` | applied, equal to the patch; code review running |
| p2 | `fix/coupled-solvers` | Rabi coupling in strang, yoshida, RK4; DDGPE uses CNLSE's `split_step`; noise, pump sampling, potential sign | reviewed; 1024 passed; committed `4c8fb80`. Its MAJOR (a cut last step takes the wrong drive sample and length) is fixed by p1a once both are merged: see step 3 |
| p3 | `fix/gpe-units` | GPE takes `V` and `g` as rates in rad/s with the GPE sign; `adapt_delta_z` on GPE; docs and example | applied, equal to the patch; code review running |
| p4 | `fix/nonlocal-response` | non-local response applied in Fourier space; `Backend.convolution` and `nl_profile` removed | reviewed; one test added for the coupled RK4 path (the review's MAJOR); 1005 passed; committed `c50a72c`; `p4.patch` refreshed |
| p5 | `refactor/kernels-cleanup` | unused CuPy and vortex kernels removed; MLX scalar cache bounded; CUDA built without `--use_fast_math` | spec reviewed (GO), its six MINOR notes applied to `p5.md` and `p5.patch`; applied, equal to the patch; code review running |
| p6 | `test/closed-forms-and-cache` | `evaluate_delta_n`; benchmark cache isolation; tests that could not fail; NLSE_3d dispersion sign | reviewed twice; cache record validated and `test_build_propagator` made to see the sign (the review's MAJORs); 1013 passed; merged as `3f3b6dd`. On CUDA the two RK4 non-local saturation cases stay red until p4 is merged |
| p7 | `docs/packaging-ci-cleanup` | README, docs, sdist, `uv.lock`, CI, examples, unused files | reviewed (PASS); 974 passed, Sphinx `-W` clean; merged as `ed00008` |

`fix/solver-state` is an older, merged branch: p1b uses `fix/solver-run-state`.

### Next steps

1. Per uncommitted package: read its review, fix CRITICAL and MAJOR findings
   (MINOR ones go in the final report), run its final check (the last
   section of each spec gives the commands and counts), commit on the
   branch. Breaking changes carry `!` and a `BREAKING CHANGE:` footer: p2,
   p3, p5 (removed kernels: `kernels_cpu.vortex`, `kernels_gpu.vortex_cp`
   and the CuPy fallbacks `rabi_coupling`, `apply_propagator`, `square_mod`,
   `rk4_axpy`, `rk4_accumulate` with their `_fused` twins) and p6 (NLSE_3d
   sign). A commit that stages a `.yml` file needs the plain PATH (the
   `yaml-parses` hook needs PyYAML, which `.venv` lacks); Python changes
   need `.venv/bin` first on PATH.
2. Merge one by one with `--no-ff`, in the order p7, p6 (both done), p3, p5,
   p1b, p1a, p4, p2. `git rerere` holds the two resolutions below. A trial
   merge of the eight patches in that order gave two
   conflicts only, and the merged tree passed on the Mac (ruff, format and
   mypy clean; 1314 passed, 123 skipped with `tests/test_packaging.py`
   left out):
   - `README.md` (p3, p7): keep p7's `NLSE/solvers/*.py` links; the GPE row's
     notes cell is p3's, ``SI units: masses in kg, times in s, `V` and `g`
     as energies over ħ (rad/s)``.
   - `NLSE/solvers/nlse.py`, constructor (p4, p1b): the block becomes the two
     lines `self._forget_device_state()` and `self.nl_length = nl_length`.
3. After the merges, by hand:
   - Run `.venv/bin/python ../NLSE_tangui_fix_specs/post_merge_cut_step_test.py .`
     and commit the two tests it adds to `tests/solvers/test_ddgpe_drive.py`:
     they pin p2's MAJOR (20 passed on the trial-merged tree, 4 failed on p2
     alone).
   - `CHANGELOG.md`: put `../NLSE_tangui_fix_specs/changelog_unreleased.md`
     above `## 4.1.0`, corrected for what the last four reviews change. No
     version bump without the user's word (5.0.0 is the suggestion).
   - `tests/physics/test_closed_forms.py`: the comment above the non-local
     case becomes "Two cells of that grid. A plane wave is its own local
     average, so the answer does not change: only the kernels that compute
     it do."
   - `NLSE/backends/cupy_backend.py`: the `kernels` docstring becomes
     `"""Return CUDA C kernels (with broadcasting fallback)."""`.
   - p2's `tests/solvers/test_potential_sign.py` cites `--use_fast_math` for
     `PHASE_ATOL`: p5 removes the flag, so re-measure on CUDA and reword.
   - The untracked local `CLAUDE.md`: `p7_claude.patch` (lines 55 and 75) and
     the non-local line 74 from p4.md, step O1.
4. Full suite on the Mac (CPU, MLX), under the conda Python for OpenCL (the two
   non-local test files above all: no architect could run CL), and on the CUDA
   box, at least six runs of each CUDA check. On CuPy also re-check a second
   RK4 `out_field` call on one solver (1024^2, `alpha=10`): it returned NaN at
   base, probably the stream race.
5. Timings before and after into `docs/optimization-log.md`, a CHANGELOG entry
   from the "Changelog lines" section of each spec, this file, then push.

### Review findings left as they are (MINOR, for the final report)

- p2: RK4 with a batched `omega` of shape `(B,)` or `(B, 1, 1, 1)` raises on
  CPU (`_add_rabi_slope` in `cnlse.py`); `_default_delta_z` leaves
  `_drive_step` set when called directly; no test for `omega` with a cut
  step, a step-changing callback or a batched `omega`; `docs/solvers_overview.md:263`
  says a positive imaginary potential removes power, which is the NLSE
  convention and now the opposite of DDGPE's.
- p6: the saturated Kerr index is written twice in `callbacks.py` (in
  `evaluate_delta_n` and `adapt_delta_z`); `evaluate_delta_n` is optical
  only (it reads component 1's `n2` and `I_sat` on a coupled solver and is
  wrong for GPE and DDGPE), which the docstring does not say; the cache's
  temporary file is named by PID alone, so two threads of one process can
  collide; `invalidate_cache` raises if the cache path is a directory;
  `docs/backends.md:51` still gives the old cache path and its table omits
  `NLSE_CACHE_DIR`; several new test docstrings lack a Parameters section.
- p7: `examples/benchmarks.py:187`, the self-check cannot see a wrong
  diffraction sign; `.github/workflows/ci.yml:287-296`, the link check only
  warns, so its comment oversells it; `docs/literature-examples.md` still
  opens with "None should be written from scratch".
- p4: `nl_length=np.inf` is accepted and gives NaN (the old window check
  raised); the smoothed intensity can dip to about -4e-7 of its maximum,
  where the zero-padded kernel could not; comments at `nlse.py:855`,
  `cnlse.py:369`, `examples/nonlocality.py:5` and two test names in
  `tests/integration/test_broadcasting.py` still say "convolve"; no test
  changes `nl_length` after construction, runs in double precision, or
  covers GPE and DDGPE; `_multiply_in_fourier_space` takes a positional
  bool. OpenCL was read, not run.

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
- Callbacks read `_current_delta_z` as the step just taken, also on a cut step.
- DDGPE: noise std goes as `sqrt(|dt|)`. `F_pump_t[j]` is the drive during
  `[j dt, (j+1) dt)` with `dt` the step asked of `out_field`; a profile that is
  too short raises ValueError. The potential gets the documented sign
  (`POTENTIAL_SCALE = -1.0`). Breaking.
- NLSE_3d: the temporal dispersion gets the documented sign
  (`dz E = +i D0/2 Omega^2 E`). Breaking.
- Not done on purpose: skipping the energy-rate estimate when `delta_z` is
  given (the cap on a caller's step needs it).

### Found while specifying, beyond the audit

Confirmed by a run, and fixed by the package named:

- CuPy: `execute_loop` runs on a non-blocking stream that does not wait for the
  null stream, so the first step raced work queued before it. On the CUDA box
  `tests/solvers/test_strang_merge.py -k CUPY` failed 11 runs of 12, and 0 of
  12 with `stream.wait_event(...)` added (p1a). One green full-suite run had
  hidden it: repeat CUDA runs.
- DDGPE applies `exp(+iVt)`, the opposite of its documented equation; the
  interaction sign is right (p2).
- DDGPE with any potential has its step capped near `pi/(V k/2)` with
  `k = 2 pi / 1e-30`, so the run never ends (p2).
- NLSE_3d applies the temporal dispersion with the sign opposite to the docs
  and to its own docstring; no test could see it (p6).
- A backward run with an adaptive callback turns forward and ends at `+z`
  (p1a).
- `docs/callbacks.md:103` documents the old `evaluate_delta_n` formula (p6).

Left open, to report:

- RK4's default step gives NaN in a trap: its limit uses the field-weighted
  kinetic rate, not the grid maximum.
- `_estimated_rates` uses an optical intensity for GPE when no field is given.
- `evaluate_delta_n` convention: `k/2` or `k`.
- Setting `simu.wl` does not update `k`.
- `get_fastest_backend` crashes on a wrong-typed `grid_size`.
- NLSE_3d's "xt at y=0" panel fixes x.
- `_normalization_factor` upcasts to float64.
- `tests/benchmarks/test_benchmark_profile.py` runs GPE with `g=n2`: check it
  after the units fix.
- `omega=0` (not None) still pays two identity rotations in strang.
- `DDGPE(omega=None)` raises TypeError in the constructor, so its
  `omega is None` branches never run.
- `evaluate_delta_n` stays optical-only: on GPE it reads `n2`, not `g`.
- DDGPE's cavity saturation used the swapped pair `(I_sat, I_sat2)`; p2 fixes
  it, visible only to a user who set `I_sat2` by hand.
- `ci.yml:5` sets an `NLSE_CACHE_DIR` nothing reads, and `ci.yml:168-172`
  caches benchmarks the tests no longer write: decide when p6 and p7 merge.
- OpenCL was not run for the non-local change (Apple's CL crashes in VkFFT).

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

- See "Fix work in progress" above for the order of work.
- CuPy now runs on the CUDA box (CuPy 14.2.0, RTX 3050). Sphinx builds with
  `-W --keep-going`, exit 0, in about 25 min. `paper/` not read.
- The project `.venv` has CPU and MLX only. OpenCL needs the conda Python with
  `PYTHONPATH` set to the repo, because that Python has an old NLSE 2.3.0
  installed.
- Running `docs/make_validation_figure.py` with any argument rewrites
  `docs/img/validation.png`.
