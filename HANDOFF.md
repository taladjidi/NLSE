# Handoff

## State (2026-11-26)

A full audit of the repo was done at commit `fa7aa59` (v4.1.0 plus six commits).
It changed no code. Nothing below is fixed yet.

Baseline at that commit, Apple M3 Max, arm64, Python 3.13.13, CPU and MLX
backends: `pytest tests/ -m "not benchmark"` gives 979 passed, 123 skipped;
`ruff check`, `ruff format --check` and `mypy NLSE/` are clean.

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

- Fix 1 to 8 first; each has a closed form to test against.
- Not checked: CuPy never run; `NLSE/backends/cupy_backend.py` and
  `NLSE/backends/opencl.py` not read; Sphinx build not run;
  `docs/optimization-log.md` and `paper/` not read.
- The project `.venv` has CPU and MLX only. OpenCL needs the conda Python with
  `PYTHONPATH` set to the repo, because that Python has an old NLSE 2.3.0
  installed.
- Running `docs/make_validation_figure.py` with any argument rewrites
  `docs/img/validation.png`.
