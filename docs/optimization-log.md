# Optimization log

What has been tried on each backend, what was kept, what was rejected, and the
numbers behind each verdict. The point of writing the rejections down is that
they can be challenged: every entry says what would make it worth re-testing.

An entry is **deployed** (in the tree today), **rejected** (measured, does not
pay), or **open** (measured as promising, not yet built).

All timings are an Apple M3 Max (16 CPU threads, 40 GPU cores, rated 400 GB/s)
unless stated. Numbers from another machine are worth little here — arm64 and
x86 disagree about several of these, and the entries say where.

## How to measure

Ad-hoc timing loops have misled this project more than once: on the GPU they
showed 21–90% spread and gave two contradictory answers on the same question.
Use the tools, and interleave variants rather than running them in sequence.

- `benchmarks/profile_backends.py --baseline <rev>` — the end-to-end guard.
  Runs the same workload against the working tree and a git worktree of `<rev>`
  in a subprocess. It measures the *slope* between two step counts, so setup
  (FFT planning, propagator build, transfers) cancels instead of being divided
  out. This is the one that decides whether a change ships. It runs lossless
  by default, which is the case every revision has in common; `--alpha 20`
  measures the solved real-space step instead, and the two answer differently
  on MLX.
- `benchmarks/profile_kernels.py` — per-kernel roofline when the change is
  inside a kernel. Reproduces to 2–5%. Percentages are against the best rate
  observed in the same table, not a synthetic ceiling.
- `benchmarks/roofline.py` — what the hardware allows, and how much of it a
  step gets. Ask this *before* optimizing: a kernel already at 80% of a bound
  has under 1.25x in it however it is rewritten.
- A kernel-level win is not a step-level win. Check what share of a step the
  kernel actually holds before investing — see the roofline below.

Never `git stash push <file>` to verify a regression; monkeypatch the old
implementation from a scratch script instead. Stashing has twice stranded the
change when a following command timed out.

**Know the floor of each tool before trusting a number against it.** Both
benchmarks compare across processes, and on a laptop they scatter more than
many of the effects they are asked about.

- `profile_backends.py --baseline` used to measure the working tree in its own
  process and the baseline in a subprocess, each side through to the end before
  the other began, so process state and drift both landed on one side. With
  *identical code* on both sides it reported 3 of 6 cells more than 10% slower,
  up to 1.19x, reproducibly. Fixed: both sides now run in the same kind of
  subprocess, alternating a round at a time and swapping which goes first, and
  the table reports per cell how much that cell moved between rounds of the
  same code. Identical code now gives 0.96-1.02x with nothing flagged, and an
  injected slowdown is still caught at 1.51x against 11.5% scatter.
- What the fix removes is the *bias*, not the scatter. Cells still move 3-13%
  between rounds on this machine, and the table says so on every line. Nothing
  under a cell's own noise figure is a result, however many times it is rerun.
- `profile_kernels.py` reproduces to 2-5% *within* a process. Between processes
  it is far worse: the same untouched kernel measured 1.695 ms and 1.198 ms in
  two runs an hour apart, a 40% swing.
- **The floor is per cell, not per machine, and the two boxes differ by an
  order of magnitude.** On the NVIDIA box the CUPY cells scatter 0.2-3.2%
  between rounds of identical code while its CPU cells scatter 20-36% in the
  same run — so a CUPY result there resolves a few percent and a CPU result
  resolves almost nothing. Read the noise column per line before reading the
  ratio; a 1.21x CPU cell on 32% scatter is not a slower cell.

So for anything smaller than the reported noise -- which is most kernel-level
work -- build both variants in one process and interleave them, as `_sincos`
above was measured. When the change is inside a kernel, compile a twin that
differs in *only* the thing under test, **matching the fast-math flags too**.
Comparing a restricted-fastmath kernel against a `fastmath=True` twin measures
the flags, not the change: it made an exact polynomial look like it had error
growing to 3e-8, when against a matched twin it was 3.5e-16 flat.

## What the machine allows

`benchmarks/roofline.py`. Measured ceilings on the M3 Max, and what one step
must pay whatever it does. The point is to know how much room is left before
spending effort on a kernel.

| ceiling | GPU | CPU |
|---|---|---|
| streaming bandwidth | 351–366 GB/s | 243–247 GB/s |
| fp32 fused multiply-add | 9.48–9.63 TFLOP/s | 0.62–0.63 TFLOP/s |
| sin + cos + exp | 74.2–77.7 G/s | 3.0–3.3 G/s |

These are the runs of 2026-11-28 tabled below. On 2026-07-31 (eedccbb) the
same probes read 357–416 GB/s, 10.2 TFLOP/s and 87–89 G/s on the GPU, and
235–242 GB/s, 0.55 TFLOP/s and 2.5–2.9 G/s on the CPU.

The GPU bandwidth is 88-92% of the 400 GB/s rating; in July it bracketed it,
read then as part of the working set being served by cache rather than DRAM.
The FMA figure is within 16% of the hardware limit (40 cores x 128 lanes x 2
at ~1.1 GHz); a float4 probe read 12.1 in July, but scalar is what CUDA can
also compile. The transcendental gap is the one that matters: **the GPU does
sin, cos and exp 24x faster than the CPU**, which is why the nonlinear kernels
dominate a CPU step and not a GPU one.

The bandwidth ceiling itself moved about 15% between runs in July and 4% in
these five. The tables keep the wider figure: nothing under ~15% apart is a
difference.

Compulsory traffic per grid point, counted on the path the step took
(`step_path`): **104 B** for a single-precision split step done as separate
passes; **64 B** where CuPy's cuFFT store callbacks carry the propagator in
the forward transform and the nonlinear step in the inverse one; **48 B** for
the CPU's three-pass blocked step, whose propagator is two vectors, not a
grid; **624 B** for RK4. Strang is charged as Lie: with no callback and no
potential or a real one, its half steps merge.

The tool used to charge every backend the separate path. Runs of it from
d28d9aa (2026-08-04, which limits the split step by dispersion) until 13345a4
also timed a 1e-5 m step that the solver lowered at 4096² (to 2.8e-6 m
on the tool's beam), so their 4096² column counted about 3.6 steps per step;
the tool now steps 1e-6 m and fails a run whose step was lowered. The
2026-07-31 tables this replaces (eedccbb, 92f9133) predate that limit.

Share of each ceiling, complex64, no potential, Lie, median of 5 runs on
2026-11-28, commit 9b685c3, macOS-27.2.1-arm64-arm-64bit-Mach-O, Python
3.13.18, 16 CPU threads, numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pyopencl
2026.1.2, mlx-metal 0.32.0, mlx 0.32.0. The ceilings in these runs: 351–366
GB/s, 9.48–9.63 TFLOP/s and 74.2–77.7 G/s on the GPU; 243–247 GB/s, 0.62–0.63
TFLOP/s and 3.0–3.3 G/s on the CPU.

| backend | grid | path | B/pt | step | memory | flops | sin+cos+exp | nearest |
|---|---|---|---|---|---|---|---|---|
| CPU | 1024² | separate | 104 | 1.714 ms | 26% | 21% | 19% | memory |
| CPU | 2048² | separate | 104 | 5.774 ms | 31% | 28% | 23% | memory / flops (within noise) |
| CPU | 4096² | blocked | 48 | 22.034 ms | 15% | 32% | 24% | flops |
| MLX | 1024² | separate | 104 | 0.301 ms | 99% | 8% | 5% | memory |
| MLX | 2048² | separate | 104 | 1.821 ms | 67% | 6% | 3% | memory |
| MLX | 4096² | separate | 104 | 7.049 ms | 69% | 6% | 3% | memory |

Strang was timed in the same runs and equals Lie within 2.5%, except MLX at
2048², where it is 5% faster (1.723 against 1.821 ms, the two ranges apart).
That one is not explained.

Three things follow.

**The GPU backends are at 67-99% of the memory ceiling**, so the remaining
headroom there is under 1.5x. Only MLX at 1024² is at it (99%).
**The CPU is at 26-32% of its nearest ceiling, so no ceiling limits it**: the
hardware allows those steps three to four times faster. Nearest is memory at
1024², memory and flops within noise at 2048², flops at 4096², with
sin+cos+exp behind at 19-24% — the transcendentals are what the `_sincos` and
`_exp` polynomials below were aimed at. The last column was headed `bound`
until 2026-11-28, and the README called it the limit: at a third of a
ceiling it is not one. The owner reaches 80-90% of the flops ceiling on other
projects, so a CPU pass is open.

**Below 1024² nothing is bandwidth-bound; it is all dispatch.** Fixed cost per
step, measured as a step on a 64² grid: **CL 0.44 ms, MLX 0.05 ms** for
split_step, and 1.83 ms against 0.11 ms for RK4. Apple's OpenCL costs almost
ten times what Metal does to put a step on the GPU, and up to 1024² that is the
whole story of CL against MLX. *Challenge this* by fusing CL's per-step
launches further, which is the only thing that can help at those sizes.

Bandwidth is measured in the DRAM regime and one number is used at every size,
so a grid small enough to sit in cache can come out **above 100%**. That is the
reading rather than an error: the grid beat DRAM and bandwidth is no longer
what limits it. A size-dependent ceiling was tried and is worse — the probe's
working set is a third of a step's, so it reports cache rates a step cannot
reach and the percentages move with the probe instead of the code.

### What is left on macOS

Per-kernel attribution at 2048², against the ceilings above.

| | time | share of step | of its own floor |
|---|---|---|---|
| CPU transform pair | 4.04 ms | 60% | 39% |
| CPU `square_mod_nl_prop` | 1.31 ms | 19% | **at the sincos ceiling** |
| CPU `apply_propagator` | 0.53 ms | 8% | 77% |
| MLX transform pair | 1.31 ms | 68% | 58% |
| MLX elementwise kernels | — | — | 85–94% |

**The nonlinear kernels are finished on both.** The CPU one runs at
3.2 Gelem/s against a machine ceiling of 3.0 — the polynomial `_sincos` and
`_exp` took it to the bound, and no further arithmetic trick can help. MLX's
elementwise kernels sit at 85–94% of bandwidth.

**What is left on both is the transform, and every substitute measured is
worse.** MLX's own CPU device is **11x slower** than scipy (57.6 ms against
5.3 ms at 2048²). pyFFTW was rejected earlier for being unvectorized on arm64.
Thread count is not it either — `workers=-1` beats every setting from 1 to 16
(7.3x scaling on 12P+4E cores).

**Apple Accelerate / vDSP — rejected, measured.** It is correct through ctypes
(`vDSP_fft2d_zip`, split-complex, max rel error 3.5e-7 against numpy) and
genuinely **1.37x faster than scipy (pocketfft then) on one thread**, 26.4 ms against
36.1 ms. It loses anyway, because vDSP has no threading of its own and
scipy's does: **5.5 ms against 26.4 ms, so scipy wins by 4.8x** in the
configuration actually used.

Threading it by hand does not rescue it. Splitting rows and columns across a
pool and calling `vDSP_fftm_zip` per chunk is *catastrophic* — 26.3 ms on one
thread becomes 249 ms on two and never recovers (105 ms at 16). The strided
column pass is what does it; giving each thread its own `FFTSetup` changes
nothing, so it is the access pattern and not setup contention. The
transpose-based way out fails on arithmetic: the contiguous row pass scales
only 2.38x (1.97 → 0.83 ms), so four of them cost 3.3 ms, and the eight plane
transposes a roundtrip needs cost more than scipy's entire roundtrip — 2.99 ms
each naive, and even a perfect blocked transpose only ties. That is all before
the split-complex conversion each step would need and before rewriting every
kernel to match.

*Challenge this* only with an FFT that is both faster per thread **and**
threads itself. Per-thread speed alone is not enough, which is the thing this
measurement settles.

### Lowering the floor instead of chasing it

Being at 60% of a floor is not the same as being finished, because the floor is
a property of the algorithm we chose. What a step must move is 104 B/point:
64 for the transform pair, 24 for the propagator multiply, 16 for the nonlinear
kernel. The transform is closed, but the other 40 are ours.

**A separable propagator — open, measured, not built.** (2026-11-28: built
since on the CPU, inside the blocked step, in CUPY's callbacks, and in MLX's
fused graph; see those sections. Measured since on OpenCL, rejected.) The
linear propagator is `exp(-i(kx²+ky²)dz/2k)`, which factors into
`exp(-i·kx²dz/2k) · exp(-i·ky²dz/2k)` — verified rank-1 to 6.7e-8, float32 eps.
So the N² array every step reads could be two N-vectors: **8 MiB → 16 KiB at
1024², 32 MiB → 32 KiB at 2048²**, and the kernel drops from 24 B/point to 16.

Measured on a prototype kernel, `A *= P` against `A *= py[i]*px[j]`:

| | 1024² | 2048² | 4096² |
|---|---|---|---|
| CPU | 0.91x | **2.76x** | 1.93x |
| MLX | 1.16x | **1.44x** | 1.41x |

The 2048² CPU figure is larger than the 1.5x traffic ratio allows, because
losing a whole grid puts the rest in cache: that kernel reads 345 GB/s against
a 248 GB/s DRAM ceiling. At 1024² there is nothing to win — the propagator
already fits — and it costs 9%.

At the step level that is ~5% on CPU and ~8% on MLX at 2048², plus whatever the
freed grid does for the transform, which is not measured. The work is not
small: `apply_propagator` on four backends, the fused split-step and RK4 paths,
the `1/N` currently folded into the propagator, the RK4 operator (a *sum*, so
separable differently), NLSE_3d, and the batched paths.

**Already built, so not a lever:** merging the Strang half-steps of adjacent
steps. `_merges_strang_halves` and the bracket pair do it, which is why double
precision costs about the same as single rather than twice.

**Rejected this round**, all on the MLX fused nonlinear kernel, which reads
110 GB/s against a 356 GB/s ceiling and looked like it had something wrong:

- Spelling `|A|²` as `A.real² + A.imag²` or `abs(A)²` instead of
  `(A·conj A).real` — 0.89x and 0.98x, inside 7–13% noise.
- Writing the rotation out as `cos + i sin` instead of a complex `mx.exp` —
  1.01x. Emitting it as a stacked real array instead: 0.29x.
- **A hand-written Metal kernel** via `mx.fast.metal_kernel`, one pass, correct
  to 4.4e-8 — **0.99x**. `mx.compile` was already matching raw Metal.

That last one also kills the inference that made the kernel look wrong: 110 GB/s
against the ceiling implied about three passes, and a kernel that provably makes
one is no faster. The apparent gap was an artefact of comparing traffic models —
`nl_prop` scores 237 GB/s only by counting a precomputed `A_sq` it did not have
to compute, and the fused kernel does less total work than the pair it replaces
(0.63 ms against 0.71 ms).

So: **the only measured lever left on macOS is the separable propagator.**

**Folding the propagator into the transform (cuFFT callbacks) — open, probe
written, nothing measured.** (2026-11-28: deployed since, see "CUDA (CuPy)";
the store callbacks of both transforms carry the propagator and the nonlinear
step, 64 B/point and not the 88 and 72 estimated here.) Better arithmetic than
the separable propagator,
because it removes the pass rather than shrinking it. cuFFT can run a store
callback as it writes each element, which absorbs the propagator multiply
entirely: **104 B/point becomes 88**, and a load callback on the inverse could
take the nonlinear step too, for **72**. On a backend already at the bandwidth
bound that is close to a straight 15–30%.

`benchmarks/cufft_callback_probe.py` (deleted since) asked the four questions
that decide whether it can be built, none of which can be answered on a
machine without an NVIDIA GPU:

1. Is `cupy.fft.config.set_cufft_callbacks` there at all? It is experimental,
   compiles the callback with nvcc and links cufft statically, so it needs a
   full toolkit rather than a driver.
2. Does a store callback compute the right thing?
3. **Does the callback see new values written into the aux array without the
   plan being rebuilt?** The one that decides the design. Callbacks bind at
   plan creation and the propagator changes whenever an adaptive step does, so
   if the array can be overwritten in place the plan is built once per run —
   and if not, every step change costs a plan and the idea is probably dead.
4. Is it actually faster than the separate multiply?

Two things to weigh before building on a green probe. It is CUDA-only, so it
widens the gap between CUPY and the other three rather than lifting them
together. And it puts a toolkit in the install path for anyone who wants it,
which is a reasonable ask of this audience but not free.

### The NVIDIA box

Same tool, `CPU CUPY`, run 2026-07-31. **The CUDA and OpenCL probes agree to
within 1% on all three ceilings** — 208 GB/s against 208, 8.10 TFLOP/s against
8.11, 83.0 G/s against 84.1. Two independent implementations landing on the
same numbers is what makes them the hardware rather than the framework.

| ceiling | GPU | CPU |
|---|---|---|
| streaming bandwidth | 208 GB/s | 32 GB/s |
| fp32 fused multiply-add | 8.1 TFLOP/s | 0.46 TFLOP/s |
| sin + cos + exp | 84 G/s | 1.3 G/s |

Re-run on 2026-11-27 with `CPU CUPY CL`, commit 67d5b6e, median of 5 runs
(Linux-7.3.4-200.fc44.x86_64-x86_64-with-glibc2.43, Python 3.12.16, 8 CPU
threads, numpy 2.5.5, scipy 1.18.1, numba 0.68.1, cupy-cuda13x 14.2.0,
pyopencl 2026.1.5). The ceilings in these runs: 208 GB/s, 8.12–8.22 TFLOP/s
and 84.7–86.0 G/s on the GPU; 32 GB/s, 0.46 TFLOP/s and 0.7–1.0 G/s on the
CPU, whose sin+cos+exp probe reads lower than the 1.3 above and moves between
runs. That is not explained.

| backend | grid | path | B/pt | step | memory | flops | sin+cos+exp | nearest |
|---|---|---|---|---|---|---|---|---|
| CUPY | 1024² | callbacks | 64 | 0.390 ms | 83% | 7% | 3% | memory |
| CUPY | 2048² | callbacks | 64 | 1.598 ms | 81% | 8% | 3% | memory |
| CUPY | 4096² | callbacks | 64 | 8.832 ms | 59% | 6% | 2% | memory |
| CL | 1024² | separate | 104 | 0.601 ms | 87% | 5% | 2% | memory |
| CL | 2048² | separate | 104 | 3.337 ms | 63% | 4% | 1% | memory |
| CL | 4096² | separate | 104 | 18.736 ms | 45% | 3% | 1% | memory |
| CPU | 1024² | blocked | 48 | 8.087 ms | 20% | 6% | 17% | memory / sin+cos+exp (within noise) |
| CPU | 2048² | blocked | 48 | 30.967 ms | 21% | 7% | 17% | memory |
| CPU | 4096² | blocked | 48 | 139.606 ms | 18% | 7% | 15% | memory |

Double precision and RK4 were not re-run; their 2026-07-31 figures are in
92f9133.

**CuPy is at 83%, 81% and 59% of the bandwidth ceiling at 1024², 2048² and
4096²**, on the 64 B per point of its callbacks path; the drop at 4096² is not
explained. Dispatch there costs 0.011 ms per step, forty times less than
Apple's OpenCL (not re-measured).

The two machines differ in which ceiling is nearest to the CPU, and it is the
hardware rather than the code. That box streams **32 GB/s against the M3 Max's
247** — a
desktop's dual-channel DDR against unified memory — so memory is nearest on
every CPU row there, where on the Mac memory and flops tie at 2048² and flops
are nearest at 4096². Neither CPU is bound by it: 18-21% there, 26-32% on the
Mac. Its sin+cos+exp rate is also a quarter of the Mac's (0.8 against 3.2
G/s), consistent with
`USING_SVML` being false on that x86 too while NEON vectorizes the polynomials.
Conclusions about *which* CPU ceiling to attack do not transfer between them.

## Where the time goes

`profile_kernels.py --backends CPU`, 2048², complex64. The split-step kernels
are elementwise: they read a couple of arrays, do a few flops and write one
back, and none of it fits in cache, so the floor is how fast the machine moves
bytes. Best observed rate in this table: 250 GB/s.

| kernel | time | GB/s | of best |
|---|---|---|---|
| `fft_roundtrip` | 4.240 ms | 31.7 | (not a streaming read) |
| `rabi_coupling` | 0.537 ms | 250.1 | 100% |
| `apply_propagator` | 0.522 ms | 192.7 | 77% |
| `square_mod` | 0.285 ms | 176.6 | 71% |
| `rk4_accumulate` | 0.521 ms | 193.3 | 77% |
| `nl_prop` | 2.707 ms | 37.2 | **15%** |
| `square_mod_nl_prop` | 1.825 ms | 36.8 | **15%** |

Two facts follow, and they drive most of what is below. The transform is the
single largest item in a CPU step. And the nonlinear kernels are the only ones
far from bandwidth — they are compute-bound on transcendentals, not memory-bound,
so bandwidth tricks do nothing for them and vice versa.

## Doing less work

Worth more than every kernel below put together, and measured with
`benchmarks/work_precision.py`, which scores wall clock against error rather
than against a step count.

**The default step is the measured optimum — REVERTED, and the reason is
worth more than the entry.** In complex64
split-step is not limited by the splitting but by round-off accumulating over
steps, so refining past the optimum costs time and accuracy together. Measured
on a self-focusing beam, 256², Strang:

| rad/step | 0.05 | 0.1 (default) | 0.2 | **0.4** | 0.8 | 1.5 |
|---|---|---|---|---|---|---|
| time | 117 ms | 59 ms | 30 ms | **16 ms** | 8.5 ms | 5.2 ms |
| rel. error | 2.1e-4 | 1.0e-4 | 5.5e-5 | **3.3e-5** | 4.3e-5 | 1.2e-4 |

The optimum sits at 0.4–0.8 rad across a 16x range of propagation distance
(0.4, 0.4, 0.8 at L = 1.25e-3, 5e-3, 2e-2), so it is not an artefact of one
problem. Against the 0.1 default, 0.4 is **3.4–4x faster and 1.8–3.8x more
accurate at the same time** — there is no trade being made.

**RK4 wants the opposite step, and shares the constant.** Its optimum is
~0.02 rad; at the shared 0.1 default it returns 5.6e-4 where 0.02 returns
2.3e-6. So the one default is four times too fine for split-step and five times
too coarse for RK4. It also means RK4 at its default is worse than split-step
at *any* step, for six times the cost: split-step at 0.4 matches RK4 at 0.05 to
within 2% of error and runs 42x faster.

**It does not generalise, and shipping it broke an example.** The sweep above
varied the propagation distance sixteenfold and never varied the *physics*.
`examples/fig2_turbulence.py` carries `kp = 2*pi*5e3` — spectral content of its
own — and aliases far below the pi-per-step cap that the criterion is written
against:

| phase/step | steps | max abs A | rel. error |
|---|---|---|---|
| 0.400 | 333 | 3034 | **1.401** |
| 0.100 | 1331 | 2860 | 0.115 |
| 0.010 | 13310 | 2853 | 0.027 |

A cliff, not a slope: 11% error becomes 140%, with the peak amplitude and the
total power both visibly wrong. `DEFAULT_PHASE_PER_STEP` is back to 0.1. The
adaptive floor went with it — a controller stopped at 0.4 rad on this problem
would report success at 140% error, which is worse than a slow run.

`RK4_PHASE_PER_STEP` stays at 0.02: that change makes the step *finer*, and
error that only falls is safe to ship.

**What a step criterion would have to measure.** The phase per step counts the
potential and the interaction, on the reasoning that split-step applies the
linear part exactly. True for a linear problem, but the splitting error goes
as the commutator of the two parts, and a field with high-k content has a large
one at a nonlinear phase per step that looks modest. Adding the kinetic rate
does not fix it — here it is 70 against an interaction of 531. The quantity
that predicts the cliff is spectral headroom, which nothing currently measures.

The rest of that batch was kept, and the consequences it had to handle:

- **The adaptive controller is held to a band**, `[optimum, 2 × optimum]`, in
  complex64. Its own error estimate goes blind above ~0.8 rad — one step and
  two halves then differ by round-off rather than by splitting error, which
  reads as "no error" and doubles the step until the answer is unrecognisable.
  Starting it at the optimum without a cap returned 28% error.
- **A `min_step` below the optimum raises** rather than being honoured or
  silently ignored. Both of those are worse than saying why.
- **`DEFAULT_MIN_STEPS` binds more often**, so on short problems the step is
  set by the sampling floor rather than by the physics. That is the floor
  doing its job, but it means a test asking whether the step tracks the field
  has to propagate far enough that the phase target is what binds.

**A step that grows must still land on `z` — fixed, and it was a real bug.**
The loop divides `z` into steps before it starts, so a callback that *grows*
the step left it taking one that did not fit. A run whose step doubled
propagated 1.1e-3 for a requested 1.0e-3 and came back with a phase error to
match — 0.283 relative, with the amplitude still correct to five figures, which
is exactly the shape of wrongness nobody notices. Trimming the last step gives
2.96e-06 on the same run. It predates all of this work; a shrinking step
overshoots by at most one small step, which is why only raising the default
exposed it.

**The adaptive controller floored at the optimum — deployed.**
`adapt_delta_z_to_error` used to shrink toward an absolute `L/1e5`, which in
complex64 means shrinking past the optimum into pure round-off. A tolerance it
cannot meet therefore bought error with time, and bought a great deal of both:

| tolerance | floor | time | rel. error |
|---|---|---|---|
| 1e-6 | optimum | **18.2 ms** | **3.11e-05** |
| 1e-6 | old `L/1e5` | 32.5 ms | 3.50e-01 |
| 1e-8 | optimum | **23.2 ms** | **3.01e-05** |
| 1e-8 | old `L/1e5` | 54,178 ms | 6.57e-02 |

At 1e-8 that is **2,300x the speed and 2,200x the accuracy**. The floor is
derived from the same rate as the π-per-step cap, so it tracks a self-focusing
beam rather than being a fixed distance. An explicit `min_step` still wins:
naming one is deliberate, often for sampling.

**Yoshida for complex128 — deployed.** Three Strang sub-steps composed to
O(dz⁴) for three transform pairs, built out of the existing step. In complex64
it buys nothing, for the reason above, and warns. In complex128 the round-off
floor is gone and it dominates outright, measured at 256²:

| | rad/step | time | rel. error |
|---|---|---|---|
| yoshida | 0.8 | **31 ms** | 1.08e-09 |
| strang | 0.005 | 1202 ms | 1.19e-09 |

**38x the speed at equal accuracy**, and at the coarse end still 2x faster and
65x more accurate than Strang's nearest point. Below ~1e-10 the table stops
resolving it — that is the reference's own floor, not Yoshida's.

The earlier entry here rejected higher-order splitting outright. That was right
about complex64 and wrong to stop there: the argument reverses entirely with
the round-off floor removed, which is the whole reason the scheme is keyed to
the float width rather than offered as a default.

**`NLSE_3d` merges its Strang half steps — deployed** (2026-11-28). It never
did: the merge declined for a solver with an `omega`, meant for a Rabi
coupling, and `NLSE_3d` calls its frequency grid `omega`. The coupled solvers
opt out by their class flag, so the check is gone. Merged against unmerged,
same process, interleaved, slope between 20 and 60 steps, 5 rounds, complex64,
lossy saturable medium (`benchmarks/strang_merge_3d.py`; M3 Max, numpy 2.4.6,
scipy 1.18.0, numba 0.66.0, mlx 0.32.0, pyopencl 2026.1.2; CUPY on an RTX
3050, CuPy 14.2.0, repeated three times at 128³ and twice at 256²×64 with
ratios within 0.01):

| backend | grid | unmerged | merged | ratio |
|---|---|---|---|---|
| CPU | 128³ | 4.44 ms | 3.59 ms | 0.81 |
| CPU | 256²×64 | 8.97 ms | 6.97 ms | 0.78 |
| MLX | 128³ | 1.21 ms | 1.10 ms | 0.91 |
| MLX | 256²×64 | 2.84 ms | 2.71 ms | 0.96 |
| CL | 128³ | 1.61 ms | 0.91 ms | 0.57 |
| CL | 256²×64 | 2.06 ms | 1.31 ms | 0.64 |
| CUPY | 128³ | 1.38 ms | 1.22 ms | 0.88 |
| CUPY | 256²×64 | 2.78 ms | 2.46 ms | 0.89 |

The two runs differ by 3e-6 to 5e-6 of the peak after 20 steps, the rounding
of complex64. Not run on CuPy.

## Solving the lossy real-space step

Not an optimization, but it belongs with them: it is the largest change in
accuracy-per-unit-work in this file, and it was found by a benchmark.

`benchmarks/work_precision.py --problem turbulence` draws the work-precision
table for examples/fig2_turbulence.py, which is lossy (`alpha = 20` over 20 cm).
Every splitting on it converged at **first order** — Strang and Yoshida
included, both returning a drift ratio of 2.00 under step halving where their
orders are 4 and 16. Switching the loss off on the same field recovered 3.57 and
13.63, so it was the loss and not the turbulence.

**The cause.** The real-space step applies `exp(-alpha*s*dz + i*g*|A|^2*s*dz)`
with `|A|^2` read once, entering. That is the exact solution of the real-space
equation only while the step preserves `|A|^2`: a pure rotation does, loss does
not. The amplitude decays *inside* the step while the interaction keeps turning
the phase at the rate the step began with, which is a local error of O(dz^2) —
and no composition wrapped around a first-order step is better than first order,
however many sub-steps it takes.

**The fix.** Two exact facts. With `y = |A|^2`, `s(y) = 1/(1 + y/Isat)` and
`u = 2*alpha*dz`, `dy/dz = -2*alpha*s*y` and `dphi/dz = g*y*s` give
`dphi/dy = -g/(2*alpha)`, so the phase over a step is
`(g/(2*alpha))*(y0 - y_end)` whatever the saturation does in between, and
`y_end` is fixed by `ln y + y/Isat` falling by `u`. The kernels solve for
`P = (1 - y_end/y0)/u` with three passes of a fixed-point iteration -- about 18
flops and a sqrt, no transcendentals -- and apply `g*y0*P*dz` for the phase and
`sqrt(1 - P*u)` for the amplitude. See `_loss_factor` in kernels/cpu.py.

At `u = 0` the iteration returns `sat` and the amplitude factor is exactly 1, so
**a lossless run is unchanged to the bit** and takes the same branch it always
did. `profile_backends.py --baseline HEAD` finds no lossless cell slower beyond
this machine's scatter.

That last sentence held on the backends that branch at runtime and **not on
MLX**, where both arms of an `mx.where` are evaluated and a lossless step came
out 1.45x slower. Fixed by compiling a lossless graph — see *MLX (Metal)*.
The lesson generalizes past this change: "the branch is not taken" is a
statement about C-like backends, and on a graph backend it has to be measured
rather than argued.

**What it bought**, on the turbulence problem at 256², complex128, against an
RK4 reference self-consistent to 3e-11 (`--problem turbulence --field
complex128`):

| method | best error before | best error after | at matched cost |
|---|---|---|---|
| lie | 4.05e-3 | 3.42e-3 | 1.2x |
| strang | 1.35e-3 | **1.96e-5** | **69x** |
| yoshida | 8.82e-3 | **1.26e-6** | **7000x** |
| RK4 | 5.08e-10 | 5.08e-10 | unchanged, as it must be |

Yoshida floors at ~1.3e-6 there, which is not its own error: the splittings are
scored against a reference built by a different method, so on a chaotic problem
the floor is where uncorrelated round-off amplification lands. The RK4 rows are
scored against RK4 and see their own round-off instead, which is why they go
lower. That is a property of the table, not of the methods.

The splittings were 14x behind RK4 at matched accuracy on this problem and are
now within ~1.5x of it.

**What it cost.** Sequential rather than interleaved, so read these as
approximate: on a lossy run at 256², roughly +16% per step for Lie and +25% for
Yoshida (the extra flops in a kernel that is already transcendental-bound), and
~11% *faster* for Strang, because merging its touching half steps is now valid
with loss and saves a nonlinear application per step. Against 69x and 7000x in
accuracy, none of that is a trade.

**Measured again with the interleaved guard** (2026-08-01), `--alpha 20`
against `--alpha 0` on the same revision, which is what `--alpha` was added to
`profile_backends.py` for. The +16% does not survive it on the backends that
branch at runtime:

| backend | cost of the solved step | scatter |
|---|---|---|
| CUPY | **nothing measurable** (0.106 → 0.106 ms at 512²) | 0.7–2.7% |
| CL | nothing measurable | 1.4–4.7% |
| CPU | ~9% at 512² | 2.3% |
| MLX | ~1.5x, **since fixed to ~1.05x** | 5–7% |

The CPU figure is the only one near the original estimate, and it is under the
10% floor the guard reports against. MLX was the outlier for the reason in
*MLX (Metal)* — it cannot branch per element, and `mx.compile` would not fuse
the iteration — and a hand-written Metal kernel has since brought it back in
line with the rest. So the solved step is now free on every backend, which
makes the accuracy it buys unpriced rather than cheap.

**Where it does not reach.** The identity holds for one decay channel and a real
potential. An absorbing (complex) potential is a second channel and is still
applied frozen, so it still costs the order — and still blocks the Strang merge.
The coupled kernels are untouched: two components decaying at their own rates do
not give `dphi/dy = const`.

**A ceiling came with it.** The iteration is a contraction only while the step
takes out a small fraction of the intensity, and past `u ~ 1` it diverges and
returns a *larger* field than it was given. `LOSS_PER_STEP_LIMIT` caps a
propagation at `u <= 0.05`, and the kernels fall back to the frozen step above
`u = 0.1` so that a kernel called directly cannot amplify either.

**Where this has actually been run.** Everything above — this entry, the
propagator fusion, and the potential-width fix — was written and measured on an
**RTX 3050 box with CPU, CUPY and OpenCL**, not on the Mac the rest of this file
is written from. That matters twice over.

The numbers are that machine's. Its OpenCL is NVIDIA's and has fp64; Apple's has
neither the doubles nor the dispatch cost, so no CL figure here transfers.

The MLX kernels were written by transcription and **could not be run at all**.
That is not a small caveat: the first Mac run of this work failed 52 tests, 49 of
them one line of numpy dtype introspection in a shared code path (see *MLX
(Metal)* above), and the other three tests of mine that assumed a backend has
double precision. All four are fixed; none of the four is verified on Metal.
Anyone picking this up on a Mac should run the suite first and treat a green run
as the first evidence that the MLX side of it works.

**Answered on an M3 Max, 2026-08-01.** The suite is green there (1199 passed).
The two questions this paragraph left open are now measured, and the second
one's answer was no: `mx.compile` does **not** fold the extra iteration in for
free. It cost a lossless run 1.45x, which is a regression the NVIDIA guard
could not see, and it costs a lossy MLX step ~1.5x where CPU and CL pay
nothing measurable. Both are written up under *MLX (Metal)*; the lossless half
is fixed and pinned by a test, the lossy half stands.

## Cross-cutting

**Propagator caching — deployed.** Linear propagators are cached by
`(grid_size, delta_z, k, precision)`. Rebuilt when an adaptive callback moves
the step.

**Propagator fused with its normalization — deployed** (`8d5fa30`). The
inverse transform's `1/N` is folded into the propagator once at build time
rather than applied per step. Backends that can skip the normalization declare
`supports_unnormalized_ifft`; on scipy that is `norm="forward"`, verified
elementwise against pyfftw's `normalise_idft=False`.

**Kernel twins generated, not written — deployed.** `kernels/templating.py`
generates the no-V / real-V / complex-V variants from one VBLOCK-marked kernel,
shared between the OpenCL and CUDA C sources, guarded by
`tests/backends/test_kernel_templating.py`.

**The audit fixes cost no step time — measured** (2026-11-26).
`profile_backends.py --baseline 38a8704 --rounds 4`, complex64, lossless, NLSE
and CNLSE, split step and RK4 at 1024² and 2048²: 4.1.0 (`38a8704`) against
`4ac2bb2`. No cell is slower by more than it moved between rounds of the same
code.

| | NLSE | CNLSE | moved between rounds |
|---|---|---|---|
| CUPY, RTX 3050 | 1.00x | 1.00–1.01x | 0.1–1% |
| MLX, M3 Max | 1.00x | 0.99–1.00x | 0.4–4.6% |
| CPU, M3 Max | 0.99–1.06x | 0.96–1.04x | 2.7–13.9% |

RK4, in ms per step, 1024² then 2048²: CUPY 2.439 → 2.444 and 9.869 → 9.887
(NLSE), 4.855 → 4.892 and 19.836 → 19.916 (CNLSE); MLX 1.594 → 1.600 and
8.779 → 8.786 (NLSE), 6.219 → 6.213 and 27.745 → 27.735 (CNLSE). The CNLSE
cell at 1024² on CUPY is +0.8% here and was +0.4% in an earlier campaign
(4.856 → 4.877): under what that box resolves, and the only cell that leaned
the same way twice.

**The tool timed more steps than it asked for — fixed.** Two cases, both the
solver lowering the step it was given. Split-step at 1024² and above: the
limit at the tool's window is 6.19e-5 m at 1024² and falls as 1/N², so
`DELTA_Z = 1e-4` became 5.57e-5 and a cell ran 1.79 steps per step asked (7.18
at 2048² before the step shrank with the grid). Both revisions lower alike,
so the ratios above stand, but a split-step "ms per step" from this tool at
1024² or above is per step asked, not per step taken: the split-step figures
behind the table are not quoted for that reason. RK4 at 2048²: 4.1.0 took
the step asked and returned NaN, its successor bounds the step and took 2.56
times as many, which read as "2.56x slower". `step_for` now shrinks the step
as `(512/N)²` past 512², under both limits: no cell of NLSE, CNLSE or NLSE_1d
from 256 to 2048 has its step lowered, on either revision.
`trace_solvers.py` and `propagator_fusion.py` asked 1e-4 at every size until
2026-11-28: both take `step_for`'s step now and raise if the solver lowers it.
The fusion table further down is from 2026-07-31, before the dispersion limit
(2026-08-04, `d28d9aa`), so its figures are per step taken: the fixed tool
reads 0.126, 0.557 and 2.242 ms unfused at 512², 1024² and 2048² on the same
box, against that table's 0.125, 0.556 and 2.269.

**A call costs less than in 4.1.0, and is still worth many steps — measured**
(2026-11-26). One `out_field` call of one step asked, on a solver that has
run before, CUPY on the RTX 3050, complex64, fastest of 7 calls in 3
alternating rounds:

| 2048² | 4.1.0 | `4ac2bb2` |
|---|---|---|
| NLSE split step | 93.3 ms | 37.7 ms |
| NLSE split step, `V` set | 107.5 ms | 57.1 ms |
| NLSE split step, default step | 133.3 ms | 56.2 ms |
| NLSE RK4 | 102.9 ms | 54.4 ms |
| CNLSE split step | 170.1 ms | 109.8 ms |
| CNLSE RK4 | 234.9 ms | 181.3 ms |

All 48 cells (512² too, both solvers, cut and whole runs) are at 0.40–0.98x.
A split step taken at 2048² is about 1.95 ms there (13.97 ms per step asked,
7.18 taken), so the NLSE call is still worth about 19 steps. What it is made
of is not broken down yet.

## CPU (numba + scipy.fft)

**scipy.fft instead of pyFFTW — deployed** (`b69a42e`). Per step, best of 5 on
the slope between 20 and 220 steps:

| case | pyFFTW | scipy | |
|---|---|---|---|
| 512 split_step | 1.778 ms | 0.681 ms | 0.38x |
| 1024 split_step | 6.528 ms | 1.846 ms | 0.28x |
| 1024 RK4 | 27.637 ms | 11.540 ms | 0.42x |

FFTW only wins where it is vectorized, and on arm64 no prebuilt pyfftw is —
neither the PyPI wheel nor conda-forge ships NEON codelets. Deleting it also
removed disk wisdom and its staleness check, the `FFTW_MEASURE` save/restore,
the vector-codelet detector, and `benchmarks/check_fftw.py`; `backends/cpu.py`
went 402 → 132 lines.

*Challenged on x86 on 2026-11-27*, in the next entry: pyFFTW is ahead only at
512². The transform stays behind the `Backend` interface all the same.

**The transform libraries, measured again — nothing taken** (2026-11-27). The
two rejections, pyFFTW above and vDSP under "What is left on macOS", were
measured on arm64 only and against older libraries. `scipy.fft` changed
underneath in the meantime: from scipy 1.18 its transform is ducc, the
successor of pocketfft ([release notes](https://docs.scipy.org/doc/scipy/release/1.18.0-notes.html#scipy-fft-improvements)).

*x86.* Ryzen 3 4100 (4 cores, 8 threads, AVX2, L3 4 MiB), `fft2` + `ifft2` in
place, min over 15 × 5, interleaved in one process, each library at its best
thread count; numpy 2.5.5, pyFFTW planned with `FFTW_MEASURE`:

| | scipy 1.18.1 | ducc0 0.42.0 wheel | mkl_fft 2.4.0 | pyfftw 0.15.1 |
|---|---|---|---|---|
| complex64 512² | 1.330 ms | 1.189 (1.12x) | 0.610 (2.18x) | 0.740 (1.80x) |
| complex64 1024² | 7.694 ms | 6.256 (1.23x) | 4.947 (1.56x) | 8.048 (0.96x) |
| complex64 2048² | 28.657 ms | 29.291 (0.98x) | **wrong** | 28.630 (1.00x) |
| complex128 512² | 2.165 ms | 1.354 (1.60x) | 1.457 (1.49x) | 1.781 (1.22x) |
| complex128 1024² | 13.894 ms | 11.732 (1.18x) | 11.940 (1.16x) | 13.986 (0.99x) |
| complex128 2048² | 46.922 ms | 42.413 (1.11x) | 51.356 (0.91x) | 59.316 (0.79x) |

- The `ducc0` wheel is a newer release of the library scipy carries (0.42.0
  against 0.40) and a different build: its binary holds AVX and AVX-512 code
  (357k `ymm` and 140k `zmm` instructions under `objdump -d`), scipy's none.
  Which of the two gives the lead was not isolated, nor why the lead is gone
  at complex64 2048².
- pyFFTW stays rejected, now on x86 too: it is ahead at 512² only. The runs
  move by more than the gaps at 1024²: an earlier run of the same script had
  pyFFTW at 7.316 ms against scipy's 7.370 there, and scipy's 512² at 1.080 ms.
- `mkl_fft` is the fastest for complex64 at 512² and 1024², and **returns a
  wrong result** for complex64 2048² on 4 or 8 threads (`ifft2(fft2(a))` off
  by 1.4; one thread, 1024², 4096² and complex128 are right).
- 4096² was not timed on the box.

*vDSP threaded by hand, through the interleaved API.*
`vDSP_DFT_Interleaved_Execute` takes interleaved complex, so the split-complex
conversion the earlier entry counted is not needed. M3 Max, rows of an n × n
complex64 array in place, ns per point per 1D transform, min of 15, against
ducc0 0.42.0, machine under load (load average 4 to 6):

| | ducc0, 1 thread | vDSP, 1 thread | ducc0, 12 threads | vDSP, `prange` on 12 |
|---|---|---|---|---|
| 1024² | 1.079 | 0.706 | 0.205 | 0.303 |
| 2048² | 1.250 | 0.704 | 0.188 | 0.241 |
| 4096² | 1.701 | 0.890 | 0.220 | 0.282 |

vDSP is 1.5 to 1.9x faster on one thread and loses on twelve, by 1.5x at
1024² and 1.3x at 2048² and 4096². It stops scaling after two threads even
when each thread has its own 1 MiB buffer and shares only the setup: 0.684 ns
per point per thread on one or two threads, 2.442 on twelve, where ducc0 goes
1.290 → 1.564. The cause was not identified. Rejected again, on rows alone;
the column pass was not reached.

*Decision.* Nothing taken. `mkl_fft` is rejected for the wrong result alone:
where it is right it is ahead for complex64, by 1.56x at 1024². pyFFTW is
ahead at 512² only. The `ducc0` wheel is right everywhere and gives 1.18x to
1.23x of the transform at 1024² and 0.98x to 1.11x at 2048²; it is not taken
as it stands, a second copy of what scipy ships for that much on one machine,
with 4096² and arm64 not timed. It is the one to time first if x86 CPU runs
matter.

*Challenge this* if Accelerate documents how the DFT routines are meant to be
called from several threads, if a `mkl_fft` release fixes the 2048² result, or
if scipy's wheels come with the AVX code the `ducc0` wheel has.

**numba threads started before any vendored OpenMP loads — deployed**
(`87b3b4e`). Two OpenMP runtimes in one process segfault at the first `prange`.
The guard lives in `NLSE/kernels/cpu.py` and is no longer about our own
dependency: a caller can still import a library that vendors one.

**`alpha == 0` rotation instead of a complex exponential — deployed.** When
lossless, the exponent is purely imaginary, so the step is a rotation computed
with `cos`/`sin` on a real angle instead of `exp` on a complex one. Worth
1.33x on the kernel (`nl_prop` 2.707 ms → `nl_prop_without_V` 1.695 ms at
2048², though that pair also differs by the `V` read).

**A polynomial sine and cosine, so the lossless kernels vectorize — deployed**
(2026-07-31). `np.sin`/`np.cos` lower to a call to `___sincos_stret` per
element, and an opaque call in the loop body stops LLVM vectorizing at all —
which is the whole reason the nonlinear kernels sat at 15% of bandwidth.
`_sincos` in `kernels/cpu.py` does Cody-Waite reduction into [-π/4, π/4] and
fdlibm's minimax polynomials instead, branchlessly.

Interleaved in one process, min-of-15, on the shipped kernel body:

| | complex64 1024²/2048² | complex128 1024²/2048² |
|---|---|---|
| libm (before) | 0.531 / 1.716 ms | 0.498 / 1.651 ms |
| polynomial | 0.399 / 1.188 ms — **1.33x / 1.44x** | 0.380 / 1.231 ms — **1.31x / 1.34x** |

End to end that is ~1.05–1.07x on a 2048² step here and within noise at 1024²,
the kernel holding only ~17% of a step on this machine.

**Confirmed on x86, and it is worth more there** (NVIDIA box, Linux, 2026-07-31,
`--baseline main --rounds 4`, the fixed guard). `1024/split_step` 9.600 → 7.850 ms,
**1.22x**, the only cell to clear its own 10.1% noise, and 0.79x on an earlier
independent run. The reason is that `numba.core.config.USING_SVML` is `False`
there too, so libm sincos is scalar on that box as well, while its FFT is
relatively much faster across many cores — for a ~1.4x kernel to give 1.22x
end to end, the nonlinear kernel must hold roughly three quarters of a step
rather than the sixth it holds here.

The RK4 cells are an unplanned control and they behave: 1.00–1.07x across both
runs, never clearing their floor. `split_step` goes through
`square_mod_nl_prop` and so through `_sincos`; RK4 goes through `rk4_nl_rhs*`,
which computes the right-hand side with no exponential at all and was not
touched. Noise does not sort itself by which kernel was edited. Accuracy: 1.5 ulp against numpy, flat from
|θ| < π/4 to |θ| < 5e8; against a libm twin with identical arithmetic, 3.5e-16
in complex128 and bit-identical in complex64. The complex128 work-precision
table is unchanged to six significant figures except the 6th digit of the two
finest steps, where the solver's own error is already within 5x of the
reference's self-consistency floor.

Two things this depends on, both easy to undo by accident. The kernels calling
`_sincos` carry a fast-math set that omits `reassoc`, because reassociation
rewrites `(x - k·P1) - k·P2` into `x - k·(P1+P2)` and collapses the split that
makes the reduction exact — with it on, the error grows with the argument
(4e-11 by |θ| = 1e6 instead of 3e-16). numba inlines `_sincos` into its caller,
so it is the *caller's* flags that reach LLVM. Isolating that flag change on
its own measured 0.97–1.02x, so it costs nothing; the win is all polynomial.

**The same treatment for the lossy kernels — deployed** (2026-07-31). `_sincos`
only reached the `alpha == 0` branch; with losses the exponent is complex and
numba calls libm, which is why `nl_prop` stayed at 15% of bandwidth against
`nl_prop_without_V`'s 28%. `_cexp` splits it as `exp(a+ib) = exp(a)(cos b + i sin b)`
— the split the OpenCL kernel has always used — over a new `_exp`, at all six
complex-exponential sites. Written through the complex argument rather than
against `alpha`, so it holds for a real or a complex potential.

| | complex64 1024²/2048² | complex128 1024²/2048² |
|---|---|---|
| libm (before) | 0.756 / 2.676 ms | 0.784 / 2.827 ms |
| polynomial | 0.534 / 1.730 ms — **1.42x / 1.55x** | 0.580 / 1.938 ms — **1.35x / 1.46x** |

1 ulp against numpy across the full double range; 5.1e-16 against a libm twin
with matched arithmetic and flags in complex128, bit-identical in complex64;
the work-precision table with losses on is unchanged to every printed digit.

Two traps worth keeping. `math.ldexp` is the obvious way to apply the `2**k`
and is *also a call*, which is the entire problem — with it the kernel measures
0.85x, slower than the libm it replaced, while writing the exponent field
directly gives 1.5x. And that write must happen in two halves: a single
`(k + 1023) << 52` overflows the field for large `|k|`, and the wrap turned
`exp(-1e4)` into `inf` instead of `0` — the opposite answer, which would spread
through the field as NaN. The argument is clamped to where `exp` saturates
anyway; clamping `k` instead would leave the polynomial an argument it is not
valid on.

**Why this stays on the CPU and is not shared C — considered, rejected.** The
tempting move is one C implementation of `_sincos`/`_exp` for all four
backends. It does not pay, and the reason is that the CPU's problem was never
the arithmetic:

- The **structure** is already shared. `cl_source/kernels.cl` splits the complex
  exponential into `exp` and `sincos` exactly as `_cexp` now does, and that C is
  already shared with CUDA through `kernels/templating.py`.
- On the **GPUs there is no call to break**. OpenCL, CUDA and Metal lower `exp`
  and `sincos` to hardware, and SIMT execution needs no auto-vectorization. A
  hand-rolled polynomial there only adds registers, and register pressure is
  the measured CL bottleneck. The ceiling is known: removing the CL exponential
  *entirely* gains ~1.3x, and `native_exp`/`native_sin`/`native_cos` gained 2%,
  inside the noise.
- **numba cannot consume C** cheaply. Reaching a C routine per element means
  ctypes/cffi — a call in the loop body, which is precisely what made the CPU
  kernels slow. The LLVM-level route (`@intrinsic`) takes llvmlite IR, not C.

So the duplication is one screenful of coefficients, and it buys each backend
the form its own compiler wants. Revisit only if a backend ever loses its
hardware transcendental.

**Grid-stride and blocked loops — rejected** (2026-07-31). The idea was that
`prange` over a flat range lets a thread start off a vector boundary and leaves
a scalar remainder per chunk, so an explicitly blocked loop would let every
thread start aligned. Measured on the memory-bound kernels, interleaved,
min-of-9, complex64:

| variant | 1024² | 2048² |
|---|---|---|
| `apply_propagator` flat (shipped) | 0.157 ms | 0.522 ms |
| rows (`prange` outer, serial inner) | 0.170 (0.93x) | 0.573 (0.91x) |
| blocked, 256–16384 elements | 0.183–0.193 (0.82–0.86x) | 0.567–0.578 (0.90–0.92x) |
| `rk4_axpy` flat (shipped) | 0.161 ms | 0.555 ms |
| blocked, 1024–16384 | 0.187–0.203 (0.79–0.86x) | 0.594–0.603 (0.92–0.94x) |

Every variant is slower. numba's default scheduling already does the right
thing and the block bookkeeping is pure overhead.

The premise is also wrong for the kernels that most need help. The nonlinear
kernels cannot be vectorized by any loop shape: the arm64 assembly shows an
indirect call to `___sincos_stret` per element (`blr` through the GOT — a
`bl _sin` search misses it). An opaque scalar call in the loop body blocks
vectorization regardless of alignment. Loop shape governs vector load/store
efficiency, which is exactly what these kernels are *not* limited by.

*Challenge this* if the nonlinear kernels ever stop calling libm — see the open
entry below — because then vectorization becomes possible and alignment starts
to matter. Note also that `backends/cpu.py` allocates with plain `np.zeros`
since pyFFTW's aligned allocator went away with `b69a42e`; restoring alignment
is one line if it ever matters.

**complex64 kernels compute in float32 — deployed** (2026-11-26). numba widens
a float32 that meets a Python float, a float64 argument or an int literal. The
solver passes `dz`, `alpha`, `g` and `Isat` as Python floats and the polynomial
constants were float64, so every nonlinear and RK4 right-hand-side kernel ran
its complex64 loop in double, on half as many lanes, and converted back at the
store. The scalars are now cast to the field's real type at kernel entry, the
shared constants are exact in float32, and a float32 angle or exponent gets its
own polynomial ([cephes `sinf.c`](https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/single/sinf.c#L108-L122),
[`expf.c`](https://github.com/jeremybarnes/cephes/blob/60f27df395b8322c2da22c83751a2366b82d50d1/single/expf.c#L70-L122)),
picked by `numba.extending.overload` on the type. complex128 keeps its code:
bit-identical per kernel and through nine solver runs.

`vec_bias.py 1024 40` (Mac) and `1024 30` (box), 1024², min over rounds × 5
calls, `main`'s kernels and the branch's interleaved in one process. Mac: M3
Max, 16 threads, numba 0.66.0. Box: Ryzen 3 4100, 8 threads, AVX2, numba 0.68.1.

| kernel | Mac, `main` → branch | box, `main` → branch |
|---|---|---|
| `_square_mod_nl_prop`, lossless | 0.423 → 0.298 ms (1.42x) | 0.942 → 0.585 ms (1.61x) |
| `_square_mod_nl_prop`, lossy | 0.694 → 0.496 (1.40x) | 1.536 → 0.696 (2.21x) |
| `_square_mod_nl_prop_v` | 0.713 → 0.496 (1.44x) | 1.599 → 0.735 (2.18x) |
| `_square_mod_rk4_nl_rhs` | 0.184 → 0.126 (1.45x) | 0.846 → 0.848 (1.00x) |
| `rk4_axpy` | 0.153 → 0.134 (1.14x) | 0.855 → 0.865 (0.99x) |
| `rk4_accumulate` | 0.132 → 0.118 (1.12x) | 0.850 → 0.865 (0.98x) |
| `_rabi_coupling` | 0.182 → 0.180 (1.01x) | 1.324 → 1.324 (1.00x) |
| complex128 `_square_mod_nl_prop` | 0.429 → 0.427 (1.01x) | 1.344 → 1.319 (1.02x) |
| complex128 `_square_mod_nl_prop_v` | 0.868 → 0.864 (1.00x) | 1.684 → 1.697 (0.99x) |

The RK4 kernels are traffic on the four-core box and do not move there. A whole
lie step (`step_cost.py 30`: both transforms, the propagator multiply and the
fused kernel) gains less, since the transforms are most of it:

| size, case | Mac, `main` → branch | box, `main` → branch |
|---|---|---|
| 512², lossless | 0.617 → 0.551 ms (1.12x) | 1.076 → 0.980 ms (1.10x) |
| 512², lossy with `V` | 0.642 → 0.588 (1.09x) | 1.591 → 1.404 (1.13x) |
| 1024², lossless | 1.688 → 1.572 (1.07x) | 8.655 → 6.833 (1.27x) |
| 1024², lossy with `V` | 2.056 → 1.860 (1.11x) | 9.478 → 7.159 (1.32x) |
| 2048², lossless | 6.277 → 5.815 (1.08x) | 35.970 → 35.015 (1.03x) |
| 2048², lossy with `V` | 7.377 → 6.755 (1.09x) | 38.702 → 35.095 (1.10x) |

On the box a step costs 3.7 ns a point at 512² and 8.3 at 2048²: past its
4 MiB of L3 the transforms are bound on memory, and they are nine tenths of the
step.

*Rounding.* In units of 2^-24 relative to the value (one rounding of a float32),
against the complex128 kernel on float32-representable inputs. `main` computed
in double and rounded once at the store, which is the best a complex64 kernel
can do: 1.0 at most, 0.42 rms.

- **Lossless rotation.** `a * (cos + i sin)` with a float32 cosine does not
  reach that: the cosine is rounded next to 1 and `|a|^2` shrinks by 0.04 a
  step on average, always the same way. `_rotate32` writes it as a sum,
  `a + a * ((cos r - 1) + i sin r)`, then multiplies by `i**k` with a swap and
  two signs. The increment is small, so is its rounding, and `a` is rounded
  once. Up to 0.1 rad a step, which is what the solver takes by default, the
  result has `main`'s rounding: 0.99 at most (1.03 on the box), 0.42 rms, and a
  mean change of `|a|^2` under 0.0001 (`vec_bias.py`, 2^20 points).
- **Phases near 1 rad.** 2.1 at most on the Mac and 2.9 on the box, where
  `main` gives 1.5; through the kernels that take a potential, up to 3.9 with a
  mean change of `|a|^2` of -0.01 to -0.03 a step (`vec_check.py`, 256²). The
  angle is reduced in float32, and that rounding is not in sum form.
- **Large phases.** The error grows as the phase times the relative error of
  the phase, on `main` too, where `|A|^2` was already a float32: 10.8 at 10 rad.
  The branch gives 12.7 on the Mac and 28.5 on the box. The difference is the
  saturation division: under `arcp`, LLVM divides float32 vectors on x86 with
  `vrcpps` and one Newton step (three `vrcpps` in the loop, no `vdivps`).
  Dropping `arcp` was not tried: at 0.1 rad it moves the maximum from 0.99 to
  1.03. Past 8192 rad the float32 reduction itself stops being exact, and past
  about 3e7 rad the result is no longer bounded, where `main`'s was: a float32
  phase is already worth nothing there, its spacing being 2 rad.
- **Decay.** cephes states 1.7e-7 for its `expf`, which is 2.9, and a decay
  multiplies the whole value. The lossy, frozen-loss and complex-potential
  branches reach 2.3 to 4.5 at most, 0.66 to 1.11 rms, with a mean change of
  `|a|^2` within 0.013 (lossy, `alpha dz` = 0.02) and of +0.03 to +0.09
  (frozen, `alpha dz` = 0.74). The RK4 right-hand sides reach 2.4 to 5.1 on
  `k`, which enters the field multiplied by the step. Not put in sum form:
  nothing is conserved there, and see the end-to-end table.
- **Rabi coupling stays in double**, on purpose. Rounded to float32, `cos` and
  `sin` leave the unit circle by up to 2^-24 and every step then moves the total
  power the same way, by up to 1. In double the mean change is 0.002, and the
  loop times the same as on `main` (table above).

End to end (`e2e.py`, then `e2e_cmp.py`: 512², default step, complex64 against
the complex128 run from the same float32-representable input), `main` and the
branch:

| run | steps | field error, rel. L2 | power drift per step |
|---|---|---|---|
| Kerr, lossless, lie | 176 | 4900 → 4852 | -2.77 → -2.75 |
| Kerr, lossless, strang | 176 | 4815 → 4880 | -2.75 → -2.78 |
| Kerr, lossless, yoshida | 176 | 14489 → 14401 | -8.28 → -8.25 |
| saturable, lossless, V | 96 | 666 → 678 | -2.38 → -2.43 |
| saturable, lossy | 95 | 669 → 676 | -2.45 → -2.49 |
| saturable, lossy, V, strang | 96 | 666 → 669 | -2.46 → -2.47 |
| Kerr, lossless, RK4 | 220 | 10.0 → 10.0 | 0.000 → 0.000 |
| coupled, lossless | 263 | 11180 → 11188 | -2.89 → -2.90 |
| coupled, Rabi, lossless | 263 | 11511 → 11542 | -2.93 → -2.93 |

The kernels are not what sets the complex64 error: the transforms are. One
complex64 `fft2` + `ifft2` round trip does not conserve power, by an amount
fixed by the size and the library (`fft_bias.py`, white noise, per round trip):

| size | scipy 1.18.0 | numpy 2.4.6 |
|---|---|---|
| 256² | -2.89 | -1.25 |
| 512² | -2.78 | -1.22 |
| 1024² | -2.82 | -1.29 |
| 2048² | -3.61 | -1.67 |
| 1000² | +2.04 | +4.06 |
| 1536² | -3.72 | -0.21 |

A split step pays it on the whole field every step (three times for yoshida),
and the lost power then turns into a phase error through the nonlinearity. RK4
transforms the field too, but adds the result as an increment times the step,
which is why its row reads 10 where the others read thousands. See *Open* for
the split step written the same way.

The regression test is `tests/backends/test_cpu_kernel_precision.py`: every
parallel loop of every kernel vectorized and free of double arithmetic for
complex64 with Python-float scalars, the rotation's rounding and modulus, and
the Rabi power. The scripts are in `~/Documents/LKB/NLSE_tangui_fix_specs/cpu_float32/`.

**Three-pass blocked split step — deployed, tuned per machine** (2026-11-27).
The unblocked step reads the field five times: row FFTs, column FFTs, the
propagator, the inverse, the nonlinear step. The blocked step
(`NLSE/kernels/cpu_blocked.py`) makes three passes over blocks that stay in
cache: forward row FFTs; column bands gathered, transformed, multiplied by the
y factor, transformed back and scattered times the x factor; inverse row FFTs
with the nonlinear step on each block. Python threads take static shares, and
everything inside runs without the GIL (`scipy.fft` with `workers=1`, `nogil`
numba kernels). It covers the 2D `NLSE` and `GPE` split step without a batch.

The best block is not a property of the code. On the Ryzen 3 4100 (lossless
Kerr, 8 rounds, before the tuner existed) the best block is 256 to 512 KiB, a
core's L2 (512 KiB): at 4096² complex64 16 rows give 1.46x, 8 rows 1.26x and
32 rows (1 MiB) 0.56x, and eight logical threads lose to four by 1.17 to
1.30x, a pair sharing one L2. On the M3 Max the best block is 1 MiB and 1024²
loses at every block. So the block (`BLOCK_BYTES`, 128 KiB to 2 MiB), the
threads (all, or half) and whether to block at all are measured per machine,
grid and dtype by `NLSE/backends/cpu_tuning.py` the first time a grid runs,
and kept in `cpu_blocking.json`.

`benchmarks/cpu_blocked_step.py`, NLSE lie with loss and saturation, 3 rounds
interleaved in one process, ms a step from the 10 to 60 step slope:

| machine | grid | dtype | chosen | unblocked | as tuned | gain |
|---|---|---|---|---|---|---|
| Ryzen 3 4100 | 1024² | complex64 | 64 rows x 4 | 8.158 | 7.158 | 1.14x |
| | 2048² | complex64 | 32 rows x 4 | 35.372 | 26.992 | 1.31x |
| | 4096² | complex64 | 16 rows x 4 | 145.085 | 109.630 | 1.32x |
| | 1024² | complex128 | 16 rows x 4 | 14.922 | 11.155 | 1.34x |
| | 2048² | complex128 | 16 rows x 4 | 60.808 | 45.344 | 1.34x |
| | 4096² | complex128 | 8 rows x 4 | 283.722 | 193.778 | 1.46x |
| M3 Max | 4096² | complex64 | 32 rows x 16 | 29.727 | 25.399 | 1.17x |
| | 4096² | complex128 | 32 rows x 16 | 50.917 | 40.770 | 1.25x |

Ryzen: 8 threads, numpy 2.5.5, scipy 1.18.1, numba 0.68.1, with the scratch
mapped apart (below; on malloc scratch 2048² complex64 read 34.597, 1.02x).
The rounds at 4096² complex64 spread by 4.4% there, and fresh processes read
106.3-106.5 ms (1.37x). M3 Max: 16
threads, numpy 2.4.6, scipy 1.18.0, numba 0.66.0, Spotlight holding one core;
1024² and 2048² stay unblocked there. `main` against the branch in alternating
fresh processes agrees, median ms a step: Ryzen 1024² 7.64-8.61 to 7.19-7.68
(complex64) and 14.07-16.05 to 11.93-12.25 (complex128), 2048² complex128
60.76-61.26 to 45.14-45.24, 4096² 145.1-145.6 to 106.6-107.6 and 286.3-287.4
to 197.1-197.2; M3 Max 4096² 30.4-39.1 to 25.4-25.8 and 52.3-52.5 to 40.6. The
gain grows with background load: with a second core of the Mac held by a
video player the same cells read 1.45x and 1.53x, and 2048² 1.10x and 1.27x.

After 20 steps (lie, strang, yoshida, 2048² and 4096²) the blocked field is
within 6e-6 of the peak of the unblocked one in complex64 and 1e-14 in
complex128, and the unblocked field equals the previous one bit for bit, on
both machines. The CuPy step, which shares the hook, did not move: medians
over 6, 6 and 12 alternating pairs on the RTX 3050, complex64, 0.387-0.391
against 0.385-0.393 ms at 1024², 1.565-1.583 against 1.560-1.576 at 2048²,
8.45-8.67 against 8.49-8.69 at 4096².

The nonlinear kernels became a serial span body and a `prange` driver over
4096-element spans, so the blocked pass and the unblocked kernel run one body:
bit-identical to the previous kernel. Both trees' `square_mod_nl_prop`
interleaved in one process on the M3 Max read 0.996-1.004, and with blocking
off the whole step on the Ryzen, `main` against the branch in four alternating
fresh processes each, takes 35.41-35.58 against 35.44-35.60 ms at 2048²
complex64 and 61.03-61.21 against 60.69-61.03 at complex128 (at 1024² the
processes spread by 5-9% and the two overlap). A
span indexed `range(start, stop)` did not vectorize (4.1 against 1.6 ns a
point); slicing first did.

*The tuner.* It ranks the blocked candidates by the best of 3 interleaved
rounds of 4 steps, then times the first against the unblocked step in 5
rounds, taking turns, and blocks only for a win over 10% (`MARGIN`). Ranking
about 10 candidates and reading the winner against 3 windows of the unblocked
step compared the best of 30 with the best of 3, and chose a losing layout on
a loaded Mac. A 5% margin was tried and withdrawn: at 1024² complex128 on the
M3 Max the tuner read 1.08x inside the benchmark process for a layout that
loses 10-18% in fresh processes (lossless, 2.55-2.57 against 3.01-3.04 ms).
The plain step's `fft2` at that size has two speeds in a long-lived process,
0.84-0.87 or 1.03-1.09 ms a transform: it flips after a tuning at another
size, only when interleaved with the numba kernels, and stays. The cause is
not established (thread placement over the P and E clusters is a guess). The
10% margin gives up a real 1.07x at 2048² complex128 on the Mac (11.66-11.80
to 10.92-11.04 ms). Repeated tunings on the Ryzen, eight a cell at 1024², six at
2048² and two at 4096², with the scratch mapped apart (below): 1.21-1.25x at
1024² complex64, 64 rows every time (on malloc scratch 0.97-1.29x, and one in
eight unblocked); 1.34-1.55x elsewhere, with 16 and 32 rows and 4 and 8
threads near ties at 2048² complex64. A tuning takes 0.4-0.9 s at 1024² and 12.1-12.5 s
at 4096² complex128 on the M3 Max; 1.2-2.4 s and 21-40 s on the Ryzen.

*The scratch is a mapping of its own — fixed* (2026-11-27). The benchmark's
own tuning on the Ryzen chose 32 rows at 2048² complex64 (512 KiB, the whole
L2) on a 1.39x reading; the solver's step with that block then ran at 1.02x in
the same process, and at 27.0 to 40.0 ms from one fresh process to the next,
against 35.3-35.8 unblocked. The block was not the cause. Each thread's
scratch came from `np.empty`, and the solver builds its step late, after
arrays of up to 32 MiB have been allocated and freed: glibc has then raised
its mmap threshold (`M_MMAP_THRESHOLD` in
[mallopt(3)](https://man7.org/linux/man-pages/man3/mallopt.3.html)) and
serves the scratch from the recycled heap, on pages in the order earlier
arrays left them, where the tuner had measured on fresh ones. The same layout
on scratch allocated before anything else, or mapped apart, runs at its tuned
speed every time. That it is the physical order of the pages (a block as
large as an 8-way L2 missing on conflicts in some orders) is inferred, not
read from a counter.

`_scratch` now cuts every thread's block from one fresh private anonymous
mapping. Alternating fresh processes on the cached 32 rows, median ms a step:
`main` 36.3, 27.1, 28.3, 27.4, 27.1, the fix 27.0-27.9 five times of five,
and 26.95-27.29 ten times of ten on its two earlier forms. The same bug
reached 4096² complex64 (16 rows, 512 KiB) once another grid had run in the
process: `main` 106.4-115.8 and once 145.8 ms, as slow as unblocked, the fix
106.6-108.2 in every run. Where the block is smaller than the L2 nothing
moves (2048² complex128 45.1-45.3 against 45.3-45.6, 4096² complex128
198.5-201.0 against 197.1-197.6, 1024² at 16 and 32 rows within 0.2 ms), and
those cells do not move on `main` after another grid ran either, so the
factors and the field stay where numpy puts them. Tuning times, the first call (2.10-2.11 s at 4096² complex64) and the
peak (2124 MiB) are unchanged, the fields are those of before (2.1e-6 to
2.7e-6 of the peak in complex64, 5e-15 in complex128, unblocked equal to
`main` bit for bit). On the M3 Max the 4096² step is the same (25.1-25.5 to
25.4-25.7 ms and 40.5-40.7 to 40.4-40.6), and so are the tuner's verdicts but
one: 2048² complex128 reads 1.07-1.11x against the 10% margin and blocked
once in two tunings (`main` 1.08-1.09x, never).
The mapping is private (`ACCESS_COPY`): mmap's default is shared, and a
forked child would write its parent's scratch.

*Huge pages for the scratch — rejected* (2026-11-27). The first form of the
fix asked for them (`MADV_HUGEPAGE`, from a 2 MiB boundary; kernel 7.3.4,
transparent huge pages on `madvise`). Against the same mapping on plain pages
no layout the tuner picks moved (2048² complex64 at 32 rows 26.96-27.29
against 26.95-27.26 ms, 4096² complex64 107.1-108.3 against 106.4-108.7), and
a 4096² complex64 tuning took 26.3-26.8 s instead of 21.4-22.0, with no
compaction stall: on huge pages a block larger than the L2 runs 2.3x slower
(4 threads: 32 rows 248-249 against 108-110 ms, 64 rows 288-304 against 134;
8 threads 143-147 against 121-129). Inferred: the column pass walks across
the block's rows at a power-of-two stride, and on physically contiguous
memory every row past what the L2's sets hold evicts another, where scattered
4 KiB pages spread them. The logs are in
`~/Documents/LKB/NLSE_tangui_fix_specs/cpu_blocked_step/ryzen/`.

*Open: the first step.* Where the blocked step is taken (and on CUPY) the
first step of a solver builds the separable factors: 4096² complex64, 1.86-2.03
to 2.09-2.12 s for the first call and 1736 to 2124 MiB peak on the Ryzen,
about 0.17 s and 530 MiB on the M3 Max. A declined call of the hook costs
1.8-3.9 us on the M3 Max and 2.4 us on the Ryzen.

*Open: thread counts.* On the M3 Max (12 P and 4 E cores) the numba kernels
are 21-34% faster on 12 threads than on 16 at every size and dtype (4096²
complex64: 2.36 against 3.60 ms), and the unblocked step on 12 against 16
threads takes 1.18 against 1.54 ms at 1024² complex64 and 2.12 against 2.40 at
complex128, while 2048² and 4096² complex64 prefer 16 (5.85 against 5.58,
27.8 against 26.0), because `fft2` alone does (11.4 against 9.6 ms at 4096²).
Kernel threads and transform workers want separate counts per machine and
grid, and the blocked candidates only try all and half the threads. The
scripts and the box log are in
`~/Documents/LKB/NLSE_tangui_fix_specs/cpu_blocked_step/`.

**The blocked candidates add the fastest cores' count — deployed.** On macOS
`sysctlbyname("hw.perflevel0.logicalcpu")` gives the logical CPUs of the
fastest cores, 12 of 16 on the M3 Max, and every block size is measured on
that count too, after all and half the threads
(`_thread_counts` in `cpu_tuning.py`; cache version 2). Per step through the
solver, ms, best of 7 rounds interleaved in one process, two runs (M3 Max, 16
threads, numpy 2.4.6, scipy 1.18.0, numba 0.66.0):

| grid | fcfba4a's step | ms | new step | ms | gain |
|---|---|---|---|---|---|
| 2048² complex128 | unblocked | 11.57, 11.49 | 32 rows × 12 | 9.77, 9.82 | 1.18x, 1.17x |
| 2048² complex128 | 64 rows × 16 | 10.99, 10.75 | 32 rows × 12 | 9.77, 9.82 | 1.13x, 1.09x |
| 4096² complex64 | 32 rows × 16 | 25.46, 23.68 | 32 rows × 12 | 23.45, 23.34 | 1.09x, 1.02x |
| 4096² complex128 | 32 rows × 16 | 39.49, 40.23 | 16 rows × 12 | 36.66, 35.06 | 1.08x, 1.15x |

At 2048² complex128 the gain is the count's: 32 rows on 16 threads took 11.12
and 10.77 ms, no faster than 64 on 16. 1024², and 2048² complex64, stay
unblocked on both; the same step timed twice there differed by up to 6%, the
runs' resolution. fcfba4a tuned 2048² complex128 unblocked twice (64 × 16
once on 402601a, the same tuner); the new tuner took 32 × 12 four times. Forced tuning, s,
two runs each, alternated:

| grid | complex64 | complex128 |
|---|---|---|
| 1024² | 0.5 → 0.6-1.0 | 0.7-0.8 → 1.0-1.7 |
| 2048² | 1.7 → 2.4-3.0 | 2.9-3.2 → 4.1-4.2 |
| 4096² | 6.8-7.9 → 9.1-9.3 | 12.0-13.0 → 17.0-18.1 |

A third run (2026-11-28, same script and machine, 7 rounds, while a macOS
indexing daemon held one core at 70-99%, which favours 12 threads): 2048²
complex128 10.70 (64 × 16) and 10.64 (32 × 16) → 9.80 ms, 1.09x; 4096²
complex64 24.99 → 24.31, 1.03x; 4096² complex128 39.25 (32 × 16) and 40.34
(16 × 16) → 37.26, 1.05x and 1.08x; 2048² complex64 unblocked on both, 6.45
and 6.44. The rounds spread 6 to 10%: the complex128 rows are a difference,
the complex64 one is not.

Scripts: `cpu_pcores_ab.py --side 2048:complex128:64:16
2048:complex128:32:16 4096:complex64:32:16 4096:complex128:32:16
4096:complex128:16:16` and `cpu_pcores_tuning_time.py`, in
`~/Documents/LKB/NLSE_tangui_fix_specs/`. Linux hybrids are not read: they
list their fast cores in `/sys/devices/cpu_core/cpus`, unmeasured here.

*Open: the unblocked step's thread count.* It stays on all 16 threads
(`workers=-1`, numba's default). A tight loop of it is 1.37x faster on 12 at
1024² complex64 (1.56 against 1.14 ms, `cpu_pcores_probe_switch.py`), but
through the solver it is 1.06x (1.82 against 1.71 ms, on the arrays the
solver hands it, `cpu_pcores_probe_arrays.py`), and the cause is not
established.

**Padded column-pass scratch rows — tuned per machine; tiled gather and
scatter — rejected in numba, measured in C** (2026-11-28).
The column pass moves 32 B per point at complex64 (gather, then scatter with
the factor). It ran at 37-42% of the Ryzen 3 4100's measured copy bandwidth
(32 GB/s) and at 29% of the M3 Max's (53 of 181 GB/s at 4096², loaded). Move
only, best of 9 rounds interleaved in one process, ms (`colmove.py`; Ryzen: 8
threads, numpy 2.5.5, scipy 1.18.1, numba 0.68.1, load 1.1-2.2; M3 Max: 16
threads, numpy 2.4.6, scipy 1.18.0, numba 0.66.0, load 10.4, to re-run quiet):

| machine, grid, layout | today | pad 128 B | numba tiles of 16 | C, today's order | C, tiles of 16 |
|---|---|---|---|---|---|
| Ryzen 1024², 32 × 8 | 2.83 | 2.12 | 1.94 (bits differ) | 2.27 | 1.96 |
| Ryzen 2048², 16 × 8 | 10.29 | 10.60 | 8.60 (bits differ) | 9.08 | 8.51 |
| Ryzen 4096², 16 × 8 | 40.47 | 40.99 | 33.76 (bits differ) | 36.18 | 33.59 |
| M3 Max 2048², 32 × 12 | 3.12 | 2.31 | 2.80 (bits differ) | 3.04 | 2.54 |
| M3 Max 4096², 32 × 12 | 10.10 | 6.42 | 8.66 (bits differ) | 9.61 | 8.22 |

*Pad sweep*, move only, best of 9 rounds, ms by pad in B (`colmove.py`;
Ryzen load 4.8-5.6; M3 Max load 10.8-11.0, to re-run quiet):

| machine | grid | layout | 0 | 64 | 128 | 256 | 384 | 512 | load |
|---|---|---|---|---|---|---|---|---|---|
| Ryzen | 1024² complex64 | 32x8 | 2.71 | 2.02 | 2.05 | 2.06 | 2.06 | 2.03 | 4.83-4.83 |
| Ryzen | 1024² complex64 | 16x4 | 2.88 | 2.79 | 2.83 | 2.82 | 2.81 | 2.83 | 4.83-4.83 |
| Ryzen | 2048² complex64 | 32x8 | 11.71 | 8.53 | 8.39 | 8.59 | 8.47 | 8.63 | 4.83-5.08 |
| Ryzen | 2048² complex64 | 16x4 | 11.09 | 10.82 | 10.88 | 10.80 | 10.87 | 10.90 | 4.83-5.08 |
| Ryzen | 4096² complex64 | 32x8 | 52.99 | 42.33 | 42.10 | 42.31 | 40.78 | 44.11 | 5.08-5.48 |
| Ryzen | 4096² complex64 | 16x4 | 43.93 | 43.70 | 43.29 | 43.38 | 42.84 | 43.34 | 5.08-5.48 |
| Ryzen | 1024² complex128 | 32x8 | 4.66 | 4.55 | 4.55 | 4.63 | 4.59 | 4.68 | 5.50-5.50 |
| Ryzen | 1024² complex128 | 16x4 | 3.90 | 3.90 | 3.98 | 4.00 | 3.96 | 3.94 | 5.50-5.50 |
| Ryzen | 2048² complex128 | 32x8 | 20.67 | 19.70 | 19.72 | 19.97 | 19.73 | 20.64 | 5.14-5.50 |
| Ryzen | 2048² complex128 | 16x4 | 15.07 | 15.29 | 15.14 | 15.47 | 15.25 | 15.30 | 5.14-5.50 |
| Ryzen | 4096² complex128 | 32x8 | 84.17 | 79.76 | 79.81 | 80.54 | 79.95 | 83.89 | 4.81-5.59 |
| Ryzen | 4096² complex128 | 16x4 | 62.60 | 63.37 | 62.77 | 62.27 | 61.18 | 61.64 | 4.81-5.59 |
| M3 Max | 1024² complex64 | 32x12 | 1.10 | 1.42 | 1.52 | 1.49 | 1.48 | 1.43 | 10.95-10.95 |
| M3 Max | 2048² complex64 | 32x12 | 2.97 | 2.18 | 2.08 | 2.18 | 1.98 | 2.23 | 10.95-10.95 |
| M3 Max | 4096² complex64 | 32x12 | 9.16 | 5.64 | 5.77 | 5.99 | 5.83 | 5.70 | 10.95-10.95 |
| M3 Max | 1024² complex128 | 32x12 | 1.09 | 0.94 | 0.86 | 0.74 | 0.84 | 0.76 | 10.84-10.84 |
| M3 Max | 2048² complex128 | 32x12 | 3.07 | 2.28 | 2.36 | 2.45 | 2.38 | 2.42 | 10.84-10.84 |
| M3 Max | 4096² complex128 | 32x12 | 9.97 | 8.83 | 8.78 | 8.49 | 8.98 | 9.01 | 10.84-10.84 |

Pads of 64-384 B moved the pass within 10% of each other at 2048² and 4096²
(27% apart at the M3 Max's 1024² complex128), so the pad is one constant,
`PAD_BYTES = 128`. Whether to pad is not: on the Ryzen it helps the
8-thread layouts (32 × 8) more than the 4-thread ones its tuner takes, and
at 1024² complex64 it cost the M3 Max 38% of the move.

*Hypotheses, not verified* (no hardware counters read): on the M3 Max, at a
power-of-two NY, every row of a band starts on the same cache sets, which the
pad breaks; on the Ryzen the cost is the field side, 128 B read per field row
at a stride of NX. A band-major field layout reached 23.6 GB/s there (74%),
but its two extra traversals made the whole step 11-30% slower on both
machines.

*Bits.* In today's loop shape numba's scatter rounds the complex product as
`fma(a, c, -(b d))`, `fma(a, d, b c)` (arm64 asm); tiled over rows, its bits
move. C with explicit `fmaf` is bit-identical in both orders. The padded and
unpadded steps are main's (97ee1c7) to the bit: 288 cases each on each machine
(square, non-square and non-power-of-two grids, both widths, 9 forced
layouts; `vs_main.py`).

*The tuner's choice.* The tuner times its best block with and without the
pad, side by side (`DUEL_ROUNDS`), keeps the pad if it wins by `MARGIN`, then
blocks only if that step beats the unblocked one by `MARGIN`. Kept per entry
(`pad_bytes`). After one forced tuning, one in-process A/B of the three steps,
best of 15 interleaved rounds, ms a step (`pad_ab.py`, same tool on both
machines; where the tuner ran unblocked, the block its ranking takes first in
that process; M3 Max on 7465343, load 5.8 before the run, no other job;
Ryzen load 5.6-5.9, its own tuning's threads):

| machine | grid | block | tuner kept | unblocked | blocked | blocked padded |
|---|---|---|---|---|---|---|
| M3 Max | 1024² complex64 | 64 × 8 | unblocked | 1.44 | 1.96 | 1.52 |
| M3 Max | 2048² complex64 | 64 × 12 | blocked padded | 6.24 | 6.12 | 5.16 |
| M3 Max | 4096² complex64 | 32 × 12 | blocked padded | 26.59 | 21.72 | 18.95 |
| M3 Max | 2048² complex128 | 32 × 12 | blocked | 10.58 | 8.65 | 8.00 |
| M3 Max | 4096² complex128 | 16 × 12 | blocked | 46.03 | 32.58 | 32.87 |
| Ryzen | 1024² complex64 | 64 × 4 | blocked | 8.11 | 6.66 | 6.17 |
| Ryzen | 2048² complex64 | 32 × 4 | blocked | 34.42 | 25.50 | 24.21 |
| Ryzen | 4096² complex64 | 16 × 4 | blocked | 144.97 | 97.62 | 97.92 |
| Ryzen | 2048² complex128 | 16 × 4 | blocked | 58.73 | 41.14 | 41.19 |
| Ryzen | 4096² complex128 | 8 × 4 | blocked | 278.26 | 178.40 | 178.98 |

The choice moves with load: under another job (M3 Max, load 14.2-15.7) the
same tool left 2048² and 4096² complex64 unblocked, its A/B there reading the
padded step 1.05x and 1.18x the unblocked one (6.47 and 6.18 ms, 25.46 and
21.65 ms). A padded step under 10% ahead of the unpadded one is also dropped
where the unpadded one is under 10% ahead of the unblocked step. Forced
tuning, main and this tree alternated twice: 4096² complex128, 17.3-17.8 s
and 18.5-19.0 s (M3 Max, 7465343), 40.4-43.0 s and 47.7-47.9 s (Ryzen,
97ee1c7); 2048² complex64 on the M3 Max, 2.5-3.8 s and 2.7 s, the pad's duel
there being 41 steps of 5-6 ms. Cache version 3. *Open:* gathering in tiles of 16
field rows is bit-safe (no arithmetic) and took the M3 Max's 4096² step a
further 1 ms; unmeasured on the Ryzen. Scripts and results:
`~/Documents/LKB/NLSE_tangui_fix_specs/cpu_tiled_gather/`.

## OpenCL (PyOpenCL + VkFFT)

**Native OpenCL C kernels instead of PyOpenCL array expressions — deployed.**
Array expressions launch an implicit kernel per operation and allocate
temporaries. `rabi_coupling` became one launch instead of six plus a buffer
allocation; `apply_propagator` one launch instead of an implicit `A *= prop`
with a temporary. Both now run at 100% and 77% of best observed bandwidth.

**A program cache — deployed.** Keyed by `(context_hash, precision)`.

**`-cl-fast-relaxed-math` — rejected on correctness; `-cl-mad-enable` alone.**
It implies `-cl-unsafe-math-optimizations`, which permits reassociation, and a
V-reading kernel and its generated no-V twin are not the same expression to
reassociate: `g * A_sq * sat + V` against `g * A_sq * sat`. POCL ordered the
multiplies differently in the two, so a potential of *exactly zero* moved the
result by about an ulp. Over 20 steps that accumulates coherently into the
nonlinear phase and reached 8.3e-7, eight times the tolerance in
`tests/integration/test_zero_potential.py` — a red CI for months. Apple and
NVIDIA compiled both twins alike and stayed bit-identical, which is why
nothing local reproduced it.

Measured on POCL 7.1, launching the two twins on one field: 1256 of 4096
elements differ under `-cl-unsafe-math-optimizations` or
`-cl-fast-relaxed-math`; none differ under `-cl-mad-enable`, under
`-cl-finite-math-only`, or under no flags. Contraction is not the problem —
`mad` is one rounding of the *same* expression.

`-cl-finite-math-only` went too: it promises no infinity reaches the kernels,
and `Isat` defaults to `np.inf`.

Free on Apple within a noisy machine's resolution — 1.05x, 0.99x, 0.86x, 0.98x
on 1024 split_step single and double, 2048 split_step and 1024 RK4, against
per-cell noise of 23%, 12%, 25%, 7%. *Challenge this* on a quiet machine, but
the correctness argument stands whatever the timing says: a relaxed flag can
only come back with something that keeps the twins on the same bits.

**Reproducing a POCL-only failure.** `conda create -n poclrepro -c conda-forge
pocl pyopencl pyvkfft matplotlib numba scipy pytest`, then run with
`PYTHONPATH` set to the repo. POCL is the only platform visible inside that
env, so the CI backend runs locally without disturbing the Apple one. Two
hypotheses were refuted from CI alone before this was tried, a full round trip
each. Note arm64 POCL shows the ulp divergence but not the test failure, and
fails `test_double_precision` for unrelated reasons.

**Strided work-items — rejected** (branch `perf/cl-strided-kernels`,
`406ed45`, `b9b9a36`, unmerged). CL is at parity with MLX on bandwidth and on
`apply_propagator`, and faster on the FFT; the one deficit is the nonlinear
kernels, 2.0–2.4x slower at every size, because the transcendentals raise
register use, occupancy falls, and memory latency stops being hidden. Giving
each work-item four consecutive elements took a probe kernel from 80 to
228 GB/s — the rate `apply_propagator` already reaches.

It fails because the optimum is per *kernel* and per *size* together. Per
kernel at fixed size the measurement does not reproduce (three trials at 1024²
agreed on one kernel of ten). Per size, timing a whole step, it reproduces
cleanly — 1 at 256², 2 at 512², 4 at 1024² and 2048² — but that answer is
measured on the split-step kernels and mis-serves the RK4 accumulate family,
which striding slows badly (`rk4_set_and_axpy` −30% at stride 4,
`rk4_acc_and_axpy` −67% at 8). End to end: 0.88x at 1024 split_step but
**1.19x slower** at 2048 split_step and 1.12x slower at 2048 RK4.

*Challenge this* with a per-kernel *and* per-size calibration table rather than
one constant — the branch has the calibration harness. It was judged not worth
the complexity for a win that only appears at one size.

**`native_exp` / `native_sin` / `native_cos` — rejected.** 2%, inside the
tool's spread. Apple's compiler already lowers `exp`/`sincos` under
`-cl-fast-relaxed-math`.

**A lossless `alpha == 0` path for CL — rejected.** Removing the exponential
entirely gains only ~1.3x on this backend, so the branch the CPU kernels carry
is not worth adding here.

**Whole-stage RK4 kernels — deployed** (2026-11-28). RTX 3050 8 GB, driver
620.70.06, pyopencl 2026.1.5. An RK4 stage was a transform pair, then
`square_mod_rk4_nl_rhs` writing the slope, then `rk4_acc_and_axpy` reading it
back to update the accumulator and form the next argument. The CUDA backend
fused the last two (`has_fused_rk4_stage`); OpenCL now has the same kernels,
`square_mod_rk4_stage` and `coupled_rk4_stage_c`, the slope finished in
registers and never written. The slope and the stage update are inline
functions (`rk4_nl_slope`, `rk4_stage_carry`, `rk4_axpy_element`) that the
per-pass kernels call too, so each sum is written once. The stage's mode is
an `Rk4StageMode` rendered as `#define RK4_*` ahead of both backends'
sources.

Per step, complex64, lossless, no potential, `step_for` step. One process,
before and after side by side, the order alternating over 8 rounds; slope
between 10 and 60 steps, median, [min, max] of the per-round gain
(`gpu_p4_r2_ab.py`):

| solver | grid | before | after | gain [range] |
|---|---|---|---|---|
| `NLSE` | 1024² | 3.344 ms | 2.895 ms | **1.16x** [1.14, 1.16] |
| `NLSE` | 2048² | 17.20 ms | 15.42 ms | **1.12x** [1.11, 1.12] |
| `NLSE` | 4096² | 89.85 ms | 83.03 ms | **1.08x** [1.08, 1.09] |
| `CNLSE` | 1024² | 6.626 ms | 5.755 ms | **1.15x** [1.15, 1.15] |
| `CNLSE` | 2048² | 32.70 ms | 29.34 ms | **1.11x** [1.09, 1.12] |
| `CNLSE` | 4096² | 178.2 ms | 165.4 ms | **1.09x** [1.05, 1.13] |
| `CNLSE`, `omega` = 1e4 | 1024² | 20.29 ms | 5.755 ms | **3.53x** [3.52, 3.53] |
| `CNLSE`, `omega` = 1e4 | 2048² | 80.51 ms | 29.24 ms | **2.75x** [2.74, 2.77] |

Without `omega` the results are identical to the bit after 60 steps (these
runs: complex64, lossless, no potential). With it, RK4 on OpenCL took the
generic path; it is fused here for the first time, through the coupling's
slope in `coupled_rk4_stage_c`, and moves by 9e-7 of the peak.
The gain falls with the grid as the transforms, which this does not touch,
take more of the step. Not measured: complex128, other OpenCL devices.

**The propagator as two vectors — rejected** (2026-11-28). The CPU's blocked
step and CUPY's forward callback read the propagator as one factor per axis;
`apply_propagator` here reads the grid. A kernel multiplying by
`rows[iy] * columns[ix]` instead, put into the solver's Lie step (`NLSE`,
complex64, lossless, no potential, `step_for` step; one process, the two sides
alternating over 8 rounds, slope between 10 and 60 steps; medians, and the
median and [min, max] of the per-round gain; `sep_prop_cl.py` in
`~/Documents/LKB/NLSE_tangui_fix_specs/`):

| device | grid | grid | two vectors | gain [range] |
|---|---|---|---|---|
| RTX 3050 | 1024² | 0.593 ms | 0.554 ms | 1.03x [0.87, 1.08] |
| RTX 3050 | 2048² | 3.333 ms | 3.183 ms | 1.05x [0.89, 1.05] |
| RTX 3050 | 4096² | 18.67 ms | 18.09 ms | 1.03x [0.92, 1.04] |
| Apple M3 Max | 1024² | 0.757 ms | 0.884 ms | 0.85x [0.63, 0.88] |
| Apple M3 Max | 2048² | 1.561 ms | 1.551 ms | 1.02x [0.94, 1.30] |
| Apple M3 Max | 4096² | 6.691 ms | 6.449 ms | 1.03x [1.02, 1.07] |

The multiply alone is 1.47 to 1.48 times faster on the RTX 3050, and the step
does not follow: one row in six gains more than its rounds spread, by 3%, and
1024² loses on Apple. Not built. RTX 3050 8 GB, driver 620.70.06, pyopencl
2026.1.5; M3 Max, macOS 27.2.1, Apple's OpenCL 1.2, pyopencl 2026.1.2;
pyvkfft 2025.1.1 on both.

## MLX (Metal)

**A lossless graph for a lossless run — deployed** (2026-08-01). The solved
lossy step made a *lossless* MLX split step **1.45x slower** — 0.104 → 0.153
ms/step at 512², 0.043 → 0.063 at 256², against 4-8% noise, while CPU and CL
were unmoved at 1.00-1.07x. Nothing was wrong with the answer: at `u = 0` the
iteration returns `sat` and the decay is exactly 1. What it cost was that
`_nl_factor` chooses between the solved and the frozen step with `mx.where`,
**and MLX evaluates both arms** — so every lossless step paid three
fixed-point iterations, a `sqrt`, and the frozen arm's `exp`. The other
backends branch at runtime in C and skip all of it, which is why only this one
regressed and why the NVIDIA-box guard never saw it.

`alpha` is read on the host, so the factor can be chosen when the graph is
compiled instead of per element. `_make_split_step` takes the loss as a
compile key and `_is_lossless` decides it; the lossless graph is bit-identical
to the general one, which `tests/backends/test_mlx.py` pins on all four
splitting × potential cases. Back to **0.93-0.96x**, 0 of 6 cells slower.

A device scalar is never called lossless — reading one costs a
synchronization, and the general kernel is correct at any `alpha`.

**Choosing the lossy arm on the host too — rejected, measured** (2026-08-01).
`u = 2*alpha*dz` is a host scalar whenever `alpha` is, so the solved arm could
be emitted alone, dropping two selects and the frozen arm's exponential. It
measured **1.54x against 1.51x** on 5-7% noise: nothing. What the solved step
costs MLX is the iteration itself, not the arm it carries alongside it.

**The solved step as a Metal kernel — deployed** (2026-08-01), which is what
that rejection said to try next, and it was right. `mx.compile` does **not**
fuse the iteration: at 512², chained so the queue pipelines, the nonlinear
kernel costs 10.3 µs lossless, 19.2 µs frozen and **66.8 µs** solved. Cost
tracks the *number of elementwise ops* rather than the flops in them — one
pass already costs 2.25x and each further pass adds ~45 µs, which is what ~9
materialized 1 MB intermediates cost at this machine's bandwidth. Horner form
and folding the leading multiply recover 11% and 20% of it, so the arithmetic
is not free either, but neither is the fix.

`mx.fast.metal_kernel` is. The same iteration in registers, with `sincos` as
one instruction instead of two passes over memory, is **25.1 µs** — and a
whole split step goes from 1.62x lossless to **1.18x**. End to end against the
graph version, `--alpha 20`: **0.73x at 512² and 0.68x at 256²**, with the
lossless guard unmoved (0.92-1.07x, nothing flagged). So the lossy penalty on
MLX is now within noise of a lossless step, where it was 1.5x.

Three things bound it, all decided on the host, all falling back to the graph:
a batched or device `alpha`, an absorbing potential (the kernel writes a real
phase), and `|u|` outside the iteration's range, where the frozen arm applies
and only the graph has one. `_loss_mode` is where that is decided and the
compile cache is keyed on it.

It is a sixth copy of one formula, and the test that keeps the other five
together skips MLX for want of fp64, so `tests/backends/test_mlx.py` scores it
against the MLX graph in float32 instead.

**What the solved lossy step costs on the M3 Max** (2026-08-01), the question
left open below. Against `main` at `alpha = 20`, `--alpha 20` on the guard:

| backend | 256² | 512² |
|---|---|---|
| CPU | 1.02x | 0.98-1.09x |
| CL | 1.00x | 0.96-1.03x |
| MLX | **1.54x** | **1.51x** |

CPU and CL are inside their own noise — the ~16% the NVIDIA box measured for
Lie does not appear here. MLX pays much more than any of them, for the reason
in the rejection above. Against 69x and 7000x in accuracy it is still not a
trade, but it is the one backend where the choice is worth making
deliberately.

The NVIDIA box has since answered the same question for CUPY, and the answer
is the same as CL's: nothing measurable, 0.106 ms at 512² with and without
loss, on 0.7% scatter. So MLX is not merely the most expensive backend for the
solved step — it is the only one that pays at all. See *Solving the lossy
real-space step* for the four backends side by side.

**Rabi scalars computed on the host — deployed.** `cos_val =
_to_mx(float(mx.cos(...)))` evaluated a single number on the GPU and dragged it
back, stalling the queue twice per call. `math.cos` instead: **1.60x** on
`rabi_coupling` at 2048² (1.200 → 0.752 ms, 112 → 178 GB/s) against 12.9%
noise, at three sites including both fused coupled steps. Only reaches CNLSE
and DDGPE with `omega` set, so the NLSE guard shows nothing — 0 of 18 cells
moved.

**One traced closure per physics case, not per signature — deployed**
(`1ba3adf`). `_make_split_step_coupled` wrote six full bodies for
potential/no-potential × Rabi/no-Rabi × single/double. Now three shared
closures; six signatures are kept because `mx.compile` traces the arguments it
is handed, so an absent potential cannot be `None`.

**pyvkfft for the MLX transform — rejected** (2026-07-31). pyvkfft 2025.1.1
ships `cuda` and `opencl` backends only; there is no Metal backend.

The workaround does run, and is numerically correct. MLX→numpy is zero-copy
(`np.asarray(mlx_array)` shares memory; the reverse copies), so an MLX array's
memory can be aliased into an OpenCL buffer with `CL_MEM_USE_HOST_PTR` and
VkFFT run on it **in place**, no copies. Wrapping costs 0.6 µs. Verified
against numpy and against `mx.fft`, roundtrip exact, MLX keeps operating on the
array afterwards. In isolation it wins: 2048² VkFFT-on-MLX-memory 1.09 ms vs
native CL buffer 1.15 ms vs `mx.fft` 1.35 ms — the host pointer costs nothing.

It still loses end to end, because Metal and OpenCL share no timeline: every
transform needs `mx.eval` before and `queue.finish` after, twice per step. Over
a 20-step split-step loop:

| grid | all-MLX (lazy) | all-MLX, eval per step | VkFFT on MLX memory |
|---|---|---|---|
| 512² | 0.211 ms/step | 0.390 | 1.043 — **0.20x** |
| 2048² | 2.666 ms/step | 2.899 | 3.846 — **0.69x** |

The barrier *call* is 0.4 µs; what it costs is MLX's overlap. VkFFT saves
~0.27 ms on the transform at 2048² and gives back ~1.2 ms in lost pipelining.
It loses even against a baseline that already syncs every step, so no usage
pattern rescues it. The solver's no-callback path syncs once after the whole
loop (`solvers/nlse.py`), which is the lazy column — MLX's best case and
interop's worst.

*Challenge this* if pyvkfft gains a Metal or Vulkan backend, which would remove
the cross-API barrier entirely, or if MLX exposes an interop primitive that
lets another API's queue wait on its own without a CPU round trip.

**Alternative spellings of the MLX transform — rejected.** Successive 1D
transforms are slower than `fftn` at 512/1024/2048 (e.g. 1.396 vs 1.329 ms at
2048²). Non-power-of-2 costs ~1.3x (1020² 0.361 ms vs 1024² 0.282 ms), which is
the usual advice to keep grid sizes at low prime factors, not a lever.

**`np.dtype()` cannot read an MLX dtype, and raises rather than comparing.**
MLX names its dtypes in its own namespace, so `np.dtype(array.dtype)` on an
`mx.array` is `TypeError: Cannot interpret 'mlx.core.float32' as a data type`.
Any width check written the obvious way is therefore a landmine that only goes
off on a Mac: one line of it in `_scale_potential` broke **every MLX run that
had a potential** — 49 failing tests, and none of them reachable from a machine
without MLX. Compare through `_as_host_array` where a real comparison is
needed, and let the device path fall through where it is not.

The same asymmetry bites a test rather than the code: MLX is single precision
throughout and Apple's OpenCL ships no fp64, so any check that scores a backend
against a double-precision reference has to skip both, or it scores them against
a reference of another width. `supports_double_precision()` is the question to
ask, and both answer no.

**The propagator as two vectors — deployed** (2026-11-28). The fused MLX step
multiplies by one vector per axis, `A * rows[:, None] * columns`, wherever the
dispersion is a sum of one term per axis (`_dispersion_axes`). That covers
every graph `split_step_fused` compiles: Lie and Strang, each with no
potential, a real one or a complex one; each loss mode; the remainder's and
Yoshida's propagators; GPE; and a batch, which broadcasts one grid's rows. The
vectors come through the CUDA callback's route, `_propagator_factors` and
`_separable_propagator`, without the 1/N that MLX's inverse transform applies
itself. `_built_step` vouches for them: a propagator the solver did not build,
or a dispersion with a cross term, keeps the grid, bit-identical to before.

The measurement ran through the solver. The base commit's `NLSE.split_step` was
bound against the new one on the same solver in one process
(`ab_two_vectors.py`: 8 rounds, the two sides alternating, the per-step cost
taken as the slope between 10 and 60 steps). Field and step were complex64 and
the `benchmarks/profile_backends.py` beam and step, on an M3 Max with macOS
27.2.1 and mlx 0.32.0, with no other process above 20% CPU. Per-step time in
ms, grid → two vectors, median gain [range]:

| case | 1024² | 2048² | 4096² |
|---|---|---|---|
| Lie, lossless | 0.294 → 0.284, 1.04x [0.92, 1.21] | 1.743 → 1.659, 1.05x [1.02, 1.11] | 6.969 → 6.641, 1.05x [1.02, 1.12] |
| Lie, real V | 0.358 → 0.338, 1.07x [0.96, 1.20] | 2.047 → 1.958, 1.05x [1.01, 1.08] | 8.332 → 7.966, 1.05x [1.02, 1.10] |
| Strang (halves merged) | 0.297 → 0.283, 1.05x [1.01, 1.16] | 1.754 → 1.645, 1.07x [1.04, 1.12] | 7.063 → 6.631, 1.06x [1.00, 1.13] |
| Strang, complex V | 0.496 → 0.465, 1.06x [1.01, 1.12] | 2.733 → 2.646, 1.03x [1.00, 1.08] | 11.490 → 11.107, 1.04x [1.01, 1.07] |
| Lie, alpha = 20/m (solved kernel) | 0.305 → 0.294, 1.02x [0.97, 1.24] | 1.775 → 1.704, 1.05x [1.00, 1.08] | 7.053 → 6.687, 1.05x [0.99, 1.09] |
| Yoshida | 0.987 → 0.931, 1.05x [1.02, 1.12] | 5.381 → 5.093, 1.05x [1.01, 1.08] | 24.085 → 23.100, 1.04x [1.03, 1.06] |
| GPE | 0.275 → 0.265, 1.05x [0.95, 1.12] | 1.744 → 1.623, 1.08x [0.98, 1.11] | 7.059 → 6.712, 1.07x [1.00, 1.13] |
| batch of 4 | 1.818 → 1.807, 1.00x [0.96, 1.04] | 7.564 → 7.492, 1.01x [0.97, 1.05] | 30.869 → 30.444, 1.01x [0.88, 1.03] |

Two earlier passes, taken with one or two other processes at 100% CPU, agree:
every median at 1.00x or above but one batch run at 4096² (0.98x [0.89, 1.07];
its repeat gave 1.01x). The batch barely gains, since its grid is read once for
four fields. It is kept for the memory, and so that no shape picks the path.
Below the range it loses: 0.93x at 256², a few µs of dispatch a step, and
1.00x at 512². The two answers differ by 2.4e-6 to 4.7e-6 of the peak after
60 steps, and by 6.8e-6 to 9.2e-6 for Yoshida, which applies three propagators
a step. The lookup costs 1.5 µs a step. The vectors' host build runs once per
configuration and takes 6, 30 and 127 ms at 1024², 2048² and 4096².

Device memory at 4096² (`mx.get_active_memory`, after a 10-step run, with the
result dropped): what a run holds falls from 320 to 192 MiB, the propagator's
128 MiB, which is never evaluated now. With Yoshida's three it falls from 448
to 192 MiB. The peak over the run falls from 6208 to 6080 MiB. Reading
`simu.propagator` still builds the grid, back to 320 MiB. The device dispersion
operator, the other 128 MiB, stays: that grid is built from it.

Left out:
- The coupled graphs (CNLSE, DDGPE): the operator is one grid per component,
  which `_dispersion_axes` declines.
- RK4: its operator is a sum, not a product.
- NLSE_1d and NLSE_3d (`FUSED_TRANSFORM_AXES`).
- The nonlocal path's `linear_step`: not measured, and its signature is shared
  by all four backends.

`tests/backends/test_mlx.py::TestMLXPropagatorFactors` pins four things: the
agreement with the grid, on white noise at 0.9 rad a step on the shortest wave
(0.24 of the round-off bound at most); the fallback; that a zeroed grid is not
read; and that a propagator the solver did not build is the one applied.

*Challenge this* if the largest grids need the memory more than a read of
`simu.propagator` needs speed. The device dispersion operator could then be
dropped once the vectors stand, and the grid built from the host's copy when
asked for.

## CUDA (CuPy)

Not measured on the Mac this file is otherwise written from — CuPy is not
installed there. The backend uses raw cuFFT plans and hand-written CUDA C
kernels (`kernels/cuda_source/kernels.cu`), not the VkFFT plans and
`cupy.fuse` kernels this section used to describe, and captures a step into a
CUDA graph. Entries below say which machine they were run on.

**Deployed: the propagator applied from a cuFFT store callback.** RTX 3050,
CUDA 13.3, CuPy 14.1.1. A split step touched the field four times — the
transform pair, the propagator multiply, the nonlinear step — and cuFFT will
run a store callback as it writes each element of a transform's output. The
multiply moves there: the pass that read the field, read the propagator and
wrote the field back becomes one extra read inside a write that was already
happening. Of the ~104 bytes per grid point a step moved, 24 were that pass and
16 of them are gone.

`benchmarks/propagator_fusion.py`, per-step cost from the 20→220 step slope,
sides interleaved, noise measured as how far a side moved between rounds:

| grid | unfused | fused | gain | noise |
|---|---|---|---|---|
| 256² | 0.020 ms | 0.018 ms | 1.12x | 15.3% |
| 512² | 0.125 ms | 0.106 ms | **1.19x** | 3.8% |
| 1024² | 0.556 ms | 0.482 ms | **1.15x** | 2.0% |
| 2048² | 2.269 ms | 1.969 ms | **1.15x** | 1.1% |

Also 1.13x for RK4 (512², 1024²) and 1.17x for CNLSE. The 256² cell is inside
its own noise and claims nothing: at that size the step is launch-bound, which
is what the CUDA graph already fixed. The isolated linear step — transform,
multiply, transform — moves 1.19–1.21x at 512²–2048², so the step-level number
is most of the kernel-level one rather than a fraction of it.

**Confirmed by the other tool** (2026-08-01), by accident: a
`profile_backends.py --baseline main` run against a *stale* local `main` —
one that predated the merge — measured the whole merge rather than the branch
it was pointed at, and returned 0.85x at 512² split_step and 0.89x for RK4
against 0.9–1.2% scatter. That is 1.18x and 1.12x, which is the table above
reproduced end to end by an independent tool, and 1.00x at 256², which is the
launch-bound cell claiming nothing again. Re-run against a current `main` the
same comparison is 1.00x on every CUPY cell to within 0.7%.

**Four things had to be true, and the fourth is the design.** (1) The callback
machinery has to be reachable: CuPy's *legacy* callbacks are not, on this
machine or any pip-installed CuPy — they compile with nvcc against the static
cuFFT and the wheels ship no `cufft.h`. The `cb_ver="jit"` route needs only
nvrtc and nvJitLink, both present, and is what is used. (2) The result agrees
with the separate multiply *bit for bit*, not to a tolerance, on every solver
and method (at most grid sizes: the nonlinear callback's entry below lists the
exceptions). (3) A callback plan captures into a CUDA graph and replays, so the
fusion and the graph compose rather than exclude each other. (4) **cuFFT binds
the callback's argument when the plan is created, and no propagator lives that
long** — an adaptive step rebuilds it, Yoshida cycles three, the solver's cache
hands back a different array whenever the step length changes. Bound directly,
every one of those costs a plan: nvrtc, nvJitLink and cuFFT planning, tens of
milliseconds, against a step of one. So the callback is handed a pointer to a
16-byte block that points at the propagator, and the plan is built once per
(shape, dtype) for the whole run. The block is rewritten by a one-thread kernel
rather than a host copy, because a Yoshida step changes propagator part way
through and a pageable host copy during a graph capture is not allowed.

Two plans, not one: a callback fires in both directions, so a plan that applied
the propagator on the way out would apply it again on the way back. The second
plan measured no extra workspace at 2048².

**What it does not cover.** CuPy reaches the callback machinery for
multi-dimensional plans only — its 1D path raises `AttributeError` on
`cb_load_aux_arr` inside `get_fft_plan`, a CuPy bug rather than a cuFFT limit —
so `NLSE_1d` and `CNLSE_1d` keep the separate multiply. Everything reports
whether it fused rather than assuming, and a run that cannot fuse loses the
pass and nothing else. `NLSE_FUSE_PROPAGATOR=0` switches it off.

`benchmarks/cufft_callback_probe.py` asked whether this was possible before any
of it was written; it is deleted rather than left to claim the question is
still open. Its four questions are the four answered above.

**Deployed: the kernels are built without `--use_fast_math`.** RTX 3050, CUDA
13.3, CuPy 14.2.0. The flag turns `sincosf` into `__sincosf`, which is accurate
on [-pi, pi] only
([CUDA Programming Guide, 5.6.9](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#use-fast-math-effect)).
`nl_prop` with a pure potential returns `exp(i V)`: over a million phases in
[-1e3, 1e3] rad its largest error was 1.36e-4 with the flag and 9.3e-8 without
(`tests/backends/test_cuda_phase_accuracy.py`).

It bought no speed. Two campaigns, both sides interleaved, complex64:

| case | with the flag | without | ratio |
|---|---|---|---|
| split step and RK4, 1024² and 2048² | — | — | 0.999 to 1.002 |
| NLSE split step, 2048², `V` set | 15.585 ms | 15.583 ms | 1.000 |
| NLSE RK4, 2048², `V` set | 18.405 ms | 18.480 ms | 1.004 |
| CNLSE split step, 1024², `omega` set | 3.788 ms | 3.791 ms | 1.001 |

The last three rows are medians of 7 pairs of 100 asked steps, each cut by the
step limit, so a row is comparable with itself only. RK4 was slower without the
flag in 7 pairs of 7, by 0.4%: under the 1.1% this box resolves at 2048², and
the price of a phase that is right.

**Deployed: the nonlinear step applied from the inverse transform's store
callback** (2026-11-27). RTX 3050 8 GB, driver 620.70.06, CuPy 14.2.0, CUDA
runtime 13.2, cuFFT 12.3.0. The other pass a split step made outside its
transforms is gone: `square_mod_nl_prop` runs as cuFFT writes each element of
the inverse transform. `NLSE` and `GPE`, 2D, local, every splitting. The
formula lives once, `nlse_nonlinear_element` in
`kernels/cuda_source/nonlinear_step.cuh`, and is compiled into the kernels and
into the callback.

Per step in ms, lie, complex64, lossless, no potential. One process, the three
sides rotating, slope between 20 and 220 steps, median of 6 rounds; "spread" is
the widest side's max minus min over its median:

| grid | no callback | propagator | both | both / none | both / propagator | spread |
|---|---|---|---|---|---|---|
| 256² | 0.021 | 0.020 | 0.023 | 0.89x | 0.89x | 15.7% |
| 512² | 0.138 | 0.120 | 0.102 | **1.36x** | 1.18x | 2.4% |
| 1024² | 0.559 | 0.481 | 0.405 | **1.38x** | 1.19x | 2.9% |
| 2048² | 2.294 | 1.975 | 1.658 | **1.38x** | 1.19x | 0.4% |
| 1280² | 0.953 | 0.833 | 0.714 | 1.34x | 1.17x | 4.2% |
| 1536² | 1.366 | 1.192 | 1.019 | 1.34x | 1.17x | 0.4% |
| 2560² | 4.409 | 3.981 | 3.548 | 1.24x | 1.12x | 0.6% |
| 3000² | 6.194 | 5.544 | 4.936 | 1.25x | 1.12x | 0.9% |
| 3072² | 5.588 | 4.871 | 4.168 | 1.34x | 1.17x | 0.3% |

Propagator callback → both, other cells (same method, two sides): lossy
1024² 0.480 → 0.405 and 2048² 1.946 → 1.629 (1.19x); lossy with a real `V`
0.503 → 0.432 and 2.014 → 1.715 (1.16x, 1.17x); 1024×2048 0.950 → 0.793
(1.20x); 4096² 12.95 → 11.88 (1.09x, the first side ranging 12.6 to 15.9);
complex128 4.152 → 4.168 at 1024² and 16.79 → 16.81 at 2048² (1.00x: 8.6 times
the complex64 step for twice the bytes, so traffic is not what binds it).

Where the bytes go, 1024² (`ncu --set full`, 8 steps after 5 warm ones, the
CUDA graph replaced by a plain loop so that `ncu` sees each launch; DRAM bytes
per grid point):

| | kernel | per step | ms | read | written |
|---|---|---|---|---|---|
| before | `regular_fft` (first pass of a transform) | 2 | 0.100 | 8.0 | 8.0 |
| | `vector_fft` (inverse, last pass) | 1 | 0.081 | 8.0 | 7.8 |
| | forward last pass, propagator callback | 1 | 0.124 | 16.0 | 7.9 |
| | `square_mod_nl_prop` | 1 | 0.082 | 8.0 | 7.9 |
| | **step** | | 0.487 | | **87.6 in all** |
| after | `regular_fft` | 2 | 0.100 | 8.0 | 7.9 |
| | forward last pass, propagator callback | 1 | 0.124 | 16.0 | 7.9 |
| | inverse last pass, nonlinear callback | 1 | 0.087 | 8.0 | 7.8 |
| | **step** | | 0.411 | | **71.6 in all** |

88 bytes per point become 72, and the inverse transform's last pass carries
the nonlinear step for 0.006 ms more than it cost alone. Against the 208 GB/s
this card streams, the untraced step moves 191 GB/s before and 185 after at
1024² (92% and 89%), 187 and 182 at 2048² (90% and 87%): it was at the ceiling
and still is, with fewer bytes to move.

**The callback calls the step out of line, and nothing else separates a gain
from a loss.** Inlined, the first implementation ran at 0.84x of the
propagator callback alone (0.499 → 0.594 ms at 1024²). cuFFT's callback kernel
holds sixteen elements per thread, and with the whole formula inlined they no
longer fit in registers. `ncu` on that one kernel, inlined against out of line:

| | inlined | out of line |
|---|---|---|
| registers per thread | 80 | 56 |
| local-memory store requests per launch | 933 888 | 0 |
| local-memory load requests per launch | 98 304 | 0 |
| DRAM written | 37.9 MB (36 B/pt) | 8.2 MB (7.8 B/pt) |
| duration | 266 µs | 87 µs |

The cost is in what is compiled, not in what runs. On a scratch tree, same
cell, propagator callback → both: an early return for `alpha == 0`, or the
exponential, the square root or the loss iteration removed one at a time,
stayed at 0.83 to 0.84x; only the square root left, 1.19x; the exponential or
the loss iteration alone, 0.83x; `__forceinline__` on the helpers, no change;
`__noinline__` on the step, 1.18x. The prototype had gained because its
formula was shorter, and its 20 to 26% loss at 1280, 1536, 2560, 3000 and 3072
points was this same spill: the table above has none.
`tests/backends/test_kernel_templating.py` pins the `__noinline__`, which no
test can time.

**Bit for bit, where cuFFT plans the same transform.** 36 runs at 256² (lie,
strang, yoshida; no, real and complex `V`; lossless and lossy; both widths)
are identical to the commit before, with callbacks on and with
`NLSE_FUSE_PROPAGATOR=0`. Against the unfused path the callbacks are identical
at 100, 243, 1000, 1009, 1280, 1536, 2187, 3125 and 4096 points. At 2560,
3000, 3072 and 3600 a plan that carries a callback is another cuFFT plan, and
every element differs by rounding: 2e-7 of the peak after one complex64 step,
4e-15 after seven complex128 ones. The propagator callback did that already,
so claim (2) of its entry above holds at the other sizes only.

**What it does not cover.** 256² loses 3 µs a step, twice in two campaigns and
inside that cell's spread; not explained. 1D and 3D runs, non-local ones (the
response is applied between the transform and the step), the coupled solvers
and RK4 keep the kernel. `NLSE_FUSE_PROPAGATOR=0` switches both callbacks off.

Reproduce none against both with `benchmarks/propagator_fusion.py`. The
three-way table, the `ncu` tables and the bitwise runs are `fp_three.py`,
`fp_ncu.py` with `fp_ncu_table.py`, `fp_bitwise.py` and `fp_sizes_diff.py`,
kept outside the repository with the session's other scratch scripts.

**Deployed: the forward callback builds the propagator from one factor per
axis.** RTX 3050, CUDA 13.3, CuPy 14.2.0. The dispersion operator of `NLSE`
and `GPE` is one term in `kx` plus one in `ky`, so the propagator is a product
`rows[iy] * columns[ix]`. The forward transform's store callback read it from
a grid-sized array, 8 bytes per point in complex64; it now reads two vectors
that stay in cache and multiplies them. The solver checks that the operator is
such a sum (to 1e-12 of its largest element; NLSE's and GPE's are within
3e-16) and hands the factors over only for the very array the kernel would
otherwise read, found by identity. In any doubt the callback reads the array,
as before.

DRAM bytes per point per step (`ncu --set full`, 10 steps after the warm
ones, plain loop), full array → factors: 73.2 → 65.2 at 1024², 73.5 → 65.7 at
2048², 74.9 → 67.0 at 4096² in complex64; 147.5 → 131.7 at 2048² in
complex128. At 2048² the forward last pass reads 12.0 → 8.1 bytes per point
and takes 0.409 → 0.375 ms.

Per step, untraced, complex64, lie, lossless, the two sides interleaved in
one process over 12 rounds, from the 20→220 step slope (`fp_sep.py`):

| grid | full array | factors | ratio | spread | of 208 GB/s |
|---|---|---|---|---|---|
| 1024² | 0.404 | 0.388 | **1.04x** | 8.5% | 91% → 85% |
| 2048² | 1.639 | 1.562 | **1.05x** | 1.1% | 90% → 85% |
| 4096² | 9.042 | 8.675 | **1.04x** | 2.2% | 67% → 62% |

Other cells: 512² 0.089 → 0.081 (1.10x, spread 8.8%), 256² 0.023 → 0.026
(0.90x, 7.4%). complex128 4.168 → 4.233 at 1024², 16.762 → 17.002 at 2048²,
72.69 → 74.93 at 4096² (0.98x, 0.99x, 0.97x; spreads 2.2%, 1.1%, 11.3%): 16
bytes fewer per point and no faster, because that step moves 37 GB/s, 18% of
what the card streams. Arithmetic binds it, and the callback now does a
second complex multiply. One path at every size and width, as for the
callbacks above: the typical runs, 1024² to 4096² in complex64, gain 4 to 5%.

A tenth fewer bytes buys half as much time: the step was at 90% of the memory
peak and is at 85%. A 32-bit divide for the row index, in place of the 64-bit
one, changed nothing (1.04x and 1.04x, then 1.02x and 1.05x, at 1024² and
2048², against 1.02x and 1.05x without; 0.98x and 0.99x in complex128):
rejected.

Not bit for bit: the product of two factors rounds differently from the
exponential of the sum. Against the full array after 220 steps, 8e-6 of
max|A| in complex64 and 1.4e-14 in complex128.
`tests/backends/test_fft_propagator_fusion.py` holds the callback against the
full array, and `tests/solvers/test_nonlinear_in_transform.py` the factors
against a closed form.

**4096² is not at the ceiling, and the gap is inside cuFFT.** The step moves
130 GB/s there, 62% of the peak, against 85% at 1024² and 2048². In the `ncu`
table the first pass of each transform (`regular_fft`) takes 2.89 ms for 16.6
bytes per point, 96 GB/s, where the same pass runs at 161 GB/s at 2048²; the
last pass with its callback runs at 188 GB/s. Two launches a step: 5.8 of its
8.9 ms.

That pass is the transform down the columns, the strided axis, and its cost
follows the length of a column, not the stride between its elements. Each axis
alone, in place, complex64, ns per point (RTX 3050, CuPy 14.2.0, best of 8
rounds of 20 transforms, `fft_axes.py`):

| rows x columns | both axes | along rows | down columns |
|---|---|---|---|
| 1024 x 1024 | 0.173 | 0.079 | 0.095 |
| 2048 x 2048 | 0.176 | 0.077 | 0.097 |
| 2048 x 4096 | 0.177 | 0.077 | 0.100 |
| 2048 x 8192 | 0.177 | 0.077 | 0.100 |
| 4096 x 2048 | 0.238 | 0.077 | 0.161 |
| 4096 x 4096 | 0.242 | 0.077 | 0.166 |
| 8192 x 2048 | 0.349 | 0.077 | 0.272 |

A row costs the same at every length. A column of 4096 costs 1.7 times a
column of 2048 whether its elements are 16 or 32 KiB apart.

**Rejected: padding the rows at 4096².** The same transform inside rows padded
by 1 to 512 columns (cuFFT's advanced layout, `inembed`), so that the stride
is no longer a power of two: 0.250 to 0.263 ns per point against 0.240
unpadded (`fft_stride.py`). The stride was not the cause.

**Rejected: a transposed layout at 4096².** Transforming rows, transposing and
transforming rows again would replace the pass down the columns, 0.166 ns per
point, by a pass along rows, 0.077, plus one transposing copy. That copy takes
0.157 ns per point (`cp.copyto(out, a.T)`, `fft_transpose.py`), more than the
0.089 it would save. Nothing in cuFFT's interface reaches the column pass
itself.

**Deployed: the Rabi coupling inside the fused coupled kernels**
(2026-11-28). RTX 3050 8 GB, driver 620.70.06, CuPy 14.2.0, CUDA runtime
13.2, cuFFT 12.3.0; pyopencl 2026.1.5 on the same card. With `omega` set, the
fused coupled split step rotated the components afterwards in a launch of
its own (`rabi_coupling_interleaved`, Lie) or through `_take_components`
copies on each side (Strang), and `CNLSE` kept RK4 off the whole-stage
kernels. `coupled_nl_prop_c` now takes the rotation's cosine and sine and a
`RabiPlacement` (`kernels/templating.py`, rendered as `#define RABI_*` ahead
of the source): before the nonlinear step to open a Strang step, after it to
close a Lie or Strang one. `coupled_rk4_stage_c` adds `-1j * omega/2` times
the other component to each slope. OpenCL and MLX keep their kernels and
rotate the interleaved field around a Strang step: OpenCL in place, MLX
restacking it.

`CNLSE`, complex64, `omega` = 1e4 rad/m, lossless, no potential, the
benchmark field (`benchmarks/profile_backends.py`, `step_for` step). One
process, base and patched trees side by side, the order alternating over 8
rounds; per step from the 10→60 step slope, median of the per-round values,
[min, max] of the per-round gain (`gpu_p4_r2_ab.py`):

| backend, scheme | grid | before | after | gain [range] |
|---|---|---|---|---|
| CUPY, Strang | 1024² | 2.096 ms | 1.114 ms | **1.88x** [1.80, 1.89] |
| CUPY, Strang | 2048² | 8.611 ms | 4.560 ms | **1.89x** [1.55, 2.02] |
| CUPY, Strang | 4096² | 41.36 ms | 22.68 ms | **1.85x** [1.74, 2.03] |
| CUPY, Lie | 1024² | 1.113 ms | 0.950 ms | **1.17x** [1.09, 1.18] |
| CUPY, Lie | 2048² | 4.550 ms | 3.902 ms | **1.17x** [1.16, 1.17] |
| CUPY, Lie | 4096² | 24.38 ms | 19.56 ms | 1.21x [0.90, 1.47] |
| CUPY, RK4 | 1024² | 10.95 ms | 4.888 ms | **2.24x** [2.22, 2.25] |
| CUPY, RK4 | 2048² | 43.95 ms | 19.93 ms | **2.21x** [2.20, 2.21] |
| CUPY, RK4 | 4096² | 198.9 ms | 97.14 ms | **2.01x** [1.86, 2.21] |
| CL, Strang | 1024² | 3.325 ms | 1.666 ms | **2.00x** [1.92, 2.05] |
| CL, Strang | 2048² | 13.57 ms | 8.187 ms | **1.66x** [1.57, 1.68] |
| CL, Strang | 4096² | 66.19 ms | 48.48 ms | 1.24x [0.71, 2.54] |

In these runs (complex64, lossless, no potential, 60 steps) the CUPY Lie and
Strang outputs and the OpenCL Strang output equal the base's to the bit
(`max_rel_diff` 0 in `gpu_p4_r2_ab.py`). Only these cases were compared, and
other steps may differ in the last bit: the old CUDA Lie rotation took its
angle from `params[0]`, the step rounded to float32, where the kernel now
rotates by the double `dz`. RK4 differs by 9.3e-7 of the peak, the slope's
sum taken in another order. Lie at 4096² on CUPY and Strang at 4096² on OpenCL vary more
between rounds than they gain and claim nothing. Closed form: a linear
`CNLSE` started in one component keeps `cos²(omega z / 2)` of its power
there; Lie, Strang and RK4 follow it to 1e-5 in complex64 over 200 steps
(`test_a_linear_rabi_run_follows_its_closed_form`). Not measured:
MLX, complex128, a batched `omega` (still the generic path).

**Deployed: the energy rates reduced once per `out_field` call**
(2026-11-28). RTX 3050 8 GB, driver 620.70.06, CuPy 14.2.0, CUDA runtime
13.2. Not a per-step cost: `out_field` sets the step from the field's energy
rates, and each limit reduced them again: `_capped_delta_z` and
`_rk4_max_dz` both for RK4, `_default_delta_z` once more without a given
step, the Yoshida cap once more. The field does not change before the
run starts, so `out_field` reduces once and passes the result down; the
limits still reduce for themselves when called alone.

One reduction, after one warm call, median and [min, max] over 7 rounds of
20 (`gpu_p4_r2_rates_cost.py`, complex64, the benchmark field):

| solver | 1024² | 2048² | 4096² |
|---|---|---|---|
| `NLSE` | 0.922 [0.920, 0.926] ms | 3.169 [3.165, 3.177] ms | 13.30 [13.28, 13.32] ms |
| `CNLSE` | 9.220 [9.210, 9.269] ms | 42.87 [42.75, 43.13] ms | 172.8 [172.6, 178.1] ms |

A `CNLSE` reduction brings the field back to the host for its per-component
means, hence the tenfold cost. A 10-step RK4 call with a given step, before
and after in one process over 8 alternating rounds (`gpu_p4_r2_ab.py`; 2 → 1
reductions counted in each call):

| solver | grid | before | after | gain [range] |
|---|---|---|---|---|
| `NLSE` | 1024² | 34.87 ms | 33.88 ms | 1.03x [1.02, 1.07] |
| `NLSE` | 2048² | 143.5 ms | 140.1 ms | 1.02x [1.02, 1.03] |
| `NLSE` | 4096² | 663.1 ms | 650.2 ms | 1.02x [1.02, 1.02] |
| `CNLSE` | 1024² | 83.38 ms | 74.07 ms | **1.13x** [1.12, 1.18] |
| `CNLSE` | 2048² | 352.0 ms | 313.6 ms | **1.12x** [1.12, 1.13] |

The saving is the one reduction each time: 1.0, 3.4, 12.8 ms and 9.3, 38.4
ms. The per-step cost does not move (0.999 to 1.002x) and the results are
identical. Without a given step or with Yoshida the call saves one reduction
more each; not timed. The cost of the `CNLSE` reduction itself is open.

**Deployed: the propagator left on the device after a run** (2026-11-28).
RTX 3050 8 GB, driver 620.70.06, CuPy 14.2.0, CUDA runtime 13.2. After every
run `_retrieve_arrays_from_gpu` copied `simu.propagator` back to the host, a
grid-sized transfer nothing in the solver reads: the next run builds its own.
`propagator` is now a property that copies the device array back the first
time it is read after a run, once; a solver that is never asked pays nothing.

A 10-step lie call, complex64, `step_for` step, before and after in one
process over 8 alternating rounds (`gpu_p4_r2_ab.py`):

| solver | grid | before | after | gain [range] |
|---|---|---|---|---|
| `NLSE` | 1024² | 10.95 ms | 9.58 ms | **1.14x** [1.14, 1.25] |
| `NLSE` | 2048² | 49.26 ms | 41.54 ms | **1.19x** [1.18, 1.19] |
| `NLSE` | 4096² | 212.8 ms | 185.3 ms | **1.15x** [1.14, 1.17] |
| `CNLSE` | 1024² | 30.53 ms | 27.86 ms | 1.10x [1.02, 1.10] |
| `CNLSE` | 2048² | 140.9 ms | 126.5 ms | **1.12x** [1.11, 1.12] |

The per-step cost does not move (0.999 to 1.005x) and the results are
identical. A caller that reads `propagator` after each run pays the transfer
as before, at the read. OpenCL and MLX take the same path; not timed.


## Open

**The absorbing-potential and coupled halves of the lossy step.** The solved
real-space step above covers one decay channel against a real potential. A
complex potential adds a second, and the coupled solvers have one per component,
and in neither case is `dphi/dy` constant, so both still freeze `|A|^2` and both
are still first order. The coupled case is the one worth measuring first: it is
where DDGPE lives, and DDGPE is driven and dissipative by construction.

**Whether a 1D plan can be made to take a callback.** `NLSE_1d` and `CNLSE_1d`
cannot fuse, because CuPy's one-dimensional plan path raises on the way into
`get_fft_plan` (`_JITCallbackManager` has no `cb_load_aux_arr`) — a CuPy bug,
not a cuFFT limit, so it may fix itself. Reaching `PlanNd` for a rank-1
transform, if it accepts one, would sidestep it without waiting.

**An `alpha == 0` branch for `_nl_prop` and `_square_mod_nl_prop_v`.** They call
`_mul_cexp` unconditionally, but with a real potential and no loss their
exponent is purely imaginary, so they could take `_rotate` alone and skip the
exponential entirely — the branch `_nl_prop_without_V` and `_square_mod_nl_prop`
already have.

**The linear step as an increment, `A + ifft((P - 1) fft(A))`.** In complex64
the transforms set the error (see the CPU entry on float32 kernels), and they
set it on the whole field. Written as an increment they only touch the part of
the field the step changes, with `P - 1` built in double and then rounded.
Prototyped through the real solver by overriding `_build_propagator` and
`_apply_linear_step` (`inc_linear.py`, 512², CPU, complex64 against complex128,
units of 2^-24):

| run | steps | field error, today → increment | power drift per step |
|---|---|---|---|
| Kerr, lossless, lie | 176 | 4852 → 18.7 | -2.75 → +0.001 |
| Kerr, lossless, strang | 176 | 4880 → 23.5 | -2.78 → 0.000 |
| Kerr, lossless, yoshida | 176 | 14401 → 37.0 | -8.25 → -0.001 |
| saturable, lossless, V | 96 | 678 → 11.7 | -2.43 → 0.000 |
| saturable, lossy | 95 | 676 → 12.5 | -2.49 → +0.002 |
| saturable, lossy, V, strang | 96 | 669 → 17.7 | -2.47 → +0.003 |

It costs traffic: the field has to survive the forward transform, so the
transform works on a copy and the nonlinear kernel reads two arrays. One
lossless lie step on the Mac, the copy as a pass of its own and nothing fused
(`inc_cost.py 15`, min over 15 × 5, interleaved): 0.613 → 0.774 ms at 512²
(1.26x), 1.640 → 1.876 at 1024² (1.14x), 7.109 → 8.164 at 2048² (1.15x),
32.65 → 34.88 at 4096² (1.07x). Not implemented: it changes the linear step of
every backend, and on CuPy the store callbacks that the fewer-passes work
builds, and it trades the bytes that work removes for accuracy. complex128 has
no use for it.

**Whether CuPy should split its complex exponential.** `kernels/cupy.py` calls
`cp.exp` on a complex array, where the OpenCL kernel splits into `exp` and
`sincos` on reals. Whether that costs anything is unmeasured — CuPy is not
installed on the development Mac.

**Whether alignment matters once the nonlinear loops vectorize.** Grid-stride
was rejected against loops that could not vectorize at all (see above). Now
that the lossless ones do, the question is live again, and `backends/cpu.py`
allocates with plain `np.zeros` since pyFFTW's aligned allocator went away.
