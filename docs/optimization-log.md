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
  out. This is the one that decides whether a change ships.
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
| streaming bandwidth | 357–416 GB/s | 235–242 GB/s |
| fp32 fused multiply-add | 10.2 TFLOP/s | 0.55 TFLOP/s |
| sin + cos + exp | 87–89 G/s | 2.5–2.9 G/s |

The GPU bandwidth brackets the 400 GB/s rating, because at these sizes part of
the working set is served by cache rather than DRAM. The FMA figure is
essentially the hardware limit (40 cores x 128 lanes x 2 at ~1.1 GHz); a
float4 probe reads 12.1, but scalar is what CUDA can also compile. The
transcendental gap is the one that matters: **the GPU does sin, cos and exp 30x
faster than the CPU**, which is why the nonlinear kernels dominate a CPU step
and not a GPU one.

The bandwidth ceiling itself moves about 15% between runs, so every percentage
below carries that: nothing under ~15% apart is a difference.

Compulsory traffic per grid point, counting each array a step must read and
write once: **104 B** for single-precision split-step, **120 B** double,
**624 B** for RK4. Two thirds of the split-step figure is the transform pair.

Fraction of the floor reached (complex64, no potential):

| | 512² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| CPU split_step | 27% | 25% | 27% | 27% |
| CL split_step | 93% | 70% | 62% | 62% |
| MLX split_step | 63% | 78% | 53% | 48% |
| MLX RK4 | 89% | 103% | 72% | 70% |

Three things follow.

**The GPU backends are at 50-80% of a hard bound**, so the remaining headroom
there is under 2x and mostly in the transform. **The CPU sits at 25-30%**, and
above 1024² its binding ceiling is not bandwidth but flops and transcendentals
— which is what the `_sincos` and `_exp` polynomials below were aimed at, and
where what is left also lies.

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
genuinely **1.37x faster than pocketfft on one thread**, 26.4 ms against
36.1 ms. It loses anyway, because vDSP has no threading of its own and
pocketfft's does: **5.5 ms against 26.4 ms, so scipy wins by 4.8x** in the
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

**A separable propagator — open, measured, not built.** The linear propagator
is `exp(-i(kx²+ky²)dz/2k)`, which factors into
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
written, nothing measured.** Better arithmetic than the separable propagator,
because it removes the pass rather than shrinking it. cuFFT can run a store
callback as it writes each element, which absorbs the propagator multiply
entirely: **104 B/point becomes 88**, and a load callback on the inverse could
take the nonlinear step too, for **72**. On a backend already at the bandwidth
bound that is close to a straight 15–30%.

`benchmarks/cufft_callback_probe.py` asks the four questions that decide
whether it can be built, none of which can be answered on a machine without an
NVIDIA GPU:

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

| of floor | 1024² | 2048² | 4096² |
|---|---|---|---|
| CPU split_step | 42% | 37% | 37% |
| CUPY split_step single | 94% | 91% | 73% |
| CUPY split_step double | 108% | 106% | 84% |
| CUPY RK4 | 114% | 113% | 93% |

**CuPy is at the bandwidth bound and has essentially nothing left**: at or over
100% up to 2048², and 73–93% at 4096² where the grid no longer fits in cache.
Dispatch there costs 0.011 ms per step, forty times less than Apple's OpenCL.

The two machines differ in which ceiling binds the CPU, and it is the hardware
rather than the code. That box streams **32 GB/s against the M3 Max's 240** — a
desktop's dual-channel DDR against unified memory — so every CPU row there is
memory-bound, where on the Mac flops and transcendentals bind above 1024². Its
sin+cos+exp rate is also half the Mac's (1.3 against 2.5 G/s), consistent with
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

Still open there, and cheap to answer with a Mac in hand: whether the solved
lossy step costs MLX what it costs the others (~16% on Lie, ~25% on Yoshida,
net-negative on Strang), and whether `mx.compile` folds the extra iteration in
for free the way it folded the hand-written Metal kernel in.

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

*Challenge this on x86*, where the wheels are vectorized and this may go the
other way. It is unmeasured there, which is why the transform stays behind the
`Backend` interface rather than being called directly.

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

## MLX (Metal)

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

**Four things had to be true, and the fourth is the design.** (1) The callback
machinery has to be reachable: CuPy's *legacy* callbacks are not, on this
machine or any pip-installed CuPy — they compile with nvcc against the static
cuFFT and the wheels ship no `cufft.h`. The `cb_ver="jit"` route needs only
nvrtc and nvJitLink, both present, and is what is used. (2) The result agrees
with the separate multiply *bit for bit*, not to a tolerance, on every solver
and method. (3) A callback plan captures into a CUDA graph and replays, so the
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

## Open

**The absorbing-potential and coupled halves of the lossy step.** The solved
real-space step above covers one decay channel against a real potential. A
complex potential adds a second, and the coupled solvers have one per component,
and in neither case is `dphi/dy` constant, so both still freeze `|A|^2` and both
are still first order. The coupled case is the one worth measuring first: it is
where DDGPE lives, and DDGPE is driven and dissipative by construction.

**The nonlinear step as a store callback on the *inverse* transform.** The
propagator fusion above removes one of the two passes a split step makes over
the field outside the transforms; this would remove the other, leaving a step
that is two transforms and nothing else. Unmeasured, and there is a real
argument that it pays much less: the propagator multiply was pure traffic, while
the nonlinear kernels run at 15% of bandwidth because they are bound on
transcendentals (see *Where the time goes*), so the 16 bytes per point this
saves may be dwarfed by what the same arithmetic costs inside a transform kernel
whose occupancy it would cut. It also needs more than a pointer through the
callback's block — the interaction strength, the loss, the saturation and the
step all change with an adaptive stepper — so the block becomes a struct that
the host rewrites per step rather than per propagator.

**Whether a 1D plan can be made to take a callback.** `NLSE_1d` and `CNLSE_1d`
cannot fuse, because CuPy's one-dimensional plan path raises on the way into
`get_fft_plan` (`_JITCallbackManager` has no `cb_load_aux_arr`) — a CuPy bug,
not a cuFFT limit, so it may fix itself. Reaching `PlanNd` for a rank-1
transform, if it accepts one, would sidestep it without waiting.

**An `alpha == 0` branch for `_nl_prop` and `_square_mod_nl_prop_v`.** They call
`_cexp` unconditionally, but with a real potential and no loss their exponent is
purely imaginary, so they could take `_sincos` alone and skip `_exp` entirely —
the branch `_nl_prop_without_V` and `_square_mod_nl_prop` already have.

**Whether CuPy should split its complex exponential.** `kernels/cupy.py` calls
`cp.exp` on a complex array, where the OpenCL kernel splits into `exp` and
`sincos` on reals. Whether that costs anything is unmeasured — CuPy is not
installed on the development Mac.

**Whether alignment matters once the nonlinear loops vectorize.** Grid-stride
was rejected against loops that could not vectorize at all (see above). Now
that the lossless ones do, the question is live again, and `backends/cpu.py`
allocates with plain `np.zeros` since pyFFTW's aligned allocator went away.
