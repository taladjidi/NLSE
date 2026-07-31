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

End to end that is ~1.05–1.07x on a 2048² step and within noise at 1024², the
kernel holding only ~17% of a step. Accuracy: 1.5 ulp against numpy, flat from
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

**`-cl-fast-relaxed-math -cl-mad-enable`, and a program cache — deployed.**
The cache is keyed by `(context_hash, precision)`.

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

## CUDA (CuPy)

Not verified on this machine — CuPy is not installed on the development Mac, so
nothing in this section has been measured here recently. The backend uses
CuPy's VkFFT plans and `cupy.fuse` kernels. Treat CUDA-specific claims as
unconfirmed until run on NVIDIA hardware.

## Open

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
