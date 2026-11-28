#!/usr/bin/env python3
"""What the machine allows, and how close a step gets to it.

    python benchmarks/roofline.py
    python benchmarks/roofline.py --sizes 1024 2048 --backends MLX

`profile_kernels.py` reports each kernel against the best rate seen in its own
table, which says whether a kernel is the odd one out but not whether the whole
table is slow. This asks the other question: given the hardware, how fast could
a step possibly be?

Three ceilings, each measured rather than taken from a spec sheet, because the
achievable fraction of a rated figure is what a step actually gets:

- streaming bandwidth,
- fused-multiply-add throughput,
- sine, cosine and exponential throughput, which is what the nonlinear
  kernels spend themselves on.

Bandwidth is measured in the DRAM regime, on arrays far larger than a grid,
and one number is used at every size. A grid small enough to sit in cache is
fed faster than that, so its memory share can come out **above 100%**: that
is the reading, not an error -- it says the grid fitted in cache and beat
DRAM, and that bandwidth is no longer what limits it. Making the ceiling
size-dependent instead was tried and is worse: the probe's working set is a
third of a step's, so it reports cache rates a step cannot reach, and the
percentages then swing with the probe rather than with the code.

Ceilings belong to the device, not the framework, so they are probed through
whatever gets closest to the metal -- numba on the CPU, OpenCL or CUDA C on
the GPU -- and every backend on that device is then held to the same number.
MLX cannot be driven to its own hardware's peak from Python, which is a fact
about the probe, not about MLX.

A step is charged for compulsory traffic only: every array it must read and
write, counted once, on the path the step took (`step_path`). Real traffic can
only be larger, so each floor is a true lower bound. A row gives the measured
step's share of each ceiling and names the largest, the ceiling the step
would meet first: only a share near 100% says the step is bound by it.
"""

import argparse
import math
import platform
import time

import numpy as np

# What a 2D transform costs per grid point: rows then columns, each pass
# reading and writing the array, and 5 N log2 N flops per 1D transform.
FFT_PASSES = 4  # array-sized reads and writes per 2D transform
# Array-sized reads and writes of a whole blocked CPU step: three passes, each
# reading and writing the field once (NLSE/kernels/cpu_blocked.py).
BLOCKED_STEP_PASSES = 6
# What one split step does per grid point besides its transforms: a complex
# multiply by the propagator, and the nonlinear step's flops and its one
# sin+cos+exp, as this tool has always charged them (not re-derived here).
PROPAGATOR_FLOPS = 6
NONLINEAR_FLOPS = 14.0
NONLINEAR_TRANSCENDENTALS = 1.0
GIGA = 1e9  # the ceilings are in G per second
MS_PER_S = 1e3
PERCENT = 100
# The timed beam, SI units: 1.05 W in a defocusing medium saturating at
# 1e5 W/m^2, in an 8 mm window over a 10 mm cell; a Gaussian of BEAM_WAIST (m).
BEAM = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 8e-3,
    "n2": -1.6e-9,
    "L": 10e-3,
    "Isat": 1e5,
}
BEAM_WAIST = 2e-3
# Step every timed run takes (m): under the split-step dispersion limit of
# BEAM's window up to 4096x4096, 3.11e-6 m there (`_dispersion_max_dz`).
# A grid where the solver lowers it fails its measurement.
STEP_DZ = 1e-6
WARMUP_STEPS = 2  # steps run before timing, which build plans and kernels
TIMING_REPEATS = 3  # a timing is the fastest of this many runs
# The import names of the packages a timing depends on, named in its header.
TIMED_PACKAGES = ("numpy", "scipy", "numba", "cupy", "pyopencl", "mlx")


def fft_bytes(itemsize):
    """Bytes per grid point for one 2D transform."""
    return FFT_PASSES * itemsize


def fft_flops(n):
    """Flops per grid point for one 2D transform of an n x n grid."""
    return 10 * math.log2(n)


def step_cost(method, splitting, n, itemsize, has_V, path):
    """Return (bytes, flops, transcendentals) per grid point for one step.

    Parameters
    ----------
    method : str
        "split_step" or "RK4".
    splitting : str
        "lie" or "strang". Strang is charged as Lie: with no callback and no
        potential or a real one, the solver merges its half steps
        (`_merges_strang_halves`); a complex potential would not merge.
    n : int
        Points per axis.
    itemsize : int
        Bytes of a complex element.
    has_V : bool
        Whether the step reads a potential.
    path : str
        How the split step ran, as `step_path` names it. RK4 is charged its
        separate passes whatever this is.

    Returns
    -------
    tuple of float
        Bytes, flops and sin+cos+exp evaluations per grid point.

    Raises
    ------
    ValueError
        For a splitting or a path this has no charge for.
    """
    c = itemsize  # a complex element
    r = itemsize // 2  # a real one
    v = r if has_V else 0
    f = fft_bytes(itemsize)
    ffts = fft_flops(n)

    if method == "split_step":
        if splitting not in ("lie", "strang"):
            raise ValueError(f"no charge for splitting {splitting!r}")
        nl_b = 2 * c + v
        moved = {
            # fft, propagator multiply, ifft, nonlinear step: a pass each.
            "separate": 2 * f + 3 * c + nl_b,
            # cuFFT store callbacks: the propagator in the forward transform's
            # store, the nonlinear step in the inverse's.
            "callbacks": 2 * f + v,
            "blocked": BLOCKED_STEP_PASSES * c + v,
        }
        if path not in moved:
            raise ValueError(f"path must be one of {sorted(moved)}, got {path!r}")
        flops = 2 * ffts + PROPAGATOR_FLOPS + NONLINEAR_FLOPS
        return moved[path], flops, NONLINEAR_TRANSCENDENTALS

    # RK4: four right-hand sides, each a transform pair plus an additive
    # nonlinear term, and the stage updates between them.
    rhs_b = 2 * f + 3 * c + (3 * c + v)
    rhs_f = 2 * ffts + 6 + 12
    stages = 5 * c + 6 * c + 6 * c + 5 * c
    return (4 * rhs_b + stages, 4 * rhs_f, 4.0)


# ── Ceiling probes ───────────────────────────────────────────────────────────

GPU_C = """
KERNEL void copy_k(GLOBAL const float4* a, GLOBAL float4* b, int n, int reps) {
    int i = TID; int s = NTHREADS;
    for (int j = i; j < n; j += s) b[j] = a[j];
}
// Scalars, not float4: CUDA's float4 is a plain struct with no arithmetic
// operators, so the vector spelling compiles only under OpenCL. Eight
// independent chains give the pipeline enough to keep busy without them.
KERNEL void fma_k(GLOBAL const float* a, GLOBAL float* b, int n, int reps) {
    int i = TID;
    float x0 = a[i], x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
    float x4 = x0 + 4.0f, x5 = x0 + 5.0f, x6 = x0 + 6.0f, x7 = x0 + 7.0f;
    for (int r = 0; r < reps; ++r) {
        x0 = x0 * 1.0000001f + 1e-7f; x1 = x1 * 1.0000001f + 1e-7f;
        x2 = x2 * 1.0000001f + 1e-7f; x3 = x3 * 1.0000001f + 1e-7f;
        x4 = x4 * 1.0000001f + 1e-7f; x5 = x5 * 1.0000001f + 1e-7f;
        x6 = x6 * 1.0000001f + 1e-7f; x7 = x7 * 1.0000001f + 1e-7f;
    }
    b[i] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
KERNEL void trans_k(GLOBAL const float* a, GLOBAL float* b, int n, int reps) {
    int i = TID;
    float x = a[i], acc = 0.0f;
    for (int r = 0; r < reps; ++r) {
        float s = SIN(x + (float)r), c = COS(x + (float)r);
        acc += s * c * EXP(-FABS(x) * 1e-3f);
    }
    b[i] = acc;
}
"""

FMA_CHAINS = 8  # independent accumulators per work item in the GPU fma_k
# Floats of FMA chains per block: 24 NEON or 12 AVX2 registers. 64 (8 AVX2)
# read 80% of one Zen 2 core; 128 spills on both. Measured 2026-11-28.
CPU_FMA_FLOATS = 96
CPU_FMA_FASTMATH = {"nnan", "ninf", "nsz", "arcp", "contract", "afn"}  # no reassoc
CPU_FMA_EPS = 1e-7  # relative growth per FMA: the chains stay near their start
CPU_FMA_REPEATS = 300  # FMAs per float in the shorter of the slope's two runs

CL_DIALECT = {
    "KERNEL": "__kernel",
    "GLOBAL": "__global",
    "TID": "get_global_id(0)",
    "NTHREADS": "get_global_size(0)",
    "SIN": "sin",
    "COS": "cos",
    "EXP": "exp",
    "FABS": "fabs",
}
CUDA_DIALECT = {
    "KERNEL": 'extern "C" __global__',
    "GLOBAL": "",
    "TID": "(blockIdx.x * blockDim.x + threadIdx.x)",
    "NTHREADS": "(gridDim.x * blockDim.x)",
    "SIN": "sinf",
    "COS": "cosf",
    "EXP": "expf",
    "FABS": "fabsf",
}


def _render(dialect):
    """Substitute one dialect's spellings into the probe source."""
    src = GPU_C
    for key, value in dialect.items():
        src = src.replace(key, value)
    return src


def slope(fn, k, sync):
    """Seconds for one call, with launch overhead cancelled by a slope."""
    for _ in range(2):
        fn(k)
        sync()

    def once(reps):
        best = float("inf")
        for _ in range(5):
            t = time.perf_counter()
            fn(reps)
            sync()
            best = min(best, time.perf_counter() - t)
        return best

    return (once(2 * k) - once(k)) / k


def stream_rate(copy_at, elements):
    """GB/s from the slope between two array sizes.

    A fixed number of repeats cannot be used here: every form of it either
    leaves the launch cost in (at 8 MiB it is twice the copy) or lets the
    compiler keep a thread's elements in registers across the repeats, which
    reported 24 TB/s on a machine rated for 400 GB/s. Growing the array
    instead cancels the launch cost and leaves nothing to fold.

    Fitted over four sizes rather than differenced over two: at these sizes a
    single difference sits close to the timer's own noise, and the noisiest
    sample is the one that looks fastest.
    """
    sizes = [elements * m for m in (1, 2, 3, 4)]
    times = [copy_at(s) for s in sizes]
    x = np.array(sizes, dtype=float)
    y = np.array(times, dtype=float)
    seconds_per_element = ((x - x.mean()) * (y - y.mean())).sum() / (
        (x - x.mean()) ** 2
    ).sum()
    return 2 * 4 / seconds_per_element / 1e9 if seconds_per_element > 0 else 0.0


def cpu_ceilings(working_bytes):
    """Measure the CPU's three ceilings, streaming at the given size."""
    import numba
    from NLSE.kernels.cpu import _exp, _sincos

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def copy_k(a, b):
        for i in numba.prange(a.size):
            b[i] = a[i]

    eps = np.float32(CPU_FMA_EPS)
    blk = CPU_FMA_FLOATS

    # acc + acc * eps keeps each chain the FMA's addend: arm64 needs no mov per
    # fmla, which acc * one + eps did, at half the peak. Without reassociation
    # LLVM cannot fold it into acc * (1 + eps), a multiply.
    @numba.njit(parallel=True, fastmath=CPU_FMA_FASTMATH)
    def fma_k(a, reps):
        total = np.float32(0.0)
        for i in numba.prange(a.size // blk):
            acc = a[i * blk : (i + 1) * blk].copy()
            for _ in range(reps):
                for j in range(blk):
                    acc[j] = acc[j] + acc[j] * eps
            for j in range(blk):
                total += acc[j]
        return total

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def trans_k(a, out, bias):
        for i in numba.prange(a.size):
            s, c = _sincos(a[i] + bias)
            out[i] = s * c * _exp(-abs(a[i]))

    n = max(working_bytes // 4, 1 << 16)
    rng = np.random.default_rng(0)
    noop = lambda: None  # noqa: E731

    def copy_at(elements):
        a = rng.random(elements, dtype=np.float32)
        b = np.empty_like(a)
        copy_k(a, b)
        return min(_wall(lambda: copy_k(a, b)) for _ in range(9))

    stream = stream_rate(copy_at, n)

    # Compute ceilings are a property of the machine, not of the working set,
    # so they are measured once at a size that keeps the units busy.
    big = rng.random(1 << 22, dtype=np.float32)
    out = np.empty_like(big)
    small = big[: 1 << 20].copy()
    # Per repeat, by slope: on all cores one call is under a ms, near its launch.
    t = slope(lambda r: fma_k(small, r), CPU_FMA_REPEATS, noop)
    fma = small.size // blk * blk * 2 / t / 1e9 if t > 0 else 0.0

    t = slope(lambda r: [trans_k(big, out, np.float32(i)) for i in range(r)], 4, noop)
    return {"stream": stream, "fma": fma, "trans": big.size / t / 1e9}


def _wall(fn):
    """Seconds for one call."""
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def cl_ceilings(working_bytes):
    """Measure the GPU's ceilings through OpenCL."""
    import pyopencl as cl
    import pyopencl.array as cla

    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, _render(CL_DIALECT)).build(options="-cl-mad-enable")
    kern = {n: cl.Kernel(prog, n) for n in ("copy_k", "fma_k", "trans_k")}

    n = max(working_bytes // 4, 1 << 16)
    rng = np.random.default_rng(0)
    sync = q.finish

    def copy_at(elements):
        a = cla.to_device(q, rng.random(elements, dtype=np.float32))
        b = cla.empty_like(a)
        args = (q, (elements // 4,), None, a.data, b.data, np.int32(elements // 4))

        def once():
            kern["copy_k"](*args, np.int32(1))
            sync()

        once()
        return min(_wall(once) for _ in range(9))

    stream = stream_rate(copy_at, n)

    n = 1 << 24
    a = cla.to_device(q, rng.random(n, dtype=np.float32))
    b = cla.empty_like(a)
    reps = np.int32(512)
    m_fma = n // 8
    t = slope(
        lambda r: [
            kern["fma_k"](q, (m_fma,), None, a.data, b.data, np.int32(m_fma), reps)
            for _ in range(r)
        ],
        2,
        sync,
    )
    fma = m_fma * FMA_CHAINS * int(reps) * 2 / t / 1e9

    tr = np.int32(64)
    m = n // 16
    t = slope(
        lambda r: [
            kern["trans_k"](q, (m,), None, a.data, b.data, np.int32(m), tr)
            for _ in range(r)
        ],
        2,
        sync,
    )
    return {"stream": stream, "fma": fma, "trans": m * int(tr) / t / 1e9}


def cupy_ceilings(working_bytes):
    """Measure the GPU's ceilings through CUDA C."""
    import cupy as cp

    mod = cp.RawModule(code=_render(CUDA_DIALECT))
    kern = {n: mod.get_function(n) for n in ("copy_k", "fma_k", "trans_k")}
    sync = cp.cuda.runtime.deviceSynchronize
    block = 256

    def launch(name, count, *args):
        kern[name]((max(count // block, 1),), (block,), args)

    # np.int32 rather than Python ints: a Python int reaches the kernel as 64
    # bits and does not match an `int` parameter.
    i32 = np.int32

    def copy_at(elements):
        a = cp.random.random(elements, dtype=cp.float32)
        b = cp.empty_like(a)

        def once():
            launch("copy_k", elements // 4, a, b, i32(elements // 4), i32(1))
            sync()

        once()
        return min(_wall(once) for _ in range(9))

    stream = stream_rate(copy_at, max(working_bytes // 4, 1 << 16))

    n = 1 << 24
    a = cp.random.random(n, dtype=cp.float32)
    b = cp.empty_like(a)
    reps = 512
    m_fma = n // 8
    t = slope(
        lambda r: [
            launch("fma_k", m_fma, a, b, i32(m_fma), i32(reps)) for _ in range(r)
        ],
        2,
        sync,
    )
    fma = m_fma * FMA_CHAINS * reps * 2 / t / 1e9

    tr, m = 64, n // 16
    t = slope(
        lambda r: [launch("trans_k", m, a, b, i32(m), i32(tr)) for _ in range(r)],
        2,
        sync,
    )
    return {"stream": stream, "fma": fma, "trans": m * tr / t / 1e9}


# A ceiling belongs to the device, so every framework that can reach it is
# tried and the best rate wins -- but only for the metrics that framework can
# be trusted on. MLX cannot be driven near the GPU's FMA peak from Python, and
# Apple's OpenCL carries a launch cost that swamps a small copy: at 2 MiB it
# read 150 GB/s where an MLX step demonstrably moved 252.
PROBES = {
    "cpu": [("numba", cpu_ceilings, {"stream", "fma", "trans"})],
    "gpu": [
        ("opencl", cl_ceilings, {"stream", "fma", "trans"}),
        ("cuda", cupy_ceilings, {"stream", "fma", "trans"}),
    ],
}

# Where the streaming ceiling is measured. Not at the working set size: below
# about 32 MiB the copy is shorter than the launch that carries it and the
# rate that comes back is the launch, not the memory system. A step at those
# sizes cannot reach the ceiling either, and the dispatch column is what says
# so.
CEILING_BYTES = 128 * 1024**2

DEVICE_OF = {"CPU": "cpu", "CL": "gpu", "MLX": "gpu", "CUPY": "gpu"}


_CEILINGS: dict = {}


def measure_ceilings(device, verbose=False):
    """Return the best {stream, fma, trans} the probes reach, or None if one is zero."""
    if device in _CEILINGS:
        return _CEILINGS[device]
    best = {"stream": 0.0, "fma": 0.0, "trans": 0.0}
    for name, probe, trusted in PROBES.get(device, ()):
        try:
            got = probe(CEILING_BYTES)
        except Exception as exc:  # pragma: no cover - depends what is installed
            if verbose:
                print(f"    ({name}: {type(exc).__name__})")
            continue
        if verbose:
            print(
                f"    {name:7s} {got['stream']:7.0f} GB/s "
                f"{got['fma'] / 1e3:6.2f} TFLOP/s {got['trans']:6.1f} G/s"
                f"   trusted for {', '.join(sorted(trusted))}"
            )
        for key in trusted:
            best[key] = max(best[key], got.get(key, 0.0))
    # Every share divides by its ceiling: one that reads zero is no ceiling.
    _CEILINGS[device] = best if min(best.values()) > 0 else None
    return _CEILINGS[device]


# ── Step timing ──────────────────────────────────────────────────────────────


def time_step(backend_name, n, method, splitting, steps=12):
    """Seconds per step, measured as a slope so setup cancels, and its path.

    Returns
    -------
    tuple of (float, str)
        Seconds per step, and the path the step took (`step_path`).

    Raises
    ------
    RuntimeError
        If the solver lowered the step, which the slope cannot divide out, if
        the path changed while the step was timed, or if twice the steps took
        no longer, which is a disturbed machine and not a step time.
    """
    simu, E = build_solver(backend_name, n)
    propagate(simu, E, WARMUP_STEPS, method, splitting)
    # The slope divides by the steps asked for: a lowered step takes more.
    if simu._current_delta_z != STEP_DZ:
        raise RuntimeError(
            f"the solver lowered the step to {simu._current_delta_z:.2e} m "
            f"from {STEP_DZ:.2e} m at {n}x{n}"
        )
    path = step_path(simu, method)
    longer = fastest_run(simu, E, 2 * steps, method, splitting)
    shorter = fastest_run(simu, E, steps, method, splitting)
    if longer <= shorter:
        raise RuntimeError(
            f"{2 * steps} steps took {longer:.3g} s and {steps} took {shorter:.3g} s "
            f"at {n}x{n}: the timing was disturbed"
        )
    seconds = (longer - shorter) / steps
    if step_path(simu, method) != path:
        raise RuntimeError(f"the step left the {path} path while it was timed")
    return seconds, path


def build_solver(backend_name, n):
    """Return the timed solver on an n x n grid of ``backend_name``, and its beam."""
    from NLSE import NLSE

    simu = NLSE(**BEAM, V=None, NX=n, NY=n, backend=backend_name)
    E = np.exp(-(simu.XX**2 + simu.YY**2) / BEAM_WAIST**2).astype(np.complex64)
    return simu, E


def propagate(simu, E, steps, method, splitting):
    """Propagate a copy of E by ``steps`` steps of STEP_DZ, and wait for it."""
    out = simu.out_field(
        E.copy(),
        steps * STEP_DZ,
        verbose=False,
        plot=False,
        normalize=False,
        delta_z=STEP_DZ,
        method=method,
        splitting=splitting,
    )
    simu._backend.synchronize(out)


def fastest_run(simu, E, steps, method, splitting):
    """Seconds of the fastest of TIMING_REPEATS propagations of ``steps`` steps."""
    return min(
        _wall(lambda: propagate(simu, E, steps, method, splitting))
        for _ in range(TIMING_REPEATS)
    )


def step_path(simu, method):
    """Name the passes the solver's split step takes, for `step_cost`.

    Parameters
    ----------
    simu : NLSE
        A solver that has propagated.
    method : str
        "split_step" or "RK4".

    Returns
    -------
    str
        "blocked" where the CPU takes its three-pass step, "callbacks" where
        CUPY's cuFFT store callbacks carry the propagator (forward transform)
        and the nonlinear step (inverse transform), "separate" everywhere else.

    Raises
    ------
    RuntimeError
        If CUPY built one of those two callbacks and not the other.
    """
    from NLSE.kernels.templating import NONLINEAR_CALLBACK, SEPARABLE_CALLBACK

    # RK4 is charged its separate passes on every backend: the callbacks in
    # CUPY's slopes are not counted.
    if method != "split_step":
        return "separate"
    name = simu._backend.name
    if name == "CPU":
        return "blocked" if simu.plans[0].blocked_step() is not None else "separate"
    if name != "CUPY":
        return "separate"
    # Plans are cached per shape, so `_fused` also holds what an RK4 case built.
    ours = {SEPARABLE_CALLBACK, NONLINEAR_CALLBACK} & set(simu.plans[0]._fused)
    if len(ours) == 2:
        return "callbacks"
    if ours:
        raise RuntimeError(f"CUPY fused only {sorted(ours)}: no charge for that step")
    return "separate"


def ceiling_shares(cost, points, ceiling, fixed, measured):
    """Return the time each ceiling allows a step, over the time it took.

    Parameters
    ----------
    cost : tuple of float
        Bytes, flops and sin+cos+exp per grid point, from `step_cost`.
    points : int
        Grid points.
    ceiling : dict
        ``stream`` in GB/s, ``fma`` in GFLOP/s, ``trans`` in G/s.
    fixed : float
        Seconds a step costs before it touches data (`overhead`).
    measured : float
        Seconds the step took (`time_step`).

    Returns
    -------
    dict
        ``memory``, ``flops``, ``sincos`` and ``dispatch``: 1.0 is a step
        running at that ceiling.
    """
    nbytes, flops, trans = cost
    floors = {
        "memory": nbytes * points / (ceiling["stream"] * GIGA),
        "flops": flops * points / (ceiling["fma"] * GIGA),
        "sincos": trans * points / (ceiling["trans"] * GIGA),
        "dispatch": fixed,
    }
    return {name: floor / measured for name, floor in floors.items()}


def configuration():
    """Return one line naming what every timing ran on, the commit excepted.

    Returns
    -------
    str
        Platform, Python, the CPU threads the blocked step may use, and the
        version of each of TIMED_PACKAGES that is installed.
    """
    from importlib.metadata import packages_distributions, version

    from NLSE.backends.cpu_tuning import available_threads

    dists = packages_distributions()
    installed = [f"{d} {version(d)}" for m in TIMED_PACKAGES for d in dists.get(m, ())]
    return (
        f"{platform.platform()}, Python {platform.python_version()}, "
        f"{available_threads()} CPU threads, " + ", ".join(installed)
    )


_OVERHEAD: dict = {}


def overhead(backend_name, method, splitting):
    """Seconds a step costs before it touches any data.

    Measured as a step on a 64x64 grid, where the arrays are a few tens of
    kilobytes and everything but dispatch has vanished. On a GPU at 512^2 this
    is most of the step, which is why fusing kernels pays there and bandwidth
    tricks do not.
    """
    key = (backend_name, method, splitting)
    if key not in _OVERHEAD:
        _OVERHEAD[key] = time_step(backend_name, 64, method, splitting)[0]
    return _OVERHEAD[key]


def main(argv=None):
    """Print the machine's ceilings and each step against them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[512, 1024, 2048])
    parser.add_argument("--backends", nargs="*")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["split_step:lie", "split_step:strang", "RK4:lie"],
    )
    parser.add_argument(
        "--probes", action="store_true", help="show each probe, not just the best"
    )
    args = parser.parse_args(argv)

    from NLSE.backends import list_available_backends

    backends = args.backends or list_available_backends()
    itemsize = 8  # complex64
    print(configuration())

    for n in args.sizes:
        working = n * n * itemsize
        print(f"\n=== {n}x{n}   ({working / 1024**2:.0f} MiB per array) ===")

        ceilings = {}
        for device in dict.fromkeys(DEVICE_OF[b] for b in backends):
            got = measure_ceilings(device, verbose=args.probes)
            if got:
                ceilings[device] = got
                print(
                    f"  {device:3s} ceiling: {got['stream']:6.0f} GB/s   "
                    f"{got['fma'] / 1e3:6.2f} TFLOP/s   "
                    f"{got['trans']:6.1f} Gsincos+exp/s"
                )

        print(
            f"\n  {'backend':<7} {'case':<20} {'path':<9} {'B/pt':>5}"
            f" {'measured':>9} {'memory':>7} {'flops':>7} {'sincos':>7}"
            f" {'dispatch':>8}  nearest"
        )
        for backend_name in backends:
            device = DEVICE_OF[backend_name]
            if device not in ceilings:
                print(
                    f"  {backend_name:<7} no {device} ceiling: no probe read all three"
                )
                continue
            ceiling = ceilings[device]
            for case in args.cases:
                method, splitting = case.split(":")
                try:
                    fixed = overhead(backend_name, method, splitting)
                    got, path = time_step(backend_name, n, method, splitting)
                    cost = step_cost(
                        method, splitting, n, itemsize, has_V=False, path=path
                    )
                except Exception as exc:
                    print(f"  {backend_name:<7} {case:<20} {type(exc).__name__}: {exc}")
                    continue
                shares = ceiling_shares(cost, n * n, ceiling, fixed, got)
                # The largest share, not their sum: a GPU overlaps dispatch
                # with the traffic of the kernel before it, so adding them is
                # not a bound the hardware has to respect.
                nearest = max(shares, key=shares.get)
                print(
                    f"  {backend_name:<7} {case:<20} {path:<9} {cost[0]:5.0f}"
                    f" {got * MS_PER_S:8.3f}m {PERCENT * shares['memory']:6.0f}%"
                    f" {PERCENT * shares['flops']:6.0f}%"
                    f" {PERCENT * shares['sincos']:6.0f}%"
                    f" {PERCENT * shares['dispatch']:7.0f}%  {nearest}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
