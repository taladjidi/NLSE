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
fed faster than that, so such a step can come out **above 100% of the floor**:
that is the reading, not an error -- it says the grid fitted in cache and beat
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
write, counted once. Real traffic can only be larger, so the floor is a true
lower bound and the percentages are upper bounds on how much room is left.
"""

import argparse
import math
import time

import numpy as np

# What a 2D transform costs per grid point: rows then columns, each pass
# reading and writing the array, and 5 N log2 N flops per 1D transform.
FFT_PASSES = 4  # array-sized reads and writes per 2D transform


def fft_bytes(itemsize):
    """Bytes per grid point for one 2D transform."""
    return FFT_PASSES * itemsize


def fft_flops(n):
    """Flops per grid point for one 2D transform of an n x n grid."""
    return 10 * math.log2(n)


def step_cost(method, precision, n, itemsize, has_V):
    """Return (bytes, flops, transcendentals) per grid point for one step.

    Counted from what the solver dispatches: see `split_step_fused` and
    `rk4_rhs_fused` in the backend kernels.
    """
    c = itemsize  # a complex element
    r = itemsize // 2  # a real one
    v = r if has_V else 0
    f = fft_bytes(itemsize)
    ffts = fft_flops(n)

    if method == "split_step":
        # fft, propagator multiply, ifft, nonlinear step.
        nl_b, nl_f, nl_t = 2 * c + v, 14.0, 1.0
        nonlinear = 2 if precision == "double" else 1
        return (
            2 * f + 3 * c + nonlinear * nl_b,
            2 * ffts + 6 + nonlinear * nl_f,
            nonlinear * nl_t,
        )

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

FMA_CHAINS = 8  # independent accumulators per work item in fma_k

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

    one, eps = np.float32(1.0000001), np.float32(1e-7)
    blk = 64

    @numba.njit(parallel=True, fastmath=True)
    def fma_k(a, reps):
        total = np.float32(0.0)
        for i in numba.prange(a.size // blk):
            acc = a[i * blk : (i + 1) * blk].copy()
            for _ in range(reps):
                for j in range(blk):
                    acc[j] = acc[j] * one + eps
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
    reps = 300
    fma_k(small, 2)
    t = min(_wall(lambda: fma_k(small, reps)) for _ in range(7))
    fma = small.size * reps * 2 / t / 1e9

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


def mlx_ceilings(working_bytes):
    """Measure the GPU's ceilings through MLX.

    Every op feeds the next, so none can be dropped: a chain whose
    intermediates nothing reads gets discarded and reports absurd rates.
    """
    import mlx.core as mx

    n = max(working_bytes // 8, 1 << 14)
    a = mx.random.uniform(shape=(n,)).astype(mx.complex64)
    mx.eval(a)
    noop = lambda: None  # noqa: E731

    def chain(reps):
        out = a
        for _ in range(reps):
            out = mx.multiply(out, complex(1.0000001))
        mx.eval(out)

    t = slope(chain, 8, noop)
    stream = 2 * n * 8 / t / 1e9

    real = mx.random.uniform(shape=(n,))
    mx.eval(real)

    def fma_chain(reps):
        out = real
        for _ in range(reps):
            out = out * 1.0000001 + 1e-7
        mx.eval(out)

    t = slope(fma_chain, 8, noop)
    fma = n * 2 / t / 1e9

    def trans_chain(reps):
        out = real
        for _ in range(reps):
            out = mx.sin(out) * mx.cos(out) * mx.exp(-mx.abs(out) * 1e-3)
        mx.eval(out)

    t = slope(trans_chain, 4, noop)
    return {"stream": stream, "fma": fma, "trans": n / t / 1e9}


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
    """Return the best {stream, fma, trans} any probe reaches, or None."""
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
    _CEILINGS[device] = best if max(best.values()) > 0 else None
    return _CEILINGS[device]


# ── Step timing ──────────────────────────────────────────────────────────────


def time_step(backend_name, n, method, precision, steps=12):
    """Seconds per step, measured as a slope so setup cancels."""
    from NLSE import NLSE

    simu = NLSE(
        alpha=0.0,
        power=1.05,
        window=8e-3,
        n2=-1.6e-9,
        V=None,
        L=10e-3,
        NX=n,
        NY=n,
        Isat=1e5,
        backend=backend_name,
    )
    E = np.exp(-(simu.XX**2 + simu.YY**2) / 2e-3**2).astype(np.complex64)
    dz = 1e-5

    def run(k):
        out = simu.out_field(
            E.copy(),
            k * dz,
            verbose=False,
            plot=False,
            normalize=False,
            delta_z=dz,
            method=method,
            precision=precision,
        )
        simu._backend.synchronize(out)

    run(2)

    def once(k):
        return min(_wall(lambda: run(k)) for _ in range(3))

    return (once(2 * steps) - once(steps)) / steps


_OVERHEAD: dict = {}


def overhead(backend_name, method, precision):
    """Seconds a step costs before it touches any data.

    Measured as a step on a 64x64 grid, where the arrays are a few tens of
    kilobytes and everything but dispatch has vanished. On a GPU at 512^2 this
    is most of the step, which is why fusing kernels pays there and bandwidth
    tricks do not.
    """
    key = (backend_name, method, precision)
    if key not in _OVERHEAD:
        _OVERHEAD[key] = time_step(backend_name, 64, method, precision)
    return _OVERHEAD[key]


def main(argv=None):
    """Print the machine's ceilings and each step against them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[512, 1024, 2048])
    parser.add_argument("--backends", nargs="*")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["split_step:single", "split_step:double", "RK4:single"],
    )
    parser.add_argument(
        "--probes", action="store_true", help="show each probe, not just the best"
    )
    args = parser.parse_args(argv)

    from NLSE.backends import list_available_backends

    backends = args.backends or list_available_backends()
    itemsize = 8  # complex64

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
            f"\n  {'backend':<7} {'case':<20} {'work':>8} {'+fixed':>8}"
            f" {'= floor':>8} {'limit':>7} {'measured':>9} {'of floor':>9}"
        )
        for backend_name in backends:
            device = DEVICE_OF[backend_name]
            if device not in ceilings:
                continue
            ceiling = ceilings[device]
            for case in args.cases:
                method, precision = case.split(":")
                b, f, t = step_cost(method, precision, n, itemsize, has_V=False)
                pts = n * n
                floors = {
                    "memory": b * pts / (ceiling["stream"] * 1e9),
                    "flops": f * pts / (ceiling["fma"] * 1e9),
                    "sincos": t * pts / (ceiling["trans"] * 1e9),
                }
                which = max(floors, key=floors.get)
                work = floors[which]
                try:
                    fixed = overhead(backend_name, method, precision)
                    got = time_step(backend_name, n, method, precision)
                except Exception as exc:
                    print(f"  {backend_name:<7} {case:<20} {type(exc).__name__}: {exc}")
                    continue
                # The larger of the two, not their sum: a GPU overlaps
                # dispatch with the traffic of the kernel before it, so
                # adding them is not a bound the hardware has to respect.
                floor = max(work, fixed)
                limit = which if work >= fixed else "dispatch"
                print(
                    f"  {backend_name:<7} {case:<20} {work * 1e3:7.3f}m"
                    f" {fixed * 1e3:7.3f}m {floor * 1e3:7.3f}m {limit:>7}"
                    f" {got * 1e3:8.3f}m {100 * floor / got:7.0f} %"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
