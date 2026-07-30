#!/usr/bin/env python3
"""Time each kernel on its own and compare it to the machine's bandwidth.

    python benchmarks/profile_kernels.py
    python benchmarks/profile_kernels.py --sizes 1024 2048 --backends MLX

The split-step kernels are elementwise: every one of them reads a couple of
arrays, does a few flops per element and writes one back. At these sizes none
of it fits in cache, so the floor is how fast the machine can move the bytes,
and a kernel's only interesting number is what fraction of that it reaches.

Percentages are against the best rate observed in the same table, not against
a synthetic ceiling: a scaling op that allocates its output is not comparable
to a kernel that writes in place, and using it as the denominator produced
kernels running at "140% of bandwidth". Pass --peak-gbs to also see what the
best observed rate is as a fraction of what the machine is rated for (an
Apple M3 Max with 40 GPU cores is 400).

FFTs are reported alongside but are not held to the same standard. They make
several passes over the array and their cost is not a single streaming read.
"""

import argparse
import statistics
import time

import numpy as np

COMPLEX = 8  # complex64
REAL = 4  # float32

# Bytes each kernel must touch per grid point, counting every array it reads
# and every array it writes. A kernel that updates an array in place is
# charged for the read and the write. Coupled kernels are charged per
# component, since they are handed one component at a time.
#
# This is the compulsory traffic, so GB/s computed from it is a lower bound on
# what the kernel actually moves.
TRAFFIC = {
    "bandwidth_ceiling": 2 * COMPLEX,
    # single component, real space
    "square_mod": COMPLEX + REAL,
    "apply_propagator": 3 * COMPLEX,
    "nl_prop": 2 * COMPLEX + 2 * REAL,
    "nl_prop_without_V": 2 * COMPLEX + REAL,
    "square_mod_nl_prop": 2 * COMPLEX,
    "square_mod_nl_prop_v": 2 * COMPLEX + REAL,
    # coupled
    "nl_prop_c": 2 * COMPLEX + 3 * REAL,
    "nl_prop_without_V_c": 2 * COMPLEX + 2 * REAL,
    "rabi_coupling": 4 * COMPLEX,
    # RK4 stages
    "rk4_axpy": 3 * COMPLEX,
    "rk4_accumulate": 3 * COMPLEX,
    "rk4_nl_rhs": 3 * COMPLEX + REAL,
    "rk4_nl_rhs_v": 3 * COMPLEX + 2 * REAL,
    "square_mod_rk4_nl_rhs": 3 * COMPLEX,
    "square_mod_rk4_nl_rhs_v": 3 * COMPLEX + REAL,
    "rk4_nl_rhs_c": 3 * COMPLEX + 2 * REAL,
    "rk4_nl_rhs_c_v": 3 * COMPLEX + 3 * REAL,
    "rk4_set_and_axpy": 5 * COMPLEX,
    "rk4_acc_and_axpy": 6 * COMPLEX,
    # whole-step kernels: a transform pair plus the real-space work
    "linear_step": 4 * COMPLEX,
    "split_step_fused": 6 * COMPLEX,
    "split_step_coupled_fused": 12 * COMPLEX,
    "rk4_rhs_fused": 6 * COMPLEX,
    "rk4_rhs_coupled_fused": 12 * COMPLEX,
    "split_step_rk4_fused": 24 * COMPLEX,
    "fft_roundtrip": 4 * COMPLEX,
}

# Which group each kernel belongs to, for the report.
GROUPS = [
    ("reference", ["bandwidth_ceiling", "fft_roundtrip"]),
    (
        "single component",
        [
            "square_mod",
            "apply_propagator",
            "nl_prop",
            "nl_prop_without_V",
            "square_mod_nl_prop",
            "square_mod_nl_prop_v",
        ],
    ),
    ("coupled", ["nl_prop_c", "nl_prop_without_V_c", "rabi_coupling"]),
    (
        "RK4 stages",
        [
            "rk4_axpy",
            "rk4_accumulate",
            "rk4_nl_rhs",
            "rk4_nl_rhs_v",
            "square_mod_rk4_nl_rhs",
            "square_mod_rk4_nl_rhs_v",
            "rk4_nl_rhs_c",
            "rk4_nl_rhs_c_v",
            "rk4_set_and_axpy",
            "rk4_acc_and_axpy",
        ],
    ),
    (
        "fused whole steps",
        [
            "linear_step",
            "split_step_fused",
            "split_step_coupled_fused",
            "rk4_rhs_fused",
            "rk4_rhs_coupled_fused",
            "split_step_rk4_fused",
        ],
    ),
]


def reset(backend):
    """Drop whatever the backend is caching, so kernels do not inherit it.

    Without this a kernel's rate depends on which kernel ran before it: MLX's
    square_mod varied by a factor of two across runs, and by less than that
    within one.
    """
    if backend.name == "MLX":
        import mlx.core as mx

        for attr in ("clear_cache", "reset_peak_memory"):
            fn = getattr(mx, attr, None) or getattr(
                getattr(mx, "metal", None), attr, None
            )
            if fn is not None:
                fn()
    elif backend.name == "CUPY":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()


def timed(fn, backend, result_of, repeats=15):
    """Return the best and median seconds for one call.

    Every call is synchronized before the clock stops. Queueing several and
    forcing only the last lets a lazy backend discard the ones whose results
    nothing asks for: MLX reported kernels running at 151% of the machine's
    bandwidth that way, which is the tell.
    """
    for _ in range(3):
        backend.synchronize(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        backend.synchronize(out if out is not None else result_of())
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times)


def make_arrays(backend, n):
    """Return every array the kernels need, on the device."""
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
    pair = np.stack([a, 0.5 * a])
    return {
        "A": backend.from_numpy(a.copy()),
        "A2": backend.from_numpy((0.5 * a).copy()),
        "B": backend.from_numpy(a.copy()),
        "k": backend.from_numpy(a.copy()),
        "acc": backend.from_numpy(a.copy()),
        "out": backend.from_numpy(a.copy()),
        "A_sq": backend.from_numpy(np.abs(a) ** 2),
        "A_sq2": backend.from_numpy(np.abs(0.5 * a) ** 2),
        "prop": backend.from_numpy(np.exp(1j * a.real).astype(np.complex64)),
        "V": backend.from_numpy(a.real.copy()),
        "pair": backend.from_numpy(pair.copy()),
        "pair_prop": backend.from_numpy(
            np.stack([np.exp(1j * a.real), np.exp(1j * a.real)]).astype(np.complex64)
        ),
    }


def ceiling_call(backend, arrays):
    """Return a callable scaling one array into another."""
    name = backend.name
    a, b = arrays["A"], arrays["B"]
    if name == "MLX":
        import mlx.core as mx

        return lambda: mx.multiply(a, 2.0)
    if name == "CUPY":
        import cupy as cp

        return lambda: cp.multiply(a, np.complex64(2.0), out=b)
    if name == "CL":
        return lambda: (b._axpbz(b, np.complex64(2.0), a, np.complex64(0.0)), b)[1]
    return lambda: np.multiply(a, np.complex64(2.0), out=b)


def kernel_calls(kernels, arrays, plan, pair_plan, dz=1e-4):
    """Return {name: callable} for every kernel this backend provides."""
    A, A2, k, acc, out = (arrays[x] for x in ("A", "A2", "k", "acc", "out"))
    A_sq, A_sq2 = arrays["A_sq"], arrays["A_sq2"]
    prop, V, pair, pair_prop = (arrays[x] for x in ("prop", "V", "pair", "pair_prop"))
    f = np.float32
    alpha, g, Isat = f(0.0), f(1e-3), f(1e5)
    g12, Isat2, omega = f(1e-4), f(1e5), f(1.0)

    candidates = {
        "square_mod": lambda: kernels.square_mod(A, A_sq),
        "apply_propagator": lambda: kernels.apply_propagator(A, prop),
        "nl_prop": lambda: kernels.nl_prop(A, A_sq, dz, alpha, V, g, Isat),
        "nl_prop_without_V": lambda: kernels.nl_prop_without_V(
            A, A_sq, dz, alpha, g, Isat
        ),
        "square_mod_nl_prop": lambda: kernels.square_mod_nl_prop(A, dz, alpha, g, Isat),
        "square_mod_nl_prop_v": lambda: kernels.square_mod_nl_prop_v(
            A, V, dz, alpha, g, Isat
        ),
        "nl_prop_c": lambda: kernels.nl_prop_c(
            A, A_sq, A_sq2, dz, alpha, V, g, g12, Isat, Isat2
        ),
        "nl_prop_without_V_c": lambda: kernels.nl_prop_without_V_c(
            A, A_sq, A_sq2, dz, alpha, g, g12, Isat, Isat2
        ),
        "rabi_coupling": lambda: kernels.rabi_coupling(A, A2, dz, omega),
        "rk4_axpy": lambda: kernels.rk4_axpy(out, A, f(0.5), k),
        "rk4_accumulate": lambda: kernels.rk4_accumulate(acc, f(0.5), k),
        "rk4_nl_rhs": lambda: kernels.rk4_nl_rhs(k, A, A_sq, alpha, g, Isat),
        "rk4_nl_rhs_v": lambda: kernels.rk4_nl_rhs_v(k, A, A_sq, V, alpha, g, Isat),
        "square_mod_rk4_nl_rhs": lambda: kernels.square_mod_rk4_nl_rhs(
            k, A, alpha, g, Isat
        ),
        "square_mod_rk4_nl_rhs_v": lambda: kernels.square_mod_rk4_nl_rhs_v(
            k, A, V, alpha, g, Isat
        ),
        "rk4_nl_rhs_c": lambda: kernels.rk4_nl_rhs_c(
            k, A, A_sq, A_sq2, alpha, g, g12, Isat, Isat2
        ),
        "rk4_nl_rhs_c_v": lambda: kernels.rk4_nl_rhs_c_v(
            k, A, A_sq, A_sq2, V, alpha, g, g12, Isat, Isat2
        ),
        "rk4_set_and_axpy": lambda: kernels.rk4_set_and_axpy(acc, out, A, k, f(0.5)),
        "rk4_acc_and_axpy": lambda: kernels.rk4_acc_and_axpy(
            acc, out, A, k, f(0.5), f(0.5)
        ),
        "linear_step": lambda: kernels.linear_step(A, prop, plan),
        "split_step_fused": lambda: kernels.split_step_fused(
            A, prop, V, dz, alpha, g, Isat, "single", plan
        ),
        "rk4_rhs_fused": lambda: kernels.rk4_rhs_fused(
            A, k, V, prop, plan, alpha, g, Isat
        ),
        "split_step_rk4_fused": lambda: kernels.split_step_rk4_fused(
            A, prop, V, dz, alpha, g, Isat, plan
        ),
        "split_step_coupled_fused": lambda: kernels.split_step_coupled_fused(
            pair,
            pair_prop,
            V,
            V,
            dz,
            alpha,
            alpha,
            g,
            g12,
            g,
            Isat,
            Isat2,
            "single",
            pair_plan,
        ),
        "rk4_rhs_coupled_fused": lambda: kernels.rk4_rhs_coupled_fused(
            pair,
            pair.copy() if hasattr(pair, "copy") else pair,
            V,
            V,
            pair_prop,
            pair_plan,
            alpha,
            alpha,
            g,
            g12,
            g,
            Isat,
            Isat2,
        ),
    }
    return {n: c for n, c in candidates.items() if hasattr(kernels, n)}


def fft_call(backend, arrays, n, plan):
    """Return a callable doing one forward and one inverse transform."""
    A = arrays["A"]

    def roundtrip():
        out = backend.fft(A, plan)
        return backend.ifft(out, plan)

    return roundtrip


def profile(backend_name, n, get_backend):
    """Return {kernel: (seconds, GB/s)} for one backend at one size."""
    backend = get_backend(backend_name, grid_size=(n, n))
    arrays = make_arrays(backend, n)
    elements = n * n
    rows = {}

    # build_fft returns a list; the fused kernels take one plan object, which
    # is how the solvers call them too (plans[0]).
    plans = backend.build_fft((n, n), (-2, -1), np.complex64, array=arrays["A"])
    plan = plans[0]
    try:
        pair_plan = backend.build_fft(
            (2, n, n), (-2, -1), np.complex64, array=arrays["pair"]
        )[0]
    except Exception:
        pair_plan = plan

    calls = {"bandwidth_ceiling": ceiling_call(backend, arrays)}
    calls.update(kernel_calls(backend.kernels, arrays, plan, pair_plan))
    calls["fft_roundtrip"] = fft_call(backend, arrays, n, plans)

    for name, call in calls.items():
        reset(backend)
        try:
            seconds, median = timed(call, backend, lambda: arrays["A"])
        except Exception as exc:
            rows[name] = (None, f"{type(exc).__name__}: {exc}", 0.0)
            continue
        gbs = TRAFFIC[name] * elements / seconds / 1e9
        spread = (median - seconds) / seconds * 100
        rows[name] = (seconds, gbs, spread)
    return rows


def main(argv=None):
    """Parse arguments and print a table per backend and size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[1024, 2048])
    parser.add_argument("--backends", nargs="*")
    parser.add_argument(
        "--peak-gbs",
        type=float,
        help="memory bandwidth this machine is rated for, in GB/s",
    )
    args = parser.parse_args(argv)

    from NLSE.backends import get_backend, list_available_backends

    for backend_name in args.backends or list_available_backends():
        for n in args.sizes:
            rows = profile(backend_name, n, get_backend)
            rates = [
                gbs
                for name, (sec, gbs, _) in rows.items()
                if sec is not None and name != "fft_roundtrip"
            ]
            best = max(rates) if rates else 0.0

            mib = n * n * COMPLEX / 1024**2
            print(f"\n=== {backend_name}  {n}x{n}  ({mib:.0f} MiB per array) ===")
            print(
                f"{'kernel':<26} {'time':>10} {'GB/s':>9} {'of best':>9} {'spread':>8}"
            )
            listed = set()
            for group, names in GROUPS:
                present = [x for x in names if x in rows]
                if not present:
                    continue
                print(f"-- {group}")
                for name in present:
                    listed.add(name)
                    seconds, gbs, spread = rows[name]
                    if seconds is None:
                        print(f"   {name:<23} {gbs}")
                        continue
                    share = (
                        "    (n/a)"
                        if name in ("fft_roundtrip", "bandwidth_ceiling")
                        else f"{100 * gbs / best:7.0f} %"
                    )
                    print(
                        f"   {name:<23} {seconds * 1e3:8.3f}ms {gbs:8.1f} {share:>9}"
                        f" {spread:6.0f} %"
                    )
            missing = [x for x in rows if x not in listed]
            if missing:
                print(f"-- ungrouped: {', '.join(missing)}")

            line = f"best observed: {best:.0f} GB/s"
            if args.peak_gbs:
                line += f"  =  {100 * best / args.peak_gbs:.0f}% of {args.peak_gbs:.0f} GB/s rated"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
