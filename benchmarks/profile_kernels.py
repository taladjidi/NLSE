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

# Kernel -> bytes touched per element, counting each array once. A kernel that
# reads an array it also writes is charged for both.
TRAFFIC = {
    "bandwidth_ceiling": 2 * COMPLEX,
    "square_mod": COMPLEX + REAL,
    "apply_propagator": 3 * COMPLEX,
    "nl_prop_without_V": 2 * COMPLEX + REAL,
    "nl_prop": 2 * COMPLEX + 2 * REAL,
    "square_mod_nl_prop": 2 * COMPLEX,
    "square_mod_nl_prop_v": 2 * COMPLEX + REAL,
    "fft_roundtrip": 4 * COMPLEX,  # a lower bound; see the module docstring
}


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
    """Return the best seconds for one call.

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
    """Return the field, its modulus squared and a propagator, on the device."""
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
    return {
        "A": backend.from_numpy(a),
        "B": backend.from_numpy(a.copy()),
        "A_sq": backend.from_numpy(np.abs(a) ** 2),
        "prop": backend.from_numpy(np.exp(1j * a.real).astype(np.complex64)),
        "V": backend.from_numpy(a.real.copy()),
    }


def ceiling_call(backend, arrays):
    """Return a callable scaling one array into another, and its result."""
    name = backend.name
    a, b = arrays["A"], arrays["B"]
    if name == "MLX":
        import mlx.core as mx

        return lambda: mx.multiply(a, 2.0)
    if name == "CUPY":
        import cupy as cp

        return lambda: cp.multiply(a, np.complex64(2.0), out=b)
    if name == "CL":
        import pyopencl.array as cla  # noqa: F401

        return lambda: (b._axpbz(b, np.complex64(2.0), a, np.complex64(0.0)), b)[1]
    return lambda: np.multiply(a, np.complex64(2.0), out=b)


def kernel_calls(kernels, arrays, dz=1e-4):
    """Return {name: callable} for every kernel this backend provides."""
    A, A_sq, prop, V = arrays["A"], arrays["A_sq"], arrays["prop"], arrays["V"]
    alpha, g, Isat = np.float32(0.0), np.float32(1e-3), np.float32(1e5)
    candidates = {
        "square_mod": lambda: kernels.square_mod(A, A_sq),
        "apply_propagator": lambda: kernels.apply_propagator(A, prop),
        "nl_prop_without_V": lambda: kernels.nl_prop_without_V(
            A, A_sq, dz, alpha, g, Isat
        ),
        "nl_prop": lambda: kernels.nl_prop(A, A_sq, dz, alpha, V, g, Isat),
        "square_mod_nl_prop": lambda: kernels.square_mod_nl_prop(A, dz, alpha, g, Isat),
        "square_mod_nl_prop_v": lambda: kernels.square_mod_nl_prop_v(
            A, V, dz, alpha, g, Isat
        ),
    }
    return {n: c for n, c in candidates.items() if hasattr(kernels, n)}


def fft_call(backend, arrays, n):
    """Return a callable doing one forward and one inverse transform."""
    A = arrays["A"]
    plan = backend.build_fft((n, n), (-2, -1), np.complex64, array=A)

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

    calls = {"bandwidth_ceiling": ceiling_call(backend, arrays)}
    calls.update(kernel_calls(backend.kernels, arrays))
    calls["fft_roundtrip"] = fft_call(backend, arrays, n)

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
                f"{'kernel':<24} {'time':>10} {'GB/s':>9} {'of best':>9} {'spread':>8}"
            )
            print("-" * 65)
            for name, (seconds, gbs, spread) in rows.items():
                if seconds is None:
                    print(f"{name:<24} {gbs}")
                    continue
                share = (
                    "    (n/a)"
                    if name == "fft_roundtrip"
                    else f"{100 * gbs / best:7.0f} %"
                )
                print(
                    f"{name:<24} {seconds * 1e3:8.3f}ms {gbs:8.1f} {share:>9}"
                    f" {spread:6.0f} %"
                )

            # What one split-step costs, from the parts, when nl_length is 0:
            # a transform pair, the propagator multiply, and the fused
            # nonlinear step. Backends with a fused whole-step kernel do it in
            # one launch, so this is the budget rather than the measured step.
            budget = {
                k: rows[k][0]
                for k in ("fft_roundtrip", "apply_propagator", "square_mod_nl_prop")
                if k in rows and rows[k][0] is not None
            }
            if len(budget) == 3:
                total = sum(budget.values())
                parts = "  ".join(
                    f"{k.replace('_roundtrip', '').replace('square_mod_', '')}"
                    f" {100 * v / total:.0f}%"
                    for k, v in budget.items()
                )
                print(f"step budget ~{total * 1e3:.2f} ms:  {parts}")

            line = f"best observed: {best:.0f} GB/s"
            if args.peak_gbs:
                line += f"  =  {100 * best / args.peak_gbs:.0f}% of {args.peak_gbs:.0f} GB/s rated"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
