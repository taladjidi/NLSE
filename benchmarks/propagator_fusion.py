#!/usr/bin/env python3
"""Measure what folding the propagator into the transform is worth.

    python benchmarks/propagator_fusion.py
    python benchmarks/propagator_fusion.py --sizes 1024 2048 --solver NLSE

The CUPY backend can apply the propagator from a cuFFT store callback, as the
forward transform writes each element, instead of in a pass of its own
(NLSE/kernels/cuda_source/fft_callbacks.cu). What that saves is traffic: a step
that read the field, read the propagator and wrote the field back now reads the
propagator inside a write that was happening anyway. On a backend at the
bandwidth bound, that is the whole of the difference.

Both sides run in this process, alternating which goes first, because the
machine drifts by more than the effect. Per-step cost is the slope between two
step counts, so planning, the propagator build and the transfers cancel rather
than being divided by the step count -- the same reason profile_backends.py
does it that way.

``NLSE_FUSE_PROPAGATOR`` is read when a plan is built and plans are cached on
the backend, so switching sides means clearing them.
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np
from NLSE import CNLSE, NLSE
from NLSE.backends import get_backend

WAIST = 2.23e-3
POWER = 1.05
N2 = -1.6e-9
ISAT = 1e5
DELTA_Z = 1e-4
STEPS_LOW = 20
STEPS_HIGH = 220
ROUNDS = 4

SOLVERS = {"NLSE": NLSE, "CNLSE": CNLSE}


def build(cls, n):
    """Return a solver of this class on an n x n grid."""
    kwargs = {
        "alpha": 1.0,
        "power": POWER,
        "window": 4 * WAIST,
        "n2": N2,
        "V": None,
        "L": 10e-3,
        "Isat": ISAT,
        "NX": n,
        "NY": n,
        "backend": "CUPY",
    }
    if cls is CNLSE:
        kwargs["n12"] = N2 / 10
    return cls(**kwargs)


def field(cls, n, dtype):
    """Return a smooth field of the shape this solver takes."""
    x = np.linspace(-2 * WAIST, 2 * WAIST, n)
    xx, yy = np.meshgrid(x, x)
    beam = np.exp(-(xx**2 + yy**2) / WAIST**2).astype(dtype)
    return np.stack([beam, beam]) if cls is CNLSE else beam


def time_run(cls, n, steps, dtype, **kw):
    """Return the wall time of a propagation of this many steps."""
    simu = build(cls, n)
    E = field(cls, n, dtype)
    start = time.perf_counter()
    simu.out_field(E, steps * DELTA_Z, delta_z=DELTA_Z, verbose=False, plot=False, **kw)
    return time.perf_counter() - start


def per_step(cls, n, fuse, dtype, **kw):
    """Return the cost of one step, as the slope between two step counts."""
    os.environ["NLSE_FUSE_PROPAGATOR"] = "1" if fuse else "0"
    get_backend("CUPY").clear_fft_plans()
    # Once through at the low count first: the first run of a size compiles
    # kernels and the callback, which belongs to neither point on the line.
    time_run(cls, n, STEPS_LOW, dtype, **kw)
    low = time_run(cls, n, STEPS_LOW, dtype, **kw)
    high = time_run(cls, n, STEPS_HIGH, dtype, **kw)
    return (high - low) / (STEPS_HIGH - STEPS_LOW)


def fused_kinds(cls, n, dtype, **kw):
    """Return what the plan ended up fusing, to prove the run took the path."""
    os.environ["NLSE_FUSE_PROPAGATOR"] = "1"
    get_backend("CUPY").clear_fft_plans()
    simu = build(cls, n)
    simu.out_field(
        field(cls, n, dtype), 3 * DELTA_Z, delta_z=DELTA_Z, verbose=False, plot=False
    )
    fused = getattr(simu.plans[0], "_fused", {})
    return ",".join("batched" if k else "direct" for k in fused) or "none"


def measure(cls, n, dtype, **kw):
    """Return per-step costs for both sides, alternating which goes first."""
    sides: dict = {"unfused": [], "fused": []}
    for i in range(ROUNDS):
        order = ("unfused", "fused") if i % 2 == 0 else ("fused", "unfused")
        for name in order:
            sides[name].append(per_step(cls, n, name == "fused", dtype, **kw))
    return sides


def main(argv=None):
    """Time the fused and unfused linear step, and say what moved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="NLSE")
    parser.add_argument("--method", choices=("split_step", "RK4"), default="split_step")
    parser.add_argument("--splitting", choices=("lie", "strang"), default="lie")
    parser.add_argument("--double", action="store_true", help="propagate complex128")
    args = parser.parse_args(argv)

    cls = SOLVERS[args.solver]
    dtype = np.complex128 if args.double else np.complex64
    kw = {"method": args.method, "splitting": args.splitting}
    if args.method == "RK4":
        kw.pop("splitting")

    print(
        f"{args.solver}, {args.method}, {np.dtype(dtype).name}, "
        f"{ROUNDS} rounds, per-step cost from the "
        f"{STEPS_LOW}->{STEPS_HIGH} step slope"
    )
    print(
        f"\n{'grid':>7s} {'fused what':11s} {'unfused':>9s} {'fused':>9s} "
        f"{'gain':>6s} {'noise':>6s}"
    )
    for n in args.sizes:
        kinds = fused_kinds(cls, n, dtype, **kw)
        sides = measure(cls, n, dtype, **kw)
        unfused, fused = min(sides["unfused"]), min(sides["fused"])
        # How much a side moved between rounds of the same code, which is the
        # smallest difference this machine can be said to have resolved.
        noise = max(max(v) / min(v) - 1 for v in sides.values())
        print(
            f"{n:5d}^2 {kinds:11s} {unfused * 1e3:8.3f}ms {fused * 1e3:8.3f}ms "
            f"{unfused / fused:5.2f}x {noise:5.1%}"
        )
        if statistics.mean(sides["fused"]) > statistics.mean(sides["unfused"]):
            print("        the fused side was slower here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
