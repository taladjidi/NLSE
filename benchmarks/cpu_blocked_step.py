#!/usr/bin/env python3
"""Measure the CPU split step as tuned on this machine, against the step before it.

    python benchmarks/cpu_blocked_step.py
    python benchmarks/cpu_blocked_step.py --sizes 1024 2048 4096 --dtypes complex128

Each grid is tuned first (``NLSE.backends.tune_cpu_step``), from the cache if
it holds the grid, and the table says what the tuning chose: blocks of so many
rows on so many threads, "unblocked", or "batch", which the blocked step
declines. "before" is the solver without the hook
(``CPUBackend.has_linear_nonlinear_step = False``), the code as it was; "after"
is with it, so on rows it declines, the gain is the cost of asking.

Both sides run in this process, alternating which goes first. Per-step cost is
the slope between two step counts, so the solver's setup cancels, as in
propagator_fusion.py. "hook" is what one declined call of the hook costs,
timed alone over HOOK_CALLS calls: a number that survives a noisy machine.
"""

import argparse
import platform
import sys
import time
import warnings

import numba
import numpy as np
import scipy
from llvmlite import binding as llvm
from NLSE import NLSE
from NLSE.backends import cpu_tuning, tune_cpu_step
from NLSE.backends.cpu import CPUBackend

WAIST = 2.23e-3  # m
POWER = 1.05  # W
N2 = -1.6e-9  # m^2/W
ISAT = 1e5  # W/m^2
# Step at 1024^2, scaled by the cell area below: the split step's dispersion
# cap scales so (NLSE/solvers/step_size.py, _dispersion_max_dz), and a capped
# step is taken as several, which the slope would count as one.
DELTA_Z_1024 = 5e-5  # m
STEPS_LOW = 10
STEPS_HIGH = 60  # 50 steps in the slope
ROUNDS = 3
BATCH = 3  # members of the batch row, on the smallest grid asked
HOOK_CALLS = 1000
HOOK_STEP = 1  # callback index the hook is timed at, after the first step
US_PER_S = 1e6
MS_PER_S = 1e3


def build(n):
    """Return an NLSE on an n x n grid, on the CPU backend."""
    return NLSE(
        alpha=1.0,
        power=POWER,
        window=4 * WAIST,
        n2=N2,
        V=None,
        L=10e-3,
        Isat=ISAT,
        NX=n,
        NY=n,
        backend="CPU",
    )


def field(n, dtype, batch):
    """Return a Gaussian beam on the grid, ``batch`` copies of it if not 0."""
    x = np.linspace(-2 * WAIST, 2 * WAIST, n)
    xx, yy = np.meshgrid(x, x)
    beam = np.exp(-(xx**2 + yy**2) / WAIST**2).astype(dtype)
    return np.stack([beam] * batch) if batch else beam


def step_of(n):
    """Return the step on an n x n grid, below its dispersion cap."""
    return DELTA_Z_1024 * (1024 / n) ** 2


def time_run(n, steps, dtype, splitting, batch):
    """Return the wall time of a propagation of this many steps."""
    simu = build(n)
    E = field(n, dtype, batch)
    start = time.perf_counter()
    simu.out_field(
        E,
        steps * step_of(n),
        delta_z=step_of(n),
        splitting=splitting,
        verbose=False,
        plot=False,
    )
    return time.perf_counter() - start


def per_step(n, hooked, dtype, splitting, batch):
    """Return the cost of one step, as the slope between two step counts."""
    CPUBackend.has_linear_nonlinear_step = hooked
    time_run(n, STEPS_LOW, dtype, splitting, batch)  # warm: compiles, pools
    low = time_run(n, STEPS_LOW, dtype, splitting, batch)
    high = time_run(n, STEPS_HIGH, dtype, splitting, batch)
    return (high - low) / (STEPS_HIGH - STEPS_LOW)


def measure(n, dtype, splitting, batch):
    """Return per-step costs for both sides, alternating which goes first."""
    sides: dict = {"before": [], "after": []}
    for i in range(ROUNDS):
        order = ("before", "after") if i % 2 == 0 else ("after", "before")
        for name in order:
            sides[name].append(per_step(n, name == "after", dtype, splitting, batch))
    return sides


def time_hook(simu, A, z, i, plans, timing):
    """Time the solver's call of the hook, HOOK_CALLS times, as a callback."""
    if i != HOOK_STEP:
        return
    args = (A, None, simu.propagator, plans, step_of(simu.NX))
    start = time.perf_counter()
    for _ in range(HOOK_CALLS):
        taken = simu._linear_nonlinear_step(*args)
    timing["us"] = (time.perf_counter() - start) / HOOK_CALLS * US_PER_S
    timing["taken"] = taken


def hook_cost(n, dtype, batch):
    """Return the hook's cost per call where it declines, or None if it takes."""
    CPUBackend.has_linear_nonlinear_step = True
    simu = build(n)
    E = field(n, dtype, batch)
    plans = simu._backend.build_fft(E.shape, axes=(-2, -1), dtype=E.dtype)
    timing: dict = {}
    simu.out_field(
        E,
        (HOOK_STEP + 2) * step_of(n),
        delta_z=step_of(n),
        verbose=False,
        plot=False,
        callback=time_hook,
        callback_args=(plans, timing),
    )
    return None if timing["taken"] else timing["us"]


def chosen(n, dtype, batch):
    """Tune the grid and return what the table says it was tuned as."""
    entry = tune_cpu_step((n, n), dtype)
    if batch:
        return f"batch of {batch}"
    if entry is None:
        return "not tuned"  # another process is tuning it
    if entry["rows"] is None:
        return "unblocked"
    return f"{entry['rows']} rows x {entry['threads']}"


def row(n, dtype, splitting, batch):
    """Measure one grid and print its row."""
    tuned_as = chosen(n, dtype, batch)
    sides = measure(n, dtype, splitting, batch)
    hook = hook_cost(n, dtype, batch)
    before, after = min(sides["before"]), min(sides["after"])
    # How much a side moved between rounds of the same code: the smallest
    # difference this machine can be said to have resolved.
    noise = max(max(v) / min(v) - 1 for v in sides.values())
    hook_text = "taken" if hook is None else f"{hook:.2f}us"
    print(
        f"{n:5d}^2 {dtype.name:10s} {tuned_as:16s} {before * MS_PER_S:9.3f}ms "
        f"{after * MS_PER_S:8.3f}ms {before / after:5.2f}x {noise:6.1%} "
        f"{hook_text:>9s}",
        flush=True,
    )


def header(splitting):
    """Print the machine and the libraries the table was measured with."""
    print(
        f"{platform.system()} {platform.machine()} ({llvm.get_host_cpu_name()}), "
        f"{cpu_tuning.available_threads()} threads; numpy {np.__version__}, "
        f"scipy {scipy.__version__}, numba {numba.__version__}"
    )
    print(
        f"NLSE, split_step {splitting}, loss and saturation, {ROUNDS} rounds, "
        f"per-step cost from the {STEPS_LOW}->{STEPS_HIGH} step slope"
    )
    print(
        f"\n{'grid':>7s} {'dtype':10s} {'tuned as':16s} {'before':>10s} "
        f"{'after':>9s} {'gain':>6s} {'noise':>7s} {'hook':>9s}"
    )


def main(argv=None):
    """Tune each grid, then time the step before and after the hook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[256, 1024, 2048, 4096])
    parser.add_argument(
        "--dtypes",
        nargs="*",
        choices=("complex64", "complex128"),
        default=["complex64", "complex128"],
    )
    parser.add_argument("--splitting", choices=("lie", "strang"), default="lie")
    args = parser.parse_args(argv)
    warnings.filterwarnings("error", message="delta_z=")  # no step cut in several
    warnings.filterwarnings("ignore", message="splitting=")  # advice, not timing
    header(args.splitting)
    try:
        for dtype in map(np.dtype, args.dtypes):
            row(min(args.sizes), dtype, args.splitting, BATCH)
            for n in args.sizes:
                row(n, dtype, args.splitting, 0)
    finally:
        CPUBackend.has_linear_nonlinear_step = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
