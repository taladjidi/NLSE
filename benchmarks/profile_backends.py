#!/usr/bin/env python3
"""Time every backend on the same workload, optionally against an older revision.

    python benchmarks/profile_backends.py
    python benchmarks/profile_backends.py --baseline 3.0.0
    python benchmarks/profile_backends.py --baseline 3.0.0 --sizes 512 --solver NLSE

With ``--baseline`` the same workload runs against the working tree and
against a git worktree of that revision, both in subprocesses, alternating a
round at a time and swapping which side goes first. Measuring one side through
and then the other put the machine's drift entirely on one side: with identical
code on both sides that reported cells up to 1.19x slower, reproducibly. The
table reports, per cell, how much that cell moved between rounds of the same
code, and calls nothing a regression unless it moved by more than that.

Steps are fixed rather than derived from the physics: the default step has
changed since 3.0.0, and a run that takes a different number of steps measures
a different amount of work. What is compared is the cost of one step.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from inspect import signature
from pathlib import Path

# Held fixed so a cell is the same work at both revisions.
WAIST = 2.23e-3
POWER = 1.05
N2 = -1.6e-9
ISAT = 1e5
DELTA_Z = 1e-4
# Per-step cost is the slope between two step counts, so everything that
# happens once per run -- FFT planning, propagator build, host-to-device
# transfers -- cancels instead of being divided by the step count. Timing a
# single run and dividing measured mostly setup on the fast backends.
STEPS_LOW = 20
STEPS_HIGH = 220
REPEATS = 5
# Repeats and rounds when comparing two revisions. Measuring one side fully
# and then the other put every bit of drift between them onto one side: with
# *identical code* on both sides this reported 3 of 6 cells more than 10%
# slower, up to 1.19x, and reproduced the direction on a second run. So the
# sides alternate instead, one round each, and swap which goes first every
# round. Both also run as subprocesses now -- measuring the working tree in
# this process and the baseline in a child compared two different process
# states as much as two revisions. REPEATS_PER_ROUND x ROUNDS is kept near
# REPEATS so the total measurement is about what it was; the extra cost is
# warmup, paid once per round per side rather than once per side.
REPEATS_PER_ROUND = 2
ROUNDS = 3

DEFAULT_SIZES = (128, 256, 512)
DEFAULT_METHODS = ("split_step", "RK4")


def load_package(root: str | None):
    """Import NLSE, from ``root`` if given, and say where it came from.

    Parameters
    ----------
    root : str or None
        Directory containing the ``NLSE`` package. None uses the default
        import path.

    Returns
    -------
    module
        The imported NLSE package.
    """
    if root is not None:
        sys.path.insert(0, root)
    import NLSE

    where = Path(NLSE.__file__).resolve().parent
    if root is not None and Path(root).resolve() not in where.parents:
        raise SystemExit(
            f"asked for the package in {root} but imported {where}. "
            f"Refusing to report a comparison against the wrong tree."
        )
    return NLSE


def build(nlse_module, solver_name, backend, n):
    """Return a solver of the requested kind on the requested backend."""
    cls = getattr(nlse_module, solver_name)
    kwargs = {
        "alpha": 0.0,
        "power": POWER,
        "window": 4 * WAIST,
        "n2": N2,
        "V": None,
        "L": 1e-2,
        "NX": n,
        "Isat": ISAT,
        "backend": backend,
    }
    if solver_name != "NLSE_1d":
        kwargs["NY"] = n
    if solver_name.startswith("CNLSE"):
        kwargs["n12"] = -1e-10
    return cls(**kwargs)


def input_field(simu, solver_name, np):
    """Return a Gaussian of the shape this solver expects."""
    if solver_name.endswith("_1d"):
        profile = np.exp(-(simu.X**2) / WAIST**2)
    else:
        profile = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2)
    field = profile.astype(np.complex64)
    if solver_name.startswith("CNLSE"):
        field = np.array([field, 0.5 * field])
    return field


def propagate(simu, field, method, steps):
    """Run ``steps`` steps, however this revision wants the step given."""
    kwargs = {"verbose": False, "plot": False, "method": method}
    if "delta_z" in signature(simu.out_field).parameters:
        kwargs["delta_z"] = DELTA_Z
    else:  # 3.0.0 and earlier carried the step on the solver
        simu.delta_z = DELTA_Z
    return simu.out_field(field.copy(), steps * DELTA_Z, **kwargs)


def time_run(nlse_module, solver_name, backend, n, method, steps, field, np):
    """Return seconds for one whole run of ``steps`` steps."""
    simu = build(nlse_module, solver_name, backend, n)
    sync = getattr(simu._backend, "synchronize", None)
    start = time.perf_counter()
    out = propagate(simu, field, method, steps)
    if sync is not None:
        sync(out)
    elif simu._backend.is_device_backend and not isinstance(out, np.ndarray):
        # Older revisions have no synchronize. Pulling the result forces the
        # work; a result already on the host means it is done.
        simu._backend.to_numpy(out)
    return time.perf_counter() - start


def time_cell(nlse_module, solver_name, backend, n, method, np, repeats=REPEATS):
    """Return milliseconds per step, from the slope between two step counts.

    The two runs differ only in how many steps they take, so subtracting them
    removes the once-per-run cost entirely. Repeats alternate between the two
    lengths so that any drift over the measurement lands on both.
    """
    simu = build(nlse_module, solver_name, backend, n)
    field = input_field(simu, solver_name, np)
    for steps in (STEPS_LOW, STEPS_HIGH):  # warm plans, JIT, autotuning
        time_run(nlse_module, solver_name, backend, n, method, steps, field, np)

    lows, highs = [], []
    for _ in range(repeats):
        lows.append(
            time_run(nlse_module, solver_name, backend, n, method, STEPS_LOW, field, np)
        )
        highs.append(
            time_run(
                nlse_module, solver_name, backend, n, method, STEPS_HIGH, field, np
            )
        )

    per_step = (min(highs) - min(lows)) / (STEPS_HIGH - STEPS_LOW) * 1e3
    spread = (
        (statistics.median(highs) - statistics.median(lows))
        / (STEPS_HIGH - STEPS_LOW)
        * 1e3
    )
    return per_step, spread


def run(args):
    """Measure every requested cell and return the results as a dict."""
    nlse_module = load_package(args.package_root)
    import numpy as np
    from NLSE.backends import list_available_backends

    backends = args.backends or list_available_backends()
    results = {}
    for backend in backends:
        for n in args.sizes:
            for method in args.methods:
                key = f"{args.solver}/{backend}/{n}/{method}"
                try:
                    best, median = time_cell(
                        nlse_module, args.solver, backend, n, method, np, args.repeats
                    )
                    results[key] = {"best": best, "median": median}
                except Exception as exc:  # a backend may not support a case
                    results[key] = {"error": f"{type(exc).__name__}: {exc}"}
                if not args.json:
                    cell = results[key]
                    shown = (
                        f"{cell['best']:8.3f} ms/step"
                        if "best" in cell
                        else cell["error"]
                    )
                    print(f"  {key:44s} {shown}", flush=True)
    return results


def measure_tree(root, args):
    """Measure every cell once against the package at ``root``, in a subprocess.

    Parameters
    ----------
    root : str
        Directory containing the ``NLSE`` package to import.
    args : argparse.Namespace
        The parsed command line, for the workload to measure.

    Returns
    -------
    dict
        The same mapping ``run`` returns.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--package-root",
        root,
        "--solver",
        args.solver,
        "--sizes",
        *[str(s) for s in args.sizes],
        "--methods",
        *args.methods,
        "--repeats",
        str(REPEATS_PER_ROUND),
        "--json",
        "-",
    ]
    if args.backends:
        cmd += ["--backends", *args.backends]
    env = dict(os.environ, NLSE_QUIET="1")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(out.stdout)


def against_baseline(revision, args):
    """Measure the working tree and ``revision`` in alternating rounds.

    Returns
    -------
    tuple of dict
        Per-cell best for the working tree, the same for ``revision``, and the
        round-to-round spread of each cell, which is what says whether a
        difference means anything.
    """
    tmp = Path(tempfile.mkdtemp(prefix="nlse-baseline-"))
    worktree = tmp / "tree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), revision],
        check=True,
        capture_output=True,
    )
    here = str(Path(__file__).resolve().parent.parent)
    samples = {"now": [], "old": []}
    try:
        for r in range(ROUNDS):
            # Swap which side goes first, so being first is not worth anything
            # over the whole measurement.
            order = ("now", "old") if r % 2 == 0 else ("old", "now")
            for side in order:
                print(f"  round {r + 1}/{ROUNDS}, {side} ...", flush=True)
                root = here if side == "now" else str(worktree)
                samples[side].append(measure_tree(root, args))
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)

    best, noise = {}, {}
    for side in ("now", "old"):
        best[side] = {}
        for key in samples[side][0]:
            got = [r[key]["best"] for r in samples[side] if "best" in r.get(key, {})]
            if got:
                best[side][key] = min(got)
                # How much this cell moved between rounds of the *same* code.
                noise[key] = max(noise.get(key, 0.0), (max(got) - min(got)) / min(got))
    return best["now"], best["old"], noise


def compare(now, before, noise, revision):
    """Print a per-cell comparison, slowest regression first.

    A cell is only called slower or faster when it moved by more than this
    machine moved the same code between rounds. Without that the table
    reports the noise as a result.
    """
    rows = []
    for key, new in now.items():
        old = before.get(key)
        if old is None:
            continue
        rows.append((new / old, key, old, new, noise.get(key, 0.0)))
    rows.sort(reverse=True)

    print(f"\n{'cell':<40} {revision:>10} {'now':>10} {'ratio':>7} {'noise':>7}")
    print("-" * 78)
    for ratio, key, old, new, cell_noise in rows:
        floor = max(0.10, cell_noise)
        flag = ""
        if ratio > 1 + floor:
            flag = "  <-- slower"
        elif ratio < 1 - floor:
            flag = "  faster"
        print(
            f"{key:<40} {old:7.3f}ms {new:7.3f}ms "
            f"{ratio:6.2f}x {cell_noise * 100:5.1f}% {flag}"
        )

    if not rows:
        print("\nNothing comparable.")
        return
    slower = [r for r in rows if r[0] > 1 + max(0.10, r[4])]
    worst = max(r[4] for r in rows)
    print(
        f"\n{len(slower)} of {len(rows)} cells slower than {revision} by more than "
        f"this machine's own scatter."
    )
    print(
        f"Worst cell moved {worst * 100:.0f}% between rounds of identical code; "
        f"anything under that is not a result."
    )


def main(argv=None):
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="git revision to compare against")
    parser.add_argument("--package-root", help="import NLSE from here (internal)")
    parser.add_argument("--solver", default="NLSE")
    parser.add_argument("--backends", nargs="*")
    parser.add_argument("--sizes", nargs="*", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--repeats", type=int, default=REPEATS, help="timed repeats per cell (internal)"
    )
    parser.add_argument("--json", help="write results as JSON ('-' for stdout)")
    args = parser.parse_args(argv)

    if args.baseline:
        # This process measures nothing itself: both sides go through the same
        # kind of subprocess, alternating, so that neither the process a run
        # happens in nor the order it happens in can be read as a difference
        # between the revisions.
        print(
            f"{args.solver}: slope of {STEPS_LOW} vs {STEPS_HIGH} steps, "
            f"best of {REPEATS_PER_ROUND} x {ROUNDS} alternating rounds\n"
        )
        now, before, noise = against_baseline(args.baseline, args)
        if args.json and args.json != "-":
            Path(args.json).write_text(json.dumps(now, indent=2))
        compare(now, before, noise, args.baseline)
        return 0

    if not args.json:
        print(
            f"{args.solver}: slope of {STEPS_LOW} vs {STEPS_HIGH} steps, "
            f"best of {args.repeats}\n"
        )
    results = run(args)

    if args.json == "-":
        json.dump(results, sys.stdout)
        return 0
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
