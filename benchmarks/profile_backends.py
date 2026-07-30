#!/usr/bin/env python3
"""Time every backend on the same workload, optionally against an older revision.

    python benchmarks/profile_backends.py
    python benchmarks/profile_backends.py --baseline 3.0.0
    python benchmarks/profile_backends.py --baseline 3.0.0 --sizes 512 --solver NLSE

With ``--baseline`` the same workload runs twice: once against the working
tree, once against a git worktree of that revision, in a subprocess with the
older package on ``sys.path``. The two runs are compared per cell.

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


def time_cell(nlse_module, solver_name, backend, n, method, np):
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
    for _ in range(REPEATS):
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
                        nlse_module, args.solver, backend, n, method, np
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


def baseline_results(revision, args):
    """Run the same workload against ``revision`` in a throwaway worktree."""
    tmp = Path(tempfile.mkdtemp(prefix="nlse-baseline-"))
    worktree = tmp / "tree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), revision],
        check=True,
        capture_output=True,
    )
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--package-root",
            str(worktree),
            "--solver",
            args.solver,
            "--sizes",
            *[str(s) for s in args.sizes],
            "--methods",
            *args.methods,
            "--json",
            "-",
        ]
        if args.backends:
            cmd += ["--backends", *args.backends]
        env = dict(os.environ, NLSE_QUIET="1")
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        return json.loads(out.stdout)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)


def compare(now, before, revision):
    """Print a per-cell comparison, slowest regression first."""
    rows = []
    for key, cell in now.items():
        old = before.get(key, {})
        if "best" not in cell or "best" not in old:
            continue
        rows.append((cell["best"] / old["best"], key, old["best"], cell["best"]))
    rows.sort(reverse=True)

    print(f"\n{'cell':<44} {revision:>11} {'now':>11} {'ratio':>8}")
    print("-" * 78)
    for ratio, key, old, new in rows:
        flag = "  <-- slower" if ratio > 1.10 else ("  faster" if ratio < 0.90 else "")
        print(f"{key:<44} {old:8.3f}ms {new:8.3f}ms {ratio:7.2f}x{flag}")

    slower = [r for r in rows if r[0] > 1.10]
    print(
        f"\n{len(slower)} of {len(rows)} cells more than 10% slower than {revision}."
        if rows
        else "\nNothing comparable."
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
    parser.add_argument("--json", help="write results as JSON ('-' for stdout)")
    args = parser.parse_args(argv)

    if not args.json:
        print(
            f"{args.solver}: slope of {STEPS_LOW} vs {STEPS_HIGH} steps, "
            f"best of {REPEATS}\n"
        )
    results = run(args)

    if args.json == "-":
        json.dump(results, sys.stdout)
        return 0
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))

    if args.baseline:
        print(f"\nmeasuring {args.baseline} ...", flush=True)
        compare(results, baseline_results(args.baseline, args), args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
