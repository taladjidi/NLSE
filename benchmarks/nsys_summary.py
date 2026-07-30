#!/usr/bin/env python3
"""Summarise an nsys report against the phases trace_solvers.py names.

    python benchmarks/nsys_summary.py report1.nsys-rep
    python benchmarks/nsys_summary.py report1.nsys-rep --top 20

nsys sees what the Python-level tracer cannot: the kernels inside a replayed
CUDA graph, the split between GPU time and driver time, and the gaps where
the GPU has nothing to do. This runs `nsys stats`, folds the kernel names into
the same phases, and reports what fraction of the wall clock the GPU was busy.

`nsys stats` report names have changed between releases, so each table is
looked up under every name it has had. Nothing is inferred from a table that
did not come back: a missing one is said to be missing.

Without a report to hand, --from-csv reads a CSV that `nsys stats --format
csv` produced earlier, which is also how the parsing is tested.
"""

import argparse
import csv
import io
import re
import shutil
import subprocess

# nsys renamed its reports; try each spelling in turn.
REPORTS = {
    "kernels": ("cuda_gpu_kern_sum", "gpukernsum"),
    "api": ("cuda_api_sum", "apisum"),
    "nvtx": ("nvtx_sum", "nvtxsum"),
    "memory": ("cuda_gpu_mem_time_sum", "gpumemtimesum"),
}

# Substrings of a CUDA kernel name -> the phase it belongs to. Matched in
# order, so put the specific ones first.
KERNEL_PHASES = [
    ("fft", "transform"),
    ("split_step_coupled", "whole step (fused)"),
    ("split_step_rk4", "whole step (fused)"),
    ("split_step", "whole step (fused)"),
    ("rk4_rhs", "RK4 rhs (fused)"),
    ("rk4_nl_rhs", "RK4 rhs"),
    ("rk4_set_and_axpy", "RK4 stage"),
    ("rk4_acc_and_axpy", "RK4 stage"),
    ("rk4_axpy", "RK4 stage"),
    ("rk4_accumulate", "RK4 stage"),
    ("linear_step", "linear (fused)"),
    ("apply_propagator", "linear"),
    ("nl_prop", "nonlinear"),
    ("square_mod", "nonlinear"),
    ("rabi", "nonlinear"),
    ("copy", "array copies"),
    ("memcpy", "array copies"),
    ("elementwise", "cupy elementwise"),
]


def phase_of(name):
    """Return the phase a CUDA kernel name belongs to."""
    lowered = name.lower()
    for needle, phase in KERNEL_PHASES:
        if needle in lowered:
            return phase
    return "other"


def run_stats(report, names):
    """Return CSV text for the first report name nsys recognises, or None."""
    for name in names:
        result = subprocess.run(
            ["nsys", "stats", "--report", name, "--format", "csv", report],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "," in result.stdout:
            return result.stdout
    return None


def parse(text):
    """Return rows as dicts from the first CSV table in nsys's output.

    nsys prints a heading and a blank line before each table, so the header
    is the first line that has commas and a recognisable time column.

    Parameters
    ----------
    text : str
        Output of ``nsys stats --format csv``.

    Returns
    -------
    list of dict
        One dict per row, keyed by column name.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "," in line and re.search(r"Time \(%\)|Total Time", line):
            start = i
            break
    if start is None:
        return []
    body = "\n".join(lines[start:])
    rows = []
    for row in csv.DictReader(io.StringIO(body)):
        if row.get("Time (%)") in (None, ""):
            break  # a blank line or the next table's heading
        rows.append(row)
    return rows


def column(row, *candidates):
    """Return the first present column among candidates."""
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def as_float(value):
    """Return a float from an nsys cell, which may carry thousands commas."""
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def summarise_kernels(rows, top):
    """Print per-kernel and per-phase GPU time."""
    entries = []
    for row in rows:
        name = column(row, "Name", "Kernel Name", "Operation") or "?"
        entries.append(
            (
                as_float(column(row, "Total Time (ns)", "Total Time")),
                int(as_float(column(row, "Instances", "Num Calls", "Count"))),
                name,
            )
        )
    total = sum(e[0] for e in entries)
    if not total:
        print("  no GPU kernel rows")
        return 0.0

    by_phase = {}
    for ns, count, name in entries:
        phase = phase_of(name)
        seconds, calls = by_phase.get(phase, (0.0, 0))
        by_phase[phase] = (seconds + ns, calls + count)

    print(f"\n  GPU kernel time by phase  (total {total / 1e6:.1f} ms)")
    print(f"  {'phase':<22} {'time':>10} {'share':>7} {'launches':>9}")
    for phase, (ns, calls) in sorted(by_phase.items(), key=lambda x: -x[1][0]):
        print(f"  {phase:<22} {ns / 1e6:8.2f}ms {100 * ns / total:6.1f} % {calls:9d}")

    print(f"\n  slowest {top} kernels")
    print(f"  {'kernel':<52} {'time':>10} {'launches':>9} {'per launch':>11}")
    for ns, count, name in sorted(entries, reverse=True)[:top]:
        short = name if len(name) <= 50 else name[:47] + "..."
        per = ns / count if count else 0.0
        print(f"  {short:<52} {ns / 1e6:8.2f}ms {count:9d} {per / 1e3:9.1f}us")
    return total


def traced_run(rows):
    """Whether these NVTX ranges came from trace_solvers.py.

    Its ranges are kernel names, and it synchronizes after each one, so the
    CUDA API time in such a report is mostly the cost of being watched.
    """
    kernels = {"linear_step", "apply_propagator", "square_mod", "rk4_axpy"}
    for row in rows:
        name = (column(row, "Range", "Name", "Operation") or "").lstrip(":")
        if name in kernels:
            return True
    return False


def summarise_nvtx(rows):
    """Print the NVTX ranges trace_solvers.py pushed, if any."""
    if not rows:
        print("\n  no NVTX ranges (run trace_solvers.py with --nvtx)")
        return
    print("\n  NVTX ranges")
    print(f"  {'range':<32} {'wall time':>11} {'instances':>10}")
    for row in rows[:20]:
        name = column(row, "Range", "Name", "Operation") or "?"
        ns = as_float(column(row, "Total Time (ns)", "Total Time"))
        count = int(as_float(column(row, "Instances", "Num Calls", "Count")))
        print(f"  {name:<32} {ns / 1e6:9.2f}ms {count:10d}")


def main(argv=None):
    """Parse arguments and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", help="path to a .nsys-rep file")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--from-csv",
        help="read one already-produced `nsys stats --format csv` table instead",
    )
    args = parser.parse_args(argv)

    if args.from_csv:
        rows = parse(open(args.from_csv).read())
        summarise_kernels(rows, args.top)
        return 0

    if not args.report:
        parser.error("give a .nsys-rep path, or --from-csv")
    if shutil.which("nsys") is None:
        print("nsys is not on PATH. Run this where the report was made, or")
        print("pass a CSV that `nsys stats --format csv` produced, via --from-csv.")
        return 1

    tables = {}
    for key, names in REPORTS.items():
        text = run_stats(args.report, names)
        tables[key] = parse(text) if text else None
        if tables[key] is None:
            print(f"  ({key} table unavailable: tried {', '.join(names)})")

    print(f"=== {args.report} ===")
    gpu_ns = summarise_kernels(tables["kernels"] or [], args.top)

    if tables["memory"]:
        moved = sum(
            as_float(column(r, "Total Time (ns)", "Total Time"))
            for r in tables["memory"]
        )
        print(f"\n  GPU memory operations: {moved / 1e6:.2f} ms")
        if gpu_ns:
            print(f"  {100 * moved / (gpu_ns + moved):.1f}% of GPU time is transfers")

    if tables["api"]:
        api_ns = sum(
            as_float(column(r, "Total Time (ns)", "Total Time")) for r in tables["api"]
        )
        print(f"\n  CUDA API time: {api_ns / 1e6:.2f} ms")
        if api_ns:
            print(f"  GPU kernel time / CUDA API time: {gpu_ns / api_ns:.2f}")
            if traced_run(tables["nvtx"] or []):
                print(
                    "  This report is of a trace_solvers.py run, which synchronizes\n"
                    "  after every kernel and steps from Python instead of replaying\n"
                    "  a CUDA graph. Most of that API time is the tracing. Profile\n"
                    "  without --nvtx, or profile a plain script, to judge the ratio."
                )
            else:
                print("  A ratio well below 1 means the GPU is waiting on the driver.")

    summarise_nvtx(tables["nvtx"] or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
