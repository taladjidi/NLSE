"""Tests of the tools in benchmarks/.

nsys_summary, propagator_fusion, roofline and trace_solvers.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import cpu_tuning

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

nsys_summary = pytest.importorskip("nsys_summary")
# A plain import: the repository's own tool failing to import is a failure.
propagator_fusion = importlib.import_module("propagator_fusion")
roofline = importlib.import_module("roofline")
trace_solvers = importlib.import_module("trace_solvers")

# What `nsys stats --report cuda_gpu_kern_sum --format csv` actually emits:
# a heading, a blank line, then a quoted CSV table.
SAMPLE = """ ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"
"48.3","412,000,000","1600","257500.0","void cufft::vector_fft<...>(...)"
"21.7","185,000,000","800","231250.0","void nlse_apply_propagator<float2>(float2*)"
"9.4","80,000,000","400","200000.0","void nlse_square_mod_nl_prop<float2>(float2*)"
"""

# `nsys stats --report cuda_api_sum --format csv`, cut to two calls: 1000 ms.
API_SAMPLE = """ ** CUDA API Summary (cuda_api_sum):

"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Name"
"90.0","900,000,000","1600","562500.0","cudaStreamSynchronize"
"10.0","100,000,000","800","125000.0","cudaLaunchKernel"
"""
# The smallest grid the tracer is run on in the tests (points per side).
TRACED_N = 64

# Hand counts of a complex64 step per grid point, each array read or written
# once: c = 8 B a complex element, v = 4 B a float32 potential.
COMPLEX64 = 8  # B per complex64 element
SEPARATE_BYTES = 104  # 2 transforms of 4c, propagator 3c, nonlinear 2c: 13c
CALLBACK_BYTES = 64  # the 2 transforms alone: 8c
BLOCKED_BYTES = 48  # 3 passes, each reading and writing the field: 6c
POTENTIAL_BYTES = 4  # v, read once on every path
SPLIT_STEP_FLOPS_1024 = 220.0  # 2 transforms of 10 log2(1024), then 6 + 14
RK4_BYTES = 624  # docs/optimization-log.md, "What the machine allows"
RK4_FLOPS_1024 = 872.0  # 4 slopes of 2 transforms (200), 6 and 12
# A step far past the split-step limit of a 64x64 grid, 1.3e-2 m (m).
OVERSIZED_STEP = 1.0
# What both run lengths are made to take, so that the slope is zero (s).
DISTURBED_RUN_SECONDS = 1.0
# A made-up step for ceiling_shares, each share a different fraction by hand:
# memory 100 B x 1000 / 1 GB/s = 1e-4 s, flops 5e-5 s, sin+cos+exp 2.5e-5 s,
# dispatch 2e-5 s, each over the 2e-4 s measured.
SHARES_COST = (100.0, 100.0, 1.0)  # B, flops, sin+cos+exp per point
SHARES_POINTS = 1000
SHARES_CEILING = {"stream": 1.0, "fma": 2.0, "trans": 0.04}  # GB/s, GFLOP/s, G/s
SHARES_FIXED = 2e-5  # s
SHARES_MEASURED = 2e-4  # s
SHARES_EXPECTED = {"memory": 0.5, "flops": 0.25, "sincos": 0.125, "dispatch": 0.1}


def test_the_heading_is_skipped():
    """Nsys puts a title and a blank line before the table."""
    rows = nsys_summary.parse(SAMPLE)
    assert len(rows) == 3, f"expected 3 kernel rows, got {len(rows)}"
    assert rows[0]["Time (%)"] == "48.3"


def test_thousands_separators_are_understood():
    """412,000,000 ns is 412 ms, not 412."""
    rows = nsys_summary.parse(SAMPLE)
    assert nsys_summary.as_float(rows[0]["Total Time (ns)"]) == 412_000_000.0


def test_a_table_that_is_not_there_is_not_invented():
    """A missing report must give no rows rather than a plausible empty one."""
    assert nsys_summary.parse("no tables here\njust prose\n") == []


@pytest.mark.parametrize(
    "kernel,expected",
    [
        ("void cufft::vector_fft<32>(...)", "transform"),
        ("void nlse_apply_propagator<float2>(float2*)", "linear"),
        ("void nlse_linear_step<float2>(float2*)", "linear (fused)"),
        ("void nlse_square_mod_nl_prop<float2>(float2*)", "nonlinear"),
        ("void nlse_rk4_axpy<float2>(float2*)", "RK4 stage"),
        ("void nlse_rk4_nl_rhs<float2>(float2*)", "RK4 rhs"),
        ("void nlse_split_step_coupled_fused<float2>(float2*)", "whole step (fused)"),
        # extern "C" in kernels.cu, so unmangled; it runs in the real-space half.
        ("rabi_coupling", "nonlinear"),
        ("some_unrelated_kernel", "other"),
    ],
)
def test_kernel_names_map_to_phases(kernel, expected):
    """The names are mangled C++; the phases have to survive that."""
    assert nsys_summary.phase_of(kernel) == expected


def test_a_more_specific_rule_wins():
    """square_mod_rk4_nl_rhs is an RK4 right-hand side, not a plain nonlinear."""
    assert nsys_summary.phase_of("nlse_square_mod_rk4_nl_rhs<float2>") == "RK4 rhs"


def test_the_columns_nsys_renamed_are_both_accepted():
    """Older releases say Num Calls where newer ones say Instances."""
    old = {"Total Time": "1000", "Num Calls": "4", "Kernel Name": "k"}
    assert nsys_summary.column(old, "Total Time (ns)", "Total Time") == "1000"
    assert nsys_summary.column(old, "Instances", "Num Calls", "Count") == "4"
    assert nsys_summary.column(old, "Name", "Kernel Name", "Operation") == "k"


def _stats_without_kernels(report, names):
    """Stand in for `nsys stats` on a report that has only its API table."""
    return API_SAMPLE if "cuda_api_sum" in names else None


def test_no_ratio_is_given_without_a_kernel_table(monkeypatch, capsys):
    """A missing kernel table is zero kernel time only by default: say so."""
    monkeypatch.setattr(nsys_summary.shutil, "which", lambda name: "/usr/bin/nsys")
    monkeypatch.setattr(nsys_summary, "run_stats", _stats_without_kernels)
    assert nsys_summary.main(["report.nsys-rep"]) == 0
    out = capsys.readouterr().out
    assert "CUDA API time: 1000.00 ms" in out
    assert "GPU kernel time / CUDA API time" not in out
    assert "GPU is waiting" not in out


@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_the_tracer_follows_a_cpu_run_step_by_step(method):
    """The CPU transforms take ``normalize``, and a step asked is a step taken."""
    _, _, recorder = trace_solvers.trace("NLSE", "CPU", TRACED_N, method, False)
    assert recorder.calls[trace_solvers.STEP_KEY] == trace_solvers.STEPS
    assert recorder.calls["fft"] > 0 and recorder.calls["ifft"] > 0


def test_the_tracer_refuses_a_run_whose_step_was_lowered(monkeypatch):
    """A lowered step takes more steps than the per-step figures divide by."""
    monkeypatch.setattr(trace_solvers, "step_for", lambda simu: OVERSIZED_STEP)
    simu = trace_solvers.build("NLSE", "CPU", TRACED_N)
    field = trace_solvers.input_field(simu, "NLSE")
    with pytest.warns(UserWarning, match="Reducing"):
        with pytest.raises(RuntimeError, match="lowered the step"):
            trace_solvers.run(simu, field, "split_step", steps=2)


def test_the_fusion_benchmark_times_the_steps_it_asks(monkeypatch):
    """Its slope is per step only while the solver keeps the step."""
    monkeypatch.setattr(propagator_fusion, "BACKEND", "CPU")
    assert propagator_fusion.time_run(NLSE, TRACED_N, 2, np.complex64) > 0
    monkeypatch.setattr(propagator_fusion, "step_for", lambda simu: OVERSIZED_STEP)
    with pytest.warns(UserWarning, match="Reducing"):
        with pytest.raises(RuntimeError, match="lowered the step"):
            propagator_fusion.time_run(NLSE, TRACED_N, 2, np.complex64)


def _failing_trace(*args):
    """Stand in for a case that cannot be traced."""
    raise TypeError("not traceable")


def test_a_case_that_fails_is_a_failed_run(monkeypatch, capsys):
    """The error under a case header must not come with exit code 0."""
    monkeypatch.setattr(trace_solvers, "trace", _failing_trace)
    args = ["--backends", "CPU", "--solvers", "NLSE", "--methods", "split_step"]
    assert trace_solvers.main(args) == 1
    assert "FAILED: TypeError: not traceable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "path,expected",
    [
        ("separate", SEPARATE_BYTES),
        ("callbacks", CALLBACK_BYTES),
        ("blocked", BLOCKED_BYTES),
    ],
)
@pytest.mark.parametrize("splitting", ["lie", "strang"])
@pytest.mark.parametrize("has_V", [False, True])
def test_a_split_step_is_charged_what_its_path_moves(path, expected, splitting, has_V):
    """Strang is charged as Lie: a run with no callback merges its halves."""
    cost = roofline.step_cost(
        "split_step", splitting, 1024, COMPLEX64, has_V=has_V, path=path
    )
    potential = POTENTIAL_BYTES if has_V else 0
    assert cost == (expected + potential, SPLIT_STEP_FLOPS_1024, 1.0)


def test_rk4_is_charged_its_separate_passes():
    """Four slopes and the stage updates, as before the paths were told apart."""
    cost = roofline.step_cost(
        "RK4", "lie", 1024, COMPLEX64, has_V=False, path="separate"
    )
    assert cost == (RK4_BYTES, RK4_FLOPS_1024, 4.0)


@pytest.mark.parametrize("splitting,path", [("yoshida", "separate"), ("lie", "fused")])
def test_a_step_with_no_charge_is_refused(splitting, path):
    """A charge for another step would be a floor for the wrong one."""
    with pytest.raises(ValueError):
        roofline.step_cost(
            "split_step", splitting, 1024, COMPLEX64, has_V=False, path=path
        )


def test_a_lowered_step_fails_the_timing(monkeypatch):
    """The slope divides by the steps asked for, so a capped run cannot count."""
    monkeypatch.setattr(roofline, "STEP_DZ", OVERSIZED_STEP)
    with (
        pytest.warns(UserWarning, match="exceeds"),
        pytest.raises(RuntimeError, match="lowered the step"),
    ):
        roofline.time_step("CPU", 64, "split_step", "lie")


def test_a_timing_that_is_not_a_slope_fails(monkeypatch):
    """Twice the steps in no more time is a disturbed machine, not a step time."""
    monkeypatch.setattr(roofline, "fastest_run", lambda *run: DISTURBED_RUN_SECONDS)
    with pytest.raises(RuntimeError, match="disturbed"):
        roofline.time_step("CPU", 64, "split_step", "lie")


def test_the_tool_step_is_timed_on_its_path():
    """The tests run the CPU unblocked (conftest.py), so the path is separate."""
    seconds, path = roofline.time_step("CPU", 64, "split_step", "lie")
    assert math.isfinite(seconds)
    assert path == "separate"


def test_each_ceiling_share_is_its_floor_over_the_time_taken():
    """Every share differs, so a wrong floor or a swapped key shows."""
    shares = roofline.ceiling_shares(
        SHARES_COST, SHARES_POINTS, SHARES_CEILING, SHARES_FIXED, SHARES_MEASURED
    )
    assert shares == pytest.approx(SHARES_EXPECTED)


@pytest.mark.parametrize("dead", [None, *sorted(SHARES_CEILING)])
def test_a_ceiling_that_reads_zero_is_no_ceiling(monkeypatch, dead):
    """A probe's slope that was not positive reads 0, which no share can divide by."""
    reading = SHARES_CEILING if dead is None else {**SHARES_CEILING, dead: 0.0}
    monkeypatch.setattr(roofline, "_CEILINGS", {})
    monkeypatch.setattr(
        roofline, "PROBES", {"cpu": [("stub", lambda _: reading, set(reading))]}
    )
    assert roofline.measure_ceilings("cpu") == (reading if dead is None else None)


def test_the_cpu_path_is_blocked_when_the_tuner_gives_a_step(monkeypatch):
    """The solver's own plan asks cpu_tuning, so its answer names the path."""
    simu, E = roofline.build_solver("CPU", 64)
    roofline.propagate(simu, E, roofline.WARMUP_STEPS, "split_step", "lie")
    monkeypatch.setattr(cpu_tuning, "blocked_step", lambda *_: object())
    assert roofline.step_path(simu, "split_step") == "blocked"
    assert roofline.step_path(simu, "RK4") == "separate"
