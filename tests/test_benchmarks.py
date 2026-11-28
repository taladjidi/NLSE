"""Tests of the tools in benchmarks/: nsys_summary.py and roofline.py."""

import importlib
import math
import sys
from pathlib import Path

import pytest
from NLSE.backends import cpu_tuning

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

nsys_summary = pytest.importorskip("nsys_summary")
# A plain import: the repository's own tool failing to import is a failure.
roofline = importlib.import_module("roofline")

# What `nsys stats --report cuda_gpu_kern_sum --format csv` actually emits:
# a heading, a blank line, then a quoted CSV table.
SAMPLE = """ ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"
"48.3","412,000,000","1600","257500.0","void cufft::vector_fft<...>(...)"
"21.7","185,000,000","800","231250.0","void nlse_apply_propagator<float2>(float2*)"
"9.4","80,000,000","400","200000.0","void nlse_square_mod_nl_prop<float2>(float2*)"
"""

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


def test_the_cpu_path_is_blocked_when_the_tuner_gives_a_step(monkeypatch):
    """The solver's own plan asks cpu_tuning, so its answer names the path."""
    simu, E = roofline.build_solver("CPU", 64)
    roofline.propagate(simu, E, roofline.WARMUP_STEPS, "split_step", "lie")
    monkeypatch.setattr(cpu_tuning, "blocked_step", lambda *_: object())
    assert roofline.step_path(simu, "split_step") == "blocked"
    assert roofline.step_path(simu, "RK4") == "separate"
