"""The profiling tools in benchmarks/ have one fiddly part worth pinning.

Parsing `nsys stats --format csv` means skipping a heading, coping with
thousands separators, and column names that have changed between nsys
releases. None of that can be exercised on a machine without CUDA, and a
silent misparse would produce a plausible-looking table of wrong numbers.
"""

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

nsys_summary = pytest.importorskip("nsys_summary")

# What `nsys stats --report cuda_gpu_kern_sum --format csv` actually emits:
# a heading, a blank line, then a quoted CSV table.
SAMPLE = """ ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

"Time (%)","Total Time (ns)","Instances","Avg (ns)","Name"
"48.3","412,000,000","1600","257500.0","void cufft::vector_fft<...>(...)"
"21.7","185,000,000","800","231250.0","void nlse_apply_propagator<float2>(float2*)"
"9.4","80,000,000","400","200000.0","void nlse_square_mod_nl_prop<float2>(float2*)"
"""


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
