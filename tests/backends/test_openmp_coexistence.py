"""The numba kernels survive sharing a process with a second OpenMP runtime."""

import importlib.util
import subprocess
import sys

import pytest

# Enough to compile and run a parfor: the crash is in the thread pool, so any
# parallel kernel reaches it, and square_mod is the smallest one that is real.
RUN_A_PARALLEL_KERNEL = """
import numpy as np
from NLSE.kernels import cpu
A = np.ones((16, 16), dtype=np.complex64)
out = np.zeros((16, 16), dtype=np.float32)
assert cpu.square_mod(A, out).sum() == 256.0
print("ok")
"""


@pytest.mark.parametrize(
    "preamble",
    ["", "import pyfftw\n"],
    ids=["nlse_first", "vendored_openmp_first"],
)
def test_a_parallel_kernel_runs_alongside_another_openmp(preamble: str) -> None:
    """A numba kernel must run whoever loaded an OpenMP runtime first.

    Parameters
    ----------
    preamble : str
        Code run before NLSE is imported, to fix which runtime loads first.
    """
    if preamble and importlib.util.find_spec("pyfftw") is None:
        pytest.skip("no pyfftw installed to load a second OpenMP runtime")
    result = subprocess.run(
        [sys.executable, "-c", preamble + RUN_A_PARALLEL_KERNEL],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"a numba parallel kernel crashed the interpreter "
        f"(exit {result.returncode}); two OpenMP runtimes were loaded and "
        f"numba's was not initialized first.\n{result.stderr}"
    )
    assert "ok" in result.stdout
