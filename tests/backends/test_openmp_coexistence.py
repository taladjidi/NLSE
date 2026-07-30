"""The numba kernels survive sharing a process with a second OpenMP runtime.

A library that vendors its own copy of libomp rather than linking the
environment's can coexist with numba's, but only if numba's initialized first:
the other order segfaults at the next ``prange`` anywhere in the process, so
the failure lands in whichever kernel happened to run first and looks nothing
like the import conflict it is.

pyfftw's PyPI wheel is the one that bit us, under ``pyfftw/.dylibs``, and this
package no longer depends on it -- which is exactly why the case is worth
keeping. NLSE cannot stop a caller importing such a library above it, and an
environment that had NLSE before the transform moved to scipy still has pyfftw
in it. So pyfftw is used here as the specimen, and skipped where absent.

That makes this untestable in-process -- by the time a test body runs the
order is already settled, and getting it wrong takes the interpreter down
rather than failing an assertion. So each case is a subprocess, and what is
asserted is that it exited at all.
"""

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

    The second case is the one that crashed: importing the other runtime
    first leaves no window to start numba's pool ahead of it, so the package
    has to notice and fall back rather than take the process down.

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
