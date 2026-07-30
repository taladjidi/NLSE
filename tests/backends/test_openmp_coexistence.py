"""The numba kernels survive sharing a process with FFTW's OpenMP runtime.

pyfftw's PyPI wheel vendors its own copy of libomp under ``pyfftw/.dylibs``
where the conda-forge build links the one in the environment, and numba's
thread pool always links the latter. Two copies coexist only if numba's is
initialized first: let FFTW's go first and the next ``prange`` anywhere in
the process segfaults, so the failure lands in whichever kernel happened to
run first and looks nothing like the import conflict it is.

That makes this untestable in-process -- by the time a test body runs the
order is already settled, and getting it wrong takes the interpreter down
rather than failing an assertion. So each case is a subprocess, and what is
asserted is that it exited at all.
"""

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
    ids=["nlse_first", "pyfftw_first"],
)
def test_a_parallel_kernel_runs_alongside_fftw(preamble: str) -> None:
    """A numba kernel must run whoever loaded an OpenMP runtime first.

    The second case is the one that crashed: importing pyfftw before NLSE
    leaves no window to start numba's pool first, and the package has to
    notice and fall back rather than take the process down.

    Parameters
    ----------
    preamble : str
        Code run before NLSE is imported, to fix which runtime loads first.
    """
    result = subprocess.run(
        [sys.executable, "-c", preamble + RUN_A_PARALLEL_KERNEL],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"a numba parallel kernel crashed the interpreter "
        f"(exit {result.returncode}); FFTW and numba each loaded an OpenMP "
        f"runtime and numba's was not initialized first.\n{result.stderr}"
    )
    assert "ok" in result.stdout
