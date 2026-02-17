"""Shared fixtures for solver tests."""

import numpy as np
import pytest
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

AVAILABLE_BACKENDS = ["CPU"]
if NLSE.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CUPY")
if NLSE.__PYOPENCL_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CL")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    """Yield each available backend as a test parameter."""
    return request.param
