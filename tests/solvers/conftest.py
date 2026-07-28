"""Shared fixtures for solver tests."""

import numpy as np
import pytest
from NLSE.backends import list_available_backends

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

# Ask the backend registry rather than rebuilding the list by hand: the
# hand-rolled version checked only CUPY and PYOPENCL, so the MLX solver
# paths were never exercised here.
AVAILABLE_BACKENDS = list_available_backends()


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    """Yield each available backend as a test parameter."""
    return request.param
