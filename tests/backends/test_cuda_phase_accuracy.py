"""The CUDA kernels' exp(i*phase) holds float32 accuracy at large phases.

``--use_fast_math`` turned sincosf into __sincosf, whose error is 2^-21.41 on
[-pi, pi] and grows with the phase outside it: 1.4e-4 at 1e3 rad on an RTX 3050.
nl_prop with A = 1, |A|^2 = 0, no loss and dz = 1 returns exactly exp(i*V), so
the potential carries the phase to sincosf unchanged.
"""

import numpy as np
import pytest
from NLSE.backends import get_backend, list_available_backends

pytestmark = pytest.mark.skipif(
    "CUPY" not in list_available_backends(), reason="CuPy not available"
)

# Largest |phase| tested, in rad: a potential term of 1e3 rad over one step,
# far past the pi per step the nonlinear phase is held to.
PHASE_MAX = 1e3
# Number of phases sampled evenly over [-PHASE_MAX, PHASE_MAX].
SAMPLES = 1 << 20
# sinf and cosf are within 2 ulp (CUDA Programming Guide, table 56, section
# 5.6.7.4); an ulp below 1 is at most 2^-23, so |error| <= sqrt(2) * 2 * 2^-23.
# https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#trigonometric-functions
TOLERANCE = np.sqrt(2.0) * 2.0 * 2.0**-23
# One step of unit length (m), so the phase applied is the potential itself.
DELTA_Z = 1.0
# No loss (1/m) and no Kerr term: only the potential acts.
ALPHA = 0.0
G = 0.0
# Saturation intensity: infinite, so nothing saturates.
I_SAT = np.inf


def test_exp_i_phase_is_float32_accurate():
    """exp(i*V) from nl_prop agrees with float64 to the sincosf ulp bound."""
    import cupy as cp

    kernels = get_backend("CUPY").kernels
    phase = np.linspace(-PHASE_MAX, PHASE_MAX, SAMPLES, dtype=np.float32)
    field = cp.ones(SAMPLES, dtype=np.complex64)
    intensity = cp.zeros(SAMPLES, dtype=np.float32)
    potential = cp.asarray(phase)
    kernels.nl_prop(field, intensity, DELTA_Z, ALPHA, potential, G, I_SAT)
    exact = np.exp(1j * phase.astype(np.float64))
    error = np.abs(cp.asnumpy(field).astype(np.complex128) - exact)
    assert error.max() <= TOLERANCE
