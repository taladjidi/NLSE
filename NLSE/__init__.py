"""
NLSE.

A package for solving the Nonlinear Schrödinger Equation (NLSE) using the
Split-Step Fourier method.
"""

__version__ = "4.0.0"
__author__ = "Tangui Aladjidi"
__license__ = "GPLv3"
__credits__ = "Laboratoire Kastler Brossel, Paris, France"
__email__ = "tangui.aladjidi@lkb.upmc.fr"


# Backward-compatible submodule aliases so that
# `from NLSE.kernels_cpu import ...` and `from NLSE.kernels_cl import ...` still work.
import sys

from . import utils as utils
from .callbacks import (
    adapt_delta_z as adapt_delta_z,
    evaluate_delta_n as evaluate_delta_n,
    norm as norm,
    sample as sample,
)
from .kernels import cpu as kernels_cpu
from .solvers import (
    CNLSE as CNLSE,
    DDGPE as DDGPE,
    GPE as GPE,
    NLSE as NLSE,
    CNLSE_1d as CNLSE_1d,
    NLSE_1d as NLSE_1d,
    NLSE_3d as NLSE_3d,
)

sys.modules[__name__ + ".kernels_cpu"] = kernels_cpu

try:
    from .kernels import cupy as kernels_gpu

    sys.modules[__name__ + ".kernels_gpu"] = kernels_gpu
except ImportError:
    pass

try:
    from .kernels import cl as kernels_cl

    sys.modules[__name__ + ".kernels_cl"] = kernels_cl
except ImportError:
    pass
