"""
NLSE.

A package for solving the Nonlinear Schrodinger Equation (NLSE) using the
Split-Step Fourier method.
"""

__version__ = "2.3.0"
__author__ = "Tangui Aladjidi"
__license__ = "GPLv3"
__credits__ = "Laboratoire Kastler Brossel, Paris, France"
__email__ = "tangui.aladjidi@lkb.upmc.fr"


from . import utils as utils
from .callbacks import adapt_delta_z as adapt_delta_z
from .callbacks import evaluate_delta_n as evaluate_delta_n
from .callbacks import norm as norm
from .callbacks import sample as sample
from .solvers import CNLSE as CNLSE
from .solvers import CNLSE_1d as CNLSE_1d
from .solvers import DDGPE as DDGPE
from .solvers import GPE as GPE
from .solvers import NLSE as NLSE
from .solvers import NLSE_1d as NLSE_1d
from .solvers import NLSE_3d as NLSE_3d

# Backward-compatible submodule aliases so that
# `from NLSE.kernels_cpu import ...` and `from NLSE.kernels_cl import ...` still work.
import sys

from .kernels import cpu as kernels_cpu

sys.modules[__name__ + ".kernels_cpu"] = kernels_cpu

try:
    from .kernels import gpu as kernels_gpu

    sys.modules[__name__ + ".kernels_gpu"] = kernels_gpu
except ImportError:
    pass

try:
    from .kernels import cl as kernels_cl

    sys.modules[__name__ + ".kernels_cl"] = kernels_cl
except ImportError:
    pass

try:
    from .kernels import metal as kernels_metal

    sys.modules[__name__ + ".kernels_metal"] = kernels_metal
except (ImportError, FileNotFoundError, OSError):
    pass

# Alias for NLSE.metal.metal_api -> NLSE.kernels.metal_native.metal_api
try:
    from .kernels import metal_native as metal

    sys.modules[__name__ + ".metal"] = metal
except (ImportError, FileNotFoundError, OSError):
    pass
