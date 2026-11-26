"""The MLX scalar cache stays bounded however many distinct values it sees.

An adaptive step hands the kernels a new delta_z every step, and each value
converted to an mx.array used to stay alive for the life of the process.
"""

import gc
import weakref

import pytest
from NLSE.utils import __MLX_AVAILABLE__

pytestmark = pytest.mark.skipif(not __MLX_AVAILABLE__, reason="MLX not available")

# Distinct scalars converted, in values: any count well above
# _SCALAR_CACHE_SIZE (128, NLSE/kernels/mlx_kernels.py) will do.
DISTINCT_VALUES = 20_000
# A value no other test converts, so nothing else holds its array.
PROBE = 0.123456789
# A value converted twice in a row, to check the cache still serves hits.
REPEATED = 0.25


def test_an_old_scalar_is_released():
    """A converted scalar must not outlive a flood of newer ones."""
    from NLSE.kernels import mlx_kernels

    probe = weakref.ref(mlx_kernels._to_mx(PROBE))
    for index in range(DISTINCT_VALUES):
        mlx_kernels._to_mx(1.0 + index)
    gc.collect()
    assert probe() is None


def test_a_repeated_scalar_is_reused():
    """The same value converts to the same array while it is cached."""
    from NLSE.kernels import mlx_kernels

    assert mlx_kernels._to_mx(REPEATED) is mlx_kernels._to_mx(REPEATED)
