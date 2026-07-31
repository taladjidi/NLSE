"""The propagator folded into the transform: same answer, and actually folded.

The CUPY backend can apply the propagator from a cuFFT store callback as the
forward transform writes each element, instead of in a pass of its own. The
arithmetic is meant to be untouched -- the callback in fft_callbacks.cu is the
apply_propagator kernel's expression, in its order -- so these compare **bit
for bit** rather than to a tolerance. A difference would mean a re-association,
which is how a fused kernel usually goes wrong.

Two of these tests are about the machinery rather than the answer, and neither
is optional.

``test_a_second_propagator_is_seen`` guards the indirection the design rests
on. cuFFT binds the callback's argument when the plan is created, so the
callback is handed a *pointer to a pointer* and one plan serves the whole run.
Bound straight to a propagator instead, a Yoshida step -- three propagators,
cycling -- would need a new plan per sub-step, and the version of this that
looks correct on a single-propagator run would be far slower on that one.

``test_the_fused_path_is_taken`` fails the run if apply_propagator is called at
all. Without it this file passes just as happily when nothing fuses, which is
what it did on every case until the plan cache was cleared between the two
sides: the fallback is silent by design, so comparing answers cannot tell the
two paths apart.
"""

import os

import numpy as np
import pytest
from helpers import gaussian, make
from NLSE import CNLSE, NLSE, NLSE_1d
from NLSE.backends import get_backend, list_available_backends

pytestmark = pytest.mark.skipif(
    "CUPY" not in list_available_backends(),
    reason="folding the propagator into the transform is a CUPY path",
)

N = 64
# Enough steps that execute_loop captures a CUDA graph and replays it, so the
# callback is exercised inside a graph and not only outside one.
STEPS = 5
DELTA_Z = 1e-4


def propagate(fuse, cls=NLSE, shape=(N, N), dtype=np.complex64, physics=(), **kw):
    """Propagate with the fusion allowed or not, and report what fused.

    ``NLSE_FUSE_PROPAGATOR`` is read when a plan is built and plans outlive the
    solver that built them, so the cache has to go for the variable to mean
    anything -- before, so this run gets its own plan, and after, so the next
    test does.

    Parameters
    ----------
    fuse : bool
        Whether to allow the fusion.
    cls : type
        Solver class.
    shape : tuple
        Shape of the input field.
    dtype : np.dtype
        Complex width of the field.
    physics : dict
        Constructor arguments to override.
    **kw
        Passed on to ``out_field``.

    Returns
    -------
    tuple
        ``(result, fused)`` -- the field, and the callbacks the plan built.

    """
    os.environ["NLSE_FUSE_PROPAGATOR"] = "1" if fuse else "0"
    try:
        get_backend("CUPY").clear_fft_plans()
        simu = make(cls, "CUPY", n=N, **dict(physics))
        out = simu.out_field(
            gaussian(shape, dtype=dtype),
            STEPS * DELTA_Z,
            delta_z=DELTA_Z,
            verbose=False,
            plot=False,
            **kw,
        )
        return np.asarray(out), dict(getattr(simu.plans[0], "_fused", {}))
    finally:
        os.environ.pop("NLSE_FUSE_PROPAGATOR", None)
        get_backend("CUPY").clear_fft_plans()


CASES = [
    ("split step", NLSE, (N, N), {}),
    ("strang", NLSE, (N, N), {"splitting": "strang"}),
    ("RK4", NLSE, (N, N), {"method": "RK4"}),
    ("coupled", CNLSE, (2, N, N), {}),
    ("coupled RK4", CNLSE, (2, N, N), {"method": "RK4"}),
    ("batched", NLSE, (3, N, N), {}),
]


@pytest.mark.parametrize("label,cls,shape,kw", CASES, ids=[case[0] for case in CASES])
def test_fusing_the_propagator_changes_nothing(label, cls, shape, kw):
    """Every path that fuses must return what it returned unfused."""
    unfused, _ = propagate(False, cls, shape, **kw)
    fused, kinds = propagate(True, cls, shape, **kw)
    assert kinds, f"{label} fused nothing, so this compares one path with itself"
    assert np.array_equal(fused, unfused), (
        f"{label} differs once fused, by at most "
        f"{np.max(np.abs(fused - unfused)):.3e}: the callback is not computing "
        f"the apply_propagator kernel's arithmetic in its order"
    )


def test_a_batch_shares_one_propagator():
    """A batched field takes the callback that wraps the propagator round it."""
    _, kinds = propagate(True, NLSE, (3, N, N))
    assert kinds == {True: kinds.get(True)}, (
        f"a batch of three fields against one propagator built {kinds}, and the "
        f"direct callback would read past the end of it"
    )


def test_yoshida_cycles_three_propagators():
    """Three propagators within one step, through one plan, unchanged.

    The case the indirection exists for. Double precision and lossless because
    Yoshida warns about both, for reasons that have nothing to do with this:
    its middle sub-step runs backwards, which amplifies a lossy field, and its
    fourth order is below single-precision round-off anyway.
    """
    kw = {
        "splitting": "yoshida",
        "dtype": np.complex128,
        "physics": {"alpha": 0.0},
    }
    unfused, _ = propagate(False, **kw)
    fused, kinds = propagate(True, **kw)
    assert kinds, "yoshida fused nothing, so this compares one path with itself"
    assert np.array_equal(fused, unfused), (
        "yoshida differs once fused: a step whose propagator changes part way "
        "is not being told about the change"
    )


def test_the_fused_path_is_taken(monkeypatch):
    """A fused run must not reach apply_propagator at all."""
    from NLSE.kernels.cupy_kernels import CUDAKernels

    def refuse(self, A, propagator):
        raise AssertionError("apply_propagator ran on a run that had fused it in")

    monkeypatch.setattr(CUDAKernels, "apply_propagator", refuse)
    result, kinds = propagate(True)
    assert kinds, "nothing fused, so nothing was under test"
    assert np.isfinite(result).all(), "the fused run returned a field with holes in it"


def test_disabling_the_fusion_restores_the_pass(monkeypatch):
    """``NLSE_FUSE_PROPAGATOR=0`` must put the multiply back in its own kernel."""
    from NLSE.kernels.cupy_kernels import CUDAKernels

    original = CUDAKernels.apply_propagator
    calls = []

    def counted(self, A, propagator):
        calls.append(A.shape)
        return original(self, A, propagator)

    monkeypatch.setattr(CUDAKernels, "apply_propagator", counted)
    _, kinds = propagate(False)
    assert not kinds, "a plan built with the fusion switched off fused anyway"
    assert calls, "the unfused path never applied the propagator"


def test_a_second_propagator_is_seen():
    """A plan must answer to whichever propagator it is handed.

    Straight at the indirection: a plan that captured the propagator it was
    built with would pass every other test in this file and still return the
    first propagator's answer for the second's.
    """
    cp = pytest.importorskip("cupy")
    from NLSE.backends.cupy_backend import _CuFFTPlan

    rng = np.random.default_rng(0)
    host = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex64)
    a = cp.asarray(host)
    transform = np.fft.fft2(host)
    out = cp.empty_like(a)
    plan = _CuFFTPlan(a, axes=(-2, -1))

    # Complex factors, not real ones: against a real propagator the multiply
    # cannot get the imaginary part wrong, and this becomes a weaker test of
    # the arithmetic than of the pointer it was written for.
    for factor in (2 + 3j, -5 + 0.5j):
        propagator = cp.asarray(np.full((N, N), factor, dtype=np.complex64))
        if not plan.fft_propagate(a, out, propagator):
            pytest.skip("this cuFFT will not fold the propagator into a transform")
        assert np.allclose(
            cp.asnumpy(out),
            transform * np.complex64(factor),
            rtol=1e-4,
            atol=1e-3 * np.abs(transform).max(),
        ), (
            f"the propagator {factor} did not reach the callback: the plan is "
            f"bound to the array it was built with rather than to the block "
            f"that points at it"
        )


def test_one_dimension_still_propagates():
    """A 1D run must come out right whether or not it can fuse.

    CuPy reaches the callback machinery for multi-dimensional plans only, and
    its one-dimensional path raises on the way in. Which of the two happens is
    CuPy's business and not asserted here; that the answer survives either is
    this project's.
    """
    unfused, _ = propagate(False, NLSE_1d, (N,))
    fused, _ = propagate(True, NLSE_1d, (N,))
    assert np.array_equal(fused, unfused), (
        "a 1D run changed once the fusion was allowed, so the fallback is not "
        "falling back to the same arithmetic"
    )
