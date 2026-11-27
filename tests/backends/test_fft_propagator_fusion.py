"""The step folded into the transforms: same answer, and actually folded.

The CUPY backend can apply the propagator from a cuFFT store callback as the
forward transform writes each element, and the nonlinear step from another as
the inverse transform does, instead of in passes of their own. The arithmetic
is meant to be untouched -- the callbacks in fft_callbacks.cu are the
apply_propagator kernel's expression and the nonlinear kernels' shared
nlse_nonlinear_element, in their order -- so these compare **bit for bit**
rather than to a tolerance. A difference would mean a re-association, which is
how a fused kernel usually goes wrong.

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
from functools import partialmethod

import numpy as np
import pytest
from helpers import WAIST, gaussian, make
from NLSE import CNLSE, NLSE, NLSE_1d
from NLSE.backends import get_backend, list_available_backends
from NLSE.kernels.templating import NONLINEAR_CALLBACK, STORE_CALLBACKS, v_suffix

pytestmark = pytest.mark.skipif(
    "CUPY" not in list_available_backends(),
    reason="folding the propagator into the transform is a CUPY path",
)

N = 64
# Enough steps that execute_loop captures a CUDA graph and replays it, so the
# callback is exercised inside a graph and not only outside one.
STEPS = 5
DELTA_Z = 1e-4

# A potential on the grid, writing about 0.04 rad per step at k = 2 pi/780 nm:
# enough to tell a twin that drops it from one that reads it.
_AXIS = np.linspace(-2 * WAIST, 2 * WAIST, N, endpoint=False)
REAL_V = (
    1e-4 * np.exp(-(_AXIS[None, :] ** 2 + _AXIS[:, None] ** 2) / WAIST**2)
).astype(np.float32)
# Imaginary part of a complex potential against its real part: an absorbing
# one, which is also what stops Strang merging its halves.
ABSORPTION = 0.1
ABSORBING = 1 + 1j * ABSORPTION  # a complex potential per unit of real one
COMPLEX_V = (REAL_V * ABSORBING).astype(np.complex64)
# 1/m, the order of k/2 at 780 nm (4.0e6), by which the solver scales V.
K_HALF = 1e6


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
    ("lossless", NLSE, (N, N), {"physics": {"alpha": 0.0}}),
    ("real V", NLSE, (N, N), {"physics": {"V": REAL_V}}),
    ("complex V", NLSE, (N, N), {"physics": {"V": COMPLEX_V}}),
    ("strang real V", NLSE, (N, N), {"splitting": "strang", "physics": {"V": REAL_V}}),
    (
        "strang complex V",
        NLSE,
        (N, N),
        {"splitting": "strang", "physics": {"V": COMPLEX_V}},
    ),
    ("double", NLSE, (N, N), {"dtype": np.complex128}),
]


@pytest.mark.parametrize("label,cls,shape,kw", CASES, ids=[case[0] for case in CASES])
def test_fusing_the_propagator_changes_nothing(label, cls, shape, kw):
    """Every path that fuses must return what it returned unfused."""
    unfused, _ = propagate(False, cls, shape, **kw)
    fused, kinds = propagate(True, cls, shape, **kw)
    assert kinds, f"{label} fused nothing, so this compares one path with itself"
    assert np.array_equal(fused, unfused), (
        f"{label} differs once fused, by at most "
        f"{np.max(np.abs(fused - unfused)):.3e}: a callback is not computing "
        f"its kernel's arithmetic in its order"
    )


def test_a_batch_shares_one_propagator():
    """A batched field takes the callback that wraps the propagator round it."""
    _, kinds = propagate(True, NLSE, (3, N, N))
    assert STORE_CALLBACKS[True] in kinds and STORE_CALLBACKS[False] not in kinds, (
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
    assert NONLINEAR_CALLBACK in kinds, (
        "yoshida did not fuse the nonlinear step, so this compares one path with itself"
    )
    assert np.array_equal(fused, unfused), (
        "yoshida differs once fused: a step whose propagator changes part way "
        "is not being told about the change"
    )


def refuse_nonlinear(self, A, *args):
    """Stand in for the nonlinear kernels on a run that has fused them in."""
    raise AssertionError("a nonlinear kernel ran on a run that had fused it in")


@pytest.mark.parametrize("V", [None, REAL_V], ids=["no V", "real V"])
def test_the_fused_path_is_taken(monkeypatch, V):
    """A fused Lie run must reach neither apply_propagator nor a nonlinear kernel."""
    from NLSE.kernels.cupy_kernels import CUDAKernels

    def refuse(self, A, propagator):
        raise AssertionError("apply_propagator ran on a run that had fused it in")

    monkeypatch.setattr(CUDAKernels, "apply_propagator", refuse)
    monkeypatch.setattr(CUDAKernels, "square_mod_nl_prop", refuse_nonlinear)
    monkeypatch.setattr(CUDAKernels, "square_mod_nl_prop_v", refuse_nonlinear)
    result, kinds = propagate(True, physics={"V": V})
    assert kinds, "nothing fused, so nothing was under test"
    assert np.isfinite(result).all(), "the fused run returned a field with holes in it"


def test_disabling_the_fusion_restores_the_pass(monkeypatch):
    """``NLSE_FUSE_PROPAGATOR=0`` must put both steps back in their kernels."""
    from NLSE.kernels.cupy_kernels import CUDAKernels

    calls = []
    nonlinear_calls = []
    for name, seen in (
        ("apply_propagator", calls),
        ("square_mod_nl_prop", nonlinear_calls),
    ):
        method = partialmethod(counted, getattr(CUDAKernels, name), seen)
        monkeypatch.setattr(CUDAKernels, name, method)
    _, kinds = propagate(False)
    assert not kinds, "a plan built with the fusion switched off fused anyway"
    assert calls, "the unfused path never applied the propagator"
    assert nonlinear_calls, "the unfused path never ran the nonlinear kernel"


def counted(self, original, seen, A, *args):
    """Stand in for a CUDAKernels method: note the field's shape, then run it."""
    seen.append(A.shape)
    return original(self, A, *args)


def refuse_nonlinear_plan(self, build, symbol):
    """Stand in for ``_build_callback_plan``: cuFFT refusing the nonlinear step.

    ``build`` is the real one, for every other callback.
    """
    if symbol.startswith(NONLINEAR_CALLBACK):
        raise RuntimeError("refused for the test")
    return build(self, symbol)


def test_a_refused_callback_leaves_the_others_fused(monkeypatch):
    """A callback cuFFT refuses must cost its own pass and nothing else."""
    from NLSE.backends.cupy_backend import _CuFFTPlan

    build = partialmethod(refuse_nonlinear_plan, _CuFFTPlan._build_callback_plan)
    monkeypatch.setattr(_CuFFTPlan, "_build_callback_plan", build)
    unfused, _ = propagate(False)
    fused, kinds = propagate(True)
    assert STORE_CALLBACKS[False] in kinds, (
        "refusing the nonlinear callback took the propagator callback with it"
    )
    assert not any(kind.startswith(NONLINEAR_CALLBACK) for kind in kinds)
    assert np.array_equal(fused, unfused)


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


# (dz, alpha) for the kernel-level comparison, by u = 2*alpha*dz: lossless,
# inside NLSE_LOSS_SOLVED_LIMIT (0.1, nonlinear_step.cuh) where the loss is
# solved, and past it where it is frozen. All three run on one plan, so the
# block's rewrite between them is under test too.
STEP_PARAMETERS = [(1e-4, 0.0), (2e-4, 25.0), (1e-4, 1000.0)]
G = 1e3  # about 0.1 rad per 1e-4 step on a unit field
ISAT = 1.0  # saturation at the field's own scale, so it is not the identity


def potential(kind, dtype):
    """Return the potential of a kind at a field's width, or None."""
    if kind == "none":
        return None
    real = np.finfo(dtype).dtype
    V = REAL_V.astype(real) * K_HALF
    return V if kind == "real" else (V * ABSORBING).astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["none", "real", "complex"])
def test_the_nonlinear_callback_is_the_kernel(monkeypatch, dtype, kind):
    """linear_nonlinear_step against linear_step and the kernel, to the bit."""
    cp = pytest.importorskip("cupy")
    from NLSE.backends.cupy_backend import _CuFFTPlan

    monkeypatch.delenv("NLSE_FUSE_PROPAGATOR", raising=False)
    kernels = get_backend("CUPY").kernels
    rng = np.random.default_rng(0)
    host = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(dtype)
    phase = rng.uniform(0, 2 * np.pi, size=(N, N))
    propagator = cp.asarray((np.exp(1j * phase) / (N * N)).astype(dtype))
    V = potential(kind, dtype)
    V = None if V is None else cp.asarray(V)
    plan = _CuFFTPlan(cp.asarray(host), axes=(-2, -1))
    for dz, alpha in STEP_PARAMETERS:
        expected = cp.asarray(host)
        kernels.linear_step(expected, propagator, plan, unnorm_ifft=True)
        if V is None:
            kernels.square_mod_nl_prop(expected, dz, alpha, G, ISAT)
        else:
            kernels.square_mod_nl_prop_v(expected, V, dz, alpha, G, ISAT)
        got = cp.asarray(host)
        if not kernels.linear_nonlinear_step(
            got, propagator, plan, V, dz, alpha, G, ISAT
        ):
            symbol = NONLINEAR_CALLBACK + v_suffix(V)
            if symbol in plan._refused:
                pytest.skip(f"this cuFFT will not link {symbol} into a transform")
            else:
                pytest.fail(f"declined although cuFFT did not refuse {symbol}")
        assert cp.array_equal(got, expected), (
            f"dz={dz}, alpha={alpha}: the callback differs from the kernel by "
            f"{float(cp.abs(got - expected).max()):.3e}"
        )
