"""The step folded into the transforms: same answer, and actually folded."""

import os
from functools import partialmethod

import numpy as np
import pytest
from helpers import WAIST, gaussian, make
from NLSE import CNLSE, NLSE, NLSE_1d
from NLSE.backends import get_backend, list_available_backends
from NLSE.kernels.templating import (
    NONLINEAR_CALLBACK,
    SEPARABLE_CALLBACK,
    STORE_CALLBACKS,
    v_suffix,
)

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


def no_separable_propagator(dtype, delta_z, rows, unnorm_ifft):
    """Stand-in for ``NLSE._separable_propagator`` that offers no factors."""
    return None


def propagate(
    fuse,
    cls=NLSE,
    shape=(N, N),
    dtype=np.complex64,
    physics=(),
    separable=False,
    **kw,
):
    """Propagate with the fusion allowed or not, and report what fused.

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
    separable : bool
        Whether the propagator may be read as one factor per axis. Off by
        default, so that the comparisons here stay bitwise.
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
        if not separable:
            simu._separable_propagator = no_separable_propagator
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
    ("two-axis batch", NLSE, (3, 2, N, N), {}),
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
    """Yoshida's three propagators in one step pass through one plan unchanged."""
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
    """A plan must answer to whichever propagator it is handed."""
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
    """A 1D run must come out right whether or not it can fuse."""
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
# Largest ratio of the separable propagator's error to the full one's, both
# against complex128. Measured 0.92-1.10 (numpy emulation, see the test).
SEPARABLE_ERROR_RATIO = 1.5
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


@pytest.mark.parametrize(
    "kw",
    [{}, {"splitting": "yoshida", "physics": {"alpha": 0.0}}],
    ids=["lie", "yoshida"],
)
def test_the_separable_propagator_is_as_accurate(kw):
    """One propagator factor per axis costs no accuracy against complex128."""
    reference, _ = propagate(True, dtype=np.complex128, **kw)
    full, _ = propagate(True, **kw)
    separable, kinds = propagate(True, separable=True, **kw)
    assert SEPARABLE_CALLBACK in kinds, "the separable propagator was not taken"
    scale = np.abs(reference).max()
    err_full = np.abs(full - reference).max() / scale
    err_separable = np.abs(separable - reference).max() / scale
    assert err_separable <= SEPARABLE_ERROR_RATIO * err_full, (
        f"separable error {err_separable:.2e} against full {err_full:.2e}"
    )


# Rows of the field the separable callback is checked on: half its columns, so
# a callback that swaps the two axes reads past the end of one of its factors.
NY = N // 2
BATCH = 3  # fields on one plan, over which the row factors repeat
# Largest difference between the separable and the full propagator's forward
# transform, in eps of its largest element. Closed form: 2 for the factors'
# product, rounded with or without a fused multiply-add, and 2 for the multiply
# by the element that follows; doubled, since neither side is exact.
SEPARABLE_ULPS = 8
SENTINEL = 7 + 7j  # what a refused transform must leave in its output
# How each refused call departs from one the plan takes.
REFUSALS = [
    "field shape",
    "field dtype",
    "field not contiguous",
    "factor dtype",
    "factor not 1D",
    "factor not contiguous",
    "columns size",
    "rows size",
    "fusion off",
]


def separable_arguments(cp, shape, dtype):
    """Return a random field and random factors of its rows and its columns.

    Parameters
    ----------
    cp : module
        CuPy.
    shape : tuple
        Shape of the field, its last two axes the transformed ones.
    dtype : np.dtype
        Complex width of the field and the factors.

    Returns
    -------
    tuple
        ``(a, rows, columns)`` on the device; ``rows`` carries 1/N and
        repeats over the batch, as ``NLSE._separable_propagator``'s does.
    """
    rng = np.random.default_rng(0)
    host = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    n_fft = shape[-2] * shape[-1]
    row_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape[-2])) / n_fft
    column_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=shape[-1]))
    rows = np.tile(row_phase, int(np.prod(shape[:-2], dtype=int)))
    return (
        cp.asarray(host.astype(dtype)),
        cp.asarray(rows.astype(dtype)),
        cp.asarray(column_phase.astype(dtype)),
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(NY, N), (BATCH, NY, N)], ids=["one", "batch"])
def test_the_separable_callback_is_the_full_propagator(monkeypatch, dtype, shape):
    """The forward transform with the factors, against the one with their product.

    Not bitwise: the callback forms each element's product itself.
    """
    cp = pytest.importorskip("cupy")
    from NLSE.backends.cupy_backend import _CuFFTPlan

    monkeypatch.delenv("NLSE_FUSE_PROPAGATOR", raising=False)
    a, rows, columns = separable_arguments(cp, shape, dtype)
    full = (rows[:, None] * columns[None, :]).reshape(shape)
    plan = _CuFFTPlan(a, axes=(-2, -1))
    expected = cp.empty_like(a)
    got = cp.empty_like(a)
    if not plan.fft_propagate(a, expected, full):
        pytest.skip("this cuFFT will not fold the propagator into a transform")
    if not plan.fft_propagate_separable(a, got, rows, columns):
        if SEPARABLE_CALLBACK in plan._refused:
            pytest.skip(f"this cuFFT will not link {SEPARABLE_CALLBACK}")
        else:
            pytest.fail(f"declined although cuFFT did not refuse {SEPARABLE_CALLBACK}")
    bound = SEPARABLE_ULPS * np.finfo(dtype).eps * float(cp.abs(expected).max())
    difference = float(cp.abs(got - expected).max())
    assert difference <= bound, f"differs by {difference:.3e}, bound {bound:.3e}"


def refused_arguments(cp, refusal, dtype):
    """Return ``(a, rows, columns)`` departing from a taken call as named.

    Parameters
    ----------
    cp : module
        CuPy.
    refusal : str
        One of REFUSALS.
    dtype : np.dtype
        Complex width of the field the plan is built for.

    Returns
    -------
    tuple
        ``(a, rows, columns)`` on the device.
    """
    a, rows, columns = separable_arguments(cp, (NY, N), dtype)
    wide = np.complex128
    if refusal == "field shape":
        a, rows, columns = separable_arguments(cp, (NY, N // 2), dtype)
    elif refusal == "field dtype":
        a, rows, columns = a.astype(wide), rows.astype(wide), columns.astype(wide)
    elif refusal == "field not contiguous":
        a = cp.asfortranarray(a)
    elif refusal == "factor dtype":
        rows = rows.astype(wide)
    elif refusal == "factor not 1D":
        rows = rows.reshape(NY, 1)
    elif refusal == "factor not contiguous":
        rows = cp.repeat(rows, 2)[::2]
    elif refusal == "columns size":
        columns = cp.ascontiguousarray(columns[1:])
    elif refusal == "rows size":
        rows = cp.ascontiguousarray(rows[1:])
    elif refusal == "fusion off":
        pass
    else:
        raise ValueError(f"no refusal named {refusal}")
    return a, rows, columns


@pytest.mark.parametrize("refusal", REFUSALS)
def test_the_separable_callback_refuses_what_it_cannot_read(monkeypatch, refusal):
    """Declined, with the output untouched, so the caller's fallback is whole."""
    cp = pytest.importorskip("cupy")
    from NLSE.backends.cupy_backend import _CuFFTPlan

    if refusal == "fusion off":
        monkeypatch.setenv("NLSE_FUSE_PROPAGATOR", "0")
    else:
        monkeypatch.delenv("NLSE_FUSE_PROPAGATOR", raising=False)
    # complex64, so that "field dtype" and "factor dtype" can widen to differ.
    plan = _CuFFTPlan(cp.zeros((NY, N), np.complex64), axes=(-2, -1))
    a, rows, columns = refused_arguments(cp, refusal, np.complex64)
    out = cp.full(a.shape, SENTINEL, dtype=a.dtype)
    assert not plan.fft_propagate_separable(a, out, rows, columns), (
        f"took a call with its {refusal}"
    )
    assert bool(cp.all(out == SENTINEL)), f"wrote the output with its {refusal}"
