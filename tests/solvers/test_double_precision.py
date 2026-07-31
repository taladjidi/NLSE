"""Tests that a double-precision run stays double precision throughout.

A run's splitting is set by the dtype of the field handed to ``out_field``,
and everything the step touches has to follow it: the scratch buffers, the
intensity buffer the coupled solvers keep, and the dispersion operator RK4
multiplies by.

Nothing did. Three places pinned the width to single -- the RK4 buffers, the
coupled intensity buffer, and ``_compute_propagator_rk4`` -- so **every
complex128 RK4 run was broken on every backend**: pyfftw refused the array
outright, and cuFFT returned NaN because the CUDA kernels pick their precision
from the field and then index an operator of the other width. split_step was
unaffected, which is why the suite stayed green: no test ran RK4 in double.
"""

import numpy as np
import pytest
from NLSE import CNLSE, GPE, NLSE, CNLSE_1d, NLSE_1d
from NLSE.backends import list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 32
L = 1e-3
window = 8.9e-3


def build(cls, backend):
    """Return a solver of this class with parameters its constructor takes."""
    if cls is GPE:
        return GPE(
            gamma=0,
            N=1e6,
            window=window,
            g=1e-3,
            V=None,
            m=1e-27,
            NX=N,
            NY=N,
            backend=backend,
        )
    params = {
        "alpha": 20,
        "power": 1.05,
        "window": window,
        "n2": -1.6e-9,
        "V": None,
        "L": L,
        "NX": N,
        "Isat": 1e5,
        "backend": backend,
    }
    if cls in (CNLSE, CNLSE_1d):
        params["n12"] = -1e-10
    if cls not in (NLSE_1d, CNLSE_1d):
        params["NY"] = N
    return cls(**params)


def initial_field(cls, dtype):
    """Return a smooth two- or one-component field of this dtype."""
    one_d = cls in (NLSE_1d, CNLSE_1d)
    x = np.linspace(-window / 2, window / 2, N)
    if one_d:
        r2 = x**2
    else:
        X, Y = np.meshgrid(x, x)
        r2 = X**2 + Y**2
    single = np.exp(-r2 / (window / 4) ** 2)
    if cls in (CNLSE, CNLSE_1d):
        return np.stack([single, single]).astype(dtype)
    return single.astype(dtype)


SOLVERS = [NLSE, NLSE_1d, CNLSE, CNLSE_1d, GPE]


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_rk4_runs_in_double_precision(backend_name, cls):
    """A complex128 RK4 run must return finite complex128, not NaN."""
    solver = build(cls, backend_name)
    if not solver._backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision")
    out = solver.out_field(
        initial_field(cls, np.complex128),
        L,
        verbose=False,
        plot=False,
        splitting="lie",
        method="RK4",
    )
    out = np.asarray(solver._backend.to_numpy(out))
    assert np.all(np.isfinite(out.view(np.float64))), (
        f"{cls.__name__} on {backend_name} returned non-finite values from a "
        f"double-precision RK4 run"
    )
    assert out.dtype == np.complex128


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_double_and_single_agree(backend_name, cls):
    """The two widths must describe the same physics.

    A double run that merely avoids NaN could still be reading a
    single-precision operator; agreement to float32 round-off is the check
    that the whole step went double.
    """
    solver = build(cls, backend_name)
    if not solver._backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no double precision")

    def run(dtype):
        s = build(cls, backend_name)
        out = s.out_field(
            initial_field(cls, dtype),
            L,
            verbose=False,
            plot=False,
            splitting="lie",
            method="RK4",
        )
        return np.asarray(s._backend.to_numpy(out)).astype(np.complex128)

    single, double = run(np.complex64), run(np.complex128)
    scale = max(float(np.max(np.abs(double))), 1e-30)
    assert float(np.max(np.abs(single - double))) / scale < 1e-5


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_the_rk4_buffers_take_the_field_dtype(backend_name, cls):
    """The scratch buffers must match the field, not a fixed width."""
    for dtype, real_dtype in (
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ):
        solver = build(cls, backend_name)
        if not solver._backend.supports_double_precision():
            pytest.skip(f"{backend_name} has no double precision")
        field = solver._backend.from_numpy(initial_field(cls, dtype))
        solver._allocate_rk4_buffers(field, "RK4")
        for name in ("_rk4_k", "_rk4_A_tmp", "_rk4_acc"):
            buffer = getattr(solver, name)
            assert np.dtype(buffer.dtype) == dtype, (
                f"{cls.__name__} {name} is {buffer.dtype} for a {dtype} field"
            )
        intensity = getattr(solver, "_rk4_A_sq_c", None)
        if intensity is not None:
            assert np.dtype(intensity.dtype) == real_dtype


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_the_rk4_operator_takes_the_field_dtype(backend_name, cls):
    """The dispersion operator must match the field it multiplies.

    The GPU kernels select single or double from the field and index the
    operator with the same flat id, so an operator of the other width is
    read as the wrong type.
    """
    solver = build(cls, backend_name)
    for dtype in (np.complex64, np.complex128):
        operator = solver._build_propagator_rk4(dtype)
        assert np.dtype(operator.dtype) == dtype

    # Both widths are cached, and the cache must not hand back the other one.
    assert np.dtype(solver._build_propagator_rk4(np.complex64).dtype) == np.complex64
