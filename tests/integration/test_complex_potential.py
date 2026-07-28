"""A complex potential is an absorbing one, and must behave the same everywhere.

``V`` may be complex: its imaginary part enters the real part of the exponent,
so it is gain or loss rather than phase. That is how an absorbing boundary is
built.

It used to work on CPU only. ``_send_arrays_to_gpu`` cast V with
``dtype=np.float32``, which on every device backend threw the imaginary part
away — silently, apart from a ``ComplexWarning`` buried in the test output. The
same input therefore produced different physics on CPU than on CUPY, CL or
MLX, and an absorbing potential simply stopped absorbing.

The kernels take V as a bare pointer, so real and complex V cannot share an
entry point. Each backend compiles a ``_cv`` twin of every V-reading kernel and
dispatches on V's dtype, which keeps a real V — the common case — on exactly
the instruction stream it always had.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 64
WAIST = 2.23e-3
WINDOW = 4 * WAIST
Z = 2e-3
DELTA_Z = 1e-4

BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": WINDOW,
    "n2": -1.6e-9,
    "L": 10e-3,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}


def grids(backend_name):
    """Return the radial grid, an input field and an absorbing ring."""
    probe = NLSE(V=None, backend=backend_name, **BASE)
    r_sq = probe.XX**2 + probe.YY**2
    field = np.exp(-r_sq / WAIST**2).astype(np.complex64)
    # A ring of loss around the beam: the standard absorbing boundary.
    ring = np.exp(-((np.sqrt(r_sq) - 2e-3) ** 2) / (3e-4) ** 2)
    return field, ring


def propagate(backend_name, V, field, dtype=np.complex64):
    """Propagate the field under V and return the result as numpy."""
    simu = NLSE(V=V, backend=backend_name, **BASE)
    simu.delta_z = DELTA_Z
    out = simu.out_field(
        field.astype(dtype), Z, verbose=False, plot=False, normalize=False
    )
    return np.asarray(
        out if isinstance(out, np.ndarray) else simu._backend.to_numpy(out)
    )


def norm(field):
    """Return the total power of a field."""
    return float(np.sum(np.abs(field) ** 2))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_an_absorbing_potential_absorbs(backend_name):
    """A positive imaginary V must remove power, not merely add phase.

    This is the test the old float32 cast would have failed on every device
    backend: with the imaginary part discarded the run is lossless, so the
    norm comes back unchanged.
    """
    field, ring = grids(backend_name)
    V = (1j * 2e2 * ring).astype(np.complex64)
    out = propagate(backend_name, V, field)

    assert np.all(np.isfinite(out)), f"{backend_name} produced non-finite values"
    absorbed = 1 - norm(out) / norm(field)
    assert absorbed > 0.01, (
        f"{backend_name}: an absorbing potential removed only "
        f"{100 * absorbed:.3f}% of the power. The imaginary part of V is "
        f"being ignored, so the potential is not absorbing at all."
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_complex_potential_matches_the_cpu_reference(backend_name):
    """Every backend must agree with CPU on a complex potential."""
    field, ring = grids(backend_name)
    V = (1j * 2e2 * ring).astype(np.complex64)

    got = propagate(backend_name, V, field)
    expected = propagate("CPU", V, field)
    np.testing.assert_allclose(
        got,
        expected,
        rtol=1e-4,
        atol=1e-5 * float(np.max(np.abs(expected))),
        err_msg=(
            f"{backend_name} disagrees with CPU on a complex potential: the "
            f"gain/loss term is not being applied the same way"
        ),
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_real_potential_is_unaffected(backend_name):
    """Adding complex-V support must not disturb the real-V path.

    A real V and the same V with a zero imaginary part describe identical
    physics, so they must give identical results even though they take
    different kernels.
    """
    field, ring = grids(backend_name)
    real = (5.0 * ring).astype(np.float32)
    as_complex = real.astype(np.complex64)

    np.testing.assert_allclose(
        propagate(backend_name, as_complex, field),
        propagate(backend_name, real, field),
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            f"{backend_name}: a complex V with zero imaginary part differs "
            f"from the equivalent real V, so the two kernels disagree"
        ),
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_imaginary_part_sets_the_direction(backend_name):
    """Positive imaginary V loses power, negative gains it.

    The amplitude is small on purpose. V enters scaled by k/2 ~ 4e6, so the
    accumulated exponent over this run is ~8e3 * Im(V): anything much larger
    saturates the lossy case to zero and overflows the gain case to inf, and
    the sign would no longer be what is under test.
    """
    field, ring = grids(backend_name)
    amplitude = 1e-4
    lossy = propagate(backend_name, (1j * amplitude * ring).astype(np.complex64), field)
    gainy = propagate(
        backend_name, (-1j * amplitude * ring).astype(np.complex64), field
    )

    assert np.all(np.isfinite(lossy)) and np.all(np.isfinite(gainy)), (
        f"{backend_name}: a modest complex potential produced non-finite values"
    )
    assert norm(lossy) < norm(field) < norm(gainy), (
        f"{backend_name}: expected loss < input < gain, got "
        f"{norm(lossy):.4g} / {norm(field):.4g} / {norm(gainy):.4g}"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize(
    "field_dtype,expected",
    [(np.complex64, np.complex64), (np.complex128, np.complex128)],
    ids=["single", "double"],
)
def test_potential_dtype_follows_the_field(backend_name, field_dtype, expected):
    """V is transferred at the field's width, complex staying complex.

    The kernels pick their precision from the field and then read V with it,
    so a mismatched V is read at the wrong width. V was previously pinned to
    float32 regardless, which both broke double precision and deleted the
    absorption.
    """
    backend = get_backend(backend_name)
    if field_dtype == np.complex128 and not backend.supports_double_precision():
        pytest.skip(f"{backend_name} has no fp64")

    field, ring = grids(backend_name)
    V = (1j * 1e2 * ring).astype(np.complex64)
    simu = NLSE(V=V, backend=backend_name, **BASE)
    simu.delta_z = DELTA_Z
    simu.out_field(
        field.astype(field_dtype), Z, verbose=False, plot=False, normalize=False
    )

    sent = simu._potential_dtype(V, field_dtype)
    assert sent == expected, (
        f"{backend_name}: a {np.dtype(field_dtype).name} field should carry a "
        f"{np.dtype(expected).name} potential, got {np.dtype(sent).name}"
    )


def test_a_real_potential_stays_real():
    """A real V must not be promoted to complex by the precision rule."""
    simu = NLSE(V=None, backend="CPU", **BASE)
    V = np.ones((N, N), dtype=np.float32)
    assert simu._potential_dtype(V, np.complex64) == np.float32
    assert simu._potential_dtype(V, np.complex128) == np.float64
