"""Broadcasting a batch of simulations must agree with running them one by one.

Broadcasting is what makes a parameter sweep cheap, so a batched run has to
be equivalent to the individual runs, not merely finite and correctly shaped.
This file previously asserted only the output shape, which is what let two
silent out-of-bounds reads survive: a shared propagator indexed with the
batched field's flat index, and a shared potential indexed the same way when
no parameter was batched at all.

Every backend runs the same matrix. A backend that cannot broadcast fails
here rather than skipping, so the four implementations cannot drift apart:

- CPU loops over the batch and calls the numba kernel per slice.
- CUPY falls back to its cp.fuse kernels, which broadcast natively.
- MLX broadcasts inside the traced graph.
- CL wraps a shared grid with ``idx % N_grid`` and places a per-simulation
  launch with ``global_offset``.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends

PRECISION_COMPLEX = np.complex64

AVAILABLE_BACKENDS = list_available_backends()

N = 32
COUNT = 3
# A batch of one is the edge case: a (1, 1, 1) parameter used to slip past the
# "is this batched?" test on CPU and CL because it holds a single element, and
# reached numba as a raw array ("No implementation of function imul found for
# signature (complex64, array(complex128, 3d, C))").
BATCH_SIZES = [1, 2, 3]
WAIST = 2.23e-3
WINDOW = 4 * WAIST
Z = 1e-3
DELTA_Z = 1e-4

N2_VALUES = np.linspace(-1.6e-9, -1e-10, COUNT)
ALPHA_VALUES = np.array([0.0, 5.0, 20.0])

BASE = {
    "power": 1.05,
    "window": WINDOW,
    "L": 10e-3,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}


def as_numpy(simu, array):
    """Return a backend array as numpy, whatever backend produced it."""
    if isinstance(array, np.ndarray):
        return array
    return simu._backend.to_numpy(array)


def make_solver(backend_name, n2, alpha, V):
    """Build a small NLSE solver with a fixed step."""
    simu = NLSE(alpha=alpha, n2=n2, V=V, backend=backend_name, **BASE)
    return simu


def grids(backend_name):
    """Return a shared potential and a single-simulation input field."""
    probe = make_solver(backend_name, N2_VALUES[0], 0.0, None)
    V = (-1e-4 * np.exp(-(probe.XX**2 + probe.YY**2) / (70e-6) ** 2)).astype(np.float32)
    field = np.exp(-(probe.XX**2 + probe.YY**2) / WAIST**2).astype(PRECISION_COMPLEX)
    return V, field


def propagate(simu, field, method, precision="single"):
    """Propagate a copy of the field and return the result as numpy."""
    out = simu.out_field(
        field.copy(),
        Z,
        verbose=False,
        plot=False,
        method=method,
        precision=precision,
        normalize=False,
        delta_z=DELTA_Z,
    )
    return np.asarray(as_numpy(simu, out))


# case -> (n2 per simulation, alpha per simulation). A case where both stay
# constant is the field-only batch: nothing but the field carries the axis.
CASES = {
    "field_only": (None, None),
    "batched_n2": (N2_VALUES, None),
    "batched_alpha": (None, ALPHA_VALUES),
}


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("with_potential", [False, True], ids=["no_V", "shared_V"])
def test_batch_matches_individual_runs(backend_name, method, case, with_potential):
    """Each slice of a batched run must equal that simulation run on its own."""
    n2_values, alpha_values = CASES[case]
    V, field = grids(backend_name)
    if not with_potential:
        V = None

    n2_batched = (
        N2_VALUES[0]
        if n2_values is None
        else np.asarray(n2_values).reshape(COUNT, 1, 1)
    )
    alpha_batched = (
        0.0 if alpha_values is None else np.asarray(alpha_values).reshape(COUNT, 1, 1)
    )

    batched = make_solver(backend_name, n2_batched, alpha_batched, V)
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), method)

    assert got.shape == (COUNT, N, N), f"batched run returned {got.shape}"
    assert np.all(np.isfinite(got)), (
        "batched run produced non-finite values: a shared grid was indexed past its end"
    )

    for index in range(COUNT):
        n2 = N2_VALUES[0] if n2_values is None else float(n2_values[index])
        alpha = 0.0 if alpha_values is None else float(alpha_values[index])
        alone = make_solver(backend_name, n2, alpha, V)
        expected = propagate(alone, field, method)
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=(
                f"{backend_name}/{method}/{case}: batch slice {index} "
                f"(n2={n2:.3e}, alpha={alpha:g}) differs from the same "
                f"simulation run on its own"
            ),
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_double_precision_batch_matches_individual_runs(backend_name):
    """Broadcasting must hold in double precision too, where supported.

    The shared-grid wrap and the per-simulation launch are indexing, not
    arithmetic, so they must be precision-agnostic. CL compiles a separate
    double-precision program, which is a second copy of every kernel.
    """
    backend = get_backend(backend_name)
    if not backend.supports_double_precision():
        pytest.skip(f"{backend_name} does not support double precision")
    V, field = grids(backend_name)
    field = field.astype(np.complex128)
    n2_batched = N2_VALUES.reshape(COUNT, 1, 1)

    batched = make_solver(backend_name, n2_batched, 0.0, V)
    got = propagate(
        batched, np.broadcast_to(field, (COUNT, N, N)), "split_step", "double"
    )

    for index in range(COUNT):
        alone = make_solver(backend_name, float(N2_VALUES[index]), 0.0, V)
        expected = propagate(alone, field, "split_step", "double")
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-8,
            atol=1e-10 * float(np.max(np.abs(expected))),
            err_msg=f"double-precision batch slice {index} differs",
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_shared_grid_is_not_indexed_past_its_end(backend_name):
    """Pin the failure mode directly: the batch must not read stale memory.

    A field of ones against a potential of zeros must come back with every
    slice identical. Reading past the end of the potential returns whatever
    the allocator left there, so the first slice is right and the rest are
    not — which is why this looked like flaky test ordering rather than an
    out-of-bounds read.
    """
    V, field = grids(backend_name)
    V = np.zeros_like(V)
    batched = make_solver(backend_name, N2_VALUES[0], 0.0, V)
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")

    for index in range(1, COUNT):
        np.testing.assert_array_equal(
            got[index],
            got[0],
            err_msg=(
                f"{backend_name}: slice {index} differs from slice 0 although "
                f"every simulation in the batch is identical"
            ),
        )


def test_every_available_backend_is_covered():
    """No backend may quietly drop out of the matrix above.

    Broadcasting used to be CPU- and CUPY-only, with CL and MLX skipping.
    The skips hid that MLX had worked all along and that CL was reading out
    of bounds, so absence of coverage is itself the failure.
    """
    assert AVAILABLE_BACKENDS, "no backends available to test"
    for backend_name in AVAILABLE_BACKENDS:
        V, field = grids(backend_name)
        batched = make_solver(backend_name, N2_VALUES.reshape(COUNT, 1, 1), 0.0, V)
        got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")
        assert got.shape == (COUNT, N, N), (
            f"{backend_name} does not broadcast: got {got.shape}"
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("count", BATCH_SIZES)
@pytest.mark.parametrize("with_potential", [False, True], ids=["no_V", "shared_V"])
def test_any_batch_size_matches_individual_runs(backend_name, count, with_potential):
    """Every batch size must work, including one.

    A batch of one is not a curiosity: it is what a sweep degenerates to at
    the ends of a scan, and what a caller writes when parametrising code that
    sometimes has a single value.

    The no-potential case is the one that matters. With a shared V the batch
    is inferred from the grid being smaller than the field, which hides a
    broken parameter test; with V=None only the parameter can reveal it.
    """
    V, field = grids(backend_name)
    if not with_potential:
        V = None
    values = np.linspace(N2_VALUES[0], N2_VALUES[-1], count)
    batched = make_solver(backend_name, values.reshape(count, 1, 1), 0.0, V)
    got = propagate(batched, np.broadcast_to(field, (count, N, N)), "split_step")

    assert got.shape == (count, N, N)
    for index, value in enumerate(values):
        alone = make_solver(backend_name, float(value), 0.0, V)
        expected = propagate(alone, field, "split_step")
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=f"{backend_name}: batch of {count}, slice {index} differs",
        )
