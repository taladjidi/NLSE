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
from NLSE import CNLSE, NLSE, CNLSE_1d
from NLSE.backends import get_backend, list_available_backends

PRECISION_COMPLEX = np.complex64

AVAILABLE_BACKENDS = list_available_backends()

N = 32
COUNT = 3
# A batch of one is the edge case: a (1, 1, 1) parameter holds a single element
# and can pass a naive "is this batched?" test, reaching numba as a raw array.
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


def propagate(simu, field, method, splitting="lie"):
    """Propagate a copy of the field and return the result as numpy."""
    out = simu.out_field(
        field.copy(),
        Z,
        verbose=False,
        plot=False,
        method=method,
        splitting=splitting,
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
    arithmetic, so they must be splitting-agnostic. CL compiles a separate
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
        batched, np.broadcast_to(field, (COUNT, N, N)), "split_step", "strang"
    )

    for index in range(COUNT):
        alone = make_solver(backend_name, float(N2_VALUES[index]), 0.0, V)
        expected = propagate(alone, field, "split_step", "strang")
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

    A skipped backend hides both a backend that works and one that reads out
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


# ---------------------------------------------------------------------------
# Coupled solvers
# ---------------------------------------------------------------------------
# A coupled field carries a component axis before the grid axes, so a batch of
# them is one axis further out. Nothing above tests this: make_solver builds
# NLSE, so every check so far runs on a single-component field.

COUPLED_BASE = {
    "power": 1.05,
    "window": WINDOW,
    "n12": -1e-10,
    "L": 10e-3,
    "Isat": 10e4,
}
# Backends whose coupled kernels take one field of exactly the coupled rank.
COUPLED_BATCH_BACKENDS = [
    b for b in AVAILABLE_BACKENDS if b not in CNLSE._no_coupled_batch_backends
]
COUPLED_NO_BATCH = [
    b for b in AVAILABLE_BACKENDS if b in CNLSE._no_coupled_batch_backends
]


def make_coupled(cls, backend_name, n2):
    """Build a small coupled solver, 1D or 2D."""
    kwargs = dict(COUPLED_BASE, alpha=0.0, n2=n2, V=None, NX=N, backend=backend_name)
    if cls is CNLSE:
        kwargs["NY"] = N
    return cls(**kwargs)


def coupled_field(simu, cls):
    """Return a two-component input field with unequal components."""
    if cls is CNLSE:
        profile = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2)
    else:
        profile = np.exp(-(simu.X**2) / WAIST**2)
    return np.array([profile, 0.5 * profile]).astype(PRECISION_COMPLEX)


COUPLED = [(CNLSE, (N, N)), (CNLSE_1d, (N,))]
COUPLED_IDS = ["CNLSE", "CNLSE_1d"]


@pytest.mark.parametrize("backend_name", COUPLED_BATCH_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_coupled_batch_matches_individual_runs(backend_name, method, cls, grid):
    """Each slice of a batched coupled run must equal that run on its own."""
    probe = make_coupled(cls, backend_name, N2_VALUES[0])
    field = coupled_field(probe, cls)

    expected = [
        propagate(make_coupled(cls, backend_name, n2), field, method)
        for n2 in N2_VALUES
    ]

    batched = make_coupled(
        cls, backend_name, N2_VALUES.reshape((COUNT, 1, *(1,) * len(grid)))
    )
    got = propagate(batched, np.broadcast_to(field, (COUNT, 2, *grid)).copy(), method)

    assert got.shape == (COUNT, 2, *grid), (
        f"a batch of {COUNT} coupled fields came back as {got.shape}"
    )
    for i in range(COUNT):
        scale = float(np.max(np.abs(expected[i])))
        np.testing.assert_allclose(
            got[i],
            expected[i],
            rtol=2e-5,
            atol=2e-5 * scale,
            err_msg=f"{backend_name}/{cls.__name__}/{method}: slice {i} differs",
        )


@pytest.mark.parametrize("backend_name", COUPLED_BATCH_BACKENDS)
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_a_coupled_batch_still_loses_light(backend_name, cls, grid):
    """A batched coupled run must apply the real-space step, not skip it.

    It skipped it. A component of a batched coupled field is strided --
    (B, NX) out of (B, 2, NX) -- and the numba kernels open with A1.ravel(),
    which copies rather than views when the input is not contiguous. The step
    was applied to that copy and dropped when the kernel returned the argument
    it had been handed, so every batched coupled run on the CPU propagated the
    linear equation with its losses and its nonlinear phase missing.

    The test above this one uses alpha=0, and a nonlinearity weak enough to
    stay inside its tolerance, so it passed throughout. Losses are what makes
    the omission impossible to miss: with alpha > 0 the amplitude has to fall
    by exp(-alpha z / 2), and a run that skips the step comes back at exactly
    the amplitude it started with.

    A flat field, so the linear step is the identity and the only thing that
    can change the amplitude is the step being tested.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    cls : type
        Coupled solver class.
    grid : tuple
        Its spatial shape.
    """
    alpha = 20.0
    z = 3e-4
    amplitude = 0.9

    simu = make_coupled(cls, backend_name, N2_VALUES[0])
    # Both, because the constructor only gives the second component the
    # first's value as a default and they are separate parameters after that.
    simu.alpha = alpha
    simu.alpha2 = alpha
    field = np.full((2, *grid), amplitude, dtype=PRECISION_COMPLEX)
    batched = np.stack([field, field])

    out = simu.out_field(
        batched, z, verbose=False, plot=False, delta_z=5e-6, normalize=False
    )
    got = float(np.max(np.abs(np.asarray(as_numpy(simu, out))))) / amplitude

    assert got == pytest.approx(np.exp(-alpha * z / 2), rel=1e-4), (
        f"a batched coupled run on {backend_name} came back at {got:.6f} of "
        f"its amplitude where {np.exp(-alpha * z / 2):.6f} was due; at 1.0 it "
        f"applied no real-space step at all"
    )


@pytest.mark.parametrize("backend_name", COUPLED_BATCH_BACKENDS)
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_a_coupled_batch_keeps_the_components_apart(backend_name, cls, grid):
    """The batch axis must not be mistaken for the component axis.

    Indexing a coupled array at axis 0 selects a component when there is no
    batch and a simulation when there is. With equal components a swap is
    invisible, so the two here differ by a factor of two.
    """
    probe = make_coupled(cls, backend_name, N2_VALUES[0])
    field = coupled_field(probe, cls)
    batched = make_coupled(
        cls, backend_name, N2_VALUES.reshape((COUNT, 1, *(1,) * len(grid)))
    )
    got = propagate(
        batched, np.broadcast_to(field, (COUNT, 2, *grid)).copy(), "split_step"
    )

    for i in range(COUNT):
        power1 = float(np.sum(np.abs(got[i, 0]) ** 2))
        power2 = float(np.sum(np.abs(got[i, 1]) ** 2))
        assert power1 > power2, (
            f"simulation {i}: component 1 should carry the larger power "
            f"({power1:.3e} vs {power2:.3e})"
        )


@pytest.mark.parametrize("backend_name", COUPLED_NO_BATCH)
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_backends_without_coupled_batching_fall_back(backend_name, cls, grid):
    """The batch is the run; the backend is how it is run, so the batch wins.

    What must not happen is a reshaped array coming back silently.
    ``CNLSE._no_coupled_batch_backends`` is what keeps the tests above off
    these backends, so it has to match what the solvers do.
    """
    probe = make_coupled(cls, backend_name, N2_VALUES[0])
    field = coupled_field(probe, cls)
    batched = make_coupled(
        cls, backend_name, N2_VALUES.reshape((COUNT, 1, *(1,) * len(grid)))
    )
    with pytest.warns(UserWarning, match=r"[Bb]roadcasting"):
        got = propagate(
            batched, np.broadcast_to(field, (COUNT, 2, *grid)).copy(), "split_step"
        )
    assert got.shape == (COUNT, 2, *grid)
    assert batched._backend.name not in CNLSE._no_coupled_batch_backends


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_an_unbatched_coupled_run_is_never_refused(backend_name, cls, grid):
    """The guard must not catch ordinary single runs on any backend."""
    simu = make_coupled(cls, backend_name, N2_VALUES[0])
    got = propagate(simu, coupled_field(simu, cls), "split_step")
    assert got.shape == (2, *grid)


@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
@pytest.mark.parametrize("rank", ["coupled", "component"], ids=["coupled", "component"])
def test_batched_constants_are_reduced_to_component_rank(cls, grid, rank):
    """The kernels are handed one component, so the constants must match it.

    A caller shapes a batched parameter against the field, which for a coupled
    solver includes the component axis -- n2 of (count, 1, 1, 1) against a
    field of (count, 2, NY, NX). ``_take_components`` then hands the kernels a
    (count, NY, NX) component, one axis short of it.

    CPU slices the batch itself and tolerates the extra axis; CuPy broadcasts
    for real and produces (count, count, NY, NX). This runs on CPU and pins
    the shape CuPy needs, so the mismatch does not have to be found on a GPU.
    """
    shape = (
        (COUNT, 1, *(1,) * len(grid))
        if rank == "coupled"
        else (COUNT, *(1,) * len(grid))
    )
    simu = make_coupled(cls, "CPU", N2_VALUES.reshape(shape))
    V = np.zeros(grid, dtype=np.float32)

    simu._precompute_step_constants(V, np.complex64)

    component_ndim = len(grid)
    for name in sorted(simu._step_constants()):
        value = getattr(simu, name)
        ndim = getattr(value, "ndim", 0)
        assert ndim <= component_ndim + 1, (
            f"{name} has shape {tuple(value.shape)}: it cannot broadcast "
            f"against a {(COUNT, *grid)} component"
        )

    # And the reduction must keep the values, not just the rank.
    batched_g = np.asarray(simu._g11).reshape(COUNT)
    single = [
        float(np.asarray(make_coupled(cls, "CPU", n2)._step_constants()["_g11"]))
        for n2 in N2_VALUES
    ]
    np.testing.assert_allclose(batched_g, single, rtol=1e-6)


@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_a_parameter_varying_over_components_is_refused(cls, grid):
    """n22/alpha2/Isat2 are how a component gets its own value, not this axis."""
    simu = make_coupled(
        cls, "CPU", np.array([-1e-9, -2e-9]).reshape(1, 2, *(1,) * len(grid))
    )
    with pytest.raises(ValueError, match="component axis"):
        simu._precompute_step_constants(None, np.complex64)
