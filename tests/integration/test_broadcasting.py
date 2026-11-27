"""Broadcasting a batch of simulations must agree with running them one by one."""

import numpy as np
import pytest
from helpers import AVAILABLE_BACKENDS
from NLSE import CNLSE, NLSE, CNLSE_1d
from NLSE.backends import get_backend
from NLSE.kernels import cpu as cpu_kernels

PRECISION_COMPLEX = np.complex64

# Every backend can run a non-local response.
NONLOCAL_BACKENDS = AVAILABLE_BACKENDS

N = 32
COUNT = 3
# A batch of one is the edge case: a (1, 1, 1) parameter holds a single element
# and can pass a naive "is this batched?" test, reaching numba as a raw array.
BATCH_SIZES = [1, 2, 3]
WAIST = 2.23e-3
WINDOW = 4 * WAIST
Z = 1e-3
DELTA_Z = 1e-4
# V/m. The runs here do not normalize the field, and at an amplitude of 1
# the ends of N2_VALUES gave the same output to 3e-7, under every tolerance
# below. At this one they are 0.69 rad apart at the centre of the beam.
FIELD_AMPLITUDE = 1e4

N2_VALUES = np.linspace(-1.6e-9, -1e-10, COUNT)
ALPHA_VALUES = np.array([0.0, 5.0, 20.0])
# Factors on n2 along a second batch axis, so that one parameter spans two.
SECOND_AXIS_SCALES = np.array([1.0, 0.5, 0.25])
# Factors on the potential, one per member, so that each has its own.
POTENTIAL_SCALES = np.linspace(1.0, 0.25, COUNT)
# An OpenCL launch reads the potential from its start, so there a potential
# per member cannot meet a parameter per member (_launch_count, kernels/cl.py).
OWN_V_BACKENDS = [b for b in AVAILABLE_BACKENDS if b != "CL"]
SHARED_V_ONLY = [b for b in AVAILABLE_BACKENDS if b == "CL"]
# Backends whose batch loop words the refusal of a shape itself.
OWN_REFUSAL = {"CPU": "does not broadcast", "CL": "does not broadcast against"}

BASE = {
    "power": 1.05,
    "window": WINDOW,
    "L": 10e-3,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}

# Saturation intensities, W/m^2, around the beam's peak of 1.3e5 W/m^2 at
# FIELD_AMPLITUDE: from deep saturation to almost none.
ISAT_VALUES = np.array([1e4, 1e5, 1e6])
# A non-local length of a few cells, so the grid resolves it, in m.
NONLOCAL_LENGTH = 4 * WINDOW / N
# Same tolerance as the batch tests below: float32 runs of the same graph.
MEMBER_RTOL = 1e-4
MEMBER_ATOL_FRACTION = 1e-5


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
    profile = np.exp(-(probe.XX**2 + probe.YY**2) / WAIST**2)
    field = (FIELD_AMPLITUDE * profile).astype(PRECISION_COMPLEX)
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
    """Broadcasting must hold in double precision too, where supported."""
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
    """A batch on a shared grid must not index past its end into stale memory."""
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


@pytest.mark.parametrize("backend_name", NONLOCAL_BACKENDS)
def test_a_batch_convolves_against_the_shared_kernel(backend_name):
    """A batched run must reach the non-local kernels too."""
    _, field = grids(backend_name)
    alpha_batched = np.asarray(ALPHA_VALUES).reshape(COUNT, 1, 1)

    batched = NLSE(
        alpha=alpha_batched,
        n2=N2_VALUES[0],
        V=None,
        backend=backend_name,
        nl_length=NONLOCAL_LENGTH,
        **BASE,
    )
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")
    assert got.shape == (COUNT, N, N)

    for index in range(COUNT):
        alone = NLSE(
            alpha=float(ALPHA_VALUES[index]),
            n2=N2_VALUES[0],
            V=None,
            backend=backend_name,
            nl_length=NONLOCAL_LENGTH,
            **BASE,
        )
        expected = propagate(alone, field, "split_step")
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=(
                f"{backend_name}: non-local batch slice {index} "
                f"(alpha={ALPHA_VALUES[index]:g}) differs from the same "
                f"simulation run on its own"
            ),
        )


def test_every_available_backend_is_covered():
    """No available backend may quietly drop out of the broadcasting matrix."""
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
    """Every batch size must work, including one."""
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


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
@pytest.mark.parametrize("with_potential", [False, True], ids=["no_V", "shared_V"])
def test_a_two_axis_batch_matches_the_same_runs_on_one_axis(
    backend_name, method, with_potential
):
    """A grid search over two parameters is the same runs laid out on a grid."""
    V, field = grids(backend_name)
    if not with_potential:
        V = None
    shape = (COUNT, ALPHA_VALUES.size)
    n2 = N2_VALUES.reshape(-1, 1, 1, 1)
    alpha = ALPHA_VALUES.reshape(1, -1, 1, 1)

    on_a_grid = make_solver(backend_name, n2, alpha, V)
    got = propagate(on_a_grid, np.broadcast_to(field, (*shape, N, N)), method)

    members = COUNT * ALPHA_VALUES.size
    in_a_row = make_solver(
        backend_name,
        np.broadcast_to(n2, (*shape, 1, 1)).reshape(members, 1, 1),
        np.broadcast_to(alpha, (*shape, 1, 1)).reshape(members, 1, 1),
        V,
    )
    expected = propagate(in_a_row, np.broadcast_to(field, (members, N, N)), method)

    assert got.shape == (*shape, N, N), f"two-axis batch returned {got.shape}"
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(
        got.reshape(expected.shape),
        expected,
        rtol=1e-4,
        atol=1e-5 * scale,
        err_msg=f"{backend_name}/{method}: a member got another member's values",
    )
    # Both parameters have to reach the kernel, or the grid is a grid of copies.
    assert np.max(np.abs(got[0, 0] - got[1, 0])) > 1e-2 * scale, "n2 was not swept"
    assert np.max(np.abs(got[0, 0] - got[0, 1])) > 1e-4 * scale, "alpha was not swept"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize(
    ("second_axis", "with_potential"),
    [(SECOND_AXIS_SCALES, False), (np.ones(1), True)],
    ids=["n2_over_both_axes", "n2_over_the_first_axis_shared_V"],
)
def test_one_parameter_on_a_two_axis_batch_keeps_every_value(
    backend_name, second_axis, with_potential
):
    """One parameter may span both batch axes, or only the first of them."""
    V, field = grids(backend_name)
    if not with_potential:
        V = None
    shape = (COUNT, SECOND_AXIS_SCALES.size)
    members = COUNT * SECOND_AXIS_SCALES.size
    n2 = N2_VALUES.reshape(-1, 1, 1, 1) * second_axis.reshape(1, -1, 1, 1)

    on_a_grid = make_solver(backend_name, n2, 0.0, V)
    got = propagate(on_a_grid, np.broadcast_to(field, (*shape, N, N)), "split_step")
    in_a_row = make_solver(
        backend_name,
        np.broadcast_to(n2, (*shape, 1, 1)).reshape(members, 1, 1),
        0.0,
        V,
    )
    expected = propagate(
        in_a_row, np.broadcast_to(field, (members, N, N)), "split_step"
    )

    np.testing.assert_allclose(
        got.reshape(expected.shape),
        expected,
        rtol=1e-4,
        atol=1e-5 * float(np.max(np.abs(expected))),
        err_msg=f"{backend_name}: a member got another member's n2",
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("with_potential", [False, True], ids=["no_V", "shared_V"])
def test_a_one_value_parameter_reaches_every_member(backend_name, with_potential):
    """A ``(1, 1, 1)`` parameter broadcasts against a batch of fields."""
    V, field = grids(backend_name)
    if not with_potential:
        V = None
    n2 = np.full((1, 1, 1), N2_VALUES[0])
    batched = make_solver(backend_name, n2, 0.0, V)
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")

    alone = make_solver(backend_name, float(N2_VALUES[0]), 0.0, V)
    expected = propagate(alone, field, "split_step")
    for index in range(COUNT):
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=f"{backend_name}: member {index} was not propagated",
        )


def test_the_cpu_loop_refuses_a_parameter_that_does_not_broadcast():
    """Three values against two fields is an error, not the first two values."""
    _, field = grids("CPU")
    fields = np.broadcast_to(field, (COUNT - 1, N, N)).copy()
    g = np.linspace(1.0, 2.0, COUNT).reshape(COUNT, 1, 1)
    with pytest.raises(ValueError, match="does not broadcast"):
        cpu_kernels.square_mod_nl_prop(fields, DELTA_Z, 0.0, g, BASE["Isat"])


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("shape", [(COUNT,), (COUNT, 1)], ids=["flat", "column"])
def test_every_backend_refuses_a_batch_parameter_without_the_grid_axes(
    backend_name, shape
):
    """``(3,)`` against a ``(3, N, N)`` field lines up with x, not the batch.

    The OpenCL loop used to read it as one value per member.
    """
    _, field = grids(backend_name)
    simu = make_solver(backend_name, N2_VALUES.reshape(shape), 0.0, None)
    with pytest.raises(ValueError, match=OWN_REFUSAL.get(backend_name, "broadcast")):
        propagate(simu, np.broadcast_to(field, (COUNT, N, N)), "split_step")


def own_potentials(backend_name):
    """Return one potential per member, a single field and a batched solver.

    Parameters
    ----------
    backend_name : str
        Backend the solver runs on.

    Returns
    -------
    tuple
        ``(potentials, field, solver)``: potentials of shape ``(COUNT, N, N)``
        and a solver with one ``n2`` per member.
    """
    V, field = grids(backend_name)
    potentials = (V * POTENTIAL_SCALES.reshape(-1, 1, 1)).astype(V.dtype)
    solver = make_solver(backend_name, N2_VALUES.reshape(-1, 1, 1), 0.0, potentials)
    return potentials, field, solver


@pytest.mark.parametrize("backend_name", OWN_V_BACKENDS)
def test_each_member_reads_its_own_potential(backend_name):
    """A potential per member and ``n2`` per member match the runs alone."""
    potentials, field, batched = own_potentials(backend_name)
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")
    for index in range(COUNT):
        alone = make_solver(
            backend_name, float(N2_VALUES[index]), 0.0, potentials[index]
        )
        expected = propagate(alone, field, "split_step")
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=f"{backend_name}: member {index} read another potential",
        )


@pytest.mark.parametrize("backend_name", SHARED_V_ONLY)
def test_opencl_refuses_a_potential_per_member_with_a_parameter_per_member(
    backend_name,
):
    """Every launch read the first potential, silently: now it is refused."""
    _, field, batched = own_potentials(backend_name)
    with pytest.raises(ValueError, match="shared by those simulations"):
        propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")


# ---------------------------------------------------------------------------
# Coupled solvers: the component axis precedes the grid axes, so the batch
# axis is one further out. The tests above only build single-component NLSE.

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
    profile = FIELD_AMPLITUDE * profile
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
    """The batch axis must not be mistaken for the component axis."""
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
    """Without coupled batching a backend warns and hands the batch to another."""
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
    """Batched constants drop to the rank of the one component a kernel gets."""
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


@pytest.mark.parametrize(
    "backend_name", [b for b in NONLOCAL_BACKENDS if b in COUPLED_BATCH_BACKENDS]
)
@pytest.mark.parametrize("cls,grid", COUPLED, ids=COUPLED_IDS)
def test_a_coupled_batch_convolves_against_the_shared_kernel(backend_name, cls, grid):
    """A coupled batch convolves with the shared kernel, once per component."""
    probe = make_coupled(cls, backend_name, N2_VALUES[0])
    field = coupled_field(probe, cls)

    def build(n2):
        kwargs = dict(
            COUPLED_BASE,
            alpha=0.0,
            n2=n2,
            V=None,
            NX=N,
            backend=backend_name,
            nl_length=NONLOCAL_LENGTH,
        )
        if cls is CNLSE:
            kwargs["NY"] = N
        return cls(**kwargs)

    values = N2_VALUES.reshape((COUNT,) + (1,) * (len(grid) + 1))
    got = propagate(
        build(values), np.broadcast_to(field, (COUNT, 2, *grid)), "split_step"
    )
    assert got.shape == (COUNT, 2, *grid)

    for index in range(COUNT):
        expected = propagate(build(float(N2_VALUES[index])), field, "split_step")
        np.testing.assert_allclose(
            got[index],
            expected,
            rtol=1e-4,
            atol=1e-5 * float(np.max(np.abs(expected))),
            err_msg=(
                f"{backend_name}/{cls.__name__}: non-local coupled batch slice "
                f"{index} differs from the same simulation run on its own"
            ),
        )


def assert_members_match(backend_name, got, expected, label):
    """Assert each member of a batched run equals its own single run.

    Parameters
    ----------
    backend_name : str
        Backend, for the message.
    got : np.ndarray
        The batched output, members along the first axis.
    expected : list of np.ndarray
        Each member's output when run alone.
    label : str
        What varies across members, for the message.
    """
    for index, alone in enumerate(expected):
        np.testing.assert_allclose(
            got[index],
            alone,
            rtol=MEMBER_RTOL,
            atol=MEMBER_ATOL_FRACTION * float(np.max(np.abs(alone))),
            err_msg=f"{backend_name}: {label}, member {index} differs from its own run",
        )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_lossy_field_batch_reads_its_shared_potential(backend_name):
    """A batch of fields runs under one scalar loss and one shared potential."""
    V, field = grids(backend_name)
    alpha = float(ALPHA_VALUES[-1])
    batched = make_solver(backend_name, N2_VALUES[0], alpha, V)
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")

    alone = propagate(
        make_solver(backend_name, N2_VALUES[0], alpha, V), field, "split_step"
    )
    assert_members_match(backend_name, got, [alone] * COUNT, "lossy, shared V")


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_batched_saturation_reaches_the_non_local_kernel(backend_name):
    """Each member saturates at its own Isat, through ``nl_prop``."""
    V, field = grids(backend_name)
    params = {"n2": N2_VALUES[0], "V": V, "nl_length": NONLOCAL_LENGTH, **BASE}
    isat = ISAT_VALUES.reshape(COUNT, 1, 1)
    batched = NLSE(alpha=0.0, backend=backend_name, **{**params, "Isat": isat})
    got = propagate(batched, np.broadcast_to(field, (COUNT, N, N)), "split_step")

    expected = [
        propagate(
            NLSE(alpha=0.0, backend=backend_name, **{**params, "Isat": float(value)}),
            field,
            "split_step",
        )
        for value in ISAT_VALUES
    ]
    assert_members_match(backend_name, got, expected, "batched Isat, non-local")
