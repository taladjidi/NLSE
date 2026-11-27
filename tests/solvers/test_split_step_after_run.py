"""A ``split_step`` taken by hand after ``out_field`` uses what it is given."""

import numpy as np
import pytest
from NLSE import CNLSE, NLSE
from scipy.constants import c, epsilon_0

N = 32
WINDOW = 4e-3  # m
L = 1e-2  # m, length of the medium
DELTA_Z = 1e-4  # m, the step taken by hand
PHASE = 0.5  # rad, what the potential or the coupling writes in that step
N2 = -1e-9  # m^2/W
# Measured error: 6.0e-8 rad on CPU and MLX, one float32 ulp of PHASE.
TOLERANCE = 1e-6  # rad


def solver(backend, cls=NLSE, **kwargs):
    """Return a small lossless solver, with any constructor argument overridden.

    Parameters
    ----------
    backend : str
        Backend name.
    cls : type
        NLSE or CNLSE.
    **kwargs
        Constructor arguments overriding the defaults.

    Returns
    -------
    NLSE
        The solver.
    """
    base = {
        "alpha": 0,
        "power": 1,
        "window": WINDOW,
        "n2": 0,
        "V": None,
        "L": L,
        "NX": N,
        "NY": N,
        "Isat": 1e30,
        "backend": backend,
    }
    if cls is CNLSE:
        base["n12"] = 0
    return cls(**{**base, **kwargs})


def with_potential(simu):
    """Give ``simu`` the uniform potential that writes ``PHASE`` in one step.

    Parameters
    ----------
    simu : NLSE
        The solver.

    Returns
    -------
    np.ndarray
        The potential.
    """
    simu.V = np.full((N, N), 2 * PHASE / (simu.k * DELTA_Z), dtype=np.float32)
    return simu.V


def run(simu, field, **kwargs):
    """Propagate ``field`` over ``2 L``, past the medium, unnormalized.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        The field.
    **kwargs
        Further out_field arguments.
    """
    simu.out_field(
        field,
        2 * L,
        delta_z=DELTA_Z,
        verbose=False,
        plot=False,
        normalize=False,
        **kwargs,
    )


def stop(simu, A, z, i):
    """Fail the run at its first step.

    Parameters
    ----------
    simu : NLSE
        The solver.
    A : Any
        The field, where the backend keeps it.
    z : float
        Distance reached, in m.
    i : int
        Step index.

    Raises
    ------
    RuntimeError
        Always.
    """
    raise RuntimeError("stopped by the test")


def phase_after_one_step(simu, field, V):
    """Return the phase at the centre after one ``split_step`` by hand.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        Field on the host.
    V : np.ndarray or None
        Potential passed to ``split_step``.

    Returns
    -------
    np.ndarray
        The phase at the centre of each component.
    """
    backend = simu._backend
    A = backend.from_numpy(field)
    A_sq = backend.allocate_real_field(field.shape, np.float32)
    propagator = simu._build_propagator(np.complex64, DELTA_Z)
    V = None if V is None else backend.from_numpy(V)
    out = simu.split_step(A, A_sq, V, propagator, simu._build_fft_plan(A), DELTA_Z)
    return np.angle(np.asarray(backend.to_numpy(out))[..., N // 2, N // 2])


@pytest.mark.parametrize("factor", [0, 1, 2])
def test_a_step_after_a_run_takes_the_potential_it_is_given(backend, factor):
    """``V`` times ``factor`` writes ``factor`` times ``PHASE``; 0 is no ``V``."""
    simu = solver(backend)
    potential = with_potential(simu)
    field = np.ones((N, N), dtype=np.complex64)
    run(simu, field)
    V = None if factor == 0 else factor * potential
    got = phase_after_one_step(simu, field, V)
    assert got == pytest.approx(factor * PHASE, abs=TOLERANCE)


def test_a_step_after_a_run_past_the_medium_keeps_the_coupling(backend):
    """The run left the medium; the step by hand still has ``k n2 I dz``."""
    simu = solver(backend, n2=N2)
    run(simu, np.ones((N, N), dtype=np.complex64))
    intensity = PHASE / (simu.k * abs(N2) * DELTA_Z)  # W/m^2
    amplitude = np.sqrt(2 * intensity / (c * epsilon_0))
    field = np.full((N, N), amplitude, dtype=np.complex64)
    got = phase_after_one_step(simu, field, None)
    assert got == pytest.approx(simu.k * N2 * intensity * DELTA_Z, abs=TOLERANCE)


def test_a_step_after_a_failed_run_takes_no_potential(backend):
    """A run that raised leaves no potential behind either."""
    simu = solver(backend)
    with_potential(simu)
    field = np.ones((N, N), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="stopped by the test"):
        run(simu, field, callback=stop)
    assert phase_after_one_step(simu, field, None) == pytest.approx(0, abs=TOLERANCE)


def test_a_coupled_step_after_a_run_takes_no_potential(backend):
    """Neither component of a CNLSE keeps the run's potential."""
    simu = solver(backend, cls=CNLSE)
    with_potential(simu)
    field = np.ones((2, N, N), dtype=np.complex64)
    run(simu, field)
    got = phase_after_one_step(simu, field, None)
    assert got == pytest.approx([0, 0], abs=TOLERANCE)
