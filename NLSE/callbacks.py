import numpy as np
from scipy.constants import c, epsilon_0

from .solvers.nlse import NLSE

# Steps per nonlinear length that adapt_delta_z aims at, dimensionless. A
# heuristic, not a derived value: the rule's value since commit bc56be6.
STEPS_PER_NONLINEAR_LENGTH = 12
# Default floor of adapt_delta_z_to_error, as a fraction of the medium length,
# or of the run for a solver that has none. Dimensionless; a choice.
MIN_STEP_FRACTION = 1e-5
# Order of the Strang splitting adapt_delta_z_to_error's trial steps run.
STRANG_ORDER = 2


def sample(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    E_samples: np.ndarray,
) -> None:
    """Save samples of the field.

    This callback will save samples every save_every steps into the E_samples
    array.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        The current field.
    z : float
        The current propagation distance.
    i : int
        Step number.
    save_every : int
        Number of propagation steps between each step.
    E_samples : np.ndarray
        Array to store the samples.
    """
    if i % save_every == 0:
        # A is whatever the backend holds, and only the CPU one holds a numpy
        # array. Assigning into E_samples copies, so nothing more is needed --
        # it used to call A.copy(), which mlx arrays do not have.
        E_samples[i // save_every] = simu._as_host_array(A)


def norm(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    norms: np.ndarray,
) -> None:
    """Save the norm of the field.

    This callback will save the norm of the field every save_every steps into the
    norms array.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        The current field.
    z : float
        The current propagation distance.
    i : int
        Step number.
    save_every : int
        Number of propagation steps between each step.
    norms : np.ndarray
        Array to store the norms.
    """
    if i % save_every == 0:
        # On the host, because .sum() is not something every backend's array
        # has -- pyopencl's does not -- and the answer is one number that has
        # to reach a numpy array anyway.
        A_host = simu._as_host_array(A)
        norms[i // save_every] = (
            A_host.real * A_host.real + A_host.imag * A_host.imag
        ).sum()


def evaluate_delta_n(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    save_every: int,
    delta_n: np.ndarray,
) -> None:
    """Evaluate the non-linear refractive index change.

    This will evaluate the weight of the non-linear refractive index change, allowing
    to adjust the step size accordingly.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        The current field.
    z : float
        The current propagation distance.
    i : int
        Step number.
    save_every : int
        Number of propagation steps between each step.
    delta_n : np.ndarray
        The array of delta_n values.
    """
    if i % save_every == 0:
        # Everything on the host: A is the backend's array, and n2 and I_sat
        # are scalars for an ordinary run but device arrays for a batched one
        # that _send_arrays_to_gpu has already moved.
        A_host = simu._as_host_array(A)
        intensity = (A_host.real * A_host.real + A_host.imag * A_host.imag) * (
            c * epsilon_0 / 2
        )
        n2 = simu._as_host_array(simu.n2)
        I_sat = simu._as_host_array(simu.I_sat)
        delta_n[i // save_every] = n2 * intensity / (1 + intensity / I_sat)


def adapt_delta_z(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    update_every: int,
    delta_z: list,
) -> float | None:
    """Update the simulation step size.

    This callback will update the simulation step size every update_every steps by
    computing the nonlinear refractive index change and adjusting the step size
    accordingly. Never past the limit of the scheme the run uses,
    ``_max_dz_in_force``, which ``adapt_delta_z_to_error`` is held to as well.

    A callback changes the step by returning it. The loop rebuilds the
    propagator to match before taking the next step; assigning the step
    somewhere would leave the propagator built from the previous one, and the
    linear part would advance by the wrong distance.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        The current field.
    z : float
        The current propagation distance.
    i : int
        Step number.
    update_every : int
        Update the step size every update_every steps.
    delta_z : list
        A list to store the size of the steps.

    Returns
    -------
    float or None
        The new step, on the steps where it is updated, else None. None too
        where nothing turns the phase (no interaction, or past the medium):
        the step in force stands.
    """
    delta_z.append(simu._current_delta_z)
    if i % update_every != 0:
        return None
    # The solver's own phase rate: k n2 I for NLSE, with the cross coupling for
    # the coupled solvers, and g n for GPE, whose k, n2 and optical intensity
    # are not physical quantities.
    peak = simu._max_interaction_rate(A)
    if peak == 0:
        # The nonlinear length is infinite, which is not a step.
        return None
    return min(1 / (STEPS_PER_NONLINEAR_LENGTH * peak), simu._max_dz_in_force(A))


def _trial_propagation(simu: NLSE, A: np.ndarray, step: float, count: int):
    """Propagate a copy of the field for a few steps, leaving the run alone.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        Current field. Not modified.
    step : float
        Step to take.
    count : int
        How many of them.

    Returns
    -------
    Any
        The trial field, left where it was computed. Bringing it back would
        cost more than the steps it took.
    """
    dtype = simu._field_dtype(A)
    trial = simu._backend.copy_field(A)
    scratch = simu._backend.allocate_real_field(trial.shape, simu._real_dtype(dtype))
    # The propagator is built from the step, so a trial at another step needs
    # its own, and the run's has to come back whatever happens.
    saved = simu._linear_state()
    try:
        simu.propagator = simu._build_propagator(dtype, step)
        simu._send_propagator_to_gpu()
        for _ in range(count):
            trial = simu.split_step(
                trial, scratch, simu.V, simu.propagator, simu.plans, step, "strang"
            )
    finally:
        simu._restore_linear_state(saved)
    return trial


def adapt_delta_z_to_error(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    tolerance: float = 1e-4,
    update_every: int = 20,
    bounds: tuple = (0.5, 2.0),
    safety: float = 0.9,
    min_step: float | None = None,
    delta_z: list | None = None,
) -> float | None:
    """Set the step from a measured local error, not from a heuristic.

    ``adapt_delta_z`` reads the step off a phase rate, which says nothing
    about the error the splitting makes. This measures it: every
    ``update_every`` steps it takes the same distance twice, once whole
    and once in halves. For a method of order p their difference is
    the local error to within a constant, and the step that would have hit
    the tolerance follows from it:

        h_new = h * (tolerance / error) ** (1 / (p + 1))

    with p = 2: the trial steps are Strang's, whatever scheme the run takes.
    Only the cap follows the scheme.

    It costs three extra steps each time it fires, 15% at the default
    ``update_every``, and as a callback it puts the run on the loop that
    dispatches per step. The smallest step is not the most accurate one:
    on a lossless complex64 beam, going from 0.1 to 0.005 rad per step
    made the answer nine times worse for twenty times the steps.
    docs/callbacks.md gives the limits on the tolerance.

    Parameters
    ----------
    simu : NLSE
        Simulation object.
    A : np.ndarray
        The current field.
    z : float
        The current propagation distance.
    i : int
        Step number.
    tolerance : float
        Relative local error to aim for.
    update_every : int
        Measure every this many steps.
    bounds : tuple
        Smallest and largest factor by which one adjustment may change the
        step. Keeps a single noisy measurement from moving it far.
    safety : float
        Factor applied to the suggested step, so that aiming at the tolerance
        does not repeatedly overshoot it.
    min_step : float, optional
        Smallest step to shrink to, in metres. Defaults to a hundred
        thousandth of the medium length, or of the run for a solver without
        one (GPE). **A tolerance below the arithmetic floor
        cannot be met at any step**, and without this the controller answers
        one it cannot reach by halving for ever: the run does not diverge, it
        simply never ends.
    delta_z : list, optional
        Appended with the step in force at each call, for plotting after.

    Returns
    -------
    float or None
        The size of the new step, on the steps where it is measured, else
        None. The loop gives it the sign of the run.
    """
    # Not the step just taken: one cut to land on L or z is a sliver, and a
    # proposal scaled from it takes several steps to grow back.
    step = simu._step_in_force
    if delta_z is not None:
        delta_z.append(step)
    if i % update_every != 0 or step is None:
        return None

    whole = _trial_propagation(simu, A, step, 1)
    halves = _trial_propagation(simu, A, step / 2, 2)
    # Reduced where the fields already are: only the two scalars cross, and
    # only the step that comes out of them has to reach the host at all,
    # because the host is what rebuilds the propagator. Bringing both fields
    # back for a numpy norm instead cost 45% of the step on CUPY.
    scale = simu._backend.norm(halves)
    if scale == 0:
        return None
    error = simu._backend.norm(whole - halves) / scale
    # Capped at the physical limit: in complex64 the error estimate can read
    # round-off as no error at all, so any tolerance above that floor keeps doubling
    # the step. The cap follows the current field and tightens as a beam focuses.
    cap = simu._max_dz_in_force(A)
    # Absolute, not relative to the step in force: a relative floor shrinks with
    # the step and never binds. Not the round-off optimum either, which is far too
    # coarse on a field with spectral content (see DEFAULT_PHASE_PER_STEP).
    length = float(simu.L) or abs(simu._run_distance)
    floor = min(min_step if min_step is not None else length * MIN_STEP_FRACTION, cap)
    if error == 0:
        return float(min(abs(step) * bounds[1], cap))

    factor = safety * (tolerance / error) ** (1.0 / (STRANG_ORDER + 1))
    # A size: the floor and the cap are sizes, and a backward step is negative.
    proposed = abs(step) * float(np.clip(factor, *bounds))
    return float(min(max(proposed, floor), cap))
