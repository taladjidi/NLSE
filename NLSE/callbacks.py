import numpy as np
from scipy.constants import c, epsilon_0

from .solvers.nlse import NLSE


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
        A_sq = A_host.real * A_host.real + A_host.imag * A_host.imag
        n2 = simu._as_host_array(simu.n2)
        I_sat = simu._as_host_array(simu.I_sat)
        delta_n[i // save_every] = c * epsilon_0 / 2 * n2 * A_sq / (1 + A_sq / I_sat)


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
    accordingly.

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
        The new step, on the steps where it is updated, else None.
    """
    delta_z.append(simu._current_delta_z)
    if i % update_every != 0:
        return None
    A_sq = (A.real * A.real + A.imag * A.imag) * c * epsilon_0 / 2
    delta_n = np.abs(simu.n2) * A_sq / (1 + A_sq / simu.I_sat)
    z_nl = float(1 / (simu.k * delta_n.max()))
    return np.abs(z_nl) / 12


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
    scratch = simu._backend.allocate_real_field(
        trial.shape, np.float32 if np.dtype(dtype).itemsize == 8 else np.float64
    )
    # The propagator is built from the step, so a trial at another step needs
    # its own, and the run's has to come back whatever happens.
    saved = (simu.propagator, simu._propagator_fft)
    try:
        simu.propagator = simu._build_propagator(dtype, step)
        simu._send_propagator_to_gpu()
        for _ in range(count):
            trial = simu.split_step(
                trial, scratch, simu.V, simu.propagator, simu.plans, step, "double"
            )
    finally:
        simu.propagator, simu._propagator_fft = saved
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

    ``adapt_delta_z`` reads the step off the peak nonlinear index and divides
    by twelve. That is a rate, not an error: it says how fast the phase turns,
    and says nothing about how much of the answer the splitting is losing. The
    two come apart badly, because **the step that minimizes the error is not
    the smallest one**. Below the optimum the splitting error is already under
    the complex64 round-off floor and taking more steps only accumulates more
    of it -- measured on a lossless beam, dropping from 0.1 to 0.005 rad per
    step makes the answer nine times *worse* while costing twenty times more.

    So the step is chosen by measuring instead. Every ``update_every`` steps
    this takes the same distance twice, once whole and once in halves, and
    compares. For a method of order p the difference between them is the local
    error to within a constant, and the step that would have hit the tolerance
    follows from it:

        h_new = h * (tolerance / error) ** (1 / (p + 1))

    with p = 2 for the Strang splitting the loop runs.

    It costs three extra steps each time it fires, so ``update_every`` trades
    the overhead against how quickly the step follows the physics: at the
    default it is 15%. Being a callback, it also puts the run on the loop that
    dispatches per step rather than the one that hands the whole propagation
    to the backend, which on CUPY is the difference between replaying a
    captured graph and not.

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
        thousandth of the medium length. **A tolerance below the arithmetic floor
        cannot be met at any step**, and without this the controller answers
        one it cannot reach by halving for ever: the run does not diverge, it
        simply never ends.
    delta_z : list, optional
        Appended with the step in force at each call, for plotting after.

    Returns
    -------
    float or None
        The new step, on the steps where it is measured, else None. Also None
        under imaginary time, which this does not control -- see below.
    """
    step = simu._current_delta_z
    if delta_z is not None:
        delta_z.append(step)
    if i % update_every != 0 or step is None:
        return None
    # Imaginary time passes a complex step, and the two ends of this compare it
    # against a floor and a cap, which no complex number answers: it raised
    # TypeError rather than adapting anything. Declining is the honest reply --
    # imaginary time is a relaxation towards a ground state, not a trajectory,
    # so a local error against the step it took is not the quantity to control.
    if isinstance(step, complex):
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
    # The step the tolerance asks for, and the largest the physics allows.
    #
    # The cap is not a formality here. Below about 0.8 rad per step the
    # measured difference is the complex64 round-off floor rather than the
    # splitting error -- it sits flat at ~3e-7 from 0.1 to 0.4 rad -- so any
    # tolerance above the floor reads as "no error at all" and asks for a
    # larger step however large the step already is. Uncapped, that doubles
    # every time it fires and the run is unrecognisable within three
    # adjustments. The cap is recomputed from the current field, so it tightens
    # as a self-focusing beam raises the phase rate.
    cap = simu._split_step_max_dz(A)
    # Absolute, not relative to the step in force: a floor recomputed as a
    # fraction of the current step shrinks with it and never binds.
    floor = min_step if min_step is not None else float(simu.L) / 1e5
    if error == 0:
        return float(min(step * bounds[1], cap))

    order = 2
    factor = safety * (tolerance / error) ** (1.0 / (order + 1))
    proposed = step * float(np.clip(factor, *bounds))
    return float(min(max(proposed, floor), cap))
