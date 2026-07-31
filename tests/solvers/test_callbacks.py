"""The built-in callbacks work on every backend, not only the CPU one.

``sample``, ``norm`` and ``evaluate_delta_n`` are exported from the package
and documented as built-ins, and nothing exercised them. All three took the
field as though it were a numpy array -- ``A.copy()``, ``.sum()``, arithmetic
assigned straight into a numpy buffer -- which is true only of the CPU
backend. CuPy mimics numpy closely enough to have carried them too, which is
presumably why it lasted: the one example that uses a callback asks for CUPY.

On the others they raised, each differently and none informatively:
``TypeError: must be real number, not Array`` from ``sample`` on CL,
``AttributeError: 'Array' object has no attribute 'sum'`` from ``norm`` on CL,
``AttributeError: 'mlx.core.array' object has no attribute 'copy'`` from
``sample`` on MLX.

So what is checked here is that they run at all, and then that they agree with
the CPU across backends -- a callback that ran but recorded the wrong thing
would be worse than one that raised.
"""

import warnings

import numpy as np
import pytest
from NLSE import NLSE, adapt_delta_z, evaluate_delta_n, norm, sample
from NLSE.backends import list_available_backends
from NLSE.callbacks import adapt_delta_z_to_error

AVAILABLE_BACKENDS = list_available_backends()

N = 32
WAIST = 2.23e-3
L = 1e-3
DELTA_Z = 1e-4
SAVE_EVERY = 2
SLOTS = round(L / DELTA_Z) // SAVE_EVERY + 2

BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 4 * WAIST,
    "n2": -1.6e-9,
    "V": None,
    "L": L,
    "NX": N,
    "NY": N,
    "Isat": 10e4,
}


def _buffers():
    """Return one output buffer per callback, keyed by name."""
    return {
        "sample": np.zeros((SLOTS, N, N), dtype=np.complex64),
        "norm": np.zeros(SLOTS),
        "evaluate_delta_n": np.zeros((SLOTS, N, N), dtype=np.float32),
    }


CALLBACKS = {"sample": sample, "norm": norm, "evaluate_delta_n": evaluate_delta_n}


def _run(backend_name, which):
    """Propagate with one built-in callback and return what it recorded.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    which : str
        Which built-in callback to pass.

    Returns
    -------
    np.ndarray
        The buffer the callback filled.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    buffer = _buffers()[which]
    simu.out_field(
        field,
        L,
        verbose=False,
        plot=False,
        delta_z=DELTA_Z,
        callback=CALLBACKS[which],
        callback_args=(SAVE_EVERY, buffer),
    )
    return buffer


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("which", sorted(CALLBACKS))
def test_a_builtin_callback_runs_on_this_backend(backend_name, which):
    """The callback must survive whatever array type the backend hands it.

    Parameters
    ----------
    backend_name : str
        Backend to run on.
    which : str
        Which built-in callback to pass.
    """
    recorded = _run(backend_name, which)
    assert np.any(recorded != 0), (
        f"{which} recorded nothing on {backend_name}; it ran but wrote no "
        f"samples, which a buffer of the wrong shape would also do"
    )
    assert np.all(np.isfinite(np.abs(recorded)))


@pytest.mark.parametrize("backend_name", [b for b in AVAILABLE_BACKENDS if b != "CPU"])
@pytest.mark.parametrize("which", sorted(CALLBACKS))
def test_a_builtin_callback_records_what_the_cpu_records(backend_name, which):
    """Running is not enough: it has to record the same physics.

    Compared as magnitudes and loosely, because the backends differ in
    summation order and in how they round a single-precision step, not
    because the quantity is approximate.

    Parameters
    ----------
    backend_name : str
        Backend to compare against the CPU.
    which : str
        Which built-in callback to pass.
    """
    np.testing.assert_allclose(
        np.abs(_run(backend_name, which)),
        np.abs(_run("CPU", which)),
        rtol=2e-2,
        atol=1e-6,
        err_msg=f"{which} disagrees with the CPU on {backend_name}",
    )


ADAPTIVE = {
    "adapt_delta_z": (adapt_delta_z, (5, [])),
    "adapt_delta_z_to_error": (adapt_delta_z_to_error, ()),
}


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("which", sorted(ADAPTIVE))
def test_an_adaptive_callback_drives_a_run_on_this_backend(which, backend_name):
    """The callbacks that choose the step must also survive a device array.

    These two were left out when the other three were fixed, because unlike
    them they had tests -- which build the solver with its default backend and
    so only ever ran on the CPU. ``adapt_delta_z`` reads the peak nonlinear
    index with ``delta_n.max()``, and pyopencl's array has no ``max``, so it
    raised AttributeError there and nowhere else.

    Parameters
    ----------
    which : str
        Which adaptive callback to drive the run with.
    backend_name : str
        Backend to run on.
    """
    simu = NLSE(backend=backend_name, **BASE)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    callback, args = ADAPTIVE[which]
    with warnings.catch_warnings():
        # The step limiter says so when it lowers the step; that is not what
        # is under test here.
        warnings.simplefilter("ignore")
        out = simu.out_field(
            field,
            L,
            verbose=False,
            plot=False,
            delta_z=DELTA_Z,
            callback=callback,
            callback_args=args,
        )
    recorded = np.asarray(
        out if isinstance(out, np.ndarray) else simu._backend.to_numpy(out)
    )
    assert np.all(np.isfinite(np.abs(recorded))), (
        f"{which} drove {backend_name} to a non-finite field"
    )
