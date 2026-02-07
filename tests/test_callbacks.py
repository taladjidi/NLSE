"""Tests for the callbacks module."""

from functools import partial

import numpy as np

from NLSE import NLSE
from NLSE.callbacks import adapt_delta_z, evaluate_delta_n, norm, sample

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 256
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 1e-3
alpha = 20


def _make_simu(alpha_val=0) -> NLSE:
    """Helper to create a basic NLSE simulation."""
    return NLSE(
        alpha_val,
        power,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="CPU",
    )


def test_sample_callback() -> None:
    """Test that the sample callback saves field snapshots."""
    simu = _make_simu()
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    n_steps = int(np.ceil(L / simu.delta_z))
    save_every = max(1, n_steps // 5)
    n_samples = n_steps // save_every + 1
    E_samples = np.zeros((n_samples, N, N), dtype=PRECISION_COMPLEX)
    cb = partial(sample, save_every=save_every, E_samples=E_samples)
    simu.out_field(E, L, verbose=False, plot=False, precision="single", callback=cb)
    # first sample should be set (step 0)
    assert not np.allclose(E_samples[0], 0), "First sample was not saved"
    # check at least one more sample is set
    any_nonzero = any(not np.allclose(E_samples[k], 0) for k in range(1, n_samples))
    assert any_nonzero, "No samples beyond the first were saved"


def test_norm_callback() -> None:
    """Test that the norm callback tracks field norm over propagation."""
    simu = _make_simu()
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    n_steps = int(np.ceil(L / simu.delta_z))
    save_every = max(1, n_steps // 5)
    n_samples = n_steps // save_every + 1
    norms = np.zeros(n_samples, dtype=PRECISION_REAL)
    cb = partial(norm, save_every=save_every, norms=norms)
    simu.out_field(E, L, verbose=False, plot=False, precision="single", callback=cb)
    # norms should be positive (field has energy)
    assert norms[0] > 0, "First norm is zero"
    # without losses (alpha=0), norms should be roughly conserved
    nonzero_norms = norms[norms > 0]
    assert len(nonzero_norms) > 1, "Not enough norm samples"
    assert np.allclose(
        nonzero_norms, nonzero_norms[0], rtol=1e-3
    ), "Norm not conserved (no losses case)"


def test_evaluate_delta_n_callback() -> None:
    """Test that evaluate_delta_n produces reasonable values."""
    simu = _make_simu()
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    n_steps = int(np.ceil(L / simu.delta_z))
    save_every = max(1, n_steps // 3)
    n_samples = n_steps // save_every + 1
    delta_n = np.zeros((n_samples, N, N), dtype=PRECISION_REAL)
    cb = partial(evaluate_delta_n, save_every=save_every, delta_n=delta_n)
    simu.out_field(E, L, verbose=False, plot=False, precision="single", callback=cb)
    # delta_n should be nonzero for a field with nonzero n2
    assert not np.allclose(delta_n[0], 0), "delta_n is zero at step 0"


def test_adapt_delta_z_callback() -> None:
    """Test that adapt_delta_z adjusts the step size."""
    simu = _make_simu(alpha_val=alpha)
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    delta_z_history = []
    update_every = 5
    cb = partial(
        adapt_delta_z,
        update_every=update_every,
        delta_z=delta_z_history,
    )
    simu.out_field(E, L, verbose=False, plot=False, precision="single", callback=cb)
    # adapt_delta_z should have recorded step sizes
    assert len(delta_z_history) > 0, "No step sizes recorded"
    # step size should be positive
    assert all(dz > 0 for dz in delta_z_history), "Negative step size found"


def test_multiple_callbacks() -> None:
    """Test passing a list of callbacks."""
    simu = _make_simu()
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    n_steps = int(np.ceil(L / simu.delta_z))
    save_every = max(1, n_steps // 3)
    n_samples = n_steps // save_every + 1
    norms = np.zeros(n_samples, dtype=PRECISION_REAL)
    E_samples = np.zeros((n_samples, N, N), dtype=PRECISION_COMPLEX)
    callbacks = [
        partial(norm, save_every=save_every, norms=norms),
        partial(sample, save_every=save_every, E_samples=E_samples),
    ]
    # callbacks passed as a list with individual arg tuples
    simu.out_field(
        E,
        L,
        verbose=False,
        plot=False,
        precision="single",
        callback=callbacks,
        callback_args=[(), ()],
    )
    assert norms[0] > 0, "Norm callback did not fire"
    assert not np.allclose(E_samples[0], 0), "Sample callback did not fire"
