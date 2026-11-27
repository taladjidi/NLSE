"""Tests for the pre-normalized propagator used by the fused linear step."""

import numpy as np
import pytest
from helpers import make
from NLSE import NLSE
from NLSE.backends import list_available_backends

PRECISION_COMPLEX = np.complex64

FUSED_BACKENDS = [b for b in list_available_backends() if b in ("CUPY", "CL")]

N = 64
waist = 2.23e-3
L = 10e-3
alpha = 0.0


def make_solver(backend, **kwargs):
    """Build a small NLSE solver on the given backend."""
    return make(NLSE, backend, n=N, **{"L": L, "alpha": alpha, **kwargs})


def to_numpy(simu, array):
    """Bring a possibly-device array back to numpy; None stays None."""
    if array is None or isinstance(array, np.ndarray):
        return array
    return simu._backend.to_numpy(array)


def record_propagators(simu, A, z, i, seen):
    """Keep the twin, the propagator and the step in force as the run holds them."""
    seen[:] = [
        to_numpy(simu, simu._propagator_fft),
        to_numpy(simu, simu.propagator),
        simu._step_in_force,
    ]


@pytest.mark.skipif(
    not FUSED_BACKENDS, reason="no CUPY/CL backend available on this machine"
)
@pytest.mark.parametrize("backend", FUSED_BACKENDS)
class TestPropagatorFFTRefresh:
    """_propagator_fft must always be derived from the current propagator."""

    def test_matches_propagator_after_normal_run(self, backend):
        """_propagator_fft is propagator / N after an ordinary run."""
        simu = make_solver(backend)
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        seen = []
        simu.out_field(
            E.copy(),
            2 * (L / 20),
            verbose=False,
            plot=False,
            delta_z=L / 20,
            callback=record_propagators,
            callback_args=(seen,),
        )

        prop_fft, prop, _ = seen
        if prop_fft is None:
            pytest.skip(f"{backend} does not use a pre-normalized propagator")
        np.testing.assert_allclose(
            prop_fft,
            prop / (N * N),
            rtol=1e-5,
            err_msg="_propagator_fft is not propagator/N",
        )

    def test_refreshed_when_step_limiter_clamps_delta_z(self, backend):
        """A clamped delta_z must refresh _propagator_fft, not just propagator."""
        simu = make_solver(backend)
        # 1000x a reasonable step reliably exceeds the split-step accuracy
        # limit. Keep z tied to the reasonable one, not to the inflated one,
        # so the run stays a handful of steps once clamped.
        dz_default = 1e-5
        z = 500 * dz_default
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)

        seen = []
        with pytest.warns(UserWarning, match="exceeds"):
            simu.out_field(
                E.copy(),
                z,
                verbose=False,
                plot=False,
                delta_z=1000 * dz_default,
                callback=record_propagators,
                callback_args=(seen,),
            )

        prop_fft, prop, delta_z = seen
        if prop_fft is None:
            pytest.skip(f"{backend} does not use a pre-normalized propagator")

        np.testing.assert_allclose(
            prop_fft,
            prop / (N * N),
            rtol=1e-5,
            err_msg=(
                "_propagator_fft is stale after the step limiter clamped "
                "delta_z: the fused linear step is using the pre-clamp "
                "propagator."
            ),
        )

        # And it must correspond to the clamped delta_z itself.
        expected = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * delta_z
        ).astype(np.complex64)
        np.testing.assert_allclose(
            prop,
            expected,
            rtol=1e-4,
            err_msg="propagator does not correspond to the clamped delta_z",
        )

    def test_clamped_run_matches_cpu(self, backend):
        """A clamped run must agree with the CPU reference."""
        reference = make_solver("CPU")
        E = np.exp(-(reference.XX**2 + reference.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        dz_default = 1e-5
        z = 500 * dz_default

        results = {}
        for name in ("CPU", backend):
            simu = make_solver(name)
            with pytest.warns(UserWarning, match="exceeds"):
                results[name] = np.asarray(
                    simu.out_field(
                        E.copy(),
                        z,
                        verbose=False,
                        plot=False,
                        delta_z=1000 * dz_default,
                    )
                )

        np.testing.assert_allclose(
            results[backend],
            results["CPU"],
            rtol=1e-3,
            atol=1e-3 * float(np.max(np.abs(results["CPU"]))),
            err_msg=f"{backend} disagrees with CPU after a clamped delta_z",
        )
