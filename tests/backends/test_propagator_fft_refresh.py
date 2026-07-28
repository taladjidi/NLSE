"""Tests for the pre-normalized propagator used by the fused linear step.

CUPY and CL absorb the inverse-FFT 1/N into a separate ``_propagator_fft``
array, and their fused kernels read it in preference to ``propagator``. Any
code path that rebuilds ``propagator`` therefore has to refresh
``_propagator_fft`` too, or the linear step keeps using the old one.

These only exercise a real code path on CUPY/CL, so they skip elsewhere.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import list_available_backends

PRECISION_COMPLEX = np.complex64

FUSED_BACKENDS = [b for b in list_available_backends() if b in ("CUPY", "CL")]

N = 64
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
power = 1.05
Isat = 10e4
L = 10e-3
alpha = 0.0


def make_solver(backend, **kwargs):
    """Build a small NLSE solver on the given backend."""
    params = {
        "alpha": alpha,
        "power": power,
        "window": window,
        "n2": n2,
        "V": None,
        "L": L,
        "NX": N,
        "NY": N,
        "Isat": Isat,
        "backend": backend,
    }
    params.update(kwargs)
    return NLSE(**params)


def to_numpy(simu, array):
    """Bring a possibly-device array back to numpy."""
    return simu._backend.to_numpy(array) if array is not None else None


@pytest.mark.skipif(
    not FUSED_BACKENDS, reason="no CUPY/CL backend available on this machine"
)
@pytest.mark.parametrize("backend", FUSED_BACKENDS)
class TestPropagatorFFTRefresh:
    """_propagator_fft must always be derived from the current propagator."""

    def test_matches_propagator_after_normal_run(self, backend):
        """_propagator_fft is propagator / N after an ordinary run."""
        simu = make_solver(backend)
        simu.delta_z = L / 20
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
        simu.out_field(E.copy(), 2 * simu.delta_z, verbose=False, plot=False)

        prop_fft = to_numpy(simu, getattr(simu, "_propagator_fft", None))
        if prop_fft is None:
            pytest.skip(f"{backend} does not use a pre-normalized propagator")
        prop = to_numpy(simu, simu.propagator)
        np.testing.assert_allclose(
            prop_fft,
            prop / (N * N),
            rtol=1e-5,
            err_msg="_propagator_fft is not propagator/N",
        )

    def test_refreshed_when_step_limiter_clamps_delta_z(self, backend):
        """A clamped delta_z must refresh _propagator_fft, not just propagator.

        _enforce_step_limit rebuilds propagator when it reduces delta_z, but
        runs after _precompute_step_constants has already derived
        _propagator_fft. Without an explicit refresh the fused linear step
        would advance by the pre-clamp dz while the nonlinear step used the
        clamped one.
        """
        simu = make_solver(backend)
        # 1000x the constructor default reliably exceeds the split-step
        # accuracy limit. Keep z tied to the default step, not to the
        # inflated one, so the run stays a handful of steps once clamped.
        dz_default = simu.delta_z
        simu.delta_z = 1000 * dz_default
        z = 500 * dz_default
        E = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)

        with pytest.warns(UserWarning, match="exceeds"):
            simu.out_field(E.copy(), z, verbose=False, plot=False)

        prop_fft = to_numpy(simu, getattr(simu, "_propagator_fft", None))
        if prop_fft is None:
            pytest.skip(f"{backend} does not use a pre-normalized propagator")
        prop = to_numpy(simu, simu.propagator)

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
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
        ).astype(np.complex64)
        np.testing.assert_allclose(
            prop,
            expected,
            rtol=1e-4,
            err_msg="propagator does not correspond to the clamped delta_z",
        )

    def test_clamped_run_matches_cpu(self, backend):
        """A clamped run must agree with the CPU reference.

        This is the end-to-end version: if _propagator_fft were stale, the
        GPU linear step would use a different dz from the CPU one.
        """
        reference = make_solver("CPU")
        E = np.exp(-(reference.XX**2 + reference.YY**2) / waist**2).astype(
            PRECISION_COMPLEX
        )
        dz_default = reference.delta_z
        z = 500 * dz_default

        results = {}
        for name in ("CPU", backend):
            simu = make_solver(name)
            simu.delta_z = 1000 * dz_default
            with pytest.warns(UserWarning, match="exceeds"):
                results[name] = np.asarray(
                    simu.out_field(E.copy(), z, verbose=False, plot=False)
                )

        np.testing.assert_allclose(
            results[backend],
            results["CPU"],
            rtol=1e-3,
            atol=1e-3 * float(np.max(np.abs(results["CPU"]))),
            err_msg=f"{backend} disagrees with CPU after a clamped delta_z",
        )
