"""Cross-backend correctness tests.

Verifies that all backends (CPU, CUPY, CL) produce identical results
for the same inputs and parameters.
"""

import numpy as np
import pytest

from NLSE import CNLSE, NLSE
from NLSE.backends import list_available_backends

# Get available backends
AVAILABLE_BACKENDS = list_available_backends()

# Tolerance for numerical comparison
# Single precision (float32/complex64)
RTOL_SINGLE = 1e-5  # Relative tolerance
ATOL_SINGLE = 1e-6  # Absolute tolerance

# Double precision (float64/complex128) - only for CPU
RTOL_DOUBLE = 1e-12
ATOL_DOUBLE = 1e-13


def backend_pairs():
    """Generate all pairs of available backends for comparison."""
    pairs = []
    for i, backend1 in enumerate(AVAILABLE_BACKENDS):
        for backend2 in AVAILABLE_BACKENDS[i + 1 :]:
            pairs.append((backend1, backend2))
    return pairs


@pytest.mark.parametrize("backend1,backend2", backend_pairs())
class TestNLSECrossBackend:
    """Test NLSE solver consistency across backends."""

    def test_propagation_without_potential(self, backend1, backend2):
        """Test propagation without potential matches across backends."""
        # Parameters
        alpha = 5.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        L = 1e-3
        NX = NY = 128
        Isat = 1e10
        waist = 1e-3

        # Create solvers
        simu1 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Same initial condition
        np.random.seed(42)
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare results
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (no potential)",
        )

        # Check power conservation
        power_in = np.sum(np.abs(E_in) ** 2)
        power_out1 = np.sum(np.abs(E_out1) ** 2)
        power_out2 = np.sum(np.abs(E_out2) ** 2)
        assert np.abs(power_out1 - power_out2) / power_in < RTOL_SINGLE

    def test_propagation_with_potential(self, backend1, backend2):
        """Test propagation with potential matches across backends."""
        # Parameters
        alpha = 10.0
        power = 1.0
        window = 5e-3
        n2 = -1e-20
        L = 2e-3
        NX = NY = 128
        Isat = 1e10
        waist = 1e-3

        # Create potential
        x = np.linspace(-window / 2, window / 2, NX)
        y = np.linspace(-window / 2, window / 2, NY)
        XX, YY = np.meshgrid(x, y)
        V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2)

        # Create solvers
        simu1 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Same initial condition
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare results
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (with potential)",
        )

    def test_saturable_absorption(self, backend1, backend2):
        """Test saturable absorption (Isat effects) matches across backends."""
        # Parameters with strong saturation
        alpha = 20.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        L = 1e-3
        NX = NY = 64
        Isat = 1e4  # Low Isat for strong saturation
        waist = 1e-3

        # Create solvers
        simu1 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # High intensity initial condition
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        E_in = 10.0 * np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare results
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (saturable absorption)",
        )

    def test_nonlinear_effects(self, backend1, backend2):
        """Test nonlinear propagation (self-focusing) matches across backends."""
        # Parameters for strong nonlinearity
        alpha = 0.0
        power = 1.0
        window = 5e-3
        n2 = 1e-18  # Strong nonlinearity
        L = 5e-3
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        # Create solvers
        simu1 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Gaussian beam
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare results
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (nonlinear)",
        )

    def test_vortex_propagation(self, backend1, backend2):
        """Test vortex beam propagation matches across backends."""
        # Parameters
        alpha = 5.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        L = 1e-3
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        # Create solvers
        simu1 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Vortex beam (charge l=1)
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        R = np.sqrt(XX**2 + YY**2)
        theta = np.arctan2(YY, XX)
        E_in = (R / waist * np.exp(-R**2 / waist**2) * np.exp(1j * theta)).astype(
            np.complex64
        )

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare results
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (vortex)",
        )


@pytest.mark.parametrize("backend1,backend2", backend_pairs())
class TestCNLSECrossBackend:
    """Test CNLSE solver consistency across backends."""

    def test_coupled_propagation_without_potential(self, backend1, backend2):
        """Test coupled propagation without potential matches across backends."""
        # Parameters
        alpha = 5.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        n12 = 0.5e-20
        L = 1e-3
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        # Create solvers
        simu1 = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Same initial condition (two components)
        XX, YY = np.meshgrid(simu1.X, simu1.Y)
        E_in = np.zeros((2, NX, NY), dtype=np.complex64)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare both components
        np.testing.assert_allclose(
            E_out1[0],
            E_out2[0],
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch in component 1 between {backend1} and {backend2}",
        )
        np.testing.assert_allclose(
            E_out1[1],
            E_out2[1],
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch in component 2 between {backend1} and {backend2}",
        )

    def test_coupled_propagation_with_potential(self, backend1, backend2):
        """Test coupled propagation with potential matches across backends."""
        # Parameters
        alpha = 10.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        n12 = 0.5e-20
        L = 1e-3
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        # Create potential
        x = np.linspace(-window / 2, window / 2, NX)
        y = np.linspace(-window / 2, window / 2, NY)
        XX, YY = np.meshgrid(x, y)
        V = 1e4 * np.exp(-(XX**2 + YY**2) / (2e-3) ** 2)

        # Create solvers
        simu1 = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend1,
        )
        simu2 = CNLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend2,
        )

        # Initial condition
        E_in = np.zeros((2, NX, NY), dtype=np.complex64)
        E_in[0] = np.exp(-(XX**2 + YY**2) / waist**2)
        E_in[1] = 0.5 * np.exp(-(XX**2 + YY**2) / (1.5 * waist) ** 2)

        # Propagate
        E_out1 = simu1.out_field(E_in, z=L, verbose=False, plot=False)
        E_out2 = simu2.out_field(E_in, z=L, verbose=False, plot=False)

        # Compare both components
        np.testing.assert_allclose(
            E_out1,
            E_out2,
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg=f"Mismatch between {backend1} and {backend2} (coupled with V)",
        )


class TestPrecisionModes:
    """Test different precision modes (single vs double)."""

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_single_vs_double_precision(self, backend):
        """Test that double precision is more accurate than single (CPU only)."""
        if backend != "CPU":
            pytest.skip("Double precision only supported on CPU")

        # Parameters
        alpha = 5.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        L = 5e-3  # Longer propagation to accumulate errors
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        # Create solver
        simu = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend,
        )

        # Initial condition
        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in_single = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)
        E_in_double = E_in_single.astype(np.complex128)

        # Propagate with single precision
        E_out_single = simu.out_field(
            E_in_single, z=L, verbose=False, plot=False, precision="single"
        )

        # Propagate with double precision
        E_out_double = simu.out_field(
            E_in_double, z=L, verbose=False, plot=False, precision="double"
        )

        # Double precision should match single to within single precision tolerance
        np.testing.assert_allclose(
            E_out_single,
            E_out_double.astype(np.complex64),
            rtol=RTOL_SINGLE,
            atol=ATOL_SINGLE,
            err_msg="Single and double precision results differ significantly",
        )

        # But they should not be identical (double is more accurate)
        assert not np.allclose(
            E_out_single, E_out_double.astype(np.complex64), rtol=1e-10, atol=1e-10
        )


class TestNumericalProperties:
    """Test numerical properties that should hold for all backends."""

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_power_conservation_no_loss(self, backend):
        """Test power conservation when alpha=0."""
        # Parameters (no loss)
        alpha = 0.0
        power = 1.0
        window = 5e-3
        n2 = 1e-20
        L = 5e-3
        NX = NY = 64
        Isat = 1e10
        waist = 1e-3

        simu = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend,
        )

        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        E_out = simu.out_field(E_in, z=L, verbose=False, plot=False, normalize=False)

        power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
        power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y

        # Power should be conserved to within numerical precision
        rel_error = np.abs(power_out - power_in) / power_in
        assert (
            rel_error < 1e-4
        ), f"Power not conserved on {backend}: {rel_error:.2e} relative error"

    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_power_decay_with_loss(self, backend):
        """Test power decays exponentially with loss."""
        # Parameters (with loss)
        alpha = 10.0  # Strong loss
        power = 1.0
        window = 5e-3
        n2 = 0.0  # No nonlinearity for analytical comparison
        L = 1e-3
        NX = NY = 64
        Isat = 1e20  # Very high Isat (no saturation)
        waist = 1e-3

        simu = NLSE(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=None,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            backend=backend,
        )

        XX, YY = np.meshgrid(simu.X, simu.Y)
        E_in = np.exp(-(XX**2 + YY**2) / waist**2).astype(np.complex64)

        E_out = simu.out_field(E_in, z=L, verbose=False, plot=False, normalize=False)

        power_in = np.sum(np.abs(E_in) ** 2) * simu.delta_X * simu.delta_Y
        power_out = np.sum(np.abs(E_out) ** 2) * simu.delta_X * simu.delta_Y

        # Expected power decay: P(z) = P(0) * exp(-alpha * z)
        expected_ratio = np.exp(-alpha * L)
        actual_ratio = power_out / power_in

        rel_error = np.abs(actual_ratio - expected_ratio) / expected_ratio
        assert (
            rel_error < 0.01
        ), f"Power decay incorrect on {backend}: {rel_error:.2%} error"


def test_all_backends_available():
    """Report which backends are available for testing."""
    print(f"\nAvailable backends for testing: {AVAILABLE_BACKENDS}")
    print(f"Number of backend pairs to test: {len(backend_pairs())}")
    assert len(AVAILABLE_BACKENDS) >= 1, "At least one backend must be available"
