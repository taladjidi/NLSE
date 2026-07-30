import numpy as np
from scipy.constants import atomic_mass, c, epsilon_0, hbar

from ..utils import __BACKEND__
from .nlse import NLSE
from .parameter import Parameter


class GPE(NLSE):
    """A class to solve GPE."""

    # A density in atoms/m^2, not an optical intensity: no conversion, and a
    # label to match. Everything else about the plot is NLSE's.
    _plot_density_scale = 1.0
    _plot_density_label = "Density (at/m^2)"

    # GPE integrates NLSE's equation with cold-gas physics on each term, so it
    # shares the storage rather than keeping a second copy of every value.
    gamma = Parameter("alpha", "Losses in Hz.")
    N = Parameter("power", "Total number of atoms.")
    m = Parameter("k", "Mass of one atom in kg.")
    g = Parameter("n2", "Interaction energy in Hz*m^2.")
    sat = Parameter("I_sat", "Saturation parameter in Hz/m^2.", scale=epsilon_0 * c / 2)

    def __init__(
        self,
        gamma: float,
        N: float,
        window: float,
        g: float,
        V: np.ndarray | None,
        m: float = 87 * atomic_mass,
        NX: int = 1024,
        NY: int = 1024,
        sat: float = np.inf,
        nl_length: float = 0,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the simulation.

        Solves an equation : d/dt psi = -1/2m(d2/dx2 + d2/dy2) psi + V psi +
          g psi**2 psi

        Parameters
        ----------
        gamma : float
            Losses in Hz.
        N : float
            Total number of atoms.
        window : float
            Window size in m.
        g : float
            Interaction energy in Hz*m^2.
        V : np.ndarray
            Potential in Hz.
        m : float, optional
            Mass of one atom in kg.
            Defaults to 87*atomic_mass for Rubidium 87.
        NX : int, optional
            Number of points in x.
            Defaults to 1024.
        NY : int, optional
            Number of points in y.
            Defaults to 1024.
        sat : float
            Saturation parameter in Hz/m^2.
        nl_length : float
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute.
        backend : str, optional
            "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=gamma,
            power=N,
            window=window,
            n2=g,
            V=V,
            L=0,
            NX=NX,
            NY=NY,
            Isat=sat,
            nl_length=nl_length,
            wvl=2 * np.pi / m,
            backend=backend,
        )
        self.V = V
        # do some conversion for the units
        self.I_sat *= epsilon_0 * c / 2
        # GPE uses quantum units (Hz), not optical (W), so norm_constant is 1.0
        self._norm_constant = np.float32(1.0)

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for GPE propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.m),
        )

    def _compute_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Compute the GPE linear propagation matrix."""
        return np.exp(
            -1j * 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m * delta_z,
            dtype=dtype,
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for GPE RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.m))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw GPE dispersion operator for RK4."""
        return -1j * 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m

    def _dispersion_operator(self) -> np.ndarray:
        """Return the GPE dispersion eigenvalues.

        The GPE dispersion operator is hbar*K^2/(2m), unlike NLSE's K^2/(2k).
        """
        return 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m
