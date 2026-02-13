import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__
from .nlse import NLSE


class NLSE_1d(NLSE):
    """A class to solve NLSE in 1d."""

    def __init__(
        self,
        alpha: float,
        power: float,
        window: float,
        n2: float,
        V: np.ndarray | None,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiate the simulation.

        Solves an equation : d/dz psi = -1/2k0(d2/dx2) psi + k0 dn psi +
          k0 n2 psi**2 psi

        Parameters
        ----------
        alpha : float
            Transmission coeff.
        power : float
            Power in W.
        n2 : float
            Non linear coeff in m^2/W.
        V : np.ndarray
            Potential.
        L : float
            Length of the medium.
        Isat : float
            Saturation intensity in W/m^2.
        nl_length : float
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute.
        wvl : float, optional
            Wavelength in m. Defaults to 780 nm.
        backend : str, optional
            "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self._last_axes = (-1,)
        self.nl_profile = self.nl_profile[0]
        self.nl_profile /= self.nl_profile.sum()

        # Override normalization factor for 1D (delta_X^2 instead of delta_X * delta_Y)
        self._norm_grid_factor = np.float32(self.delta_X**2)

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Uses caching to avoid recomputing propagators with identical parameters.

        Parameters
        ----------
        precision : str, optional
            "single" or "double" precision. Defaults to "single".

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        # Create cache key (1D version)
        cache_key = (self.NX, float(self.delta_z), precision, float(self.k))

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        dtype = np.complex128 if precision == "double" else np.complex64
        propagator = np.exp(
            -1j * 0.5 * (self.Kx**2) / self.k * self.delta_z, dtype=dtype
        )

        # Cache for future use
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _build_propagator_rk4(self) -> np.ndarray:
        """Build raw 1D dispersion operator for RK4 (no exp, no delta_z).

        Returns
        -------
        np.ndarray
            The raw dispersion operator.
        """
        cache_key = (self.NX, "RK4", float(self.k))
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        propagator = (-1j * 0.5 * self.Kx**2 / self.k).astype(np.complex64)
        self._propagator_cache[cache_key] = propagator
        return propagator

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance in m.
        """
        fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        if A_plot.ndim == 2:
            for i in range(A_plot.shape[0]):
                ax[0].plot(
                    self.X * 1e3,
                    1e-4 * c / 2 * epsilon_0 * np.abs(A_plot[i, :]) ** 2,
                )
                ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot[i, :])))
        elif A_plot.ndim == 1:
            ax[0].plot(self.X * 1e3, 1e-4 * c / 2 * epsilon_0 * np.abs(A_plot) ** 2)
            ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot)))
        ax[0].set_title(r"$|\psi|^2$")
        ax[0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi|^2$ in $W/cm^2$")
        ax[1].set_title(r"Phase $\mathrm{arg}(\psi)$")
        ax[1].set_ylabel(r"Phase arg$(\psi)$")
        for a in ax:
            a.set_xlabel("Position x in mm")
        plt.show()
