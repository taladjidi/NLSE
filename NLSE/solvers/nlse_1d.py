import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__
from .nlse import NLSE, show_if_possible


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
    ) -> None:
        """Instantiate the simulation.

        Solves the equation of `NLSE` with one transverse dimension.

        Parameters
        ----------
        alpha : float
            Intensity loss in 1/m: I = I0 exp(-alpha z).
        power : float
            Power in W, of a beam taken as ``delta_X`` thick: the field is
            scaled so that ``c eps0 / 2 * sum(|E|^2) * delta_X**2`` is the
            power, and the peak intensity at fixed power grows with ``NX``.
        window : float
            Computational window in m.
        n2 : float
            Non linear coeff in m^2/W.
        V : np.ndarray
            Potential, dimensionless as in `NLSE`: twice the local change of
            the refractive index.
        L : float
            Length of the medium in m.
        NX : int, optional
            Number of points. Defaults to 1024.
        Isat : float
            Saturation intensity in W/m^2.
        nl_length : float
            Non-local length in m, of a diffusive non-locality: the
            interaction reads the intensity smoothed by
            1 / (1 + nl_length^2 K^2) in Fourier space. 0 is local.
        wvl : float, optional
            Wavelength in m. Defaults to 780 nm.
        backend : str, optional
            Backend name: "CPU", "CUPY", "CL", "MLX" or "auto" (the installed
            one with the fastest transform at this grid size). Defaults to
            the backend NLSE picked when it was imported.
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

        # Override normalization factor for 1D (delta_X^2 instead of delta_X * delta_Y)
        self._norm_grid_factor = np.float32(self.delta_X**2)

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for 1D propagator."""
        return (self.NX, float(delta_z), np.dtype(dtype).str, float(self.k))

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for 1D RK4 dispersion operator."""
        return (self.NX, "RK4", float(self.k))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw 1D dispersion operator for RK4."""
        return -1j * 0.5 * self.Kx**2 / self.k

    def _dispersion_operator(self) -> np.ndarray:
        """Return the 1D dispersion eigenvalues."""
        return 0.5 * self.Kx**2 / self.k

    def _transverse_k_squared(self) -> np.ndarray:
        """Return the squared wavenumber over the 1D Fourier grid."""
        return self.Kx**2

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance in m.
        """
        # Every other solver's plot_field starts here, and this one did not:
        # it called np.abs and np.angle on whatever the backend handed it,
        # which works on the CPU and on MLX and raises "setting an array
        # element with a sequence" on CL.
        A_plot = self._to_plot_array(A_plot, 2)
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
        show_if_possible()
