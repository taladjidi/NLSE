import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__
from .cnlse import CNLSE


class CNLSE_1d(CNLSE):
    """A class to solve the 1D coupled NLSE."""

    def __init__(
        self,
        alpha: float,
        power: float,
        window: float,
        n2: float,
        n12: float,
        V: np.ndarray,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        omega: float | None = None,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the class with all the relevant physical parameters.

        Parameters
        ----------
        alpha : float
            Alpha through the cell.
        power : float
            Optical power in W.
        window : float
            Computational window in m.
        n2 : float
            Non linear index of the 1st component in m^2/W.
        n12 : float
            Inter component interaction parameter.
        V : np.ndarray
            Potential landscape in a.u.
        L : float
            Length of the cell in m.
        NX : int, optional
            Number of points along x. Defaults to 1024.
        Isat : float, optional
            Saturation intensity, assumed to be the same
            for both components. Defaults to infinity.
        nl_length : float
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute.
        wvl : float, optional
            Wavelength in m. Defaults to 780 nm.
        omega : float, optional
            Rabi coupling. Defaults to None.
        backend : str, optional
            "GPU" or "CPU". Defaults to __BACKEND__.

        """
        super().__init__(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            n12=n12,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            omega=omega,
            backend=backend,
        )
        self._last_axes = (-1,)
        self.nl_profile = self.nl_profile[0]
        self.nl_profile /= self.nl_profile.sum()

        # Override normalization factor for 1D (delta_X^2 instead of delta_X * delta_Y)
        self._norm_grid_factor = np.float32(self.delta_X**2)

    def _dispersion_operator(self) -> np.ndarray:
        """Return the 1D coupled dispersion eigenvalues.

        CNLSE's is built from the 2D Fourier grid, which is the wrong shape
        here. It went unnoticed while only the maximum was taken; weighting
        by the field needs the operator itself, on the right grid.
        """
        return 0.5 * self.Kx**2 / min(self.k, self.k2)

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for 1D coupled propagator."""
        return (
            self.NX,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.k),
            float(self.k2),
        )

    def _compute_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Compute the 1D coupled linear propagation matrices."""
        propagator1 = np.exp(-1j * 0.5 * (self.Kx**2) / self.k * delta_z, dtype=dtype)
        propagator2 = np.exp(-1j * 0.5 * (self.Kx**2) / self.k2 * delta_z, dtype=dtype)
        return np.array([propagator1, propagator2])

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for 1D coupled RK4 dispersion operator."""
        return (self.NX, "RK4", float(self.k), float(self.k2))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw 1D coupled dispersion operators for RK4."""
        prop1 = (-1j * 0.5 * self.Kx**2 / self.k).astype(np.complex64)
        prop2 = (-1j * 0.5 * self.Kx**2 / self.k2).astype(np.complex64)
        return np.array([prop1, prop2])

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance in m.
        """
        A_plot = self._to_plot_array(A_plot, 2)
        A_1_plot = A_plot[0]
        A_2_plot = A_plot[1]
        fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        # plot amplitudes and phases
        ax[0, 0].plot(self.X * 1e3, np.abs(A_1_plot) ** 2 * epsilon_0 * c / 2 * 1e-4)
        ax[0, 0].set_title(r"$|\psi_1|^2$")
        ax[0, 0].set_xlabel("x in mm")
        ax[0, 0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$")
        ax[0, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_1_plot)))
        ax[0, 1].set_title(r"$\mathrm{arg}(\psi_1)$")
        ax[0, 1].set_xlabel("x in mm")
        ax[0, 1].set_ylabel(r"Phase in rad")
        ax[1, 0].plot(self.X * 1e3, np.abs(A_2_plot) ** 2 * epsilon_0 * c / 2 * 1e-4)
        ax[1, 0].set_title(r"$|\psi_2|^2$")
        ax[1, 0].set_xlabel("x in mm")
        ax[1, 0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi_2|^2$ in $W/cm^2$")
        ax[1, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_2_plot)))
        ax[1, 1].set_title(r"$\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x in mm")
        ax[1, 1].set_ylabel(r"Phase in rad")
        plt.show()
