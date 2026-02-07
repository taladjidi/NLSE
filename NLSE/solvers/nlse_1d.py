from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__
from .nlse import NLSE


class NLSE_1d(NLSE):
    """A class to solve the 1D NLSE."""

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

        Solves an equation : d/dz psi = -1/2k0(d2/dx2) psi + k0 dn psi +
        k0 n2 psi**2 psi

        Parameters
        ----------
        alpha : float
            Absorption coefficient.
        power : float
            Power in W.
        window : float
            Computational window in the transverse direction in m.
        n2 : float
            Non linear coefficient in m^2/W.
        V : np.ndarray or None
            Potential.
        L : float
            Length of the medium in m.
        NX : int, optional
            Number of points in x. Defaults to 1024.
        Isat : float, optional
            Saturation intensity in W/m^2. Defaults to np.inf.
        nl_length : float, optional
            Non local length in m. Defaults to 0.
        wvl : float, optional
            Wavelength in m. Defaults to 780e-9.
        backend : str, optional
            Compute backend. Defaults to ``__BACKEND__``.
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

    def _compute_norm_factor(self, E_in):
        """1D normalization uses delta_X**2 instead of delta_X * delta_Y."""
        arr = E_in.real * E_in.real + E_in.imag * E_in.imag
        arr = (arr * self.delta_X**2).astype(E_in.real.dtype)
        integral = self._backend.sum(arr, axis=self._last_axes)
        integral = integral * c * epsilon_0 / 2
        E_00 = self._backend.sqrt(self.power / integral)
        return E_00

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Parameters
        ----------
        precision : str, optional
            ``"single"``, ``"double"`` or ``"RK4"``. Defaults to ``"single"``.

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        match precision:
            case "single" | "double":
                propagator = np.exp(
                    -1j * 0.5 * (self.Kx**2) / self.k * self.delta_z
                ).astype(np.complex64)
            case "RK4":
                propagator = (-1j * 0.5 * (self.Kx**2) / self.k).astype(np.complex64)
        return propagator  # type: ignore[no-any-return]

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance in m.
        """
        A_plot = self._backend.to_host(A_plot)
        if not isinstance(A_plot, np.ndarray):
            A_plot = np.asarray(A_plot)
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
