from __future__ import annotations

from typing import Any

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
        V: np.ndarray | None,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        omega: float | None = None,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the simulation.

        Parameters
        ----------
        alpha : float
            Absorption coefficient.
        power : float
            Optical power in W.
        window : float
            Computational window in m.
        n2 : float
            Non linear index of the 1st component in m^2/W.
        n12 : float
            Inter-component interaction parameter.
        V : np.ndarray or None
            Potential landscape.
        L : float
            Length of the cell in m.
        NX : int, optional
            Number of points along x. Defaults to 1024.
        Isat : float, optional
            Saturation intensity, assumed to be the same for both
            components. Defaults to np.inf.
        nl_length : float, optional
            Non local length in m. Defaults to 0.
        wvl : float, optional
            Wavelength in m. Defaults to 780e-9.
        omega : float or None, optional
            Rabi coupling. Defaults to ``None``.
        backend : str, optional
            Compute backend. Defaults to ``__BACKEND__``.
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

    def _prepare_output_array(self, E: np.ndarray, normalize: bool) -> tuple[Any, Any]:
        """Prepare the output arrays for 1D CNLSE.

        Two-component normalization with ``delta_X**2``.

        Parameters
        ----------
        E : np.ndarray
            Input array of shape ``(2, NX)``.
        normalize : bool
            Normalize the field to the total power.

        Returns
        -------
        A : array-like
            Output field array.
        A_sq : array-like
            Output field modulus squared array.
        """
        A, A_sq = self._backend.allocate_pair(E.shape, E.dtype)
        E_dev = self._backend.to_device(E)
        if normalize:
            puiss_arr = np.array([self.power, self.power2], dtype=E.dtype)
            integral = ((E.real * E.real + E.imag * E.imag) * self.delta_X**2).sum(
                axis=self._last_axes
            )
            integral *= c * epsilon_0 / 2
            E_00 = (puiss_arr / integral) ** 0.5
            A[:] = (E_00.T * E_dev.T).T
        else:
            A[:] = E_dev
        return A, A_sq

    def _take_components(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Take the components of the field.

        Parameters
        ----------
        A : np.ndarray
            Field to retrieve the components of.

        Returns
        -------
        tuple of np.ndarray
            The two components ``(A1, A2)``.
        """
        A1 = A[..., 0, :]
        A2 = A[..., 1, :]
        return A1, A2

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
                propagator1 = np.exp(
                    -1j * 0.5 * (self.Kx**2) / self.k * self.delta_z
                ).astype(np.complex64)
                propagator2 = np.exp(
                    -1j * 0.5 * (self.Kx**2) / self.k2 * self.delta_z
                ).astype(np.complex64)
            case "RK4":
                propagator1 = (-1j * 0.5 * (self.Kx**2) / self.k).astype(np.complex64)
                propagator2 = (-1j * 0.5 * (self.Kx**2) / self.k2).astype(np.complex64)
        return np.array([propagator1, propagator2])

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance in m.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        A_plot = self._backend.to_host(A_plot)
        if not isinstance(A_plot, np.ndarray):
            A_plot = np.asarray(A_plot)
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
        ax[1, 0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi_1|^2$ in $W/cm^2$")
        ax[1, 1].plot(self.X * 1e3, np.unwrap(np.angle(A_2_plot)))
        ax[1, 1].set_title(r"$\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x in mm")
        ax[1, 1].set_ylabel(r"Phase in rad")
        plt.show()
