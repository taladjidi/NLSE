from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from .nlse import NLSE
from ..utils import __BACKEND__


class NLSE_3d(NLSE):
    """A class to solve the 3D NLSE i.e propagation of pulses
    of light in nonlinear media.
    """

    def __init__(
        self,
        alpha: float,
        energy: float,
        window: Union[list, tuple],
        n2: float,
        D0: float,
        vg: float,
        V: Union[np.ndarray, None],
        L: float,
        NX: int = 1024,
        NY: int = 1024,
        NZ: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiate the simulation.

        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi +
          D0/2 (d2/dt2) psi + k0 dn psi +
          k0 n2 psi**2 psi

        Args:
            alpha (float): alpha
            energy (float): Total energy in J
            window (np.ndarray): Computanional window in the transverse plane
                (index 0) in m and longitudinal direction (index 1) in s.
                Can also be window = [window_x, window_y, window_t]
            n2 (float): Non linear coeff in m^2/W.
            D0 (float): Dispersion in s^2/m.
            vg (float): Group velocity in m/s.
            V (np.ndarray): Potential.
            L (float): Length in m of the nonlinear medium
            NX (int, optional): Number of points in the x direction.
                Defaults to 1024.
            NY (int, optional): Number of points in the y direction.
                Defaults to 1024.
            NZ (int, optional): Number of points in the t direction.
                Defaults to 1024.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non local length in m.
                The non-local kernel is the instantiated as a Bessel function
                to model a diffusive non-locality stored in the nl_profile
                attribute.
            wvl (float): Wavelength in m
            backend (str, optional): "CUPY" or "CPU".
                Defaults to __BACKEND__.
        """
        if len(window) == 2:
            window = [window[0], window[0], window[-1]]
        super().__init__(
            alpha=alpha,
            power=energy,
            window=window[0:2],
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self.energy = self.power
        self.NZ = NZ
        self.window_t = window[-1]
        self.power = self.energy / self.window_t
        Dn = self.n2 * self.power / min(self.window[0:2]) ** 2
        z_nl = 1 / (self.k * abs(Dn))
        if isinstance(z_nl, np.ndarray):
            z_nl = z_nl.min()
        self.delta_z = 0.5e-2 * z_nl
        self.T, self.delta_T = np.linspace(
            -self.window_t / 2, self.window_t / 2, self.NZ, retstep=True
        )
        self.omega = 2 * np.pi * np.fft.fftfreq(self.NZ, self.delta_T)
        self.D0 = D0
        self.vg = vg
        self.XX, self.YY, self.TT = np.meshgrid(self.X, self.Y, self.T)
        self.Kxx, self.Kyy, self.Omega = np.meshgrid(self.Kx, self.Ky, self.omega)
        self._last_axes = (-3, -2, -1)  # Axes are x, y, t

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Args:
            precision (str): "single", "double" or "RK4". Defaults to "single".
        Returns:
            np.ndarray: The propagator
        """
        prop_2d = super()._build_propagator(precision=precision)
        match precision:
            case "single" | "double":
                prop_t = np.exp(
                    -1j * self.D0 / 2 * self.Omega**2
                ).astype(np.complex64)
            case "RK4":
                prop_t = (
                    -1j * self.D0 / 2 * self.Omega**2
                ).astype(np.complex64)
        # prop_t *= np.exp(1 / self.vg * self.Omega)
        return (prop_2d * prop_t).astype(np.complex64)

    def _compute_norm_factor(self, E_in):
        """3D normalization includes delta_T and uses energy instead of power."""
        arr = E_in.real * E_in.real + E_in.imag * E_in.imag
        arr = (arr * self.delta_X * self.delta_Y * self.delta_T).astype(
            E_in.real.dtype
        )
        integral = self._backend.sum(arr, axis=self._last_axes)
        integral = integral * c * epsilon_0 / 2
        E_00 = self._backend.sqrt(self.energy / integral)
        return E_00

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
            z (float): Propagation distance in m.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        A_plot = self._backend.to_host(A_plot)
        if not isinstance(A_plot, np.ndarray):
            A_plot = np.asarray(A_plot)
        fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
        ]
        ext_time = [
            np.min(self.T) * 1e6,
            np.max(self.T) * 1e6,
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
        ]
        rho = np.abs(A_plot) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi = np.angle(A_plot)
        rho_xy = rho[:, :, self.NZ // 2]
        phi_xy = phi[:, :, self.NZ // 2]
        rho_xt = rho[:, self.NY // 2, :]
        phi_xt = phi[:, self.NY // 2, :]
        im0 = ax[0, 0].imshow(rho_xy, extent=ext_real)
        ax[0, 0].set_title(r"Intensity in $xy$ plane at $t$=0")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label="Intensity (W/cm^2)")
        im1 = ax[0, 1].imshow(
            phi_xy,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[0, 1].set_title(r"Phase in $xy$ plane at $t$=0")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho_xt, extent=ext_time, aspect="auto")
        ax[1, 0].set_title(r"Intensity in $xt$ plane at $y$=0")
        ax[1, 0].set_ylabel(r"$x$ ($mm$)")
        ax[1, 0].set_xlabel(r"$t$ ($\mu s$)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label="Intensity (a.u.)")
        im3 = ax[1, 1].imshow(
            phi_xt,
            extent=ext_time,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
            aspect="auto",
        )
        ax[1, 1].set_title(r"Phase in $xt$ plane at $y$=0")
        ax[1, 1].set_ylabel(r"$x$ ($mm$)")
        ax[1, 1].set_xlabel(r"$t$ ($\mu s$)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Intensity (a.u.)")
        plt.show()
