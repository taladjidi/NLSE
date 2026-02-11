import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, c, epsilon_0, hbar

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .nlse import NLSE

if __CUPY_AVAILABLE__:
    import cupy as cp


class GPE(NLSE):
    """A class to solve GPE."""

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
    ) -> object:
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
        # listof physical parameters
        self.g = g
        self.V = V
        self.gamma = self.alpha
        self.N = self.power
        self.m = self.k
        self.sat = sat
        self.delta_t = self.delta_z
        # do some conversion for the units
        self.I_sat *= epsilon_0 * c / 2
        # GPE uses quantum units (Hz), not optical (W), so norm_constant is 1.0
        self._norm_constant = np.float32(1.0)

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
        # Create cache key (GPE uses delta_t and m instead of delta_z and k)
        cache_key = (self.NX, self.NY, float(self.delta_t), precision, float(self.m))

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        dtype = np.complex128 if precision == "double" else np.complex64
        propagator = np.exp(
            -1j * 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m * self.delta_t,
            dtype=dtype,
        )

        # Cache for future use
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
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
        fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
        ]
        ext_fourier = [
            np.min(self.Kx) * 1e-3,
            np.max(self.Kx) * 1e-3,
            np.min(self.Ky) * 1e-3,
            np.max(self.Ky) * 1e-3,
        ]
        rho = np.abs(A_plot) ** 2
        phi = np.angle(A_plot)
        im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot)))
        im0 = ax[0].imshow(rho, extent=ext_real)
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label="Density (at/m^2)")
        im1 = ax[1].imshow(
            phi,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x (mm)")
        ax[1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[1], shrink=0.6, label="Phase (rad)")
        im2 = ax[2].imshow(
            im_fft,
            extent=ext_fourier,
            cmap="nipy_spectral",
        )
        ax[2].set_title("Fourier space")
        ax[2].set_xlabel(r"$k_x$ ($mm^{-1}$)")
        ax[2].set_ylabel(r"$k_y$ ($mm^{-1}$)")
        fig.colorbar(im2, ax=ax[2], shrink=0.6, label="Intensity (a.u.)")
        plt.show()
