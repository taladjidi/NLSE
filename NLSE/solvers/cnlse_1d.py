import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .cnlse import CNLSE

if __CUPY_AVAILABLE__:
    import cupy as cp


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
    ) -> object:
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

        Returns
        -------
        object
            CNLSE class instance.
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

    def _take_components(self, A: np.ndarray) -> tuple:
        """Take the components of the field.

        Parameters
        ----------
        A : np.ndarray
            Field to retrieve the components of.

        Returns
        -------
        tuple
            Tuple of the two components.
        """
        A1 = A[..., 0, :]
        A2 = A[..., 1, :]

        # OpenCL/CUPY backends don't support offset arrays - make contiguous copies
        if self._backend.name in ["CL", "CUPY", "MLX"]:
            if hasattr(A1, "copy"):
                A1 = A1.copy()
                A2 = A2.copy()

        return A1, A2

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Uses caching to avoid recomputing propagators with identical parameters.

        Parameters
        ----------
        precision : str, optional
            "single" or "double" application of the
            propagator. Defaults to "single".

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        # Create cache key (1D version, includes k2 for second component)
        cache_key = (
            self.NX,
            float(self.delta_z),
            precision,
            float(self.k),
            float(self.k2),
        )

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        dtype = np.complex128 if precision == "double" else np.complex64
        propagator1 = np.exp(
            -1j * 0.5 * (self.Kx**2) / self.k * self.delta_z, dtype=dtype
        )
        propagator2 = np.exp(
            -1j * 0.5 * (self.Kx**2) / self.k2 * self.delta_z, dtype=dtype
        )
        propagator = np.array([propagator1, propagator2])

        # Cache for future use
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _build_propagator_rk4(self) -> np.ndarray:
        """Build raw 1D 2-component dispersion operator for RK4.

        Returns
        -------
        np.ndarray
            The raw dispersion operators for both components.
        """
        cache_key = (self.NX, "RK4", float(self.k), float(self.k2))
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        prop1 = (-1j * 0.5 * self.Kx**2 / self.k).astype(np.complex64)
        prop2 = (-1j * 0.5 * self.Kx**2 / self.k2).astype(np.complex64)
        propagator = np.array([prop1, prop2])
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
