import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .nlse import NLSE

if __CUPY_AVAILABLE__:
    import cupy as cp


class NLSE_3d(NLSE):
    """A class to solve the 3D NLSE.

    Propagation of pulses of light in nonlinear media.
    """

    def __init__(
        self,
        alpha: float,
        energy: float,
        window: list | tuple,
        n2: float,
        D0: float,
        vg: float,
        V: np.ndarray | None,
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

        Parameters
        ----------
        alpha : float
            alpha.
        energy : float
            Total energy in J.
        window : np.ndarray
            Computational window in the transverse plane
            (index 0) in m and longitudinal direction (index 1) in s.
            Can also be window = [window_x, window_y, window_t].
        n2 : float
            Non linear coeff in m^2/W.
        D0 : float
            Dispersion in s^2/m.
        vg : float
            Group velocity in m/s.
        V : np.ndarray
            Potential.
        L : float
            Length in m of the nonlinear medium.
        NX : int, optional
            Number of points in the x direction.
            Defaults to 1024.
        NY : int, optional
            Number of points in the y direction.
            Defaults to 1024.
        NZ : int, optional
            Number of points in the t direction.
            Defaults to 1024.
        Isat : float
            Saturation intensity in W/m^2.
        nl_length : float
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute.
        wvl : float
            Wavelength in m.
        backend : str, optional
            "GPU" or "CPU".
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

        # Override normalization factor for 3D (includes delta_T)
        self._norm_grid_factor = np.float32(self.delta_X * self.delta_Y * self.delta_T)

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
            The propagator.
        """
        # Create cache key (3D version includes D0 and NZ)
        cache_key = (
            self.NX,
            self.NY,
            self.NZ,
            float(self.delta_z),
            precision,
            float(self.k),
            float(self.D0),
        )

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        dtype = np.complex128 if precision == "double" else np.complex64
        prop_2d = super()._build_propagator(precision)
        prop_t = np.exp(-1j * self.D0 / 2 * self.Omega**2, dtype=dtype)
        # prop_t *= np.exp(1 / self.vg * self.Omega)
        propagator = prop_2d * prop_t

        # Cache for future use
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare the output arrays depending on backend.

        Prepares the A and A_sq arrays to store the field and its modulus.
        Overrides base class to normalize to energy instead of power.

        Parameters
        ----------
        E_in : np.ndarray
            Input array.
        normalize : bool
            Normalize the field to the total energy.

        Returns
        -------
        A : np.ndarray
            Output field array.
        A_sq : np.ndarray
            Output field modulus squared array.
        """
        A = self._backend.allocate_field(E_in.shape, E_in.dtype)
        A_sq = self._backend.allocate_real_field(E_in.shape, E_in.real.dtype)
        E_in = self._backend.from_numpy(E_in)

        if normalize:
            arr = (E_in * E_in.conj()).real
            arr = arr * self._norm_grid_factor
            if self._backend.name in ["CL", "MLX"]:
                arr_np = self._backend.to_numpy(arr)
                E_in_np = self._backend.to_numpy(E_in)
                integral = np.sum(arr_np, axis=self._last_axes)
                integral = integral * self._norm_constant
                E_00 = (self.energy / integral) ** 0.5
                result = (E_00.T * E_in_np.T).T.astype(E_in_np.dtype)
                A = self._backend.from_numpy(result)
            else:
                integral = np.sum(arr, axis=self._last_axes)
                integral = integral * self._norm_constant
                E_00 = (self.energy / integral) ** 0.5
                A[:] = (E_00.T * E_in.T).T
        else:
            if self._backend.name == "MLX":
                A = E_in
            else:
                A[:] = E_in
        return A, A_sq

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
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        if self.__CUPY_AVAILABLE__ and isinstance(A_plot, cp.ndarray):
            A_plot = A_plot.get()
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
