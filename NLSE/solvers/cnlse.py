import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__, __CUPY_AVAILABLE__, __PYOPENCL_AVAILABLE__
from .nlse import NLSE

if __CUPY_AVAILABLE__:
    import cupy as cp

if __PYOPENCL_AVAILABLE__:
    from pyopencl import array as cla


class CNLSE(NLSE):
    """A class to solve the coupled NLSE."""

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
        NY: int = 1024,
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
        NY : int, optional
            Number of points along y. Defaults to 1024.
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
            V=V,
            L=L,
            NX=NX,
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self.I_sat2 = self.I_sat
        self.n12 = n12
        # initialize intra component 2 interaction parameter
        # to be the same as intra component 1
        self.n22 = self.n2
        # Rabi coupling
        self.omega = omega
        # same for the losses, this is to leave separate attributes so
        # the the user can chose whether or not to unbalence the rates
        self.alpha2 = self.alpha
        # wavenumbers
        self.k2 = self.k
        # powers
        self.power2 = self.power
        # waists
        self.propagator1 = None
        self.propagator2 = None

    def _prepare_output_array(
        self, E: np.ndarray, normalize: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare the output arrays depending on __BACKEND__.

        Prepares the A and A_sq arrays to store the field and its modulus.

        Parameters
        ----------
        E_in : np.ndarray
            Input array.
        normalize : bool
            Normalize the field to the total power.

        Returns
        -------
        A : np.ndarray
            Output field array.
        A_sq : np.ndarray
            Output field modulus squared array.
        """
        puiss_arr = np.array([self.power, self.power2], dtype=E.dtype)
        A = self._backend.allocate_field(E.shape, E.dtype)
        A_sq = self._backend.allocate_real_field(E.shape, E.real.dtype)
        if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
            puiss_arr = cp.array(puiss_arr)
        if normalize:
            # normalization of the field (use contiguous formula)
            match self.backend:
                case "CUPY" | "CPU":
                    arr = (E * E.conj()).real * self._norm_grid_factor
                    integral = arr.sum(axis=self._last_axes)
                    integral = integral * self._norm_constant
                    E_00 = (puiss_arr / integral) ** 0.5
                case "CL":
                    arr = (E * E.conj()).real * self._norm_grid_factor
                    integral = arr.sum(axis=self._last_axes)
                    integral = integral * self._norm_constant
                    E_00 = (puiss_arr / integral) ** 0.5
                    E_00 = cla.to_device(self._backend.queue, E_00.astype(E.dtype))
                    E = cla.to_device(self._backend.queue, E.astype(E.dtype))
            A[0] = E_00[0] * E[0]
            A[1] = E_00[1] * E[1]
        else:
            A[:] = E
        return A, A_sq

    def _send_arrays_to_gpu(self) -> None:
        """Send arrays to GPU."""
        super()._send_arrays_to_gpu()
        # for broadcasting of parameters in case they are
        # not already on the device
        if self._backend.name in ["CUPY", "CL"]:
            if isinstance(self.n22, np.ndarray):
                self.n22 = self._backend.from_numpy(self.n22)
            if isinstance(self.n12, np.ndarray):
                self.n12 = self._backend.from_numpy(self.n12)

    def _retrieve_arrays_from_gpu(self) -> None:
        """Retrieve arrays from GPU."""
        super()._retrieve_arrays_from_gpu()
        match self.backend:
            case "CUPY":
                if isinstance(self.n22, cp.ndarray):
                    self.n22 = self.n22.get()
                if isinstance(self.n12, cp.ndarray):
                    self.n12 = self.n12.get()
            case "CL":
                if isinstance(self.n22, cla.Array):
                    self.n22 = self.n22.get()
                if isinstance(self.n12, cla.Array):
                    self.n12 = self.n12.get()

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the propagators.

        Parameters
        ----------
        precision : str, optional
            "single" or "double" precision. Defaults to "single".

        Returns
        -------
        np.ndarray
            Array containing propagators for both components.
        """
        # Create cache key (includes k2 for second component)
        cache_key = (
            self.NX,
            self.NY,
            float(self.delta_z),
            precision,
            float(self.k),
            float(self.k2),
        )

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        dtype = np.complex128 if precision == "double" else np.complex64
        propagator1 = super()._build_propagator(precision=precision)
        match precision:
            case "single" | "double":
                propagator2 = np.exp(
                    -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2 * self.delta_z,
                    dtype=dtype,
                )
            case "RK4":
                propagator2 = (
                    -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2
                ).astype(dtype)

        propagator = np.array([propagator1, propagator2])

        # Cache for future use
        self._propagator_cache[cache_key] = propagator
        return propagator

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
        A1 = A[..., 0, :, :]
        A2 = A[..., 1, :, :]

        # OpenCL/CUPY backends don't support offset arrays - make contiguous copies
        if self._backend.name in ["CL", "CUPY"]:
            # For GPU backends, we need contiguous arrays (no offset)
            # This creates copies but ensures kernel compatibility
            if hasattr(A1, "copy"):  # GPU array
                A1 = A1.copy()
                A2 = A2.copy()

        return A1, A2

    def _RK4_rhs_non_mutating(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
    ) -> np.ndarray:
        """Compute the RHS of NLSE in a non-mutating manner for RK4.

        Split step function for one propagation step using a 4th order Runge-Kutta method (RK4).

        Parameters
        ----------
        A : np.ndarray
            Field to propagate.
        A_sq : np.ndarray
            Field modulus squared.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects.
            Either a single FFT plan for both directions (GPU case)
            or distinct FFT and IFFT plans for FFTW.
        precision : str, optional
            Single or double application of
            the nonlinear propagation step. Defaults to "single".
        """
        # prepare output array, this kills performance but we need it
        A_prop = A.copy()
        A_sq = (A * A.conj()).real
        self._backend.fft(A_prop, plans)
        A_prop *= propagator
        self._backend.ifft(A_prop, plans)
        if self.nl_length > 0:
            A_sq[:] = self._convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
        # Linear prop
        arg = A_prop
        # saturation
        sat = 1 / (
            1
            + A_sq[0] * 1 / (2 * self.I_sat / (epsilon_0 * c))
            + A_sq[1] * 1 / (2 * self.I_sat2 / (epsilon_0 * c))
        )
        # Interactions
        arg[0] += 1j * (
            self.k / 2 * self.n2 * c * epsilon_0 * A_sq[0] * sat
            + self.k / 2 * self.n12 * c * epsilon_0 * A_sq[1] * sat
        )
        arg[1] += 1j * (
            self.k / 2 * self.n22 * c * epsilon_0 * A_sq[1] * sat
            + self.k / 2 * self.n12 * c * epsilon_0 * A_sq[0] * sat
        )
        # Losses
        arg[0] -= self.alpha / 2 * sat * A[0]
        arg[1] -= self.alpha2 / 2 * sat * A[1]
        if V is not None:
            V_ = 1j * self.k / 2 * V * A
            arg += V_
        return arg

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step.

        Parameters
        ----------
        A : np.ndarray
            Fields to propagate of shape (2, NY, NX).
        A_sq : np.ndarray
            Intensity of the fields.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix for both fields
            [propagator1, propagator2].
        plans : list
            List of FFT plan objects. Either a single FFT plan for
            both directions (GPU case) or distinct FFT and IFFT plans for
            FFTW.
        precision : str, optional
            Single or double application of the
            nonlinear propagation step. Defaults to "single".

        Returns
        -------
        None
        """
        # Precompute scaled potentials with correct dtype
        if V is not None:
            V_scaled = V * np.float32(self.k / 2)
            V2_scaled = V * np.float32(self.k2 / 2)
        else:
            V_scaled = V2_scaled = None

        A1, A2 = self._take_components(A)
        if precision == "double":
            self._backend.kernels.square_mod(A, A_sq)
            A_sq_1, A_sq_2 = self._take_components(A_sq)
            if self.nl_length > 0:
                A_sq_1 = self._convolution(
                    A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
                )
                A_sq_2 = self._convolution(
                    A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
                )

            if V is None:
                self._backend.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._backend.kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    V_scaled,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    V2_scaled,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
        # For GPU backends with double precision, write back modified
        # components before FFT (A1/A2 are copies, not views)
        if precision == "double" and self._backend.name in ["CL", "CUPY"]:
            A[0] = A1
            A[1] = A2
        self._backend.fft(A, plans)
        kernels = self._backend.kernels
        if hasattr(kernels, "apply_propagator"):
            kernels.apply_propagator(A, propagator)
        else:
            A *= propagator
        self._backend.ifft(A, plans)
        # Re-extract components after FFT/IFFT (CL/CUPY copies are now stale)
        A1, A2 = self._take_components(A)
        self._backend.kernels.square_mod(A, A_sq)
        A_sq_1, A_sq_2 = self._take_components(A_sq)
        if self.nl_length > 0:
            A_sq_1 = self._convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._convolution(
                A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                self._backend.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._backend.kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    V_scaled,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    V2_scaled,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
        else:
            if V is None:
                self._backend.kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._backend.kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    V_scaled,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._backend.kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    V2_scaled,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
            if self.omega is not None:
                self._backend.kernels.rabi_coupling(
                    A1, A2, self.delta_z, self.omega / 2
                )

        # For GPU backends, copy modified components back to original array
        if self._backend.name in ["CL", "CUPY"]:
            A[0] = A1
            A[1] = A2

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot the field.

        Parameters
        ----------
        A_plot : np.ndarray
            The field to plot.
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
        rho0 = np.abs(A_plot[0]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        ax[0, 0].set_title(r"$|\psi_1|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=r"Intensity $(W/cm^2)$")
        im1 = ax[0, 1].imshow(
            phi0,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[0, 1].set_title(r"Phase $\mathrm{arg}(\psi_1)$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(r"$|\psi_2|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=r"Intensity $(W/cm^2)$")
        im3 = ax[1, 1].imshow(
            phi1,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_2)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()
