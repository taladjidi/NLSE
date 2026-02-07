from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from .nlse import NLSE
from ..utils import __BACKEND__


class CNLSE(NLSE):
    """A class to solve the coupled NLSE"""

    def __init__(
        self,
        alpha: float,
        power: float,
        window: float,
        n2: float,
        n12: float,
        V: Union[np.ndarray, None],
        L: float,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        omega: float = None,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiates the class with all the relevant physical parameters

        Args:
            alpha (float): alpha through the cell
            power (float): Optical power in W
            window (float): Computational window in m
            n2 (float): Non linear index of the 1 st component in m^2/W
            n12 (float): Inter component interaction parameter
            V (np.ndarray): Potential landscape in a.u
            L (float): Length of the cell in m
            NX (int, optional): Number of points along x. Defaults to 1024.
            NY (int, optional): Number of points along y. Defaults to 1024.
            Isat (float, optional): Saturation intensity, assumed to be the same
                for both components. Defaults to infinity.
            nl_length (float): Non local length in m.
                The non-local kernel is the instantiated as a Bessel function
                to model a diffusive non-locality stored in the nl_profile
                attribute.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            omega (float, optional): Rabi coupling. Defaults to None.
            backend (str, optional): "CUPY" or "CPU". Defaults to __BACKEND__.
        Returns:
            object: CNLSE class instance
        """
        if backend == "Metal":
            raise NotImplementedError(
                "Metal backend is not yet supported for CNLSE. "
                "Use CPU, CUPY or CL backend."
            )
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
        """Prepare the output arrays for CNLSE.

        Two-component normalization requires per-component broadcasting.

        Args:
            E (np.ndarray): Input array of shape (2, NY, NX)
            normalize (bool): Normalize the field to the total power.
        Returns:
            A (np.ndarray): Output field array
            A_sq (np.ndarray): Output field modulus squared array
        """
        A, A_sq = self._backend.allocate_pair(E.shape, E.dtype)
        E_dev = self._backend.to_device(E)
        if normalize:
            puiss_arr = np.array([self.power, self.power2], dtype=E.real.dtype)
            arr = E.real * E.real + E.imag * E.imag
            arr = (arr * self.delta_X * self.delta_Y).astype(E.real.dtype)
            integral = arr.sum(axis=self._last_axes)
            integral = integral * c * epsilon_0 / 2
            E_00 = (puiss_arr / integral) ** 0.5
            if self.backend == "CL":
                from pyopencl import array as cla
                E_00 = cla.to_device(self._cl_queue, E_00.astype(E.dtype))
                E_dev = cla.to_device(self._cl_queue, E.astype(E.dtype))
            A[0] = E_00[0] * E_dev[0]
            A[1] = E_00[1] * E_dev[1]
        else:
            A[:] = E_dev
        return A, A_sq

    def _send_arrays_to_gpu(self) -> None:
        """
        Send arrays to GPU.
        """
        super()._send_arrays_to_gpu()
        # for broadcasting of parameters in case they are
        # not already on the GPU
        for attr in ("n22", "n12"):
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, self._backend.to_device(val))

    def _retrieve_arrays_from_gpu(self) -> None:
        """
        Retrieve arrays from GPU.
        """
        super()._retrieve_arrays_from_gpu()
        for attr in ("n22", "n12"):
            val = getattr(self, attr)
            if self._backend.is_device_array(val):
                setattr(self, attr, self._backend.to_host(val))

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the propagators.

        Returns:
            propagator1 (np.ndarray): The propagator for the first component.
            propagator2 (np.ndarray): The propagator for the second component.
        """
        propagator1 = super()._build_propagator(precision=precision)
        match precision:
            case "single" | "double":
                propagator2 = np.exp(
                    -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z
                ).astype(np.complex64)
            case "RK4":
                propagator2 = -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k
        return np.array([propagator1, propagator2])

    def _take_components(self, A: np.ndarray) -> tuple:
        """Take the components of the field.

        Args:
            A (np.ndarray): Field to retrieve the components of
        Returns:
            tuple: Tuple of the two components
        """
        A1 = A[..., 0, :, :]
        A2 = A[..., 1, :, :]
        return A1, A2

    def _RK4_rhs_non_mutating(
        self,
        A: np.ndarray,
        V: Union[np.ndarray, None],
        propagator: np.ndarray,
        plans: list,
    ) -> np.ndarray:
        """Compute the RHS of NLSE in a non-mutating manner for RK4.

        Split step function for one propagation step using a 4th order Runge-Kutta method (RK4).

        Args:
            A (np.ndarray): Field to propagate
            A_sq (np.ndarray): Field modulus squared.
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects.
                Either a single FFT plan for both directions (GPU case)
                or distinct FFT and IFFT plans for FFTW.
            precision (str, optional): Single or double application of
                the nonlinear propagation step. Defaults to "single".
        """
        # prepare output array, this kills performance but we need it
        A_prop = A.copy()
        A_sq = A.real * A.real + A.imag * A.imag
        self._linear_step(A_prop, propagator)
        if self.nl_length > 0:
            A_sq[:] = self._backend.convolution(
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
        V: Union[np.ndarray, None],
        propagator: np.ndarray,
        plans,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step

        Args:
            A (np.ndarray): Fields to propagate of shape (2, NY, NX)
            A_sq (np.ndarray): Intensity of the fields.
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix for both fields
                [propagator1, propagator2].
            plans: FFT plan object.
            precision (str, optional): Single or double application of the
                nonlinear propagation step. Defaults to "single".
        Returns:
            None
        """
        A1, A2 = self._take_components(A)
        if precision == "double":
            self._kernels.square_mod(A, A_sq)
            A_sq_1, A_sq_2 = self._take_components(A_sq)
            if self.nl_length > 0:
                A_sq_1 = self._backend.convolution(
                    A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
                )
                A_sq_2 = self._backend.convolution(
                    A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
                )

            if V is None:
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
        self._linear_step(A, propagator)
        # fft normalization
        self._kernels.square_mod(A, A_sq)
        A_sq_1, A_sq_2 = self._take_components(A_sq)
        if self.nl_length > 0:
            A_sq_1 = self._backend.convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._backend.convolution(
                A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z / 2,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_without_V_c(
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
                self._kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    self.k / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                    2 * self.I_sat2 / (epsilon_0 * c),
                )
                self._kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    self.delta_z,
                    self.alpha2 / 2,
                    self.k2 / 2 * V,
                    self.k2 / 2 * self.n22 * c * epsilon_0,
                    self.k2 / 2 * self.n12 * c * epsilon_0,
                    2 * self.I_sat2 / (epsilon_0 * c),
                    2 * self.I_sat / (epsilon_0 * c),
                )
            if self.omega is not None:
                self._kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot the field.

        Args:
            A_plot (np.ndarray): The field to plot
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
