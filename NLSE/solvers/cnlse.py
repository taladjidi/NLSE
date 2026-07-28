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

    _gpu_param_attrs = (*NLSE._gpu_param_attrs, "n22", "n12")
    # Both intra- and inter-component couplings switch off past the medium.
    _nonlinearity_attrs = (*NLSE._nonlinearity_attrs, "n12", "n22")

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
        E : np.ndarray
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
        if self._backend.name == "CUPY":
            E = cp.asarray(E)
            puiss_arr = cp.array(puiss_arr)
        if normalize:
            # normalization of the field (use contiguous formula)
            match self._backend.name:
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
                case "MLX":
                    E_np = E if isinstance(E, np.ndarray) else self._backend.to_numpy(E)
                    arr = (E_np * E_np.conj()).real * self._norm_grid_factor
                    integral = arr.sum(axis=self._last_axes)
                    integral = integral * self._norm_constant
                    E_00 = (puiss_arr / integral) ** 0.5
                    result = np.zeros_like(E_np)
                    result[0] = E_00[0] * E_np[0]
                    result[1] = E_00[1] * E_np[1]
                    return self._backend.from_numpy(result.astype(E_np.dtype)), A_sq
            A[0] = E_00[0] * E[0]
            A[1] = E_00[1] * E[1]
        else:
            if self._backend.name == "CL":
                E = cla.to_device(self._backend.queue, E)
            if self._backend.name == "MLX":
                A = self._backend.from_numpy(E)
            else:
                A[:] = E
        return A, A_sq

    def _propagator_cache_key(self, precision: str) -> tuple:
        """Return cache key for coupled propagator."""
        return (
            self.NX,
            self.NY,
            float(self.delta_z),
            precision,
            float(self.k),
            float(self.k2),
        )

    def _compute_propagator(self, precision: str) -> np.ndarray:
        """Compute the coupled linear propagation matrices."""
        dtype = np.complex128 if precision == "double" else np.complex64
        propagator1 = super()._compute_propagator(precision)
        propagator2 = np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2 * self.delta_z,
            dtype=dtype,
        )
        return np.array([propagator1, propagator2])

    def _rk4_max_dz(self) -> float:
        """Compute the maximum stable RK4 step size for both components."""
        K_sq = 0.5 * (self.Kxx**2 + self.Kyy**2)
        k_min = min(self.k, self.k2)
        D_max = float(np.max(K_sq / k_min))
        if D_max == 0:
            return np.inf
        return 2.83 / D_max

    def _split_step_max_dz(self, A: np.ndarray) -> float:
        """Compute the maximum split-step dz for coupled components."""
        A_np = self._backend.to_numpy(A)
        I1_peak = float(np.max(np.abs(A_np[0]) ** 2))
        I2_peak = float(np.max(np.abs(A_np[1]) ** 2))
        g11 = abs(getattr(self, "_g11", self.k / 2 * self.n2 * c * epsilon_0))
        g12 = abs(getattr(self, "_g12", self.k / 2 * self.n12 * c * epsilon_0))
        g22 = abs(getattr(self, "_g22", self.k2 / 2 * self.n22 * c * epsilon_0))
        Isat1 = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        Isat2 = getattr(self, "_Isat_conv2", 2 * self.I_sat2 / (epsilon_0 * c))
        sat = 1 / (1 + I1_peak / Isat1 + I2_peak / Isat2)
        nl_rate_1 = (g11 * I1_peak + g12 * I2_peak) * sat
        nl_rate_2 = (g22 * I2_peak + g12 * I1_peak) * sat
        nl_rate = max(nl_rate_1, nl_rate_2)
        if nl_rate == 0:
            return np.inf
        return np.pi / nl_rate

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for coupled RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k), float(self.k2))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw coupled dispersion operators for RK4."""
        prop1 = super()._compute_propagator_rk4()
        prop2 = (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2).astype(np.complex64)
        return np.array([prop1, prop2])

    def _precompute_step_constants(
        self, V: np.ndarray | None, precision: str = "single"
    ) -> None:
        """Pre-compute constants for coupled propagation steps."""
        super()._precompute_step_constants(V, precision)
        fp = np.float32 if precision == "single" else np.float64
        self._g11 = fp(self.k / 2 * self.n2 * c * epsilon_0)
        self._g12 = fp(self.k / 2 * self.n12 * c * epsilon_0)
        self._g22 = fp(self.k2 / 2 * self.n22 * c * epsilon_0)
        self._alpha2_half = fp(self.alpha2 / 2)
        self._Isat_conv2 = fp(2 * self.I_sat2 / (epsilon_0 * c))
        self._k2_half = fp(self.k2 / 2)
        if V is not None:
            self._V2_scaled = V * self._k2_half
        else:
            self._V2_scaled = None

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

        # GPU backends don't support offset arrays - make contiguous copies
        if self._backend.is_device_backend:
            if hasattr(A1, "copy"):
                A1 = A1.copy()
                A2 = A2.copy()

        return A1, A2

    def _allocate_rk4_buffers(self, A: np.ndarray, method: str) -> None:
        """Pre-allocate scratch buffers for coupled RK4 stepper."""
        super()._allocate_rk4_buffers(A, method)
        if method == "RK4":
            self._rk4_A_sq_c = self._backend.allocate_real_field(A.shape, np.float32)

    def _RK4_rhs(
        self,
        A_in: np.ndarray,
        k: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
    ) -> np.ndarray:
        """Compute coupled RK4 RHS into pre-allocated buffer k.

        Parameters
        ----------
        A_in : np.ndarray
            Input field (not modified).
        k : np.ndarray
            Output buffer for RHS result (modified in-place for non-MLX).
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects.
        """
        kernels = self._backend.kernels

        # Pre-computed constants (shared by all code paths)
        alpha_half = getattr(self, "_alpha_half", self.alpha / 2)
        alpha2_half = getattr(self, "_alpha2_half", self.alpha2 / 2)
        g11 = getattr(self, "_g11", self.k / 2 * self.n2 * c * epsilon_0)
        g12 = getattr(self, "_g12", self.k / 2 * self.n12 * c * epsilon_0)
        g22 = getattr(self, "_g22", self.k2 / 2 * self.n22 * c * epsilon_0)
        Isat1 = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        Isat2 = getattr(self, "_Isat_conv2", 2 * self.I_sat2 / (epsilon_0 * c))
        V_scaled = getattr(self, "_V_scaled", None)
        V2_scaled = getattr(self, "_V2_scaled", None)
        if V_scaled is None and V is not None:
            k_half = getattr(self, "_k_half", np.float32(self.k / 2))
            k2_half = getattr(self, "_k2_half", np.float32(self.k2 / 2))
            V_scaled = V * k_half
            V2_scaled = V * k2_half

        # Fused fast path (CL, MLX): zero component copies
        if self._backend.has_fused_coupled_rk4_rhs and self.nl_length == 0:
            prop_fft = getattr(self, "_propagator_fft", None)
            return kernels.rk4_rhs_coupled_fused(
                A_in,
                k,
                V_scaled,
                V2_scaled,
                prop_fft if prop_fft is not None else propagator,
                plans[0],
                alpha_half,
                alpha2_half,
                g11,
                g12,
                g22,
                Isat1,
                Isat2,
                unnorm_ifft=(prop_fft is not None),
            )

        if self._backend.name == "MLX":
            k = self._apply_linear_step(A_in, propagator, plans)
        else:
            k[:] = A_in
            k = self._apply_linear_step(k, propagator, plans)

        A1, A2 = self._take_components(A_in)
        k1, k2 = self._take_components(k)

        A_sq = kernels.square_mod(A_in, self._rk4_A_sq_c)
        A_sq_1, A_sq_2 = self._take_components(A_sq)

        if self.nl_length > 0:
            A_sq_1 = self._convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._convolution(
                A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
            )

        if V is None:
            k1 = kernels.rk4_nl_rhs_c(
                k1,
                A1,
                A_sq_1,
                A_sq_2,
                alpha_half,
                g11,
                g12,
                Isat1,
                Isat2,
            )
            k2 = kernels.rk4_nl_rhs_c(
                k2,
                A2,
                A_sq_2,
                A_sq_1,
                alpha2_half,
                g22,
                g12,
                Isat2,
                Isat1,
            )
        else:
            k1 = kernels.rk4_nl_rhs_c_v(
                k1,
                A1,
                A_sq_1,
                A_sq_2,
                V_scaled,
                alpha_half,
                g11,
                g12,
                Isat1,
                Isat2,
            )
            k2 = kernels.rk4_nl_rhs_c_v(
                k2,
                A2,
                A_sq_2,
                A_sq_1,
                V2_scaled,
                alpha2_half,
                g22,
                g12,
                Isat2,
                Isat1,
            )

        k[0] = k1
        k[1] = k2

        return k

    def _apply_nl_prop_c(
        self,
        A1,
        A2,
        A_sq_1,
        A_sq_2,
        dz,
        V_scaled,
        V2_scaled,
        alpha_half,
        alpha2_half,
        g11,
        g12,
        g22,
        Isat_conv,
        Isat_conv2,
    ):
        """Apply coupled nonlinear propagation to both components."""
        kernels = self._backend.kernels
        if V_scaled is None:
            A1 = kernels.nl_prop_without_V_c(
                A1,
                A_sq_1,
                A_sq_2,
                dz,
                alpha_half,
                g11,
                g12,
                Isat_conv,
                Isat_conv2,
            )
            A2 = kernels.nl_prop_without_V_c(
                A2,
                A_sq_2,
                A_sq_1,
                dz,
                alpha2_half,
                g22,
                g12,
                Isat_conv2,
                Isat_conv,
            )
        else:
            A1 = kernels.nl_prop_c(
                A1,
                A_sq_1,
                A_sq_2,
                dz,
                alpha_half,
                V_scaled,
                g11,
                g12,
                Isat_conv,
                Isat_conv2,
            )
            A2 = kernels.nl_prop_c(
                A2,
                A_sq_2,
                A_sq_1,
                dz,
                alpha2_half,
                V2_scaled,
                g22,
                g12,
                Isat_conv2,
                Isat_conv,
            )
        return A1, A2

    def _compute_A_sq_components(self, A, A_sq):
        """Compute squared modulus and extract/convolve components."""
        A_sq = self._backend.kernels.square_mod(A, A_sq)
        A_sq_1, A_sq_2 = self._take_components(A_sq)
        if self.nl_length > 0:
            A_sq_1 = self._convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._convolution(
                A_sq_2, self.nl_profile, mode="same", axes=self._last_axes
            )
        return A_sq, A_sq_1, A_sq_2

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> np.ndarray:
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
        np.ndarray
            The propagated field.
        """
        # Use pre-computed constants with fallbacks for direct calls
        alpha_half = getattr(self, "_alpha_half", self.alpha / 2)
        alpha2_half = getattr(self, "_alpha2_half", self.alpha2 / 2)
        g11 = getattr(self, "_g11", self.k / 2 * self.n2 * c * epsilon_0)
        g12 = getattr(self, "_g12", self.k / 2 * self.n12 * c * epsilon_0)
        g22 = getattr(self, "_g22", self.k2 / 2 * self.n22 * c * epsilon_0)
        Isat_conv = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        Isat_conv2 = getattr(self, "_Isat_conv2", 2 * self.I_sat2 / (epsilon_0 * c))
        V_scaled = getattr(self, "_V_scaled", None)
        V2_scaled = getattr(self, "_V2_scaled", None)
        if V is not None and V_scaled is None:
            V_scaled = V * np.float32(self.k / 2)
            V2_scaled = V * np.float32(self.k2 / 2)
        nl_args = (
            V_scaled,
            V2_scaled,
            alpha_half,
            alpha2_half,
            g11,
            g12,
            g22,
            Isat_conv,
            Isat_conv2,
        )

        kernels = self._backend.kernels

        # Fused fast path (CL, MLX): zero component copies
        if self._backend.has_fused_coupled_split_step and self.nl_length == 0:
            dz = self.delta_z / 2 if precision == "double" else self.delta_z
            prop_fft = getattr(self, "_propagator_fft", None)
            omega_half = (
                self.omega / 2
                if (precision == "single" and self.omega is not None)
                else None
            )
            return kernels.split_step_coupled_fused(
                A,
                prop_fft if prop_fft is not None else propagator,
                V_scaled,
                V2_scaled,
                dz,
                alpha_half,
                alpha2_half,
                g11,
                g12,
                g22,
                Isat_conv,
                Isat_conv2,
                precision,
                plans[0],
                omega=omega_half,
                unnorm_ifft=(prop_fft is not None),
            )

        # First half-step (double precision only)
        if precision == "double":
            A1, A2 = self._take_components(A)
            A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)
            A1, A2 = self._apply_nl_prop_c(
                A1, A2, A_sq_1, A_sq_2, self.delta_z / 2, *nl_args
            )
            A[0] = A1
            A[1] = A2

        # Linear propagation in Fourier domain
        A = self._apply_linear_step(A, propagator, plans)

        # Second half-step (always)
        A1, A2 = self._take_components(A)
        A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)
        dz_step = self.delta_z / 2 if precision == "double" else self.delta_z
        A1, A2 = self._apply_nl_prop_c(A1, A2, A_sq_1, A_sq_2, dz_step, *nl_args)

        if precision == "single" and self.omega is not None:
            A1, A2 = self._backend.kernels.rabi_coupling(
                A1, A2, self.delta_z, self.omega / 2
            )

        A[0] = A1
        A[1] = A2
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot the field.

        Parameters
        ----------
        A_plot : np.ndarray
            The field to plot.
        z : float
            Propagation distance in m.
        """
        A_plot = self._to_plot_array(A_plot, 3)
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
