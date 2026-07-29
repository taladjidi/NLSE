from collections.abc import Callable

import numpy as np

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .cnlse import CNLSE
from .parameter import Parameter

if __CUPY_AVAILABLE__:
    import cupy as cp


class DDGPE(CNLSE):
    """A class to solve the 2D driven dissipative Gross-Pitaevskii equation."""

    # A polariton density on a time axis, with the two components named for
    # the exciton and the cavity. Everything else about the plot is CNLSE's.
    _plot_density_scale = 1.0
    _plot_density_label = "Density"
    _plot_axis_symbol = "t"
    _plot_axis_unit = "ps"
    _plot_axis_format = "g"
    _plot_components = (r"\psi_x", r"\psi_c")

    # Polariton physics on CNLSE's terms, sharing its storage. g carries the
    # sign: the two parametrisations write the interaction term opposite ways,
    # and the kernels want CNLSE's.
    gamma = Parameter("alpha", "Losses in Hz.")
    g = Parameter("n2", "Intra-component interaction parameter.", scale=-1)
    g12 = Parameter("n12", "Inter-component interaction parameter.")
    T = Parameter("L", "Total propagation time in s.")

    # gamma, g and g12 are not listed: they are views onto alpha, n2 and n12,
    # which the inherited tuples already carry, and transferring the same
    # storage twice would convert it twice.
    _gpu_param_attrs = (
        *CNLSE._gpu_param_attrs,
        "omega",
        "k_z",
        "omega_exc",
        "omega_cav",
        "detuning",
        "omega_pump",
    )
    # g2 belongs to the second component and has storage of its own, so unlike
    # g and g12 it is not covered by the inherited entries.
    #
    # split_step passes n2 and n12 rather than g and g12: the kernels want the
    # NLSE sign convention, which is what the storage holds.
    _nonlinearity_attrs = (*CNLSE._nonlinearity_attrs, "g2")

    def __init__(
        self,
        gamma: float,
        power: float,
        window: float,
        g: float,
        omega: float,
        T: float,
        omega_exc: float,
        omega_cav: float,
        detuning: float,
        k_z: float,
        V: np.ndarray = None,
        g12: float = 0,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the class with all the relevant physical parameters.

        Parameters
        ----------
        gamma : float
            Losses coefficient in s^-1.
        power : float
            Optical power in W.
        window : float
            Computational window in m.
        g : float
            Interaction parameter.
        omega : float
            Rabi coupling strength.
        T : float
            Total propagation time in s.
        omega_exc : float
            Exciton frequency in rad/s.
        omega_cav : float
            Cavity frequency in rad/s.
        detuning : float
            Detuning from lower polariton in rad/s.
        k_z : float
            Longitudinal wavevector in m^-1.
        V : np.ndarray, optional
            Potential landscape in a.u. Defaults to None.
        g12 : float, optional
            Inter component interaction parameter. Defaults to 0.
        NX : int, optional
            Number of points along x. Defaults to 1024.
        NY : int, optional
            Number of points along y. Defaults to 1024.
        Isat : float, optional
            Saturation intensity, assumed to be the same
            for both components. Defaults to infinity.
        nl_length : float, optional
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute. Defaults to 0.
        backend : str, optional
            "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=gamma,
            power=power,
            window=window,
            n2=-g,
            n12=g12,
            V=V,
            L=T,
            NX=NX,
            NY=NY,
            Isat=Isat,
            nl_length=nl_length,
            wvl=1e-30,
            omega=omega,
            backend=backend,
        )
        self.g2 = 0
        self.k_z = k_z
        self.gamma2 = gamma
        self.omega_exc = omega_exc
        self.omega_cav = omega_cav
        self.detuning = detuning
        omega_lp = (omega_exc + omega_cav) / 2 - 0.5 * np.sqrt(
            (omega_exc - omega_cav) ** 2 + (omega) ** 2
        )
        self.omega_pump = omega_lp + detuning
        if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
            self._random = cp.random.normal
        else:
            self._random = np.random.normal

    @staticmethod
    def add_noise(
        simu: object,
        A: np.ndarray,
        t: float,
        i: int,
        noise: float = 0,
    ) -> None:
        """Add noise to the propagation step.

        Follows the callback convention of NLSE.

        Parameters
        ----------
        simu : object
            DDGPE object.
        A : np.ndarray
            Field array.
        t : float
            Propagation time in s.
        i : int
            Propagation step.
        """
        rand1 = simu._random(
            loc=0, scale=simu._current_delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(
            loc=0, scale=simu._current_delta_z, size=(simu.NY, simu.NX)
        )
        rand2 = simu._random(
            loc=0, scale=simu._current_delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(
            loc=0, scale=simu._current_delta_z, size=(simu.NY, simu.NX)
        )
        A[..., 0, :, :] += (
            noise * np.sqrt(simu.gamma / (4 * (simu.delta_X * simu.delta_Y))) * rand1
        )
        A[..., 1, :, :] += (
            noise * np.sqrt((simu.gamma2) / (4 * (simu.delta_X * simu.delta_Y))) * rand2
        )

    @staticmethod
    def laser_excitation(
        simu: object,
        A: np.ndarray,
        t: float,
        i: int,
        F_pump_r: np.ndarray,
        F_pump_t: np.ndarray,
        F_probe_r: np.ndarray,
        F_probe_t: np.ndarray,
    ) -> None:
        """Add the pump and probe laser.

        This function adds a pump field with a spatial profile F_pump_r and a temporal
        profile F_pump_t and a probe field with a spatial profile F_probe_r and a
        temporal profile F_probe_t. The pump and probe fields are added to the
        cavity field at each propagation step.

        Parameters
        ----------
        simu : object
            The simulation object.
        A : np.ndarray
            The field array.
        t : float
            The current solver time.
        i : int
            The current solver step.
        F_pump_r : np.ndarray
            The spatial profile of the pump field.
        F_pump_t : np.ndarray
            The temporal profile of the pump field.
        F_probe_r : np.ndarray
            The spatial profile of the probe field.
        F_probe_t : np.ndarray
            The temporal profile of the probe field.
        """
        A[..., 1, :, :] -= F_pump_r * F_pump_t[i] * simu._current_delta_z * 1j
        A[..., 1, :, :] -= F_probe_r * F_probe_t[i] * simu._current_delta_z * 1j

    def _precompute_step_constants(
        self, V: np.ndarray | None, precision: str = "single"
    ) -> None:
        """Pre-compute constants for DDGPE propagation steps."""
        super()._precompute_step_constants(V, precision)
        fp = np.float32 if precision == "single" else np.float64
        self._gamma_half = fp(self.gamma / 2)
        self._gamma2_half = fp(self.gamma2 / 2)
        # DDGPE's couplings go to the kernels as they are, with none of the
        # optical conversion CNLSE applies. Restate them, because the step
        # limits read these: left at CNLSE's, they carry a factor k/2 built
        # from a wavelength DDGPE only supplies to keep the base class happy,
        # which makes the interaction rate about 1e26 times too large and
        # collapses the step to nothing.
        self._g11 = fp(self.n2)
        self._g12 = fp(self.n12)
        self._g22 = fp(self.g2)

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for DDGPE propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.omega_exc),
            float(self.omega_cav),
            float(self.omega_pump),
            float(self.k_z),
        )

    def _compute_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Compute the DDGPE polariton propagation matrices."""
        propagator1 = np.exp(
            -1j * (self.omega_exc * (1 + 0 * self.Kxx**2) - self.omega_pump) * delta_z,
            dtype=dtype,
        )
        propagator2 = np.exp(
            -1j
            * (
                self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
                - self.omega_pump
            )
            * delta_z,
            dtype=dtype,
        )
        return np.array([propagator1, propagator2])

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for DDGPE RK4 dispersion operator."""
        return (
            self.NX,
            self.NY,
            "RK4",
            float(self.omega_exc),
            float(self.omega_cav),
            float(self.omega_pump),
            float(self.k_z),
        )

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw DDGPE polariton dispersion operators for RK4."""
        prop1 = (-1j * (self.omega_exc - self.omega_pump)).astype(np.complex64)
        prop2 = (
            -1j
            * (
                self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
                - self.omega_pump
            )
        ).astype(np.complex64)
        return np.array([prop1, prop2])

    def _dispersion_operator(self) -> np.ndarray:
        """Return the DDGPE polariton dispersion eigenvalues.

        Use the actual polariton branches instead of K^2/(2k). The exciton
        branch is flat, so the larger of the two is taken pointwise.
        """
        D_exc = abs(self.omega_exc - self.omega_pump)
        D_cav = np.abs(
            self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
            - self.omega_pump
        )
        return np.maximum(D_exc, D_cav)

    def _prepare_output_array(self, E_in: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare the output array depending on __BACKEND__.

        Parameters
        ----------
        E_in : np.ndarray
            Input array.
        normalize : bool
            Normalize the field to the total power.

        Returns
        -------
        np.ndarray
            Output array.
        """
        A_sq = self._backend.allocate_real_field(E_in.shape, E_in.real.dtype)
        A = self._backend.from_numpy(E_in)
        if normalize:
            pass
        return A, A_sq

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray,
        propagator: np.ndarray,
        plans: list,
        delta_z: float,
        precision: str = "single",
    ) -> np.ndarray:
        """Split step function for one propagation step.

        Parameters
        ----------
        A : np.ndarray
            Fields to propagate of shape (2, NY, NX).
        A_sq : np.ndarray
            Squared modulus of the fields.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix for both fields
            [propagator1, propagator2].
        plans : list
            List of FFT plan objects. Either a single FFT plan for
            both directions (GPU case) or distinct FFT and IFFT plans for FFTW.
        delta_z : float
            Step to take. Must match the propagator, which was built
            from it.
        precision : str, optional
            Single or double application of the nonlinear
            propagation step. Defaults to "single".

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        # Use pre-computed constants with fallbacks
        gamma_half = getattr(self, "_gamma_half", self.gamma / 2)
        gamma2_half = getattr(self, "_gamma2_half", self.gamma2 / 2)
        kernels = self._backend.kernels

        A1, A2 = self._take_components(A)

        # First half-step (double precision only)
        if precision == "double":
            A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)
            if V is None:
                A1 = kernels.nl_prop_without_V_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    delta_z / 2,
                    gamma_half,
                    self.n2,
                    self.n12,
                    self.I_sat,
                    self.I_sat2,
                )
                A2 = kernels.nl_prop_without_V_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    delta_z / 2,
                    gamma2_half,
                    self.g2,
                    self.n12,
                    self.I_sat,
                    self.I_sat2,
                )
            else:
                A1 = kernels.nl_prop_c(
                    A1,
                    A_sq_1,
                    A_sq_2,
                    delta_z / 2,
                    gamma_half,
                    V,
                    self.n2,
                    self.n12,
                    self.I_sat,
                    self.I_sat2,
                )
                A2 = kernels.nl_prop_c(
                    A2,
                    A_sq_2,
                    A_sq_1,
                    delta_z / 2,
                    gamma2_half,
                    V,
                    self.g2,
                    self.n12,
                    self.I_sat,
                    self.I_sat2,
                )
            A[0] = A1
            A[1] = A2

        # Linear propagation in Fourier domain
        A = self._apply_linear_step(A, propagator, plans)

        # Second half-step (always)
        A1, A2 = self._take_components(A)
        A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)

        dz_step = delta_z / 2 if precision == "double" else delta_z
        if V is None:
            A1 = kernels.nl_prop_without_V_c(
                A1,
                A_sq_1,
                A_sq_2,
                dz_step,
                gamma_half,
                self.n2,
                self.n12,
                self.I_sat,
                self.I_sat2,
            )
            A2 = kernels.nl_prop_without_V_c(
                A2,
                A_sq_2,
                A_sq_1,
                dz_step,
                gamma2_half,
                self.g2,
                self.n12,
                self.I_sat,
                self.I_sat2,
            )
        else:
            A1 = kernels.nl_prop_c(
                A1,
                A_sq_1,
                A_sq_2,
                dz_step,
                gamma_half,
                V,
                self.n2,
                self.n12,
                self.I_sat,
                self.I_sat2,
            )
            A2 = kernels.nl_prop_c(
                A2,
                A_sq_2,
                A_sq_1,
                dz_step,
                gamma2_half,
                V,
                self.g2,
                self.n12,
                self.I_sat,
                self.I_sat2,
            )
        if precision == "single" and self.omega is not None:
            A1, A2 = kernels.rabi_coupling(A1, A2, delta_z, self.omega / 2)

        A[0] = A1
        A[1] = A2
        return A

    def out_field(
        self,
        E_in: np.ndarray,
        t: float,
        laser_excitation: Callable | None,
        delta_z: float | complex | None = None,
        plot: bool = False,
        precision: str = "single",
        verbose: bool = True,
        callback: list[Callable] | Callable | None = None,
        callback_args: list[tuple] | tuple | None = None,
    ) -> np.ndarray:
        """Propagate a field to time T.

        Parameters
        ----------
        E_in : np.ndarray
            Input field where E_in[0] is the exciton field and
            E_in[1] is the cavity field.
        t : float
            Time to propagate to in s.
        delta_z : float or complex, optional
            Time step. Defaults to None, meaning the solver derives one from
            the field.
        laser_excitation : callable or None
            The excitation function.
            This represents the laser pump and probe. Defaults to None which uses
            the static method defined in the class. In this case you still need
            to pass the correct arguments to the callback_args.
        plot : bool, optional
            Whether to plot the results. Defaults to False.
        precision : str, optional
            Whether to apply the nonlinear terms in a
            single or double step. Defaults to "single".
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        callback : list[callable] or callable, optional
            A list of functions
            to execute at every solver step. Defaults to None.
        callback_args : list[tuple] or tuple, optional
            A list of callback
            arguments passed to the callbacks. Defaults to None.

        Returns
        -------
        np.ndarray
            Propagated field.
        """
        if callback is None:
            callback = []
        elif callable(callback):
            callback = [callback]
        if laser_excitation is None:
            callback.insert(0, self.laser_excitation)
        else:
            callback.insert(0, laser_excitation)
        return super().out_field(
            E_in=E_in,
            z=t,
            delta_z=delta_z,
            plot=plot,
            precision=precision,
            verbose=verbose,
            normalize=False,
            callback=callback,
            callback_args=callback_args,
        )
