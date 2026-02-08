from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .cnlse import CNLSE

if __CUPY_AVAILABLE__:
    import cupy as cp


class DDGPE(CNLSE):
    """A class to solve the 2D driven-dissipative Gross-Pitaevskii equation."""

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
        V: np.ndarray | None = None,
        g12: float = 0,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the simulation.

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
            Rabi coupling.
        T : float
            Propagation time in s.
        omega_exc : float
            Exciton frequency.
        omega_cav : float
            Cavity frequency.
        detuning : float
            Detuning from the lower polariton.
        k_z : float
            Longitudinal wave-vector.
        V : np.ndarray or None, optional
            Potential landscape. Defaults to ``None``.
        g12 : float, optional
            Inter-component interaction parameter. Defaults to 0.
        NX : int, optional
            Number of points along x. Defaults to 1024.
        NY : int, optional
            Number of points along y. Defaults to 1024.
        Isat : float, optional
            Saturation intensity. Defaults to np.inf.
        nl_length : float, optional
            Non local length in m. Defaults to 0.
        backend : str, optional
            Compute backend. Defaults to ``__BACKEND__``.
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
        self.g = self.n2
        self.g12 = self.n12
        self.g2 = 0
        self.k_z = k_z
        self.gamma = gamma
        self.gamma2 = self.gamma
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
        simu: DDGPE,
        A: np.ndarray,
        t: float,
        i: int,
        noise: float = 0,
    ) -> None:
        """Add noise to the propagation step.

        Follows the callback convention of NLSE.

        Parameters
        ----------
        simu : DDGPE
            Simulation object.
        A : np.ndarray
            Field array.
        t : float
            Propagation time in s.
        i : int
            Propagation step.
        noise : float, optional
            Noise amplitude. Defaults to 0.
        """
        rand1 = simu._random(
            loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX))
        rand2 = simu._random(
            loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX)
        ) + 1j * simu._random(loc=0, scale=simu.delta_z, size=(simu.NY, simu.NX))
        A[..., 0, :, :] += (
            noise * np.sqrt(simu.gamma / (4 * (simu.delta_X * simu.delta_Y))) * rand1
        )
        A[..., 1, :, :] += (
            noise * np.sqrt((simu.gamma2) / (4 * (simu.delta_X * simu.delta_Y))) * rand2
        )

    @staticmethod
    def laser_excitation(
        simu: DDGPE,
        A: np.ndarray,
        t: float,
        i: int,
        F_pump_r: np.ndarray,
        F_pump_t: np.ndarray,
        F_probe_r: np.ndarray,
        F_probe_t: np.ndarray,
    ) -> None:
        """Add the pump and probe laser.

        This function adds a pump field with a spatial profile ``F_pump_r``
        and a temporal profile ``F_pump_t`` and a probe field with a spatial
        profile ``F_probe_r`` and a temporal profile ``F_probe_t``.

        Parameters
        ----------
        simu : DDGPE
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
        A[..., 1, :, :] -= F_pump_r * F_pump_t[i] * simu.delta_z * 1j
        A[..., 1, :, :] -= F_probe_r * F_probe_t[i] * simu.delta_z * 1j

    def _send_arrays_to_gpu(self, force_refresh: bool = False) -> None:
        """Send arrays to GPU.

        Parameters
        ----------
        force_refresh : bool, optional
            Force re-uploading arrays even if already on GPU. Defaults to False.
        """
        super()._send_arrays_to_gpu(force_refresh=force_refresh)

        # Lazy GPU transfer: skip if arrays are already on device
        if not force_refresh and hasattr(self, "_gpu_initialized") and self._gpu_initialized:
            return

        for attr in (
            "gamma",
            "g",
            "omega",
            "k_z",
            "omega_exc",
            "omega_cav",
            "detuning",
            "omega_pump",
        ):
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, self._backend.to_device(val))

    def _retrieve_arrays_from_gpu(self) -> None:
        """
        Retrieve arrays from GPU.
        """
        super()._retrieve_arrays_from_gpu()
        for attr in (
            "gamma",
            "g",
            "omega",
            "k_z",
            "omega_exc",
            "omega_cav",
            "detuning",
            "omega_pump",
        ):
            val = getattr(self, attr)
            if self._backend.is_device_array(val):
                setattr(self, attr, self._backend.to_host(val))

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the propagators.

        Parameters
        ----------
        precision : str, optional
            ``"single"``, ``"double"`` or ``"RK4"``. Defaults to ``"single"``.

        Returns
        -------
        np.ndarray
            Array of linear propagators for each component.
        """
        propagator1 = np.exp(
            -1j
            * (self.omega_exc * (1 + 0 * self.Kxx**2) - self.omega_pump)
            * self.delta_z
        ).astype(np.complex64)
        propagator2 = np.exp(
            -1j
            * (
                self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
                - self.omega_pump
            )
            * self.delta_z
        ).astype(np.complex64)
        return np.array([propagator1, propagator2])

    def _compute_norm_factor(self, E_in):
        """DDGPE: no normalization (returns 1)."""
        return 1

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> tuple[Any, Any]:
        """Prepare the output array depending on backend.

        Parameters
        ----------
        E_in : np.ndarray
            Input array.
        normalize : bool
            Normalize the field to the total power.

        Returns
        -------
        A : array-like
            Output field array.
        A_sq : array-like
            Output field modulus squared array.
        """
        A, A_sq = self._backend.allocate_pair(E_in.shape, E_in.dtype)
        A[:] = self._backend.to_device(E_in)
        return A, A_sq

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: Any,
        precision: str = "single",
    ) -> None:
        """Perform one split-step propagation step.

        Parameters
        ----------
        A : np.ndarray
            Fields to propagate of shape ``(2, NY, NX)``.
        A_sq : np.ndarray
            Squared modulus of the fields.
        V : np.ndarray or None
            Potential field (can be ``None``).
        propagator : np.ndarray
            Propagator matrix for both fields.
        plans : FFTPlan
            FFT plan object.
        precision : str, optional
            Single or double application of the nonlinear propagation
            step. Defaults to ``"single"``.
        """
        A1, A2 = self._take_components(A)
        if precision == "double":
            # Use fused kernels when available and no convolution needed
            use_fused = (
                self.nl_length == 0
                and hasattr(self._kernels, "nl_prop_c_fused")
                and hasattr(self._kernels, "nl_prop_without_V_c_fused")
            )

            if use_fused:
                # Fused path: compute |A1|² and |A2|² inline within nl_prop_c
                if V is None:
                    self._kernels.nl_prop_without_V_c_fused(
                        A1,
                        A2,
                        self.delta_z / 2,
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c_fused(
                        A2,
                        A1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c_fused(
                        A1,
                        A2,
                        self.delta_z / 2,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c_fused(
                        A2,
                        A1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
            else:
                # Non-fused path: compute |A|² separately
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
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c(
                        A1,
                        A_sq_1,
                        A_sq_2,
                        self.delta_z,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
            self._put_components(A, A1, A2)
        self._linear_step(A, propagator)
        A1, A2 = self._take_components(A)

        # Use fused kernels when available and no convolution needed
        use_fused = (
            self.nl_length == 0
            and hasattr(self._kernels, "nl_prop_c_fused")
            and hasattr(self._kernels, "nl_prop_without_V_c_fused")
        )

        if use_fused:
            # Fused path: compute |A1|² and |A2|² inline within nl_prop_c
            if precision == "double":
                if V is None:
                    self._kernels.nl_prop_without_V_c_fused(
                        A1,
                        A2,
                        self.delta_z / 2,
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c_fused(
                        A2,
                        A1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c_fused(
                        A1,
                        A2,
                        self.delta_z / 2,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c_fused(
                        A2,
                        A1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
            else:
                if V is None:
                    self._kernels.nl_prop_without_V_c_fused(
                        A1,
                        A2,
                        self.delta_z,
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c_fused(
                        A2,
                        A1,
                        self.delta_z,
                        self.gamma2 / 2,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c_fused(
                        A1,
                        A2,
                        self.delta_z,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c_fused(
                        A2,
                        A1,
                        self.delta_z,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
            if self.omega is not None:
                self._kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)
        else:
            # Non-fused path: compute |A|² separately
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
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c(
                        A1,
                        A_sq_1,
                        A_sq_2,
                        self.delta_z / 2,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z / 2,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
            else:
                if V is None:
                    self._kernels.nl_prop_without_V_c(
                        A1,
                        A_sq_1,
                        A_sq_2,
                        self.delta_z,
                        self.gamma / 2,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_without_V_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z,
                        self.gamma2 / 2,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                else:
                    self._kernels.nl_prop_c(
                        A1,
                        A_sq_1,
                        A_sq_2,
                        self.delta_z,
                        self.gamma / 2,
                        V,
                        self.g,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                    self._kernels.nl_prop_c(
                        A2,
                        A_sq_2,
                        A_sq_1,
                        self.delta_z,
                        self.gamma2 / 2,
                        V,
                        self.g2,
                        self.g12,
                        self.I_sat,
                        self.I_sat2,
                    )
                if self.omega is not None:
                    self._kernels.rabi_coupling(A1, A2, self.delta_z, self.omega / 2)
        self._put_components(A, A1, A2)

    def plot_field(self, A_plot: np.ndarray, t: float) -> None:
        """Plot the field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            The field to plot.
        t : float
            The time at which the field was sampled.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 3:
            while len(A_plot.shape) > 3:
                A_plot = A_plot[0]
        A_plot = self._backend.to_host(A_plot)
        if not isinstance(A_plot, np.ndarray):
            A_plot = np.asarray(A_plot)
        fig, ax = plt.subplots(2, 2, layout="constrained", figsize=(10, 10))
        fig.suptitle(rf"Field at $t$ = {t:} ps")
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
        ]
        rho0 = np.abs(A_plot[0]) ** 2
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        ax[0, 0].set_title(r"$|\psi_x|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=r"Density")
        im1 = ax[0, 1].imshow(
            phi0,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[0, 1].set_title(r"Phase $\mathrm{arg}(\psi_x)$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(r"$|\psi_c|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=r"Density")
        im3 = ax[1, 1].imshow(
            phi1,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1, 1].set_title(r"Phase $\mathrm{arg}(\psi_c)$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        plt.show()

    def out_field(  # type: ignore[override]
        self,
        E_in: np.ndarray,
        t: float,
        laser_excitation: Callable[..., Any] | None,
        plot: bool = False,
        precision: str = "single",
        verbose: bool = True,
        callback: list[Callable[..., Any]] | Callable[..., Any] | None = None,
        callback_args: list[tuple[Any, ...]] | tuple[Any, ...] | None = None,
    ) -> Any:
        """Propagate a field to time *t*.

        Parameters
        ----------
        E_in : np.ndarray
            Input field where ``E_in[0]`` is the exciton field and
            ``E_in[1]`` is the cavity field.
        t : float
            Time to propagate to in s.
        laser_excitation : callable or None
            The excitation function representing the laser pump and
            probe. ``None`` uses the static method defined in the class.
        plot : bool, optional
            Whether to plot the results. Defaults to ``False``.
        precision : str, optional
            ``"single"`` or ``"double"`` application of the nonlinear
            terms. Defaults to ``"single"``.
        verbose : bool, optional
            Whether to print progress. Defaults to ``True``.
        callback : callable or list of callable, optional
            Function(s) to execute at every solver step.
            Defaults to ``None``.
        callback_args : tuple or list of tuple, optional
            Arguments passed to the callbacks. Defaults to ``None``.

        Returns
        -------
        np.ndarray
            Propagated field.
        """
        if callback is None:
            callback = []
        elif callable(callback):
            callback = [callback]
        if callback_args is None:
            callback_args = [()]
        if laser_excitation is None:
            callback.insert(0, self.laser_excitation)
        else:
            callback.insert(0, laser_excitation)
        return super().out_field(
            E_in=E_in,
            z=t,
            plot=plot,
            precision=precision,
            verbose=verbose,
            normalize=False,
            callback=callback,
            callback_args=callback_args,
        )
