from collections.abc import Callable
from typing import Any

import numpy as np

from ..utils import __BACKEND__, __CUPY_AVAILABLE__
from .cnlse import CNLSE
from .parameter import Parameter

if __CUPY_AVAILABLE__:
    import cupy as cp

# Scale from DDGPE's potential to the phase rate the kernels apply, unitless.
# The kernels advance by exp(+i V_scaled dt) (NLSE's sign, V_scaled = k V / 2);
# DDGPE's V is an energy rate, i dpsi/dt = V psi (docs/solvers_overview.md).
POTENTIAL_SCALE = -1.0


def _drive_sample(profile: Any, simu: Any, t: float, i: int) -> Any:
    """Return the sample of a temporal drive profile for the step just taken.

    ``profile[j]`` is the drive during ``[j dt, (j + 1) dt)``, ``dt`` the step
    out_field was asked for (its own default if none), before any cap. The
    step just taken ends at ``t``, so its midpoint picks ``j``: an uncapped
    run takes sample ``i`` at step ``i``, a capped one samples by time.

    Parameters
    ----------
    profile : Any
        Samples of the drive, one per asked step.
    simu : Any
        The DDGPE running the step.
    t : float
        Time at the end of the step just taken.
    i : int
        Index of that step, used when not inside out_field.

    Returns
    -------
    Any
        The sample in force over the step.

    Raises
    ------
    ValueError
        If the profile ends before the step does.
    """
    if simu._drive_step is None:
        # Called outside out_field: no asked step to time by.
        index = i
    else:
        midpoint = abs(t) - abs(simu._current_delta_z) / 2
        index = int(np.floor(midpoint / abs(simu._drive_step)))
    if index >= len(profile):
        raise ValueError(
            f"The drive profile has {len(profile)} samples but the step "
            f"ending at t={abs(t):.6g} needs {index + 1}: give one sample "
            "per asked step over the whole run."
        )
    return profile[index]


class DDGPE(CNLSE):
    """A class to solve the 2D driven dissipative Gross-Pitaevskii equation.

    The exciton and cavity equations are Eq. (1.87) of [1]_, here in the frame
    of the pump and with a cross term ``g12`` and a saturation added. Its
    Table 2.1 gives the parameters in ps, µm and meV over hbar.

    References
    ----------
    .. [1] K. Guerrero-Feuillet, "Rotational superradiance in a polariton
       quantum fluid", PhD thesis, Sorbonne Université, 2026.
       https://theses.hal.science/tel-05677012
    """

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
    gamma2 = Parameter("alpha2", "Cavity losses in Hz.")
    g = Parameter("n2", "Intra-component interaction parameter.", scale=-1)
    g12 = Parameter("n12", "Inter-component interaction parameter.")
    T = Parameter("L", "Total propagation time in s.")

    # gamma, gamma2, g and g12 are not listed: views onto alpha, alpha2, n2, n12,
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
    # g2 has storage of its own, so it is not covered by the inherited entries.
    # _step_constants passes n2 and n12, not g and g12 (NLSE sign convention).
    _medium_attrs = (*CNLSE._medium_attrs, "g2")

    def __init__(
        self,
        *,
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
            Potential in rad/s: it advances the phase by -V t. Defaults to
            None.
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
            Non-local length in m, of a diffusive non-locality: the
            interaction reads the intensity smoothed by
            1 / (1 + nl_length^2 K^2) in Fourier space. Defaults to 0, local.
        backend : str, optional
            Backend name: "CPU", "CUPY", "CL", "MLX" or "auto" (the installed
            one with the fastest transform at this grid size). Defaults to
            the backend NLSE picked when it was imported.
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
        noise : float
            Amplitude of the noise, as a multiple of sqrt(gamma / 4 dX dY).
            Zero, the default, adds nothing. Each step adds a Wiener
            increment: each quadrature has standard deviation
            ``noise * sqrt(gamma / (4 dX dY)) * sqrt(dt)``, so a component
            gains ``noise**2 * gamma / (2 dX dY)`` of mean intensity per unit
            time, whatever the step.
        """
        # A Wiener increment's variance, not its deviation, grows with dt.
        increment = np.sqrt(abs(simu._current_delta_z))
        area = 4 * (simu.delta_X * simu.delta_Y)
        # Not A.dtype: MLX names its own dtypes and numpy cannot interpret
        # them, and handing a complex128 draw to a complex64 field is how the
        # noise arrives scaled by whatever the reinterpretation makes of it.
        dtype = simu._field_dtype(A)
        for component, gamma in ((0, simu.gamma), (1, simu.gamma2)):
            draw = simu._complex_noise(increment)
            delta = (noise * np.sqrt(gamma / area) * draw).astype(dtype)
            # Read, add, write back: pyopencl silently discards `+=` on a slice. The
            # noise also has to reach the device before it is added.
            A[..., component, :, :] = A[
                ..., component, :, :
            ] + simu._backend.from_numpy(delta)

    def _complex_noise(self, scale: float) -> Any:
        """Return one grid of complex Gaussian noise.

        Parameters
        ----------
        scale : float
            Standard deviation of the real part, and of the imaginary part.

        Returns
        -------
        Any
            An (NY, NX) grid from the solver's generator.
        """
        shape = (self.NY, self.NX)
        real = self._random(loc=0, scale=scale, size=shape)
        return real + 1j * self._random(loc=0, scale=scale, size=shape)

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
            The temporal profile of the pump field: ``F_pump_t[j]`` is the
            drive during ``[j dt, (j + 1) dt)``, ``dt`` the step out_field was
            asked for (its default if None), before any cap. Each step takes
            the sample at its midpoint; a profile that ends first raises
            ValueError.
        F_probe_r : np.ndarray
            The spatial profile of the probe field.
        F_probe_t : np.ndarray
            The temporal profile of the probe field, sampled as F_pump_t.
        """
        # Read, subtract, write back, one profile at a time: pyopencl takes `-=` on
        # a slice, raises nothing and leaves it zero, which zeroed the cavity on CL.
        dtype = simu._field_dtype(A)
        dt = simu._current_delta_z
        for profile_r, profile_t in ((F_pump_r, F_pump_t), (F_probe_r, F_probe_t)):
            amplitude = _drive_sample(profile_t, simu, t, i)
            delta = profile_r * amplitude * dt * 1j
            if isinstance(delta, np.ndarray):
                delta = simu._backend.from_numpy(delta.astype(dtype))
            A[..., 1, :, :] = A[..., 1, :, :] - delta

    def _step_constants(self) -> dict[str, Any]:
        """Override the constants, which DDGPE gives the kernels unconverted.

        CNLSE scales the couplings by ``k / 2 * c * epsilon_0``, the potential
        by ``k / 2`` and the saturation by ``2 / (epsilon_0 * c)``. DDGPE's
        ``k`` comes from a wavelength it supplies only to satisfy the base
        constructor, so that conversion would inflate the interaction rate by
        ~1e26. The losses are CNLSE's, ``gamma2`` being ``alpha2``.
        """
        return {
            **super()._step_constants(),
            "_Isat_conv": self.I_sat,
            "_Isat_conv2": self.I_sat2,
            "_k_half": POTENTIAL_SCALE,
            "_k2_half": POTENTIAL_SCALE,
            "_g11": self.n2,
            # Both components are the same field at the same energy scale here,
            # so the two cross constants CNLSE splits by k coincide.
            "_g12": self.n12,
            "_g21": self.n12,
            "_g22": self.g2,
        }

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
        prop2 = -1j * (
            self.omega_cav * np.sqrt(1 + (self.Kxx**2 + self.Kyy**2) / self.k_z**2)
            - self.omega_pump
        )
        # Broadcast to the grid: the exciton branch is dispersionless, a scalar, and
        # np.array of a scalar beside a grid raises (numpy >= 1.24).
        prop1 = np.full_like(prop2, -1j * (self.omega_exc - self.omega_pump))
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

    # Its step drives the cavity and adds noise, so it is not a product of
    # the same two operators taken in halves.
    _lie_step_is_strang_body = False

    # The step out_field was asked for, which the drive profiles are sampled
    # by; None outside out_field.
    _drive_step: float | complex | None = None

    def _default_delta_z(
        self,
        A: np.ndarray,
        method: str = "split_step",
        z: float | None = None,
        rates: dict[str, float] | None = None,
    ) -> float:
        """Return the solver's default step, kept as the drive's sampling step.

        Parameters
        ----------
        A : np.ndarray
            Field, as for the parent.
        method : str, optional
            Integration method, as for the parent.
        z : float, optional
            Total time, as for the parent.
        rates : dict, optional
            The field's rates, as for the parent.

        Returns
        -------
        float
            The parent's default step.
        """
        delta_z = super()._default_delta_z(A, method, z, rates)
        self._drive_step = delta_z
        return delta_z

    def out_field(
        self,
        E_in: np.ndarray,
        t: float,
        laser_excitation: Callable | None,
        delta_z: float | None = None,
        plot: bool = False,
        splitting: str = "lie",
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
        laser_excitation : callable or None
            The pump and probe, called as a callback before the others. None
            uses `DDGPE.laser_excitation`. Either way its arguments are the
            first tuple of `callback_args`.
        delta_z : float, optional
            Time step. Defaults to None, meaning the solver derives one from
            the field.
        plot : bool, optional
            Whether to plot the results. Defaults to False.
        splitting : str, optional
            Splitting scheme, as in `NLSE.out_field`. Defaults to "lie".
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        callback : list[callable] or callable, optional
            A list of functions
            to execute at every solver step. Defaults to None.
        callback_args : list[tuple], optional
            One tuple of arguments per callback, the excitation's first.
            Defaults to None.

        Returns
        -------
        np.ndarray
            Propagated field.
        """
        excitation = (
            self.laser_excitation if laser_excitation is None else laser_excitation
        )
        # A new list, so the caller's own can be passed again unchanged.
        if callback is None:
            callbacks = [excitation]
        elif callable(callback):
            callbacks = [excitation, callback]
        else:
            callbacks = [excitation, *callback]
        # None asks for the default, which _default_delta_z records instead.
        self._drive_step = delta_z
        try:
            return super().out_field(
                E_in=E_in,
                z=t,
                delta_z=delta_z,
                plot=plot,
                splitting=splitting,
                verbose=verbose,
                normalize=False,
                callback=callbacks,
                callback_args=callback_args,
            )
        finally:
            self._drive_step = None
