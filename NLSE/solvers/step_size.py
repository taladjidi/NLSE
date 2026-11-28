"""How big a step the solver may take.

Methods that answer one question, and answer it in three stages: what the
field's energy is in each term of the equation, what step those energies allow,
and whether the step the caller asked for is inside that. They came out of
NLSE, where they sat between the propagator and the propagation loop and
accounted for a sixth of the file without being either.

They are a mixin rather than functions taking a solver. CNLSE overrides
``_energy_rates`` -- its two components carry two of everything -- and a
function would have to be told that, where a method is simply overridden. What
they need of the solver is what any of its methods need: its grid, its
backend, its scaled potential and its precomputed constants. Nothing here
writes to it except ``_dispersion_cache``, which is a cache.

Splitting a class across files is a real cost, so this is the seam it is worth
paying for: every method below is called by the ones below it or by the
propagation loop, and nothing else in NLSE calls in.
"""

import math
import warnings
from typing import Any

import numpy as np

# Explicit RK4 is stable for |lambda * dz| up to ~2*sqrt(2) along the
# imaginary axis. Every solver's step limit is derived from it.
RK4_STABILITY_RADIUS = 2.83

# Phase per step, at the peak, past which RK4 diverges on the cubic term
# alone: psi' = -i|psi|^2 psi from |psi| = 1 blows up from a step of 1.547
# (tests/solvers/test_solver_state.py iterates that map), rounded down.
RK4_CUBIC_RADIUS = 1.5

# Phase per step the linear step may put on the shortest resolved wave before
# split-step goes unstable on a finite background (Weideman and Herbst, SIAM J.
# Numer. Anal. 23, 485 (1986)). On a vortex dipole: unstable from 1.6 rad,
# converged below 1.3 (CHANGELOG.md, 4.1.0), so 1.0 leaves a margin.
SPLIT_STEP_DISPERSION_PHASE_LIMIT = 1.0

# Phase per step the default aims at, under the limits (pi for split-step, 2.83
# for RK4). Split-step stays at 0.1: the smooth-beam optimum 0.4 gave 140% error
# on examples/fig2_turbulence.py. RK4's error still falls at 0.02 (2.3e-6 against
# 5.6e-4 at 0.1). Both are in docs/optimization-log.md.
DEFAULT_PHASE_PER_STEP = 0.1
RK4_PHASE_PER_STEP = 0.02

# Fewest steps a default may take over the requested distance, so that a run
# is something a callback can sample and a plot can show rather than one jump.
DEFAULT_MIN_STEPS = 10

# Bound on 2*alpha*dz, alpha the medium's loss: a real-space step takes out
# at most 2.5% of the intensity, a quarter of the kernels' _LOSS_SOLVED_LIMIT
# (0.1, on alpha*dz), past which their solved step falls back to the frozen
# one. A kernel called directly carries its own limit.
LOSS_PER_STEP_LIMIT = 0.05

# Names _grid_on_device keeps the potential's real part and absorption under.
# Their key holds the grid and not V, so they are dropped when V is rescaled.
POTENTIAL_GRIDS = ("V_re", "V_im")

# Fraction of a limit that a step past it is lowered to (dimensionless): a
# margin under the bound rather than the bound itself. A design choice, not a
# derived value: the 0.9 the cap has used since commit c6449e4.
CAPPED_STEP_FRACTION = 0.9


class StepSize:
    """The step-size half of a solver, mixed into NLSE.

    Every method is private and reached through the solver, so this class is
    never instantiated on its own and carries no state of its own.
    """

    # Step constants each holding one component's loss as the kernels take it,
    # alpha / 2 (NLSE._step_constants). CNLSE adds its second component's.
    _loss_constants = ("_alpha_half",)

    def _peak_loss_half(self) -> float:
        """Return the largest ``alpha / 2`` over the components and the batch.

        Returns
        -------
        float
            That rate, in the units of the solver's loss.
        """
        # Through the host: a batched run holds its constants on the device.
        return float(
            max(
                np.max(np.abs(self._as_host_array(self._constant(name))))
                for name in self._loss_constants
            )
        )

    def _energy_rates(self, A: np.ndarray) -> dict[str, float]:
        """Return the phase rate each term contributes, weighted by the field.

        Each entry is an expectation value, ``<psi|O|psi> / <psi|psi>``: the
        kinetic term over the spectral density, the rest over the intensity.
        That is the rate at which a term rotates the field's phase, so
        multiplying by dz gives the phase it adds in one step — the energy in
        that term, in the units the step limits are written in.

        This is what the limiters reduce with, rather than ``max`` over each
        operator. A maximum is a property of the grid, not of the solution: a
        tall potential in a corner the field never reaches, or a high-K corner
        with no spectral weight, would set the step for a run it has no effect
        on. Weighting by the field follows the physics instead. The potential
        is weighted by its magnitude: a signed one whose mean over the beam
        vanishes still turns the phase where the beam is.

        Parameters
        ----------
        A : np.ndarray
            Field, possibly on a device.

        Returns
        -------
        dict
            ``kinetic``, ``potential``, ``interaction`` and ``loss`` rates.
        """
        if self._can_rate_on_device(A):
            return self._energy_rates_on_device(A)
        return self._energy_rates_on_host(A)

    def _can_rate_on_device(self, A: np.ndarray) -> bool:
        """Whether the rates can be reduced where the field is.

        The device path indexes every array with one shape, so a parameter
        carrying a value per simulation rules it out: only CUPY broadcasts one
        of those natively, and pyopencl does not broadcast at all. It also
        needs the transforms, which the caller may be asking before they are
        planned.

        Parameters
        ----------
        A : np.ndarray
            The field.

        Returns
        -------
        bool
            True if the device path applies.
        """
        if self.plans is None:
            return False
        scalars = (
            self._constant("_g"),
            self._constant("_Isat_conv"),
            self._constant("_alpha_half"),
        )
        return not any(getattr(value, "ndim", 0) > 0 for value in scalars)

    def _energy_rates_on_device(self, A: np.ndarray) -> dict[str, float]:
        """Return the rates, reducing where the field already is.

        The host version brings the field back and takes a numpy FFT of it,
        and the transform is most of the cost -- 6.0 ms of 9.3 at 512x512,
        against 0.41 for the transfer. The transform is planned already, so
        this uses it.

        Parameters
        ----------
        A : np.ndarray
            Normalized field, wherever the backend keeps it.

        Returns
        -------
        dict
            The rates, in radians per metre.
        """
        backend = self._backend
        zero = {"kinetic": 0.0, "potential": 0.0, "interaction": 0.0, "loss": 0.0}
        real_dtype = (
            np.float32 if np.dtype(self._field_dtype(A)).itemsize == 8 else np.float64
        )
        weight = backend.allocate_real_field(A.shape, real_dtype)
        weight = backend.kernels.square_mod(A, weight)
        total = backend.sum(weight)
        if total == 0:
            return zero

        # The transform is in place, so it gets a copy rather than the field.
        spectrum_field = backend.copy_field(A)
        spectrum_field = backend.fft(spectrum_field, self.plans)
        spectrum = backend.allocate_real_field(A.shape, real_dtype)
        spectrum = backend.kernels.square_mod(spectrum_field, spectrum)
        spectral_total = backend.sum(spectrum)
        dispersion = self._grid_on_device(
            "rate", self._dispersion_eigenvalues(), real_dtype, A.shape
        )
        kinetic = (
            backend.sum(spectrum * dispersion) / spectral_total
            if spectral_total > 0
            else 0.0
        )

        V_scaled = self._V_scaled
        if V_scaled is None:
            potential = absorption = 0.0
        else:
            host_V = self._as_host_array(V_scaled)
            real_name, imag_name = POTENTIAL_GRIDS
            real_V = self._grid_on_device(
                real_name, np.abs(np.real(host_V)), real_dtype, A.shape
            )
            imag_V = self._grid_on_device(
                imag_name, np.abs(np.imag(host_V)), real_dtype, A.shape
            )
            potential = backend.sum(weight * real_V) / total
            absorption = backend.sum(weight * imag_V) / total

        g = float(np.max(np.abs(self._as_host_array(self._constant("_g")))))
        Isat = float(self._as_host_array(self._constant("_Isat_conv")))
        saturated = weight * weight / (real_dtype(1.0) + weight / real_dtype(Isat))
        mean_intensity = backend.sum(saturated) / total
        interaction = g * mean_intensity

        return {
            "kinetic": abs(kinetic),
            "potential": abs(potential),
            "interaction": interaction,
            "loss": self._peak_loss_half() + absorption,
        }

    def _grid_on_device(self, name: str, values: Any, real_dtype: Any, shape: tuple):
        """Return a grid-shaped quantity on the device, matching the field.

        Stretched to the field's shape first. A coupled field carries a
        component axis that the dispersion eigenvalues and the potential do
        not, and pyopencl does not broadcast at all -- it raises rather than
        stretching, and only for the coupled solvers, so the plain ones would
        have shipped fine.

        Parameters
        ----------
        name : str
            What this is, to key the cache by.
        values : Any
            The quantity, on the host.
        real_dtype : Any
            Real width matching the field.
        shape : tuple
            Shape of the field it will multiply.

        Returns
        -------
        Any
            The quantity, where the field is.
        """
        key = (*self._propagator_rk4_cache_key(), np.dtype(real_dtype).str, shape, name)
        if key not in self._dispersion_cache:
            grid = np.ascontiguousarray(
                np.broadcast_to(np.asarray(values).astype(real_dtype), shape)
            )
            self._dispersion_cache[key] = (
                self._backend.from_numpy(grid)
                if self._backend.is_device_backend
                else grid
            )
        return self._dispersion_cache[key]

    def _forget_potential_grids(self) -> None:
        """Drop the potential's grids, which another potential would reuse."""
        stale = [key for key in self._dispersion_cache if key[-1] in POTENTIAL_GRIDS]
        for key in stale:
            del self._dispersion_cache[key]

    def _energy_rates_on_host(self, A: np.ndarray) -> dict[str, float]:
        """Return the rates by bringing the field back. The reference."""
        A_np = np.asarray(self._backend.to_numpy(A))
        weight = np.abs(A_np) ** 2
        total = float(np.sum(weight))
        zero = {"kinetic": 0.0, "potential": 0.0, "interaction": 0.0, "loss": 0.0}
        if total == 0:
            return zero

        spectrum = np.abs(np.fft.fftn(A_np, axes=self._last_axes)) ** 2
        spectral_total = float(np.sum(spectrum))
        dispersion = np.asarray(self._dispersion_eigenvalues())
        kinetic = (
            float(np.sum(spectrum * dispersion) / spectral_total)
            if spectral_total > 0
            else 0.0
        )

        # Only the real part of V rotates the phase; its imaginary part is
        # gain or loss and belongs with the losses.
        V_scaled = self._as_host_array(self._V_scaled)
        if V_scaled is None:
            potential = absorption = 0.0
        else:
            potential = float(np.sum(weight * np.abs(np.real(V_scaled))) / total)
            absorption = float(np.sum(weight * np.abs(np.imag(V_scaled))) / total)

        g = self._as_host_array(self._constant("_g"))
        Isat = self._as_host_array(self._constant("_Isat_conv"))
        # Batched runs carry one value per simulation; the step has to satisfy
        # the fastest of them, so reduce with max after weighting.
        mean_intensity = float(np.sum(weight * weight / (1 + weight / Isat)) / total)
        interaction = float(np.max(np.abs(g) * mean_intensity))

        return {
            "kinetic": abs(kinetic),
            "potential": abs(potential),
            "interaction": interaction,
            "loss": self._peak_loss_half() + absorption,
        }

    def _default_delta_z(
        self,
        A: np.ndarray,
        method: str = "split_step",
        z: float | None = None,
        rates: dict[str, float] | None = None,
    ) -> float:
        """Return the step to use when the caller has not chosen one.

        Aims at a fixed phase per step, ``DEFAULT_PHASE_PER_STEP``, against
        the same rate the limit for this method is built from: every term for
        RK4, which approximates the whole right-hand side, and the real-space
        terms alone for split-step, which applies the linear part exactly.

        Costs one FFT, against a propagation about to run thousands.

        Parameters
        ----------
        A : np.ndarray
            Normalized field (possibly on device).
        method : str
            Integration method ("split_step" or "RK4").
        z : float or None
            Distance about to be propagated, which bounds the step from
            above. Without it the medium length stands in.
        rates : dict or None
            ``_energy_rates(A)``, if the caller already has it. Without it
            the rates are reduced here.

        Returns
        -------
        float
            Step size.
        """
        if rates is None:
            rates = self._energy_rates(A)
        if method == "RK4":
            rate = sum(rates.values())
            phase = RK4_PHASE_PER_STEP
        else:
            rate = rates["potential"] + rates["interaction"]
            phase = DEFAULT_PHASE_PER_STEP
        # A rate of zero means nothing rotates the phase, so only the bound
        # below decides.
        dz = phase / rate if rate > 0 else np.inf

        span = abs(float(z)) if z is not None else float(self.L)
        if span > 0:
            dz = min(dz, span / DEFAULT_MIN_STEPS)
        if not np.isfinite(dz):
            dz = self.k * min(self.window) ** 2
        return dz

    def _max_local_linear_rate(self) -> float:
        """Return the fastest linear rate RK4 evaluates point by point.

        Here that is the potential, absorption included. The coupled solvers
        add what else they have.

        Returns
        -------
        float
            ``max |V k / 2|`` over the grid, or 0 without a potential.
        """
        V_scaled = self._as_host_array(self._V_scaled)
        if V_scaled is None:
            return 0.0
        return float(np.max(np.abs(V_scaled)))

    def _max_interaction_rate(self, A: np.ndarray) -> float:
        """Return the fastest interaction rate on the grid, at the field's peak.

        ``|g| |A|^2 / (1 + |A|^2 / Isat)``, maximized over the grid rather than
        weighted by the field as in ``_energy_rates``. The coupled solvers add
        the cross coupling.

        Parameters
        ----------
        A : np.ndarray
            Field, possibly on a device.

        Returns
        -------
        float
            The largest rate, in radians per unit of the propagation axis.
        """
        # On the host: pyopencl's array has no .max(), and g and Isat are device
        # arrays once a batched run has been sent.
        A_host = self._as_host_array(A)
        A_sq = A_host.real * A_host.real + A_host.imag * A_host.imag
        g = np.abs(self._as_host_array(self._constant("_g")))
        Isat = self._as_host_array(self._constant("_Isat_conv"))
        return float(np.max(g * A_sq / (1 + A_sq / Isat)))

    def _rk4_max_dz(
        self, A: np.ndarray | None = None, rates: dict[str, float] | None = None
    ) -> float:
        """Compute the maximum stable step size for explicit RK4.

        RK4's stability region reaches ~2.83 along the imaginary axis, so the
        step is bounded by ``2.83 / |lambda|``, and every term it evaluates
        explicitly adds to that eigenvalue: dispersion, potential, coupling,
        interaction and loss.

        The linear terms count by their maximum over the grid, not by the
        field-weighted rates of ``_energy_rates``: past the radius RK4
        amplifies whatever a mode holds, and round-off puts something in
        every mode. The interaction counts at its peak too, and against
        ``RK4_CUBIC_RADIUS``: the cubic term diverges well inside the linear
        radius. The absorption of a complex potential counts, unlike in
        ``_split_step_max_dz``: split-step applies it exactly.

        Parameters
        ----------
        A : np.ndarray, optional
            Field, for the interaction and the loss. Without it only the
            linear terms count.
        rates : dict or None
            ``_energy_rates(A)``, if the caller already has it. Without it
            the rates are reduced here.

        Returns
        -------
        float
            Largest stable step, or infinity if every rate vanishes.
        """
        rate = self._max_dispersion_rate() + self._max_local_linear_rate()
        if A is not None:
            if rates is None:
                rates = self._energy_rates(A)
            cubic = RK4_STABILITY_RADIUS / RK4_CUBIC_RADIUS
            rate += cubic * self._max_interaction_rate(A) + rates["loss"]
        if rate == 0:
            return np.inf
        return RK4_STABILITY_RADIUS / rate

    def _split_step_max_dz(
        self, A: np.ndarray, rates: dict[str, float] | None = None
    ) -> float:
        """Compute the maximum step size for split-step accuracy.

        The smallest of three steps: the one at which the potential and the
        interaction imprint pi of phase, ``_dispersion_max_dz`` and
        ``_loss_max_dz``. The kernels put the potential and the interaction in
        one exponent, ``arg_imag = (g |A|^2 sat + V) dz``, so their rates add.

        The potential counts although the exponential applies it exactly:
        what limits the step is the phase imprinted. Scaled by k/2 ~ 4e6, it
        dominates. The field-weighted kinetic rate is absent, because the
        linear part is applied exactly; dispersion enters through its
        stability bound alone.

        Parameters
        ----------
        A : np.ndarray
            Normalized field (possibly on device).
        rates : dict or None
            ``_energy_rates(A)``, if the caller already has it. Without it
            the rates are reduced here.

        Returns
        -------
        float
            Largest step keeping the real-space phase per step below pi, and
            the linear phase on the shortest resolved wave below a radian.
        """
        if rates is None:
            rates = self._energy_rates(A)
        phase_rate = rates["potential"] + rates["interaction"]
        limit = np.inf if phase_rate == 0 else np.pi / phase_rate
        return min(limit, self._dispersion_max_dz(), self._loss_max_dz())

    def _dispersion_max_dz(self) -> float:
        """Return the largest step split-step stays stable at.

        A stability bound, not an accuracy one. Split-step solves a linear
        problem exactly at any step, but on a finite-amplitude background the
        real-space step couples modes, and past about a radian of linear
        phase per step the shortest resolved waves grow out of round-off
        (Weideman and Herbst 1986; CHANGELOG, 4.1.0). The result reads as
        turbulence, not as a numerical fault.

        The rate is the grid's maximum, not the field-weighted "kinetic" rate
        of ``_energy_rates``: the nonlinearity seeds modes the field has no
        weight in (on one smooth Fourier mode, a 1% perturbation reaches 100%
        in twenty nonlinear lengths).

        Returns
        -------
        float
            Largest step, or infinity if the problem has no dispersion.
        """
        rate = self._max_dispersion_rate()
        if rate == 0:
            return np.inf
        return SPLIT_STEP_DISPERSION_PHASE_LIMIT / rate

    def _loss_max_dz(self) -> float:
        """Return the largest step the lossy real-space step is solved at.

        The scalar kernels solve the lossy step by an iteration
        (``_loss_factor`` in kernels/cpu.py) while its ``u``, ``alpha*dz`` in
        the medium's ``alpha``, is under their ``_LOSS_SOLVED_LIMIT``; past it
        they fall back to the frozen step, which is first order.
        ``LOSS_PER_STEP_LIMIT`` holds ``u`` to a quarter of that. The coupled
        kernels always freeze the step, so there the limit is an accuracy
        ceiling.

        Written against the largest component's ``alpha``
        (``_loss_constants``), not the loss energy rate: the iteration has to
        converge where the saturation is 1 and the rate is ``alpha`` itself.

        Returns
        -------
        float
            Largest step keeping the intensity lost per step inside the
            iteration's region, or infinity for a lossless medium.

        """
        peak = 2 * self._peak_loss_half()
        if peak == 0:
            return np.inf
        return LOSS_PER_STEP_LIMIT / (2 * peak)

    def _capped_delta_z(
        self,
        delta_z: float,
        A: np.ndarray,
        method: str,
        rates: dict[str, float] | None = None,
        *,
        warn: bool = True,
    ) -> float:
        """Return delta_z, lowered if it leaves the method's convergence region.

        Only ever lowers its magnitude, and keeps its sign, which is the
        direction of the run. The limit of either method can bind on the
        default step as well as on one the caller passed: both count the
        grid's shortest wave, which the default weights by the field or
        leaves out. Only lowering a step the caller passed warns.

        Parameters
        ----------
        delta_z : float
            Proposed step.
        A : np.ndarray
            Normalized field (possibly on device).
        method : str
            Integration method ("split_step" or "RK4").
        rates : dict or None
            ``_energy_rates(A)``, if the caller already has it. Without it
            the rates are reduced here, once.
        warn : bool, optional
            Warn when the step is lowered. Defaults to True; False for the
            solver's own default, which the caller did not choose.

        Returns
        -------
        float
            The step to actually take.
        """
        if rates is None:
            rates = self._energy_rates(A)
        if method == "RK4":
            max_dz = self._rk4_max_dz(A, rates)
            label = "RK4 stability"
            # Stability is not accuracy: with V scaled by k/2, a stable step can
            # still be a large phase per step. Split-step applies V exactly and has
            # no such limit.
            extra = (
                " This is a stability bound, not an accuracy one: a smaller "
                "delta_z may still be needed. split_step applies the "
                "potential exactly and is not limited this way."
                if rates["potential"] + rates["loss"] > rates["kinetic"]
                else ""
            )
        else:
            max_dz = self._split_step_max_dz(A, rates)
            label = "split-step accuracy"
            extra = ""
        # By magnitude: a backward step leaves the region of convergence at
        # the same length a forward one does.
        if abs(delta_z) > max_dz:
            capped = math.copysign(CAPPED_STEP_FRACTION * max_dz, delta_z)
            if warn:
                warnings.warn(
                    f"delta_z={delta_z:.2e} exceeds {label} limit "
                    f"({max_dz:.2e}). Reducing to {capped:.2e}.{extra}",
                    stacklevel=2,
                )
            return capped
        return delta_z
