"""How big a step the solver may take.

Ten methods that answer one question, and answer it in three stages: what the
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

import warnings
from typing import Any

import numpy as np
from scipy.constants import c, epsilon_0

# Explicit RK4 is stable for |lambda * dz| up to ~2*sqrt(2) along the
# imaginary axis. Every solver's step limit is derived from it.
RK4_STABILITY_RADIUS = 2.83

# Phase in radians the default step imprints per step, per method. The limits
# are ceilings (pi for split-step aliasing, 2.83 for RK4 stability); these are
# where a default sits under them.
#
# Both are the measured optimum rather than a margin under the ceiling, and
# the two methods want opposite steps. Split-step in complex64 is limited by
# round-off accumulating over steps, not by the splitting, so its error has a
# minimum: refining past ~0.4 rad costs time and accuracy together. RK4's
# truncation error is still falling steeply at 0.02, and at 0.1 it returns
# 5.6e-4 where 0.02 returns 2.3e-6. Measured with
# benchmarks/work_precision.py; see docs/optimization-log.md.
DEFAULT_PHASE_PER_STEP = 0.4
RK4_PHASE_PER_STEP = 0.02

# The same optimum, used to bound the adaptive controller. It is stable at
# 0.4-0.8 rad across a sixteenfold range of propagation distance, so in
# complex64 the controller is held to that band: below it there is only
# round-off, and above it the controller's own estimate goes blind -- one step
# and two halves then differ by round-off rather than by splitting error, which
# reads as "no error" and doubles the step until the answer is unrecognisable.
COMPLEX64_OPTIMUM_PHASE = 0.4
COMPLEX64_OPTIMUM_BAND = 2.0

# Fewest steps a default may take over the requested distance, so that a run
# is something a callback can sample and a plot can show rather than one jump.
DEFAULT_MIN_STEPS = 10


class StepSize:
    """The step-size half of a solver, mixed into NLSE.

    Every method is private and reached through the solver, so this class is
    never instantiated on its own and carries no state of its own.
    """

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
        on. Weighting by the field follows the physics instead.

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
        batched = (
            self._V_scaled,
            self._constant("_g"),
            self._constant("_Isat_conv"),
            self._constant("_alpha_half"),
        )
        return not any(getattr(value, "ndim", 0) > 0 for value in batched[1:])

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
            "rate", self._dispersion_operator(), real_dtype, A.shape
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
            real_V = self._grid_on_device("V_re", np.real(host_V), real_dtype, A.shape)
            imag_V = self._grid_on_device(
                "V_im", np.abs(np.imag(host_V)), real_dtype, A.shape
            )
            potential = backend.sum(weight * real_V) / total
            absorption = backend.sum(weight * imag_V) / total

        g = float(np.max(np.abs(self._as_host_array(self._constant("_g")))))
        Isat = float(self._as_host_array(self._constant("_Isat_conv")))
        saturated = weight * weight / (real_dtype(1.0) + weight / real_dtype(Isat))
        mean_intensity = backend.sum(saturated) / total
        interaction = g * mean_intensity

        alpha_half = float(
            np.max(np.abs(self._as_host_array(self._constant("_alpha_half"))))
        )
        return {
            "kinetic": abs(kinetic),
            "potential": abs(potential),
            "interaction": interaction,
            "loss": alpha_half + absorption,
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
        dispersion = np.asarray(self._dispersion_operator())
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
            potential = float(np.sum(weight * np.real(V_scaled)) / total)
            absorption = float(np.sum(weight * np.abs(np.imag(V_scaled))) / total)

        g = self._as_host_array(self._constant("_g"))
        Isat = self._as_host_array(self._constant("_Isat_conv"))
        # Batched runs carry one value per simulation; the step has to satisfy
        # the fastest of them, so reduce with max after weighting.
        mean_intensity = float(np.sum(weight * weight / (1 + weight / Isat)) / total)
        interaction = float(np.max(np.abs(g) * mean_intensity))

        alpha_half = self._as_host_array(self._constant("_alpha_half"))
        loss = float(np.max(np.abs(alpha_half))) + absorption
        return {
            "kinetic": abs(kinetic),
            "potential": abs(potential),
            "interaction": interaction,
            "loss": loss,
        }

    def _estimated_rates(self) -> dict[str, float]:
        """Return the phase rates without a field, for use before a run.

        Same quantities as ``_energy_rates``, from the grid rather than from
        the solution: the largest dispersion eigenvalue, the extremes of V,
        and the intensity the given power would have spread over the window.
        Coarser than the field-weighted version, and only used to give
        ``delta_z`` a value before anything has been propagated.

        Returns
        -------
        dict
            ``kinetic``, ``potential``, ``interaction`` and ``loss`` rates.
        """
        kinetic = float(np.max(np.abs(self._dispersion_operator())))

        V = self._as_host_array(self._V_scaled)
        if V is None and self.V is not None:
            V = self._as_host_array(self.V) * self._constant("_k_half")
        potential = float(np.max(np.abs(np.real(V)))) if V is not None else 0.0
        loss = float(np.max(np.abs(np.imag(V)))) if V is not None else 0.0

        area = float(np.prod([float(w) for w in self.window[:2]]))
        intensity = np.abs(
            2 * np.asarray(self.power, dtype=float) / (epsilon_0 * c * area)
        )
        Isat = np.abs(self._as_host_array(self._constant("_Isat_conv")))
        g = np.abs(self._as_host_array(self._constant("_g")))
        interaction = float(np.max(g * intensity / (1 + intensity / Isat)))

        return {
            "kinetic": kinetic,
            "potential": potential,
            "interaction": interaction,
            "loss": loss,
        }

    def _default_delta_z(
        self,
        A: np.ndarray | None = None,
        method: str = "split_step",
        z: float | None = None,
    ) -> float:
        """Return the step to use when the caller has not chosen one.

        Aims at a fixed phase per step, ``DEFAULT_PHASE_PER_STEP``, against
        the same rate the limit for this method is built from: every term for
        RK4, which approximates the whole right-hand side, and the real-space
        terms alone for split-step, which applies the linear part exactly.

        Costs one FFT, against a propagation about to run thousands.

        Parameters
        ----------
        A : np.ndarray or None
            Normalized field (possibly on device). Without one the rates are
            estimated from the grid instead.
        method : str
            Integration method ("split_step" or "RK4").
        z : float or None
            Distance about to be propagated, which bounds the step from
            above. Without it the medium length stands in.

        Returns
        -------
        float
            Step size.
        """
        rates = self._energy_rates(A) if A is not None else self._estimated_rates()
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

    def _rk4_max_dz(self, A: np.ndarray | None = None) -> float:
        """Compute the maximum stable step size for explicit RK4.

        RK4's stability region reaches ~2.83 along the imaginary axis, so the
        step is bounded by ``2.83 / |lambda|`` for the right-hand side it
        integrates. Every term it evaluates explicitly belongs in that
        eigenvalue — dispersion, potential, interaction and loss — and they
        add.

        Dispersion alone is almost never the largest: V is scaled by k/2, so
        omitting it puts RK4 outside its stability region whenever there is a
        potential.

        The absorption of a complex potential counts here, unlike in
        ``_split_step_max_dz``: split-step applies it exactly through the
        exponential, whereas RK4 approximates it, so it is as much part of the
        eigenvalue as the phase is.

        Parameters
        ----------
        A : np.ndarray, optional
            Field. Without it the field-weighted rates cannot be formed, so a
            conservative grid maximum is used instead.

        Returns
        -------
        float
            Largest stable step, or infinity if every rate vanishes.
        """
        if A is None:
            rate = float(np.max(np.abs(self._dispersion_operator())))
            V_scaled = self._as_host_array(self._V_scaled)
            if V_scaled is not None:
                rate += float(np.max(np.abs(V_scaled)))
        else:
            rate = sum(self._energy_rates(A).values())
        if rate == 0:
            return np.inf
        return RK4_STABILITY_RADIUS / rate

    def _split_step_max_dz(self, A: np.ndarray) -> float:
        """Compute the maximum step size for split-step accuracy.

        Keep the phase imprinted in one real-space step below pi. The kernels
        put the potential and the interaction in the same exponent —
        ``arg_imag = (g |A|^2 sat + V) dz`` — so both contribute, and their
        energies add.

        The potential counts even though the exponential applies it exactly:
        what limits the step is the phase imprinted, however exactly it was
        computed. Scaled by k/2 ~ 4e6, it dominates.

        The kinetic term is deliberately absent, and that is the real
        difference from ``_rk4_max_dz``. Split-step applies the linear part
        exactly in Fourier space, so a purely linear problem is solved exactly
        at any step size, and dispersion cannot limit accuracy on its own. RK4 approximates the whole right-hand side, so
        for it the kinetic term binds like everything else.

        Parameters
        ----------
        A : np.ndarray
            Normalized field (possibly on device).

        Returns
        -------
        float
            Largest step keeping the real-space phase per step below pi.
        """
        rates = self._energy_rates(A)
        phase_rate = rates["potential"] + rates["interaction"]
        if phase_rate == 0:
            return np.inf
        return np.pi / phase_rate

    def _capped_delta_z(self, delta_z: float, A: np.ndarray, method: str) -> float:
        """Return delta_z, lowered if it leaves the method's convergence region.

        Only ever lowers it. A step the solver chose itself is already well
        inside the limit, so this binds on a step the caller passed.

        Parameters
        ----------
        delta_z : float
            Proposed step.
        A : np.ndarray
            Normalized field (possibly on device).
        method : str
            Integration method ("split_step" or "RK4").

        Returns
        -------
        float
            The step to actually take.
        """
        if method == "RK4":
            rates = self._energy_rates(A)
            max_dz = self._rk4_max_dz(A)
            label = "RK4 stability"
            # Stability is not accuracy. At the edge of the region the scheme
            # merely stops diverging; a potential is scaled by k/2, so a step
            # that is stable can still turn a large phase per step into a
            # large error. Split-step applies V through the exponential and
            # has no such limit, which is usually the better answer.
            extra = (
                " This is a stability bound, not an accuracy one: a smaller "
                "delta_z may still be needed. split_step applies the "
                "potential exactly and is not limited this way."
                if rates["potential"] + rates["loss"] > rates["kinetic"]
                else ""
            )
        else:
            max_dz = self._split_step_max_dz(A)
            label = "split-step accuracy"
            extra = ""
        if delta_z > max_dz:
            warnings.warn(
                f"delta_z={delta_z:.2e} exceeds {label} limit "
                f"({max_dz:.2e}). Reducing to {0.9 * max_dz:.2e}.{extra}",
                stacklevel=2,
            )
            return 0.9 * max_dz
        return delta_z
