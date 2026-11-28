from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__
from .nlse import NLSE, show_if_possible


class CNLSE(NLSE):
    """A class to solve the coupled NLSE."""

    _gpu_param_attrs = (*NLSE._gpu_param_attrs, "n22", "n12")
    # The second component's potential; None until a run scales it.
    _V2_scaled: Any = None
    _run_state_attrs = (*NLSE._run_state_attrs, "_V2_scaled")
    # Both intra- and inter-component couplings switch off past the medium.
    _medium_attrs = (*NLSE._medium_attrs, "n12", "n22", "alpha2")
    # The second component's loss bounds the step as the first one's does.
    _loss_constants = (*NLSE._loss_constants, "_alpha2_half")
    # A merged Strang run brackets its Lie steps with the scalar
    # `_nonlinear_step`, which has neither the cross coupling nor the second
    # loss, so no coupled run merges.
    _lie_step_is_strang_body = False

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
            Non-local length in m, of a diffusive non-locality: the
            interaction reads the intensity smoothed by
            1 / (1 + nl_length^2 K^2) in Fourier space. 0 is local.
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

    @property
    def _norm_target(self) -> np.ndarray:
        """Normalize each component to its own power."""
        return np.array([self.power, self.power2])

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for coupled propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.k),
            float(self.k2),
        )

    def _dispersion_operator(self) -> np.ndarray:
        """Return the dispersion eigenvalues of the faster component."""
        return 0.5 * (self.Kxx**2 + self.Kyy**2) / min(self.k, self.k2)

    def _energy_rates(self, A: np.ndarray) -> dict[str, float]:
        """Return the phase rates for the coupled system.

        Each component carries its own potential and its own pair of
        couplings, so both are evaluated and the more restrictive one sets
        the step. The intensities are means weighted by each component's own
        density, matching how the base class forms its rates.
        """
        rates = super()._energy_rates(A)
        A_np = np.asarray(self._backend.to_numpy(A))
        w1 = np.abs(A_np[self._component(0)]) ** 2
        w2 = np.abs(A_np[self._component(1)]) ** 2
        t1, t2 = float(np.sum(w1)), float(np.sum(w2))
        if t1 == 0 and t2 == 0:
            return rates

        def mean(weight, total, field):
            return float(np.sum(weight * field) / total) if total > 0 else 0.0

        # Each component sees the potential scaled by its own k.
        V1_scaled = self._as_host_array(self._V_scaled)
        V2_scaled = self._as_host_array(self._V2_scaled)
        potential = 0.0
        for weight, total, V in ((w1, t1, V1_scaled), (w2, t2, V2_scaled)):
            if V is not None:
                potential = max(potential, abs(mean(weight, total, np.real(V))))
        rates["potential"] = potential

        rates["interaction"] = self._fastest_interaction_rate(
            mean(w1, t1, w1), mean(w2, t2, w2)
        )
        return rates

    def _fastest_interaction_rate(self, I1: Any, I2: Any) -> float:
        """Return the fastest interaction rate of either component.

        Component i turns at ``(|g_ii| I_i + |g_ij| I_j) sat``, with one
        saturation ``1 / (1 + I1/Isat1 + I2/Isat2)`` for both, as the coupled
        kernels apply it.

        Parameters
        ----------
        I1 : float or np.ndarray
            First component's ``|A|^2``.
        I2 : float or np.ndarray
            Second component's ``|A|^2``.

        Returns
        -------
        float
            The larger of the two rates, at its peak over the grid and batch.
        """
        g11, g12, g21, g22 = (
            np.abs(self._as_host_array(self._constant(name)))
            for name in ("_g11", "_g12", "_g21", "_g22")
        )
        Isat1 = self._as_host_array(self._constant("_Isat_conv"))
        Isat2 = self._as_host_array(self._constant("_Isat_conv2"))
        sat = 1 / (1 + I1 / Isat1 + I2 / Isat2)
        # Batched runs carry one value per simulation; reduce with max so the
        # step satisfies the fastest component of the fastest simulation.
        return float(np.max(np.maximum(g11 * I1 + g12 * I2, g22 * I2 + g21 * I1) * sat))

    def _max_local_linear_rate(self) -> float:
        """Return the fastest point-by-point linear rate of the coupled system.

        Each component sees the potential scaled by its own k, and the faster
        of the two counts. The Rabi coupling turns one component into the
        other at ``omega / 2``, on top of either potential.
        """
        rate = super()._max_local_linear_rate()
        V2_scaled = self._as_host_array(self._V2_scaled)
        if V2_scaled is not None:
            rate = max(rate, float(np.max(np.abs(V2_scaled))))
        if self.omega is not None:
            rate += float(np.max(np.abs(self._as_host_array(self.omega)))) / 2
        return rate

    def _max_interaction_rate(self, A: np.ndarray) -> float:
        """Return the fastest interaction rate of either component, at its peak."""
        A_host = self._as_host_array(A)
        A_sq = A_host.real * A_host.real + A_host.imag * A_host.imag
        return self._fastest_interaction_rate(
            A_sq[self._component(0)], A_sq[self._component(1)]
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for coupled RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k), float(self.k2))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw coupled dispersion operators for RK4."""
        prop1 = super()._compute_propagator_rk4()
        prop2 = -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2
        return np.array([prop1, prop2])

    def _nonlocal_response(self) -> np.ndarray:
        """Return the response for both components, laid out as the propagator."""
        response = super()._nonlocal_response()
        return np.array([response, response])

    def _step_constants(self) -> dict[str, Any]:
        """Add the second component's constants to NLSE's."""
        base = super()._step_constants()
        return {
            **base,
            # The intra-component coupling is the base class's, under the name
            # the coupled kernels take it by.
            "_g11": base["_g"],
            # One n12, two constants: each component's equation carries its own
            # k, so the index the other component writes turns into a phase
            # rate at that component's own wavenumber.
            "_g12": self.k / 2 * self.n12 * c * epsilon_0,
            "_g21": self.k2 / 2 * self.n12 * c * epsilon_0,
            "_g22": self.k2 / 2 * self.n22 * c * epsilon_0,
            "_alpha2_half": self.alpha2 / 2,
            "_Isat_conv2": 2 * self.I_sat2 / (epsilon_0 * c),
            "_k2_half": self.k2 / 2,
        }

    def _precompute_step_constants(
        self, V: np.ndarray | None, field_dtype: np.dtype = np.complex64
    ) -> None:
        """Pre-compute constants for coupled propagation steps."""
        super()._precompute_step_constants(V, field_dtype)
        # Through _scale_potential for the reason the first component's is: the
        # second component's kernel reads this at the field's width as well.
        self._V2_scaled = (
            None
            if V is None
            else self._scale_potential(
                V, self._k2_half, self._potential_dtype(V, field_dtype)
            )
        )
        # The kernels see one component at a time, so the constants they take
        # must broadcast against a component rather than against the pair.
        for name in (*self._step_constants(), "_V_scaled", "_V2_scaled"):
            setattr(self, name, self._per_component(getattr(self, name)))

    def _per_component(self, value: Any) -> Any:
        """Drop the component axis from a batched parameter.

        A caller batching a run shapes each parameter to broadcast against the
        field, which for a coupled solver includes the component axis: n2 of
        shape (count, 1, 1, 1) against a field of (count, 2, NY, NX). The
        kernels are handed one component at a time, of shape (count, NY, NX),
        so that axis is one too many by the time it reaches them.

        Parameters
        ----------
        value : Any
            A step constant, possibly batched, possibly None.

        Returns
        -------
        Any
            The same value, reduced to component rank if it was above it.
        """
        rank = len(self._last_axes)
        if getattr(value, "ndim", 0) <= rank + 1:
            return value
        axis = -(rank + 1)
        if value.shape[axis] != 1:
            raise ValueError(
                f"a batched parameter may not vary over the component axis; "
                f"got shape {tuple(value.shape)}. Use n2/n22, alpha/alpha2 "
                f"and Isat/Isat2 to give the components different values."
            )
        return value.reshape(value.shape[:axis] + value.shape[axis + 1 :])

    def _component(self, i: int) -> tuple:
        """Return the index selecting component ``i`` of a coupled array.

        The component axis sits just before the grid axes, so it is counted
        from the end: a 1D pair is (2, NX) and a 2D pair (2, NY, NX), and
        either may carry leading batch axes.

        Parameters
        ----------
        i : int
            Component, 0 or 1.

        Returns
        -------
        tuple
            Index tuple for ``A[...]``.
        """
        return (..., i) + (slice(None),) * len(self._last_axes)

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
        A1 = A[self._component(0)]
        A2 = A[self._component(1)]

        # Contiguous components, since the kernels need it: the numba kernels open with
        # A1.ravel(), which copies a strided batched component, so the step landed on
        # the copy and was dropped. CUPY and CL always copy, MLX arrays have no .copy,
        # the CPU copies a view that is not contiguous. _set_components writes back.
        if self._backend.is_device_backend:
            if hasattr(A1, "copy"):
                A1 = A1.copy()
                A2 = A2.copy()
        elif not A1.flags.c_contiguous:
            A1 = np.ascontiguousarray(A1)
            A2 = np.ascontiguousarray(A2)

        return A1, A2

    def _is_batched(self, A: np.ndarray, params: tuple = ()) -> bool:
        """Whether this run carries a batch, of fields or of parameters.

        An unbatched coupled field is one component axis plus the grid axes;
        anything above that is a batch of simulations. A parameter given per
        simulation is the other kind, and either rules out a kernel that
        expects scalars and one field.

        Parameters
        ----------
        A : np.ndarray
            The coupled field.
        params : tuple
            Parameters a kernel would take as scalars. Pass only those; a
            grid-shaped argument such as V is array-valued either way.

        Returns
        -------
        bool
            True if a batch is present.
        """
        if A.ndim > len(self._last_axes) + 1:
            return True
        return any(getattr(p, "ndim", 0) > 0 for p in params)

    # Backends whose coupled kernels take one field of exactly the coupled
    # rank. The generic path they fall back to is not batched-clean either,
    # so a batched coupled run is refused rather than quietly reshaped.
    _no_coupled_batch_backends = ("CL", "MLX")

    def _check_batch_support(self, E_in: np.ndarray) -> None:
        """Refuse a batched coupled run where the kernels cannot serve one."""
        if self._backend.name not in self._no_coupled_batch_backends:
            return
        params = (self.n2, self.n12, self.n22, self.alpha, self.alpha2)
        if self._is_batched(E_in, params):
            self._require_backend(
                lambda backend: backend.name not in self._no_coupled_batch_backends,
                "broadcasting a coupled solver over a batch",
            )

    def _can_fuse_components(self, A: np.ndarray, params: tuple) -> bool:
        """Whether the interleaved coupled kernels can serve this call.

        They read both components out of the one array with a flat index and
        take their parameters as scalars, so a batch -- of fields or of
        parameters -- has to go the generic way, as does a non-local run,
        which needs the intensity convolved between the two.

        CL and MLX refuse a batched coupled run outright, in
        ``_check_batch_support``, so for them this only ever restates what is
        already true. CUPY serves one, and needs the fallback.

        Parameters
        ----------
        A : np.ndarray
            The coupled field.
        params : tuple
            The parameters the kernel would take as scalars.

        Returns
        -------
        bool
            True if the fused path applies.
        """
        return self.nl_length == 0 and not self._is_batched(A, params)

    def _set_components(self, A: np.ndarray, A1: np.ndarray, A2: np.ndarray) -> None:
        """Write the two components back into ``A``.

        The counterpart to ``_take_components``, which returns copies on the
        device backends: without this the components are computed and dropped.

        Parameters
        ----------
        A : np.ndarray
            Coupled array to write into, modified in place.
        A1, A2 : np.ndarray
            The components.
        """
        A[self._component(0)] = A1
        A[self._component(1)] = A2

    def _allocate_rk4_buffers(self, A: np.ndarray, method: str) -> None:
        """Pre-allocate scratch buffers for coupled RK4 stepper.

        The intensity buffer takes the real width matching the field, for the
        reason given on the base method: the kernels pick their precision
        from the field and write this buffer at that width.
        """
        super()._allocate_rk4_buffers(A, method)
        if method == "RK4":
            real_dtype = self._real_dtype(self._field_dtype(A))
            self._rk4_A_sq_c = self._backend.allocate_real_field(A.shape, real_dtype)

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
        alpha_half = self._constant("_alpha_half")
        alpha2_half = self._constant("_alpha2_half")
        g11 = self._constant("_g11")
        g12 = self._constant("_g12")
        g21 = self._constant("_g21")
        g22 = self._constant("_g22")
        Isat1 = self._constant("_Isat_conv")
        Isat2 = self._constant("_Isat_conv2")
        V_scaled = self._V_scaled
        V2_scaled = self._V2_scaled
        if V_scaled is None and V is not None:
            k_half = self._constant("_k_half")
            k2_half = self._constant("_k2_half")
            V_scaled = V * k_half
            V2_scaled = V * k2_half

        # Fused fast path: zero component copies, and the transform writes
        # straight into k rather than the stage starting with a copy. The
        # fused kernel has no Rabi term, so a coupled run takes the generic one.
        if (
            self.omega is None
            and self._backend.has_fused_coupled_rk4_rhs
            and self._can_fuse_components(
                A_in, (alpha_half, alpha2_half, g11, g12, g21, g22, Isat1, Isat2)
            )
        ):
            prop, unnorm = self._fused_propagator(propagator)
            return kernels.rk4_rhs_coupled_fused(
                A_in,
                k,
                V_scaled,
                V2_scaled,
                prop,
                plans[0],
                alpha_half,
                alpha2_half,
                g11,
                g12,
                g21,
                g22,
                Isat1,
                Isat2,
                unnorm_ifft=unnorm,
            )

        k[:] = A_in
        k = self._apply_linear_step(k, propagator, plans)

        A1, A2 = self._take_components(A_in)
        k1, k2 = self._take_components(k)

        if self.nl_length > 0:
            A_sq = self._nonlocal_intensity(A_in, self._rk4_A_sq_c)
        else:
            A_sq = kernels.square_mod(A_in, self._rk4_A_sq_c)
        A_sq_1, A_sq_2 = self._take_components(A_sq)

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
                g21,
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
                g21,
                Isat2,
                Isat1,
            )

        if self.omega is not None:
            k1, k2 = self._add_rabi_slope(k1, k2, A1, A2)
        self._set_components(k, k1, k2)

        return k

    def _add_rabi_slope(
        self, k1: np.ndarray, k2: np.ndarray, A1: np.ndarray, A2: np.ndarray
    ) -> tuple:
        """Add the Rabi coupling to each component's right-hand side.

        The term ``i dA1/dz = (omega / 2) A2``, and its mirror for ``A2``,
        which ``rabi_coupling`` integrates exactly in the split step.

        Parameters
        ----------
        k1, k2 : np.ndarray
            Right-hand sides of the two components.
        A1, A2 : np.ndarray
            The components they were evaluated at.

        Returns
        -------
        tuple
            The two right-hand sides with the coupling added.
        """
        rate = -1j * self.omega / 2
        return k1 + rate * A2, k2 + rate * A1

    def _fusable_params(self) -> tuple:
        """Return the parameters the coupled whole-stage kernel takes as scalars."""
        return (
            self._constant("_alpha_half"),
            self._constant("_alpha2_half"),
            self._constant("_g11"),
            self._constant("_g12"),
            self._constant("_g21"),
            self._constant("_g22"),
            self._constant("_Isat_conv"),
            self._constant("_Isat_conv2"),
        )

    def _can_fuse_rk4_stage(self, A: np.ndarray) -> bool:
        """Whether the whole-stage kernels can serve this run.

        Not with a Rabi coupling, which those kernels do not apply.

        Parameters
        ----------
        A : np.ndarray
            The coupled field.

        Returns
        -------
        bool
            True if the fused path applies.
        """
        return self.omega is None and super()._can_fuse_rk4_stage(A)

    def _RK4_fused_stage(
        self,
        A_in: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        out: np.ndarray,
        A: np.ndarray,
        w: float,
        c: float,
        mode: int,
    ) -> np.ndarray:
        """Run one whole coupled RK4 stage, both components at once."""
        V_scaled = self._V_scaled
        V2_scaled = self._V2_scaled
        if V_scaled is None and V is not None:
            V_scaled = V * self._constant("_k_half")
            V2_scaled = V * self._constant("_k2_half")
        prop, unnorm = self._fused_propagator(propagator)
        return self._backend.kernels.rk4_stage_coupled_fused(
            A_in,
            self._rk4_k,
            V_scaled,
            V2_scaled,
            prop,
            plans[0],
            self._rk4_acc,
            out,
            A,
            self._constant("_alpha_half"),
            self._constant("_alpha2_half"),
            self._constant("_g11"),
            self._constant("_g12"),
            self._constant("_g21"),
            self._constant("_g22"),
            self._constant("_Isat_conv"),
            self._constant("_Isat_conv2"),
            w,
            c,
            mode,
            unnorm_ifft=unnorm,
        )

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
        g21,
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
                g21,
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
                g21,
                Isat_conv2,
                Isat_conv,
            )
        return A1, A2

    def _compute_A_sq_components(self, A, A_sq):
        """Compute the squared modulus, smoothed if non-local, and its components."""
        if self.nl_length > 0:
            A_sq = self._nonlocal_intensity(A, A_sq)
        else:
            A_sq = self._backend.kernels.square_mod(A, A_sq)
        A_sq_1, A_sq_2 = self._take_components(A_sq)
        return A_sq, A_sq_1, A_sq_2

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        delta_z: float,
        splitting: str = "lie",
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
            Propagators of the two fields, one per component.
        plans : list
            List of FFT plan objects: one plan used in both directions on
            the GPU backends, and the axes to transform over on CPU.
        delta_z : float
            Step to take. Must match the propagator, which was built
            from it.
        splitting : str, optional
            Single or double application of the
            nonlinear propagation step. Defaults to "lie".

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        # Half a Rabi rotation on each side keeps the Strang step symmetric,
        # hence second order, and Yoshida's sub-steps with it.
        rabi_halves = splitting == "strang" and self.omega is not None
        if rabi_halves:
            A = self._rabi_rotation(A, delta_z / 2)
        A = self._split_step_body(A, A_sq, V, propagator, plans, delta_z, splitting)
        if rabi_halves:
            A = self._rabi_rotation(A, delta_z / 2)
        return A

    def _rabi_rotation(self, A: np.ndarray, dz: float) -> np.ndarray:
        """Exchange the two components through the Rabi coupling over dz.

        Parameters
        ----------
        A : np.ndarray
            Coupled field, written back in place.
        dz : float
            Step to rotate over.

        Returns
        -------
        np.ndarray
            The field.
        """
        A1, A2 = self._take_components(A)
        A1, A2 = self._backend.kernels.rabi_coupling(A1, A2, dz, self.omega / 2)
        self._set_components(A, A1, A2)
        return A

    def _split_step_body(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        delta_z: float,
        splitting: str,
    ) -> np.ndarray:
        """Take the step ``split_step`` describes, a Lie step's Rabi rotation included.

        Parameters
        ----------
        A, A_sq, V, propagator, plans, delta_z, splitting
            As for ``split_step``.

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        # Use pre-computed constants with fallbacks for direct calls
        alpha_half = self._constant("_alpha_half")
        alpha2_half = self._constant("_alpha2_half")
        g11 = self._constant("_g11")
        g12 = self._constant("_g12")
        g21 = self._constant("_g21")
        g22 = self._constant("_g22")
        Isat_conv = self._constant("_Isat_conv")
        Isat_conv2 = self._constant("_Isat_conv2")
        V_scaled = self._V_scaled
        V2_scaled = self._V2_scaled
        if V is not None and V_scaled is None:
            # The solver's own scale: DDGPE's is not k / 2.
            V_scaled = V * np.float32(self._constant("_k_half"))
            V2_scaled = V * np.float32(self._constant("_k2_half"))
        nl_args = (
            V_scaled,
            V2_scaled,
            alpha_half,
            alpha2_half,
            g11,
            g12,
            g21,
            g22,
            Isat_conv,
            Isat_conv2,
        )

        kernels = self._backend.kernels

        # Fused fast path: zero component copies. It takes one field of
        # exactly the coupled rank and scalar parameters.
        if self._backend.has_fused_coupled_split_step and self._can_fuse_components(
            A,
            (alpha_half, alpha2_half, g11, g12, g21, g22, Isat_conv, Isat_conv2),
        ):
            dz = delta_z / 2 if splitting == "strang" else delta_z
            prop, unnorm = self._fused_propagator(propagator)
            omega_half = (
                self.omega / 2
                if (splitting == "lie" and self.omega is not None)
                else None
            )
            return kernels.split_step_coupled_fused(
                A,
                prop,
                V_scaled,
                V2_scaled,
                dz,
                alpha_half,
                alpha2_half,
                g11,
                g12,
                g21,
                g22,
                Isat_conv,
                Isat_conv2,
                splitting,
                plans[0],
                omega=omega_half,
                unnorm_ifft=unnorm,
            )

        # First half-step (strang only)
        if splitting == "strang":
            A1, A2 = self._take_components(A)
            A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)
            A1, A2 = self._apply_nl_prop_c(
                A1, A2, A_sq_1, A_sq_2, delta_z / 2, *nl_args
            )
            self._set_components(A, A1, A2)

        # Linear propagation in Fourier domain
        A = self._apply_linear_step(A, propagator, plans)

        # Second half-step (always)
        A1, A2 = self._take_components(A)
        A_sq, A_sq_1, A_sq_2 = self._compute_A_sq_components(A, A_sq)
        dz_step = delta_z / 2 if splitting == "strang" else delta_z
        A1, A2 = self._apply_nl_prop_c(A1, A2, A_sq_1, A_sq_2, dz_step, *nl_args)

        if splitting == "lie" and self.omega is not None:
            A1, A2 = self._backend.kernels.rabi_coupling(
                A1, A2, delta_z, self.omega / 2
            )

        self._set_components(A, A1, A2)
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
        fig.suptitle(self._plot_title(z))
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
        ]
        rho0 = np.abs(A_plot[0]) ** 2 * self._plot_density_scale
        phi0 = np.angle(A_plot[0])
        rho1 = np.abs(A_plot[1]) ** 2 * self._plot_density_scale
        phi1 = np.angle(A_plot[1])
        # plot amplitudes and phases
        # Row 0 is the smallest y: without origin="lower" it is drawn on top.
        im0 = ax[0, 0].imshow(rho0, extent=ext_real, origin="lower")
        first, second = self._plot_components
        ax[0, 0].set_title(rf"$|{first}|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=self._plot_density_label)
        im1 = ax[0, 1].imshow(
            phi0,
            extent=ext_real,
            origin="lower",
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[0, 1].set_title(rf"Phase $\mathrm{{arg}}({first})$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real, origin="lower")
        ax[1, 0].set_title(rf"$|{second}|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=self._plot_density_label)
        im3 = ax[1, 1].imshow(
            phi1,
            extent=ext_real,
            origin="lower",
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1, 1].set_title(rf"Phase $\mathrm{{arg}}({second})$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        show_if_possible()
