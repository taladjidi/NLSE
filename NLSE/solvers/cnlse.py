from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__, __CUPY_AVAILABLE__, __PYOPENCL_AVAILABLE__
from .nlse import NLSE, show_if_possible

if __CUPY_AVAILABLE__:
    pass

if __PYOPENCL_AVAILABLE__:
    pass


class CNLSE(NLSE):
    """A class to solve the coupled NLSE."""

    _gpu_param_attrs = (*NLSE._gpu_param_attrs, "n22", "n12")
    # The second component's potential; None until a run scales it.
    _V2_scaled: Any = None
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

        g11 = np.abs(self._as_host_array(self._constant("_g11")))
        g12 = np.abs(self._as_host_array(self._constant("_g12")))
        g22 = np.abs(self._as_host_array(self._constant("_g22")))
        Isat1 = self._as_host_array(self._constant("_Isat_conv"))
        Isat2 = self._as_host_array(self._constant("_Isat_conv2"))

        I1, I2 = mean(w1, t1, w1), mean(w2, t2, w2)
        sat = 1 / (1 + I1 / Isat1 + I2 / Isat2)
        # Batched runs carry one value per simulation; reduce with max so the
        # step satisfies the fastest component of the fastest simulation.
        rates["interaction"] = float(
            np.max(np.maximum((g11 * I1 + g12 * I2) * sat, (g22 * I2 + g12 * I1) * sat))
        )
        return rates

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for coupled RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k), float(self.k2))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw coupled dispersion operators for RK4."""
        prop1 = super()._compute_propagator_rk4()
        prop2 = -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k2
        return np.array([prop1, prop2])

    def _step_constants(self) -> dict[str, Any]:
        """Add the second component's constants to NLSE's."""
        base = super()._step_constants()
        return {
            **base,
            # The intra-component coupling is the base class's, under the name
            # the coupled kernels take it by.
            "_g11": base["_g"],
            "_g12": self.k / 2 * self.n12 * c * epsilon_0,
            "_g22": self.k2 / 2 * self.n22 * c * epsilon_0,
            "_alpha2_half": self.alpha2 / 2,
            "_Isat_conv2": 2 * self.I_sat2 / (epsilon_0 * c),
            "_k2_half": self.k2 / 2,
        }

    def _precompute_step_constants(
        self, V: np.ndarray | None, precision: str = "single"
    ) -> None:
        """Pre-compute constants for coupled propagation steps."""
        super()._precompute_step_constants(V, precision)
        self._V2_scaled = None if V is None else V * self._k2_half
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

        # Contiguous, whoever is asking, because the kernels need it and
        # neither of them says so.
        #
        # The device backends cannot index an offset array at all. The numba
        # kernels can, and quietly do the wrong thing: they open with
        # A1.ravel(), which returns a view of a contiguous array and a COPY of
        # one that is not, so the step is applied to the copy and thrown away
        # when the kernel returns the argument it was given. A coupled field
        # is (2, NX) unbatched and each component is contiguous, so this never
        # showed. Add a batch and it is (B, 2, NX), each component is a
        # strided (B, NX), and every batched coupled run on the CPU silently
        # dropped its whole real-space step -- losses and nonlinear phase --
        # and propagated the linear equation instead.
        #
        # The copy is only made where the view is not already contiguous, so
        # an unbatched run pays nothing and keeps writing through to A.
        # _set_components puts the components back either way.
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
            raise NotImplementedError(
                f"Broadcasting a coupled solver over a batch is not supported "
                f"with the {self._backend.name} backend. Use CPU or CUPY."
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
            real_dtype = (
                np.float32
                if np.dtype(self._field_dtype(A)).itemsize == 8
                else np.float64
            )
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
        # straight into k rather than the stage starting with a copy.
        if self._backend.has_fused_coupled_rk4_rhs and self._can_fuse_components(
            A_in, (alpha_half, alpha2_half, g11, g12, g22, Isat1, Isat2)
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
                g22,
                Isat1,
                Isat2,
                unnorm_ifft=unnorm,
            )

        k[:] = A_in
        k = self._apply_linear_step(k, propagator, plans)

        A1, A2 = self._take_components(A_in)
        k1, k2 = self._take_components(k)

        A_sq = kernels.square_mod(A_in, self._rk4_A_sq_c)
        A_sq_1, A_sq_2 = self._take_components(A_sq)

        if self.nl_length > 0:
            A_sq_1 = self._backend.convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._backend.convolution(
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

        self._set_components(k, k1, k2)

        return k

    def _fusable_params(self) -> tuple:
        """Return the parameters the coupled whole-stage kernel takes as scalars."""
        return (
            self._constant("_alpha_half"),
            self._constant("_alpha2_half"),
            self._constant("_g11"),
            self._constant("_g12"),
            self._constant("_g22"),
            self._constant("_Isat_conv"),
            self._constant("_Isat_conv2"),
        )

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
            A_sq_1 = self._backend.convolution(
                A_sq_1, self.nl_profile, mode="same", axes=self._last_axes
            )
            A_sq_2 = self._backend.convolution(
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
        delta_z: float,
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
            List of FFT plan objects: one plan used in both directions on
            the GPU backends, and the axes to transform over on CPU.
        delta_z : float
            Step to take. Must match the propagator, which was built
            from it.
        precision : str, optional
            Single or double application of the
            nonlinear propagation step. Defaults to "single".

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
        g22 = self._constant("_g22")
        Isat_conv = self._constant("_Isat_conv")
        Isat_conv2 = self._constant("_Isat_conv2")
        V_scaled = self._V_scaled
        V2_scaled = self._V2_scaled
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

        # Fused fast path: zero component copies. It takes one field of
        # exactly the coupled rank and scalar parameters.
        if self._backend.has_fused_coupled_split_step and self._can_fuse_components(
            A,
            (alpha_half, alpha2_half, g11, g12, g22, Isat_conv, Isat_conv2),
        ):
            dz = delta_z / 2 if precision == "double" else delta_z
            prop, unnorm = self._fused_propagator(propagator)
            omega_half = (
                self.omega / 2
                if (precision == "single" and self.omega is not None)
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
                g22,
                Isat_conv,
                Isat_conv2,
                precision,
                plans[0],
                omega=omega_half,
                unnorm_ifft=unnorm,
            )

        # First half-step (double precision only)
        if precision == "double":
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
        dz_step = delta_z / 2 if precision == "double" else delta_z
        A1, A2 = self._apply_nl_prop_c(A1, A2, A_sq_1, A_sq_2, dz_step, *nl_args)

        if precision == "single" and self.omega is not None:
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
        im0 = ax[0, 0].imshow(rho0, extent=ext_real)
        first, second = self._plot_components
        ax[0, 0].set_title(rf"$|{first}|^2$")
        ax[0, 0].set_xlabel("x (mm)")
        ax[0, 0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0, 0], shrink=0.6, label=self._plot_density_label)
        im1 = ax[0, 1].imshow(
            phi0,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[0, 1].set_title(rf"Phase $\mathrm{{arg}}({first})$")
        ax[0, 1].set_xlabel("x (mm)")
        ax[0, 1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[0, 1], shrink=0.6, label="Phase (rad)")
        im2 = ax[1, 0].imshow(rho1, extent=ext_real)
        ax[1, 0].set_title(rf"$|{second}|^2$")
        ax[1, 0].set_xlabel("x (mm)")
        ax[1, 0].set_ylabel("y (mm)")
        fig.colorbar(im2, ax=ax[1, 0], shrink=0.6, label=self._plot_density_label)
        im3 = ax[1, 1].imshow(
            phi1,
            extent=ext_real,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1, 1].set_title(rf"Phase $\mathrm{{arg}}({second})$")
        ax[1, 1].set_xlabel("x (mm)")
        ax[1, 1].set_ylabel("y (mm)")
        fig.colorbar(im3, ax=ax[1, 1], shrink=0.6, label="Phase (rad)")
        show_if_possible()
