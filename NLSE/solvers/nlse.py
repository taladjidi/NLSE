#!/usr/bin/env python3
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import contextlib
import math
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.constants import c, epsilon_0

from ..backends import Backend, fastest_backend_supporting, get_backend
from ..utils import (
    __BACKEND__,
    __CUPY_AVAILABLE__,
    __MLX_AVAILABLE__,
    __PYOPENCL_AVAILABLE__,
)
from .step_size import (
    DEFAULT_MIN_STEPS as DEFAULT_MIN_STEPS,
    DEFAULT_PHASE_PER_STEP as DEFAULT_PHASE_PER_STEP,
    RK4_STABILITY_RADIUS as RK4_STABILITY_RADIUS,
    StepSize,
)

# How many propagators to keep. Building one costs one exponential on the
# device, so the cache stays small and bounded: an adaptive step asks for a
# new one every few steps.
PROPAGATOR_CACHE_SIZE = 8

# Largest departure of the dispersion operator from one term per axis, as a
# fraction of its largest element, at which the propagator is still built as a
# product of one factor per axis. NLSE's and GPE's sit at round-off, 1.8e-16 to
# 3.0e-16 measured at 256 to 4096 points; a cross term sits orders above.
SEPARABLE_RTOL = 1e-12

# Yoshida's triple jump, which makes a fourth-order step out of three Strang
# ones: S4(h) = S2(w1 h) S2(w0 h) S2(w1 h), with 2*w1 + w0 = 1. The middle
# weight is negative, which is what buys the order and what makes the scheme
# unsuitable for a lossy medium -- a backwards step there amplifies.
_CBRT2 = 2.0 ** (1.0 / 3.0)
YOSHIDA_WEIGHTS = (
    1.0 / (2.0 - _CBRT2),
    -_CBRT2 / (2.0 - _CBRT2),
    1.0 / (2.0 - _CBRT2),
)
# The longest sub-step a Yoshida step takes, in units of the step itself. The
# phase limits are written per real-space application, so they bind on this.
YOSHIDA_SPAN = max(abs(w) for w in YOSHIDA_WEIGHTS)

# The compositions of the linear and nonlinear parts, named for the schemes
# (formerly "single" and "double", which read as float widths).
SPLITTINGS = ("lie", "strang", "yoshida")

# Fraction of a step below which a distance left over is round-off in
# distance / step rather than real propagation (dimensionless): z / dz for 237
# steps comes to 237.00000000000003, which must still be 237 whole steps.
STEP_ROUNDOFF = 1e-9

# Transformed axes the nonlinear step is folded into the linear step at: 2D,
# the plans cuFFT links store callbacks into here, checked against in
# tests/backends/test_fft_propagator_fusion.py, and the CPU's blocked step.
FUSED_TRANSFORM_AXES = 2


# Backends that render to a file rather than a window. Calling show() on one
# does nothing except warn, and a library should not warn a user for the
# backend matplotlib picked.
_NON_INTERACTIVE_BACKENDS = frozenset(
    {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf"}
)


def show_if_possible() -> bool:
    """Show the current figure, where the backend can show anything.

    Returns
    -------
    bool
        True if the figure was shown.
    """
    if plt.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        return False
    plt.show()
    return True


class NLSE(StepSize):
    """A class to solve NLSE."""

    # What plot_field draws and how it labels it. GPE and DDGPE integrate the
    # same equation for a density rather than an optical intensity, and their
    # axis is a time; stating the difference is enough, so they inherit the
    # plotting itself rather than restating fifty lines of matplotlib.
    _plot_density_scale = c * epsilon_0 / 2 * 1e-4  # |E|^2 in V^2/m^2 -> W/cm^2
    _plot_density_label = r"Intensity ($W/cm^2$)"
    _plot_axis_symbol = "z"
    _plot_axis_unit = "m"
    _plot_axis_format = ".2e"
    _plot_components = (r"\psi_1", r"\psi_2")

    # Step just taken, for callbacks that need it -- callbacks receive
    # (simu, A, z, i, *args) and have no other way to see it. Written by the
    # propagation loop, meaningless outside one, and not a way to set the step:
    # a callback changes it by *returning* a new one.
    _current_delta_z: float | None = None
    # Step the loop takes next. It is not the one just taken when that one was
    # cut to land on L or z, and it is the one a controller has to scale.
    _step_in_force: float | None = None

    # Set per run rather than derived from the parameters, so they are None
    # until a run sets them up. Declared here so reading one before that is a
    # plain attribute access rather than a getattr with a default at each site.
    _V_scaled: Any = None
    _propagator_fft: Any = None
    # (twin, dtype, delta_z, cache_key): a twin _update_propagator_fft made
    # from a propagator _build_propagator built, what for and under which
    # state. Factors only while the twin and the state are both still these.
    _propagator_fft_origin: tuple | None = None
    # Run state besides the step constants; _forget_run_state clears it.
    _run_state_attrs: tuple[str, ...] = (
        "_V_scaled",
        "_propagator_fft",
        "_propagator_fft_origin",
    )
    # Key, dispersion eigenvalues and their largest magnitude, for the last
    # configuration only: one float64 grid, 33.5 MB at 2048x2048.
    _eigenvalues: tuple[tuple, np.ndarray, float] | None = None
    # Declared for type checkers: whichever array type the backend allocates and
    # a list of its plan objects (assigned None in __init__).
    propagator: Any = None
    plans: Any = None

    # All three, so nothing has to know which backends are worth asking about.
    __CUPY_AVAILABLE__ = __CUPY_AVAILABLE__
    __PYOPENCL_AVAILABLE__ = __PYOPENCL_AVAILABLE__
    __MLX_AVAILABLE__ = __MLX_AVAILABLE__

    def __init__(
        self,
        alpha: float | np.floating,
        power: float | np.floating,
        window: float | tuple[float, float] | list[float],
        n2: float | np.floating,
        V: np.typing.NDArray[np.complexfloating | np.floating] | None,
        L: float | np.floating,
        NX: int = 1024,
        NY: int = 1024,
        Isat: float | np.floating = np.inf,
        nl_length: float | np.floating = 0,
        wvl: float | np.floating = 780e-9,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the simulation.

        Solves an equation : d/dz psi = -1/2k0(d2/dx2 + d2/dy2) psi +
          k0 dn psi + k0 n2 psi**2 psi

        Parameters
        ----------
        alpha : float
            alpha
        power : float
            Power in W
        window : float, list or tuple
            Computational window in the
            transverse plane in m.
            Can be different in x and y.
        n2 : float
            Non linear coeff in m^2/W
        V : np.ndarray
            Potential.
        L : float
            Length in m of the nonlinear medium
        NX : int, optional
            Number of points in the x direction.
            Defaults to 1024.
        NY : int, optional
            Number of points in the y direction.
            Defaults to 1024.
        Isat : float
            Saturation intensity in W/m^2
        nl_length : float
            Non-local length in m, of a diffusive non-locality: the
            interaction reads the intensity smoothed by
            1 / (1 + nl_length^2 K^2) in Fourier space. 0 is local.
        wvl : float
            Wavelength in m
        backend : str, optional
            Backend name ("CPU", "CUPY", "CL", or "auto").
            When "auto", automatically selects the fastest backend for your hardware.
            Defaults to __BACKEND__.
        """
        # list of physical parameters
        self._backend: Backend = get_backend(backend, grid_size=(NX, NY))
        self.n2 = n2
        self.V = V
        self.wl = wvl
        self.k = 2 * np.pi / self.wl
        self.L = L  # length of the non linear medium
        self.alpha = alpha
        self.power = power
        self.I_sat = Isat
        # number of grid points in X (even, best is power of 2 or low prime
        # factors)
        self.NX = NX
        self.NY = NY
        # self.window = window
        if isinstance(window, (list, tuple)):
            self.window = window
        else:
            self.window = [window, window]
        # transverse coordinate
        self.X, self.delta_X = np.linspace(
            -self.window[0] / 2,
            self.window[0] / 2,
            num=NX,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
        self.Y, self.delta_Y = np.linspace(
            -self.window[1] / 2,
            self.window[1] / 2,
            num=NY,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
        # define last axes for broadcasting operations
        self._last_axes = (-2, -1)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        self.Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)
        self.Kxx, self.Kyy = np.meshgrid(self.Kx, self.Ky)
        self._forget_device_state()
        self.nl_length = nl_length

        # Pre-compute normalization factors to avoid runtime upcasting
        self._norm_grid_factor = np.float32(self.delta_X * self.delta_Y)
        self._norm_constant = np.float32(c * epsilon_0 / 2)

    # The public surface: a solver is built, asked to propagate, and plotted.
    # split_step and split_step_RK4 are the single-step primitives out_field
    # drives; they are public because a caller may want one step at a time.
    @property
    def backend(self) -> str:
        """Return the backend used for the simulation."""
        return self._backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the backend, dropping the arrays the previous one built."""
        self._backend = get_backend(value, grid_size=(self.NX, self.NY))
        self._forget_device_state()

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
        delta_z: float | None = None,
        plot: bool = False,
        splitting: str = "lie",
        method: str = "split_step",
        verbose: bool = True,
        normalize: bool = True,
        callback: list[Callable] | Callable | None = None,
        callback_args: list[tuple] | tuple = (),
    ) -> np.ndarray:
        """Propagate the field at a distance z.

        This function propagates the field E_in over a distance z by
        calling the split step function in a loop.

        Parameters
        ----------
        E_in : np.ndarray
            Normalized input field (between 0 and 1).
        z : float
            Propagation distance in m, not negative.
        delta_z : float, optional
            Step to propagate with. Defaults to None, meaning the solver
            derives one from the field: a step that imprints a fixed phase
            per step, against the same energy rates the stability and
            accuracy limits are built from. A step given here is used as
            given, capped only where it would leave the method's region of
            convergence. A negative step propagates backwards over z.
        plot : bool, optional
            Plots the results. Defaults to False.
        splitting : str, optional
            How the linear and nonlinear parts are composed, *not* the
            floating-point width. Defaults to "lie".

            - "lie": one nonlinear application per step, O(dz), one
              transform pair.
            - "strang": the nonlinear step split around the linear one,
              O(dz^2) globally, one transform pair. The scalar solvers
              also merge the touching half steps of consecutive steps.
            - "yoshida": three Strang sub-steps composed to O(dz^4), three
              transform pairs. Worth it only in complex128 -- see the
              warning it raises otherwise.

            The floating-point width comes from ``E_in``: pass a complex128
            field for float64 arithmetic. A backend without it (MLX, an
            OpenCL device without fp64) hands the run, with a warning, to the
            fastest one that has it. The propagator is built to match,
            because the kernels select their precision from the field and
            then read the propagator with it.
        method : str, optional
            Integration method: "split_step" or "RK4".
            Defaults to "split_step".
        verbose : bool, optional
            Prints progress and time.
            Defaults to True.
        normalize : bool, optional
            Normalize the field to the total power.
            Defaults to True.
        callback : callable, optional
            Callback function.
            Defaults to None.
        callback_args : tuple or list of tuples, optional
            Additional arguments for the callback, or one tuple per callback
            when ``callback`` is a list. Defaults to no arguments for every
            callback.

        Returns
        -------
        np.ndarray
            Propagated field in proper units V/m.

        Raises
        ------
        TypeError
            If delta_z is complex: imaginary time is not supported.
        ValueError
            If delta_z is zero or not finite, or if z is negative or not
            finite.
        """
        self._check_distance_and_step(z, delta_z)
        # Backward compat: splitting="RK4" maps to method="RK4"
        if splitting == "RK4":
            method = "RK4"
            splitting = "lie"
        if splitting not in SPLITTINGS:
            raise ValueError(
                f"splitting must be one of {SPLITTINGS}, got {splitting!r}"
            )

        grid_shape = self.XX.shape[self._last_axes[0] :]
        if E_in.shape[self._last_axes[0] :] != grid_shape:
            raise ValueError(
                f"E_in has shape {E_in.shape}, which does not end in the "
                f"grid's {grid_shape}"
            )
        if E_in.dtype not in (np.complex64, np.complex128):
            raise TypeError(f"E_in must be complex64 or complex128, not {E_in.dtype}")
        self._check_batch_support(E_in)
        field_dtype = self._field_dtype(E_in)
        self._check_precision_support(field_dtype)
        self._warn_about_splitting(splitting, field_dtype)
        # Rebuilt below once delta_z is settled; drop the previous run's so
        # the transfer does not carry it.
        self.propagator = None
        with self._arrays_on_device(field_dtype):
            V = self.V
            A, A_sq = self._prepare_output_array(E_in, normalize)
            self.plans = self._build_fft_plan(A)
            self._allocate_rk4_buffers(A, method)
            self._precompute_step_constants(V, field_dtype)

            # Settle the step before building anything that depends on it.
            if delta_z is None:
                delta_z = self._default_delta_z(A, method, z)
            delta_z = self._capped_delta_z(delta_z, A, method)
            if splitting == "yoshida":
                # The phase limits are written per real-space application, and
                # Yoshida's longest sub-step is 1.7 times the step itself.
                yoshida_max = self._split_step_max_dz(A) / YOSHIDA_SPAN
                delta_z = math.copysign(min(abs(delta_z), yoshida_max), delta_z)

            # The propagator depends on delta_z, k, the grid and the precision,
            # any of which may have changed since the last run. _build_propagator
            # is cache-backed, so an unchanged configuration is not recomputed.
            if method == "RK4":
                self.propagator = self._build_propagator_rk4(field_dtype)
            else:
                # Yoshida builds its own three alongside this one, which is
                # still what a partial step and the solver's own reporting
                # read.
                self.propagator = self._build_propagator(field_dtype, delta_z)
            self._send_propagator_to_gpu()
            if verbose:
                pbar = tqdm.tqdm(
                    total=100,
                    position=4,
                    desc="Propagation",
                    leave=False,
                    unit="%",
                    unit_scale=True,
                )

            with self._backend.timed() as timing:
                A = self._run_propagation(
                    A,
                    A_sq,
                    V,
                    z,
                    delta_z,
                    splitting,
                    method,
                    callback,
                    callback_args,
                    verbose,
                    pbar if verbose else None,
                )
                # Before the clock stops: a queue submitted is not work done.
                self._backend.synchronize(A)

            if verbose:
                pbar.n = 100
                pbar.refresh()
                pbar.close()
                print(f"\n{timing}\n")
            # _run_propagation restores the nonlinear coupling itself.
            return_np_array = isinstance(E_in, np.ndarray)
            if self._backend.is_device_backend and return_np_array:
                A = self._backend.to_numpy(A)

        if plot:
            self.plot_field(A, z)
        return A

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
            Field to propagate.
        A_sq : np.ndarray
            Field modulus squared.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.
        delta_z : float
            Step to take. Must match the propagator, which was built from it.
        splitting : str, optional
            Order of the split step: "lie" applies the nonlinear step
            once, "strang" splits it around the linear step. Not the
            floating-point width, which follows the field. Defaults to
            "lie".

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        kernels = self._backend.kernels
        # Use pre-computed constants (set in out_field), with fallbacks
        # for direct split_step calls outside out_field.
        alpha_half = self._constant("_alpha_half")
        g = self._constant("_g")
        Isat_conv = self._constant("_Isat_conv")
        k_half = self._constant("_k_half")
        V_scaled = self._V_scaled
        if V_scaled is None and V is not None:
            V_scaled = V * k_half

        # Fused fast path (CL, MLX). Not eligible with a non-local response,
        # which smooths |A|^2 between the linear and the nonlinear step.
        if self._backend.has_fused_split_step and self.nl_length == 0:
            dz = delta_z / 2 if splitting == "strang" else delta_z
            prop, unnorm = self._fused_propagator(propagator)
            return kernels.split_step_fused(
                A,
                prop,
                V_scaled,
                dz,
                alpha_half,
                g,
                Isat_conv,
                splitting,
                plans[0],
                unnorm_ifft=unnorm,
            )

        # First half-step (only for splitting == "strang")
        if splitting == "strang":
            A = self._nonlinear_step(A, A_sq, V_scaled, delta_z / 2)

        # Linear propagation, then the second half-step (always executed),
        # the whole step for Lie splitting
        dz_step = delta_z / 2 if splitting == "strang" else delta_z
        return self._linear_then_nonlinear(
            A, A_sq, V_scaled, propagator, plans, dz_step
        )

    def _nonlinear_step(self, A, A_sq, V_scaled, dz):
        """Apply the real-space part of the step: interaction, potential, loss.

        Both halves of a Strang step do exactly this and differed only in the
        distance, so they are one method. It is also what brackets a merged
        run -- see ``_open_strang_bracket`` and ``_close_strang_bracket``.

        Parameters
        ----------
        A : np.ndarray
            Field, modified in place on the array backends.
        A_sq : np.ndarray
            Scratch for the intensity. Only used by the non-local path, which
            has to smooth it before the kernel reads it.
        V_scaled : np.ndarray or None
            Pre-scaled potential, or None.
        dz : float
            Distance this application advances by. May be negative, which is
            how the bracket undoes its last half step.

        Returns
        -------
        np.ndarray
            The field.
        """
        kernels = self._backend.kernels
        alpha_half = self._constant("_alpha_half")
        g = self._constant("_g")
        Isat_conv = self._constant("_Isat_conv")

        if self.nl_length > 0:
            # The smoothing needs the intensity as an array of its own, so
            # the kernel that computes it inline cannot be used.
            A_sq = self._nonlocal_intensity(A, A_sq)
            if V_scaled is None:
                return kernels.nl_prop_without_V(A, A_sq, dz, alpha_half, g, Isat_conv)
            return kernels.nl_prop(A, A_sq, dz, alpha_half, V_scaled, g, Isat_conv)
        if V_scaled is None:
            return kernels.square_mod_nl_prop(A, dz, alpha_half, g, Isat_conv)
        return kernels.square_mod_nl_prop_v(A, V_scaled, dz, alpha_half, g, Isat_conv)

    def _fuses_nonlinear_into_transform(self) -> bool:
        """Whether the nonlinear step may run inside the linear step's last pass.

        Only where the backend has a step for it (a store callback on CUPY,
        blocked passes on CPU), the response is local (a non-local one smooths
        |A|^2 between the transform and the step), and the transform is 2D.

        Returns
        -------
        bool
            True to try ``kernels.linear_nonlinear_step``.
        """
        return (
            self._backend.has_linear_nonlinear_step
            and self.nl_length == 0
            and len(self._last_axes) == FUSED_TRANSFORM_AXES
        )

    def _linear_then_nonlinear(self, A, A_sq, V_scaled, propagator, plans, dz):
        """Apply the linear step, then ``_nonlinear_step`` over ``dz``.

        In one pass over the field fewer where the backend can run the
        nonlinear step as the inverse transform stores its output; as the two
        steps one after the other where it cannot.

        Parameters
        ----------
        A : np.ndarray
            Field, modified in place on the array backends.
        A_sq : np.ndarray
            Scratch for the intensity, as in ``_nonlinear_step``.
        V_scaled : np.ndarray or None
            Pre-scaled potential, or None.
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.
        dz : float
            Distance the nonlinear step advances by.

        Returns
        -------
        np.ndarray
            The field.
        """
        if self._linear_nonlinear_step(A, V_scaled, propagator, plans, dz):
            return A
        A = self._apply_linear_step(A, propagator, plans)
        return self._nonlinear_step(A, A_sq, V_scaled, dz)

    def _linear_nonlinear_step(self, A, V_scaled, propagator, plans, dz) -> bool:
        """Take both steps in the backend's own step for them, where it takes them.

        Parameters
        ----------
        A : np.ndarray
            Field, stepped in place if the step is taken.
        V_scaled : np.ndarray or None
            Pre-scaled potential, or None.
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.
        dz : float
            Distance the nonlinear step advances by.

        Returns
        -------
        bool
            True if taken; False, with A untouched, where the backend has no
            such step or declines this call.
        """
        if not self._fuses_nonlinear_into_transform():
            return False
        prop, unnorm = self._fused_propagator(propagator)
        # The callback's inverse transform does not normalize, so it needs
        # the propagator that carries 1/N. The factors are built when the
        # kernel asks: a step it declines costs no pass over the dispersion.
        return unnorm and self._backend.kernels.linear_nonlinear_step(
            A,
            prop,
            plans[0],
            V_scaled,
            dz,
            self._constant("_alpha_half"),
            self._constant("_g"),
            self._constant("_Isat_conv"),
            factors=partial(self._propagator_fft_factors, prop, A.size // A.shape[-1]),
        )

    def _require_backend(self, supports: Callable, what: str) -> None:
        """Move to the fastest backend that can serve this run, and say so.

        A backend is a performance choice, not a physics one, so asking for
        one that cannot run the problem is answered by running the problem
        rather than by refusing it. Only raises when nothing installed can.

        Parameters
        ----------
        supports : Callable
            Called with a backend; True if it can serve the run.
        what : str
            What is being asked for, for the message.
        """
        if supports(self._backend):
            return
        asked = self._backend.name
        replacement = fastest_backend_supporting(supports, grid_size=(self.NX, self.NY))
        if replacement is None:
            raise NotImplementedError(
                f"No installed backend supports {what}. The {asked} backend "
                f"does not, and neither does anything else available here."
            )
        warnings.warn(
            f"The {asked} backend does not support {what}, so this solver "
            f"will use {replacement.name} instead — the fastest one available "
            f"that does. Pass backend='{replacement.name}' to choose it "
            f"yourself and silence this.",
            stacklevel=3,
        )
        self._backend = replacement
        self._forget_device_state()

    def _warn_about_splitting(self, splitting: str, field_dtype: np.dtype) -> None:
        """Say so when the splitting and the float width do not go together.

        The two are chosen separately and the useful combinations are not the
        obvious ones. In complex64 the answer is limited by round-off
        accumulating over steps rather than by the splitting, so a
        fourth-order scheme pays three transform pairs for accuracy the
        arithmetic cannot hold. In complex128 that floor is gone and the
        splitting is what is left, so Strang leaves a large factor on the
        table. Both are measured in docs/optimization-log.md.

        Parameters
        ----------
        splitting : str
            The composition asked for.
        field_dtype : np.dtype
            Complex dtype the run will use.
        """
        single = np.dtype(field_dtype) == np.dtype(np.complex64)
        if splitting == "yoshida":
            if single:
                warnings.warn(
                    "splitting='yoshida' on a complex64 field: it costs three "
                    "transform pairs per step for an order the arithmetic "
                    "cannot use, because round-off over steps sets the error "
                    "long before the splitting does. Pass a complex128 field, "
                    "or use splitting='strang'.",
                    stacklevel=3,
                )
            if self._has_loss():
                warnings.warn(
                    "splitting='yoshida' with loss: its middle sub-step runs "
                    "backwards, which amplifies instead of decaying. The "
                    "composition is still fourth order, but the intermediate "
                    "field is not physical and may overflow.",
                    stacklevel=3,
                )
        elif not single:
            warnings.warn(
                f"splitting={splitting!r} on a complex128 field: with the "
                "round-off floor out of the way the splitting error is what "
                "is left, and splitting='yoshida' reaches a given accuracy "
                "for much less work.",
                stacklevel=3,
            )

    def _has_loss(self) -> bool:
        """Whether anything in the run removes norm from the field."""
        if np.any(np.asarray(self.alpha) != 0):
            return True
        V = self.V
        return V is not None and np.iscomplexobj(V) and np.any(np.imag(V) != 0)

    def _yoshida_propagators(self, dtype: np.dtype, delta_z: float) -> list:
        """Build and place the propagator each Yoshida sub-step needs.

        Three sub-steps of different length need three propagators, and they
        are built once per step size rather than per step. Each is kept with
        its pre-normalized twin because ``_fused_propagator`` reads that twin
        off the solver rather than off its argument, so swapping one without
        the other would hand a kernel a propagator built for another distance.
        The twin's origin goes with it, or the sub-step is offered no factors.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field.
        delta_z : float
            The whole Yoshida step.

        Returns
        -------
        list
            ``_linear_state()`` for each sub-step, in order.
        """
        saved = self._linear_state()
        states = []
        try:
            for weight in YOSHIDA_WEIGHTS:
                self.propagator = self._build_propagator(dtype, weight * delta_z)
                self._send_propagator_to_gpu()
                states.append(self._linear_state())
        finally:
            self._restore_linear_state(saved)
        return states

    def split_step_yoshida(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagators: list,
        plans: list,
        delta_z: float,
    ) -> np.ndarray:
        """Take one fourth-order Yoshida step.

        Three Strang sub-steps with weights summing to one, the middle of them
        backwards. That is what raises the order from two to four, and what
        rules the scheme out for a lossy medium, where a backwards step
        amplifies instead of decaying.

        Parameters
        ----------
        A : np.ndarray
            Field, modified in place on the array backends.
        A_sq : np.ndarray
            Scratch for the intensity, used by the non-local path.
        V : np.ndarray or None
            Potential.
        propagators : list
            ``_linear_state()`` per sub-step, from ``_yoshida_propagators``.
        plans : list
            FFT plans.
        delta_z : float
            The whole step. Each sub-step takes its own fraction of it.

        Returns
        -------
        np.ndarray
            The field.
        """
        saved = self._linear_state()
        try:
            for state, weight in zip(propagators, YOSHIDA_WEIGHTS):
                self._restore_linear_state(state)
                A = self.split_step(
                    A, A_sq, V, self.propagator, plans, weight * delta_z, "strang"
                )
        finally:
            self._restore_linear_state(saved)
        return A

    def split_step_RK4(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        delta_z: float,
    ) -> np.ndarray:
        """Propagate one step using classic 4th-order Runge-Kutta.

        Uses pre-allocated scratch buffers (k, A_tmp, acc) to avoid
        per-step memory allocations.

        Parameters
        ----------
        A : np.ndarray
            Field to propagate.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.
        delta_z : float
            Step to take. Must match the propagator, which was built from it.

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        kernels = self._backend.kernels

        # Whole-stage fused path: the slope never reaches memory.
        if self._backend.has_fused_rk4_stage and self._can_fuse_rk4_stage(A):
            return self._split_step_RK4_fused(A, V, propagator, plans, delta_z)

        # Whole-step fused fast path (MLX), single component only: a field with
        # nothing but its spatial axes, so no component axis and no batch. Testing
        # A.ndim == 2 sent CNLSE_1d's (2, NX) field down this path, which read its
        # component axis as a spatial one.
        if (
            self._backend.has_fused_rk4_step
            and self.nl_length == 0
            and A.ndim == len(self._last_axes)
        ):
            alpha_half = self._constant("_alpha_half")
            g = self._constant("_g")
            k_half = self._constant("_k_half")
            Isat_conv = self._constant("_Isat_conv")
            V_scaled = self._V_scaled
            if V_scaled is None and V is not None:
                V_scaled = V * k_half
            return kernels.split_step_rk4_fused(
                A,
                propagator,
                V_scaled,
                delta_z,
                alpha_half,
                g,
                Isat_conv,
                plans[0],
            )

        if not hasattr(self, "_rk4_k"):
            self._allocate_rk4_buffers(A, "RK4")
        k = self._rk4_k
        A_tmp = self._rk4_A_tmp
        acc = self._rk4_acc
        h = delta_z

        has_fused = self._backend.has_fused_rk4_stage_update

        # Stage 1: k1 = f(A)
        k = self._RK4_rhs(A, k, V, propagator, plans)
        if has_fused:
            acc, A_tmp = kernels.rk4_set_and_axpy(acc, A_tmp, A, k, h / 2)
        else:
            acc = kernels.rk4_axpy(acc, k, 0.0, k)  # acc = k (copy via axpy)
            A_tmp = kernels.rk4_axpy(A_tmp, A, h / 2, k)  # A_tmp = A + h/2*k1

        # Stage 2: k2 = f(A + h/2*k1)
        k = self._RK4_rhs(A_tmp, k, V, propagator, plans)
        if has_fused:
            acc, A_tmp = kernels.rk4_acc_and_axpy(acc, A_tmp, A, k, 2.0, h / 2)
        else:
            acc = kernels.rk4_accumulate(acc, 2.0, k)  # acc = k1 + 2*k2
            A_tmp = kernels.rk4_axpy(A_tmp, A, h / 2, k)  # A_tmp = A + h/2*k2

        # Stage 3: k3 = f(A + h/2*k2)
        k = self._RK4_rhs(A_tmp, k, V, propagator, plans)
        if has_fused:
            acc, A_tmp = kernels.rk4_acc_and_axpy(acc, A_tmp, A, k, 2.0, h)
        else:
            acc = kernels.rk4_accumulate(acc, 2.0, k)  # acc = k1 + 2*k2 + 2*k3
            A_tmp = kernels.rk4_axpy(A_tmp, A, h, k)  # A_tmp = A + h*k3

        # Stage 4: k4 = f(A + h*k3)
        k = self._RK4_rhs(A_tmp, k, V, propagator, plans)
        if self._backend.has_fused_rk4_final_update:
            # A += h/6 * (acc + k4), without writing acc only to read it back
            A = kernels.rk4_final_update(A, acc, k, h / 6)
        else:
            acc = kernels.rk4_accumulate(acc, 1.0, k)  # acc = k1+2*k2+2*k3+k4
            # Final update: A += h/6 * acc
            A = kernels.rk4_accumulate(A, h / 6, acc)

        self._rk4_k = k
        self._rk4_A_tmp = A_tmp
        self._rk4_acc = acc
        return A

    def _is_batched(self, A: np.ndarray, params: tuple = ()) -> bool:
        """Whether this run carries a batch, of fields or of parameters.

        An unbatched field is the grid axes and nothing else; anything above
        that is a batch of simulations. A parameter given per simulation is
        the other kind, and either rules out a kernel taking scalars and one
        field.

        Parameters
        ----------
        A : np.ndarray
            The field.
        params : tuple
            Parameters a kernel would take as scalars.

        Returns
        -------
        bool
            True if a batch is present.
        """
        if A.ndim > len(self._last_axes):
            return True
        return any(getattr(p, "ndim", 0) > 0 for p in params)

    def _can_fuse_rk4_stage(self, A: np.ndarray) -> bool:
        """Whether the whole-stage kernels can serve this run.

        They take their parameters as scalars and index one field with a flat
        id, so a batch goes the generic way; and they compute the intensity
        from the field in registers, which a non-local run cannot do because
        it has to convolve it first.

        Parameters
        ----------
        A : np.ndarray
            The field.

        Returns
        -------
        bool
            True if the fused path applies.
        """
        return self.nl_length == 0 and not self._is_batched(A, self._fusable_params())

    def _fusable_params(self) -> tuple:
        """Return the parameters the whole-stage kernel takes as scalars."""
        return (
            self._constant("_alpha_half"),
            self._constant("_g"),
            self._constant("_Isat_conv"),
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
        """Run one whole RK4 stage through the backend's fused kernel.

        The one thing the coupled solver has to say differently, so that
        ``_split_step_RK4_fused`` states the scheme once.

        Parameters
        ----------
        A_in : np.ndarray
            Field this stage evaluates the slope at.
        V : np.ndarray or None
            Potential field.
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            FFT plan objects from the backend.
        out : np.ndarray
            Where this stage's result goes.
        A : np.ndarray
            The field the step started from.
        w, c : float
            Weights of this stage's slope in the accumulator and the output.
        mode : int
            0 to set the accumulator, 1 to add to it, 2 to spend it.

        Returns
        -------
        np.ndarray
            The array this stage wrote.
        """
        V_scaled = self._V_scaled
        if V_scaled is None and V is not None:
            V_scaled = V * self._constant("_k_half")
        prop, unnorm = self._fused_propagator(propagator)
        return self._backend.kernels.rk4_stage_fused(
            A_in,
            self._rk4_k,
            V_scaled,
            prop,
            plans[0],
            self._rk4_acc,
            out,
            A,
            self._constant("_alpha_half"),
            self._constant("_g"),
            self._constant("_Isat_conv"),
            w,
            c,
            mode,
            unnorm_ifft=unnorm,
        )

    def _split_step_RK4_fused(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        delta_z: float,
    ) -> np.ndarray:
        """Take one RK4 step with each stage fused into a single kernel.

        The same scheme as ``split_step_RK4``: four slopes, the first three
        accumulated with weights 1, 2, 2 and the argument advanced by h/2,
        h/2, h, then the field advanced by h/6 times the total. What differs
        is that no slope is written to memory.

        Parameters
        ----------
        A : np.ndarray
            Field to propagate, modified in place by the last stage.
        V : np.ndarray or None
            Potential field.
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            FFT plan objects from the backend.
        delta_z : float
            Step to take.

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        if not hasattr(self, "_rk4_k"):
            self._allocate_rk4_buffers(A, "RK4")
        A_tmp = self._rk4_A_tmp
        h = delta_z
        stage = self._RK4_fused_stage
        #        argument  out     w    c      mode
        stage(A, V, propagator, plans, A_tmp, A, 1.0, h / 2, 0)
        stage(A_tmp, V, propagator, plans, A_tmp, A, 2.0, h / 2, 1)
        stage(A_tmp, V, propagator, plans, A_tmp, A, 2.0, h, 1)
        stage(A_tmp, V, propagator, plans, A, A, 1.0, h / 6, 2)
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance, in this solver's axis units.
        """
        A_plot = self._to_plot_array(A_plot, 2)
        fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))
        fig.suptitle(self._plot_title(z))
        ext_real = [
            np.min(self.X) * 1e3,
            np.max(self.X) * 1e3,
            np.min(self.Y) * 1e3,
            np.max(self.Y) * 1e3,
        ]
        ext_fourier = [
            np.min(self.Kx) * 1e-3,
            np.max(self.Kx) * 1e-3,
            np.min(self.Ky) * 1e-3,
            np.max(self.Ky) * 1e-3,
        ]
        rho = np.abs(A_plot) ** 2 * self._plot_density_scale
        phi = np.angle(A_plot)
        im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot)))
        # Row 0 is the smallest y: without origin="lower" it is drawn on top.
        im0 = ax[0].imshow(rho, extent=ext_real, origin="lower")
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label=self._plot_density_label)
        im1 = ax[1].imshow(
            phi,
            extent=ext_real,
            origin="lower",
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax[1].set_title("Phase")
        ax[1].set_xlabel("x (mm)")
        ax[1].set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax[1], shrink=0.6, label="Phase (rad)")
        im2 = ax[2].imshow(
            im_fft,
            extent=ext_fourier,
            origin="lower",
            cmap="nipy_spectral",
        )
        ax[2].set_title("Fourier space")
        ax[2].set_xlabel(r"$k_x$ ($mm^{-1}$)")
        ax[2].set_ylabel(r"$k_y$ ($mm^{-1}$)")
        fig.colorbar(im2, ax=ax[2], shrink=0.6, label="Intensity (a.u.)")
        show_if_possible()

    # The non-local response. The interaction reads the intensity smoothed by
    # the Green function of a diffusion equation, applied as its transform on
    # the periodic grid the propagator uses.
    def _transverse_k_squared(self) -> np.ndarray:
        """Return the squared transverse wavenumber over the Fourier grid.

        The non-locality is the index diffusing across the beam, so only the
        transverse wavenumbers enter it: in 3D the time axis is left alone.
        The solvers on a 1D grid override this alone.

        Returns
        -------
        np.ndarray
            ``Kx^2 + Ky^2``, with the propagator's grid shape.
        """
        return self.Kxx**2 + self.Kyy**2

    def _nonlocal_response(self) -> np.ndarray:
        """Return the non-local response in Fourier space.

        ``1 / (1 + nl_length^2 K^2)`` is the transform of the diffusion
        equation's Green function: ``K0(r / l) / (2 pi l^2)`` in 2D and
        ``exp(-|x| / l) / (2 l)`` in 1D. It is 1 at ``K = 0``, so the smoothed
        intensity has the same total as the intensity.

        Returns
        -------
        np.ndarray
            The response, with the propagator's shape.
        """
        return 1 / (1 + self.nl_length**2 * self._transverse_k_squared())

    def _nonlocal_response_on_device(self, dtype: np.dtype) -> tuple[Any, bool]:
        """Return the response where the backend can use it, and if it has 1/N.

        Built once per grid, length, width and backend, and kept with the
        dispersion operators. Complex at the field's width, which is what the
        propagator kernels multiply by. Where the backend can skip the inverse
        transform's 1/N it is folded in here, as it is into the propagator.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field whose intensity it smooths.

        Returns
        -------
        tuple
            ``(response, unnorm_ifft)``.
        """
        unnorm = self._backend.supports_unnormalized_ifft
        key = (
            *self._propagator_rk4_cache_key(),
            "nl_response",
            float(self.nl_length),
            np.dtype(dtype).str,
            self._backend.name,
        )
        if key not in self._dispersion_cache:
            response = self._nonlocal_response()
            if unnorm:
                response = response / math.prod(
                    response.shape[ax] for ax in self._last_axes
                )
            self._dispersion_cache[key] = self._backend.from_numpy(
                np.ascontiguousarray(response, dtype=dtype)
            )
        return self._dispersion_cache[key], unnorm

    # The linear operator, and the propagator built from it. Subclasses
    # override the compute/key pairs alone; the caching around them is shared.
    def _dispersion_operator(self) -> np.ndarray:
        """Return the dispersion eigenvalues on the Fourier grid.

        The linear part of the right-hand side, as an array rather than one
        number, so the limiters can weight it by where the field actually has
        spectral weight. Subclasses override this alone.

        Returns
        -------
        np.ndarray
            ``K^2 / (2 k)`` over the Fourier grid.
        """
        return 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k

    def _dispersion_eigenvalues(self) -> np.ndarray:
        """Return ``_dispersion_operator()``, built once per configuration.

        The step limiters read it on every ``out_field`` call, and building it
        took 4.8 ms at 2048x2048, twice per call. A new
        ``_propagator_rk4_cache_key()`` replaces the grid kept.

        Returns
        -------
        np.ndarray
            The dispersion eigenvalues over the Fourier grid.
        """
        key = self._propagator_rk4_cache_key()
        if self._eigenvalues is None or self._eigenvalues[0] != key:
            grid = self._dispersion_operator()
            self._eigenvalues = (key, grid, float(np.max(np.abs(grid))))
        return self._eigenvalues[1]

    def _max_dispersion_rate(self) -> float:
        """Return the largest dispersion eigenvalue in magnitude, cached alike.

        Returns
        -------
        float
            ``max |_dispersion_operator()|``.
        """
        self._dispersion_eigenvalues()
        return self._eigenvalues[2]

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for the split-step propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.k),
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for the RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw dispersion operator for RK4 (no caching).

        Left at the width the grid gives it. ``_build_propagator_rk4`` casts
        it to the field's, which is the one place that knows what the field
        is: computing it narrow and widening it afterwards would carry
        single-precision values in a double-precision array.
        """
        return -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k

    def _build_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Build the linear propagation matrix with caching.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field the propagator will multiply.
        delta_z : float
            Step the propagator advances by. It is part of the cache key, so
            a run with a different step gets its own.

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        cache_key = self._propagator_cache_key(dtype, delta_z)
        if cache_key in self._propagator_cache:
            self._propagator_cache.move_to_end(cache_key)
            return self._propagator_cache[cache_key][0]
        operator = self._dispersion_on_device(dtype)
        propagator = self._backend.exp(operator * np.dtype(dtype).type(delta_z))
        # Kept with what it was built for, so that _built_step can name the
        # step of an array from its identity alone.
        self._propagator_cache[cache_key] = (propagator, np.dtype(dtype), delta_z)
        while len(self._propagator_cache) > PROPAGATOR_CACHE_SIZE:
            self._propagator_cache.popitem(last=False)
        return propagator

    def _built_step(self, propagator: Any) -> tuple | None:
        """Return what ``_build_propagator`` built this very array for.

        By identity against the arrays it still holds, so an array assigned
        by hand, or one a subclass's override made of its own, has none.

        Parameters
        ----------
        propagator : Any
            The array to look up.

        Returns
        -------
        tuple or None
            ``(dtype, delta_z, cache_key)``, the key being the solver's state
            at the build; None for an array it did not build.
        """
        for cache_key, (built, dtype, delta_z) in self._propagator_cache.items():
            if built is propagator:
                return dtype, delta_z, cache_key
        return None

    def _dispersion_axes(self) -> tuple | None:
        """Split the dispersion operator into one term per axis, if it is so.

        Returns
        -------
        tuple or None
            ``(along_y, along_x)``, complex128, with the operator equal to
            ``along_y[:, None] + along_x[None, :]`` to within SEPARABLE_RTOL;
            None for an operator with a cross term, or not of one 2D grid.
        """
        cache_key = (*self._propagator_rk4_cache_key(), "axes")
        if cache_key in self._dispersion_cache:
            return self._dispersion_cache[cache_key]
        operator = np.ascontiguousarray(
            self._compute_propagator_rk4(), dtype=np.complex128
        )
        if operator.ndim == FUSED_TRANSFORM_AXES:
            # Copied, or the cache would keep the whole grid alive through a view.
            along_x = operator[0, :].copy()
            along_y = operator[:, 0] - operator[0, 0]
            departure = np.subtract(operator, along_x[None, :])
            departure -= along_y[:, None]
            residual = self._largest_modulus(departure, departure)
            if residual <= SEPARABLE_RTOL * self._largest_modulus(operator, departure):
                axes = (along_y, along_x)
            else:
                axes = None
        else:
            axes = None
        self._dispersion_cache[cache_key] = axes
        return axes

    @staticmethod
    def _largest_modulus(values: np.ndarray, scratch: np.ndarray) -> float:
        """Return ``max |values|`` without allocating a grid.

        Squares the real and imaginary parts into ``scratch`` and sums them in
        place there, so ``scratch`` is overwritten; it may be ``values``.

        Parameters
        ----------
        values : np.ndarray
            C-contiguous complex128 grid.
        scratch : np.ndarray
            C-contiguous complex128 grid of the same shape.

        Returns
        -------
        float
            The largest modulus.
        """
        parts = scratch.view(np.float64)
        np.square(values.view(np.float64), out=parts)
        np.add(parts[..., 0::2], parts[..., 1::2], out=parts[..., 0::2])
        return float(np.sqrt(parts[..., 0::2].max()))

    def _separable_propagator(
        self, dtype: np.dtype, delta_z: float, rows: int
    ) -> tuple | None:
        """Return the propagator as one factor per axis, on the device.

        The product of the two is the propagator divided by the transform
        size, as ``_propagator_fft`` is. Each factor is exponentiated in
        complex128 and rounded once, to the field's width.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field.
        delta_z : float
            Step the propagator advances by.
        rows : int
            Rows of the field, over the whole of a batch.

        Returns
        -------
        tuple or None
            ``(rows, columns)`` factors, or None where the dispersion has a
            cross term.
        """
        axes = self._dispersion_axes()
        if axes is None:
            return None
        cache_key = (*self._propagator_cache_key(dtype, delta_z), rows)
        if cache_key in self._separable_cache:
            self._separable_cache.move_to_end(cache_key)
            return self._separable_cache[cache_key]
        along_y, along_x = axes
        n_fft = along_y.size * along_x.size
        row_factors = np.tile(np.exp(along_y * delta_z) / n_fft, rows // along_y.size)
        factors = (
            self._backend.from_numpy(row_factors.astype(dtype)),
            self._backend.from_numpy(np.exp(along_x * delta_z).astype(dtype)),
        )
        self._separable_cache[cache_key] = factors
        while len(self._separable_cache) > PROPAGATOR_CACHE_SIZE:
            self._separable_cache.popitem(last=False)
        return factors

    def _dispersion_on_device(self, dtype: np.dtype) -> Any:
        """Return the dispersion operator, where the backend can use it.

        Built on the host once per grid and splitting, then left alone.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field it will propagate.

        Returns
        -------
        Any
            The operator.
        """
        key = (*self._propagator_rk4_cache_key(), np.dtype(dtype).str, "device")
        if key not in self._dispersion_cache:
            if self._backend.is_device_backend:
                # Built and transferred without going through the host cache,
                # so the host copy is a temporary rather than a second full
                # grid kept for the life of the solver.
                operator = self._backend.from_numpy(
                    np.asarray(self._compute_propagator_rk4()).astype(dtype)
                )
            else:
                operator = self._build_propagator_rk4(dtype)
            self._dispersion_cache[key] = operator
        return self._dispersion_cache[key]

    def _build_propagator_rk4(self, dtype: np.dtype) -> np.ndarray:
        """Build raw dispersion operator for RK4 with caching.

        Parameters
        ----------
        dtype : np.dtype
            Complex dtype of the field the operator will multiply. The GPU
            kernels choose their precision from the field and index the
            operator with the same flat id, so an operator of the other
            width is read as the wrong type and comes back NaN. It is part
            of the cache key for the same reason.

        Returns
        -------
        np.ndarray
            The raw dispersion operator, at the field's precision.
        """
        cache_key = (*self._propagator_rk4_cache_key(), np.dtype(dtype).str)
        if cache_key in self._dispersion_cache:
            return self._dispersion_cache[cache_key]
        propagator = np.asarray(self._compute_propagator_rk4()).astype(dtype)
        self._dispersion_cache[cache_key] = propagator
        return propagator

    def _fused_propagator(self, propagator: Any) -> tuple:
        """Return the propagator a kernel should use, and whether it carries 1/N.

        Two answers that have to agree. ``_update_propagator_fft`` builds a
        pre-normalized twin where the backend can skip the inverse
        transform's 1/N, and a kernel handed that twin must be told to skip
        it -- handed it and told to normalize anyway, the field is divided by
        N twice, and told to skip it while handed the plain propagator, not
        at all. Neither shows up as an error, only as a field of the wrong
        amplitude.

        Every fused call site chose both by hand, eight of them across this
        class and CNLSE, each writing ``prop_fft if prop_fft is not None else
        propagator`` next to ``unnorm_ifft=(prop_fft is not None)``. Deciding
        it here is one fact rather than eight pairs of facts.

        Parameters
        ----------
        propagator : Any
            The plain propagator, used when there is no pre-normalized twin.

        Returns
        -------
        tuple
            ``(propagator, unnorm_ifft)`` -- what to pass, and what to pass
            it as.
        """
        prop_fft = self._propagator_fft
        if prop_fft is None:
            return propagator, False
        return prop_fft, True

    def _update_propagator_fft(self) -> None:
        """Derive the pre-normalized propagator used by fused linear steps.

        Absorbs the 1/N of the inverse FFT into the propagator so that
        ``linear_step`` can skip a separate normalization multiply. Only
        CUPY (cuFFT) and CL (VkFFT norm=0) expose an unnormalized IFFT.

        Must be called whenever ``self.propagator`` changes, otherwise the
        fused kernels keep using a propagator derived from the previous one.
        Records the twin's origin where ``_build_propagator`` built the
        propagator, which is what ``_propagator_fft_factors`` offers from.
        """
        if self._backend.supports_unnormalized_ifft and self.propagator is not None:
            N_fft = 1
            for ax in self._last_axes:
                N_fft *= self.propagator.shape[ax]
            inv_N = (
                np.float32(1.0 / N_fft)
                if self.propagator.dtype == np.complex64
                else np.float64(1.0 / N_fft)
            )
            self._propagator_fft = self.propagator * inv_N
            built = self._built_step(self.propagator)
            self._propagator_fft_origin = (
                None if built is None else (self._propagator_fft, *built)
            )
        else:
            self._propagator_fft = None
            self._propagator_fft_origin = None

    def _propagator_fft_factors(self, propagator_fft: Any, rows: int) -> tuple | None:
        """Return the twin a kernel is handed as one factor per axis, if known.

        Only for the twin ``_update_propagator_fft`` recorded an origin for,
        only while it is that very array, and only while the solver's cache
        key for it is unchanged. Anything else gets None and the kernel reads
        the full array.

        Parameters
        ----------
        propagator_fft : Any
            The pre-normalized propagator the kernel is handed.
        rows : int
            Rows of the field, over the whole of a batch.

        Returns
        -------
        tuple or None
            ``(rows, columns)`` from ``_separable_propagator``, or None.
        """
        origin = self._propagator_fft_origin
        if origin is None or origin[0] is not propagator_fft:
            return None
        _, dtype, delta_z, cache_key = origin
        # The factors are computed from the solver as it is now: a k or a grid
        # changed since the build would give those of another propagator.
        if self._propagator_cache_key(dtype, delta_z) != cache_key:
            return None
        return self._separable_propagator(dtype, delta_z, rows)

    def _linear_state(self) -> tuple:
        """Return what the linear step reads off the solver, to restore later.

        A step on a propagator of its own swaps all three and puts them back
        as one, so the twin's origin cannot describe another twin.

        Returns
        -------
        tuple
            ``(propagator, _propagator_fft, _propagator_fft_origin)``.
        """
        return (self.propagator, self._propagator_fft, self._propagator_fft_origin)

    def _restore_linear_state(self, state: tuple) -> None:
        """Put back what ``_linear_state`` returned.

        Parameters
        ----------
        state : tuple
            ``(propagator, _propagator_fft, _propagator_fft_origin)``.
        """
        self.propagator, self._propagator_fft, self._propagator_fft_origin = state

    # Setting up one run: the dtypes, the arrays, the FFT plans and the
    # constants that do not change between steps.
    @staticmethod
    def _field_dtype(E_in: np.ndarray) -> np.dtype:
        """Return the complex dtype the propagator has to match.

        The propagator multiplies the field, so the two must share a dtype.
        The GPU kernels select single or double precision from the *field*,
        then index the propagator with it: a complex128 propagator against a
        complex64 field was read as pairs of float32 and came back NaN.

        Parameters
        ----------
        E_in : np.ndarray
            Input field, on the host or on a device.

        Returns
        -------
        np.dtype
            ``complex128`` if the field is double precision, else
            ``complex64``.
        """
        dtype = getattr(E_in, "dtype", np.complex64)
        try:
            double = np.dtype(dtype).itemsize == 16
        except TypeError:
            # MLX names its own dtypes, which numpy cannot interpret.
            double = "128" in str(dtype)
        return np.dtype(np.complex128) if double else np.dtype(np.complex64)

    @staticmethod
    def _potential_dtype(V: np.ndarray, field_dtype: np.dtype) -> np.dtype:
        """Return the dtype a potential must take for a given field.

        The width follows the field, exactly as the propagator's does: the
        kernels pick their single- or double-precision variant from the field
        and then read V with it. Whether V is complex follows V itself — a
        complex potential is an absorbing (or amplifying) one, its imaginary
        part entering as gain/loss, and that has to survive the transfer.
        Casting it to a real dtype silently deleted the absorption.

        Parameters
        ----------
        V : np.ndarray
            Potential, real or complex.
        field_dtype : np.dtype
            Complex dtype of the field V will act on.

        Returns
        -------
        np.dtype
            ``float32``/``float64`` for a real V, ``complex64``/
            ``complex128`` for a complex one.
        """
        single = np.dtype(field_dtype).itemsize == 8
        if np.iscomplexobj(V):
            return np.dtype(np.complex64 if single else np.complex128)
        return np.dtype(np.float32 if single else np.float64)

    @property
    def _norm_target(self) -> Any:
        """Return what ``normalize=True`` fixes the field's integral to.

        A power here, an energy for ``NLSE_3d``, one value per component for
        the coupled solvers. Trailing axes broadcast against the integral, so
        a per-component target works for a batch of them too.
        """
        return self.power

    def _check_batch_support(self, E_in: np.ndarray) -> None:
        """Refuse a batched run the backend cannot serve. Every backend can.

        Parameters
        ----------
        E_in : np.ndarray
            The input field.
        """

    def _check_precision_support(self, field_dtype: np.dtype) -> None:
        """Move a complex128 run off a backend that has no float64.

        MLX rounded such a field to complex64 without a word, and an OpenCL
        device without fp64 refuses it when its kernels compile. Either way
        the run moves, as ``_require_backend`` moves any run a backend
        cannot serve.

        Parameters
        ----------
        field_dtype : np.dtype
            Complex dtype the run will use, from ``_field_dtype``.
        """
        if np.dtype(field_dtype) == np.dtype(np.complex64):
            return
        self._require_backend(
            lambda backend: backend.supports_double_precision(),
            "double precision (a complex128 field)",
        )

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> tuple[np.ndarray | Any, np.ndarray | Any]:
        """Prepare the output arrays depending on backend.

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
        # Allocate arrays on the backend
        A = self._backend.allocate_field(E_in.shape, E_in.dtype)
        A_sq = self._backend.allocate_real_field(E_in.shape, E_in.real.dtype)
        E_in = self._backend.from_numpy(E_in)

        if normalize:
            # normalization of the field (use contiguous formula)
            arr = (E_in * E_in.conj()).real
            # Use pre-computed grid factor to avoid runtime upcasting
            arr = arr * self._norm_grid_factor
            if self._backend.normalizes_on_host:
                arr_np = self._backend.to_numpy(arr)
                E_in_np = self._backend.to_numpy(E_in)
                integral = np.sum(arr_np, axis=self._last_axes)
                integral = integral * self._norm_constant
                E_00 = self._normalization_factor(self._norm_target, integral)
                # E_00 is float64, so the bare product would be a complex128
                # temporary twice the field's size; write it at the field's.
                result = np.empty_like(E_in_np)
                np.multiply(E_00.T, E_in_np.T, out=result.T)
                A = self._backend.from_numpy(result)
            else:
                integral = np.sum(arr, axis=self._last_axes)
                integral = integral * self._norm_constant
                target = self._norm_target
                if getattr(target, "ndim", 0) > 0:
                    # One target per component: an array operand has to be on
                    # the same device as the integral it divides.
                    target = self._backend.from_numpy(
                        np.asarray(target, dtype=integral.dtype)
                    )
                E_00 = self._normalization_factor(target, integral)
                np.multiply(E_00.T, E_in.T, out=A.T)
        else:
            A[:] = E_in
        return A, A_sq

    @staticmethod
    def _normalization_factor(target, integral):
        """Return the scale that carries this field to the target power.

        A field with no power in it has no factor that scales it to one, and
        dividing anyway gives infinity where the caller asked for a beam --
        which then multiplies the zeros it came from into NaN, and the run
        reports nothing worse than a RuntimeWarning on the way past. A field
        that is zero is left as it is instead. It is a legitimate initial
        condition: a driven cavity starts from one.

        Parameters
        ----------
        target : Any
            Power asked for, per component where there are several.
        integral : Any
            Power the field carries.

        Returns
        -------
        Any
            The factor, 1 wherever the field carries nothing.
        """
        empty = integral == 0
        safe = integral + empty  # 1 where the field is empty, untouched elsewhere
        return ((target / safe) ** 0.5) * (1 - empty) + empty

    def _build_fft_plan(self, A: np.ndarray) -> list:
        """Build the FFT plan objects for propagation.

        Parameters
        ----------
        A : np.ndarray
            Array to transform.

        Returns
        -------
        list
            List of FFT plan objects from the backend.
        """
        return self._backend.build_fft(A.shape, self._last_axes, A.dtype, array=A)

    def _allocate_rk4_buffers(self, A: np.ndarray, method: str) -> None:
        """Pre-allocate scratch buffers for the RK4 stepper.

        The buffers take the field's own splitting. Pinning them to
        complex64 made a double-precision run write the field into a stage
        buffer half its width, and the transform -- planned from the field --
        then met an array of the wrong dtype, which cuFFT answers with NaN
        rather than an error.
        """
        if method == "RK4":
            dtype = self._field_dtype(A)
            self._rk4_k = self._backend.allocate_field(A.shape, dtype)
            self._rk4_A_tmp = self._backend.allocate_field(A.shape, dtype)
            self._rk4_acc = self._backend.allocate_field(A.shape, dtype)

    def _step_constants(self) -> dict[str, Any]:
        """Return the per-step constants, from the physical parameters.

        The single statement of what each one is. ``_precompute_step_constants``
        casts these to the run's float width and stores them under the same
        names, and ``_constant`` reads them back, falling back here when a
        caller asks before a run has set them up.

        Before this, each expression appeared twice: once here and once inside
        every ``self._constant("_g")`` that read it. Two
        copies of a physical definition drift, and one of them being wrong is
        invisible while the other is in use.

        Returns
        -------
        dict
            Attribute name to value, in physical units.
        """
        return {
            "_alpha_half": self.alpha / 2,
            "_g": self.k / 2 * self.n2 * c * epsilon_0,
            "_Isat_conv": 2 * self.I_sat / (epsilon_0 * c),
            "_k_half": self.k / 2,
        }

    def _constant(self, name: str) -> Any:
        """Return a per-step constant, precomputed if a run has set it up.

        Parameters
        ----------
        name : str
            Attribute name, as it appears in _step_constants.

        Returns
        -------
        Any
            The stored value, or the same quantity computed afresh.
        """
        value = getattr(self, name, None)
        return self._step_constants()[name] if value is None else value

    @staticmethod
    def _real_dtype(field_dtype: np.dtype) -> type:
        """Return the real width that goes with a complex field width."""
        return np.float32 if np.dtype(field_dtype) == np.complex64 else np.float64

    def _precompute_step_constants(
        self, V: np.ndarray | None, field_dtype: np.dtype = np.complex64
    ) -> None:
        """Pre-compute constants that are invariant across propagation steps.

        The width of everything here follows the *field*, which is the rule the
        whole solver is written to: the GPU kernels pick single or double
        precision from the field and then index the propagator, the potential
        and the scratch buffers with it, so anything of the other width is read
        as the wrong type. ``_field_dtype`` says it of the propagator and
        test_double_precision.py of the RK4 buffers.

        It used to follow the *splitting* -- float32 for Lie and float64 for
        everything else -- which is a leftover from when "single" and "double"
        named the splittings rather than the float width. It was wrong in both
        directions. A complex64 Strang or Yoshida run scaled the potential by a
        float64 scalar, and NEP 50 promotion handed a float64 array to a kernel
        reading it through a ``float*``: NaN from the first step above 128x128,
        and 30% wrong below it, which is worse for looking like an answer. A
        complex128 Lie run had the opposite fault and rounded every physical
        constant to float32.

        Parameters
        ----------
        V : np.ndarray or None
            Potential field.
        field_dtype : np.dtype
            Complex dtype of the field this run propagates, which fixes the
            width of every constant and of the scaled potential.
        """
        fp = self._real_dtype(field_dtype)
        for name, value in self._step_constants().items():
            # A batched parameter is an array and stays one; only scalars are
            # narrowed, so the kernels get a value of the right width.
            setattr(
                self,
                name,
                fp(value) if isinstance(value, (int, float, np.floating)) else value,
            )
        self._V_scaled = (
            None
            if V is None
            else self._scale_potential(
                V, self._k_half, self._potential_dtype(V, field_dtype)
            )
        )
        self._forget_potential_grids()
        self._update_propagator_fft()

    @staticmethod
    def _scale_potential(V: Any, k_half: Any, dtype: np.dtype) -> Any:
        """Return ``V * k/2`` at the dtype the field asks of a potential.

        Neither operand decides that. ``V`` is stored at whatever width it was
        assigned at and the CPU backend never moves it, so it can be narrower
        than the field; ``k_half`` is a numpy scalar, and NEP 50 promotion lets
        it widen the product past the field. Both have been wrong here.

        The dtype comes from ``_potential_dtype``, which is also what the
        transfer uses -- so the width follows the field and the complexity
        follows V. Not ``_real_dtype``: an absorbing potential is complex, and
        casting it to a real dtype deletes the absorption without saying so.

        Parameters
        ----------
        V : Any
            Potential, wherever it lives.
        k_half : Any
            Half the wavenumber, scalar or batched.
        dtype : np.dtype
            What ``_potential_dtype`` asks for, given the field.

        Returns
        -------
        Any
            The scaled potential, at that dtype.

        """
        scaled = V * k_half
        # numpy cannot read MLX dtypes, so the cast is a guarantee only where it can;
        # elsewhere k_half, already narrowed to the field's width in
        # _precompute_step_constants, keeps the product at that width.
        try:
            mismatched = np.dtype(scaled.dtype) != np.dtype(dtype)
        except TypeError:
            return scaled
        return scaled.astype(dtype) if mismatched else scaled

    # Moving arrays on and off the device.
    #
    # Arrays are always transferred; params only if they are np.ndarray, i.e.
    # batched. Subclasses extend these tuples with their own attributes.
    _gpu_array_attrs = ("V", "propagator")
    _gpu_param_attrs = ("power", "n2", "alpha", "I_sat")

    def _as_host_array(self, value: Any) -> Any:
        """Return a parameter as a host numpy array, or None if unset.

        Parameters are scalars for an ordinary run, but arrays for a batched
        one, and _send_arrays_to_gpu may have moved those arrays onto the
        device. Callers that need to compare or reduce them have to bring
        them back first.

        Parameters
        ----------
        value : Any
            Scalar, numpy array, or device array.

        Returns
        -------
        np.ndarray or None
            The value on the host, or None if it was None.
        """
        if value is None:
            return None
        if isinstance(value, np.ndarray) or np.isscalar(value):
            return np.asarray(value)
        return np.asarray(self._backend.to_numpy(value))

    def _forget_device_state(self) -> None:
        """Drop everything built as arrays of the current backend.

        The propagator, the plans, the RK4 scratch buffers and both caches are
        arrays of the backend that built them. Kept across a change of
        backend they reach kernels that cannot read them: MLX to CPU raised,
        and CPU to MLX propagated with a propagator built on the host.
        """
        self.propagator = None
        self.plans = None
        # Bounded, least-recently-used. Dispersion operators live apart:
        # there are only a handful, one per grid and splitting, and evicting
        # one would cost the host build the propagators no longer pay.
        self._propagator_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._dispersion_cache: dict[tuple, Any] = {}
        # The propagator as one factor per axis, keyed as the propagators are
        # and by the rows of a batch, bounded the same way.
        self._separable_cache: OrderedDict[tuple, Any] = OrderedDict()
        # Host array and device copy of each attribute a run sent, by name.
        self._sent_arrays: dict[str, tuple[Any, Any]] = {}
        # The RK4 steppers reallocate every buffer when _rk4_k is missing.
        for name in ("_rk4_k", "_rk4_A_tmp", "_rk4_acc"):
            vars(self).pop(name, None)

    def _forget_run_state(self) -> None:
        """Clear what a run set up for its own steps.

        The step constants, the scaled potential and the pre-normalized
        propagator describe one run. Left set, a later ``split_step`` used
        them instead of the ``V`` and propagator it was given, and after a
        run past ``L`` kept a zero coupling.
        """
        for name in (*self._step_constants(), *self._run_state_attrs):
            setattr(self, name, None)

    @contextlib.contextmanager
    def _arrays_on_device(self, field_dtype: np.dtype) -> Iterator[None]:
        """Hold the solver's arrays on the device for the duration of a run.

        V, the propagator and any batched parameter are moved onto
        the device and put back afterwards, on the way out of a failed run as
        well as a finished one. Left there, they break the next run rather than
        the one that failed: from_numpy is handed a device array. The run's
        step constants are cleared on the way out as well.

        Parameters
        ----------
        field_dtype : np.dtype
            Complex dtype of the field, so V goes at a matching width.

        Yields
        ------
        None
        """
        self._send_arrays_to_gpu(field_dtype)
        try:
            yield
        finally:
            self._retrieve_arrays_from_gpu()
            self._forget_run_state()

    def _send_arrays_to_gpu(self, field_dtype: np.dtype = np.complex64) -> None:
        """Send arrays to device using backend.

        Parameters
        ----------
        field_dtype : np.dtype
            Complex dtype of the field, so the potential is transferred at a
            matching width rather than always as float32.
        """
        if not self._backend.is_device_backend:
            return
        for attr in self._gpu_array_attrs:
            val = getattr(self, attr, None)
            if val is None or not isinstance(val, np.ndarray):
                # Already where it belongs. The propagator is built there now,
                # and from_numpy handed a device array does not complain about
                # the type: it tries to enqueue a write *from* it.
                continue
            sent = val
            if attr == "V":
                sent = np.ascontiguousarray(
                    val, dtype=self._potential_dtype(val, field_dtype)
                )
            self._send_to_device(attr, val, sent)
        if not self._backend.broadcasts_parameters_natively:
            # The kernels take one simulation's scalar value per launch, so a
            # batched parameter is picked apart on the host and never reaches
            # the device as an array.
            return
        for attr in self._gpu_param_attrs:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                self._send_to_device(attr, val, val)

    def _send_to_device(self, attr: str, host: np.ndarray, sent: np.ndarray) -> None:
        """Replace an attribute by a device copy, remembering its host array.

        Parameters
        ----------
        attr : str
            Attribute name.
        host : np.ndarray
            The array the attribute held, which the run gives back.
        sent : np.ndarray
            What is transferred: ``host`` itself, or V at the field's width.
        """
        device = self._backend.from_numpy(sent)
        self._sent_arrays[attr] = (host, device)
        setattr(self, attr, device)

    def _send_propagator_to_gpu(self) -> None:
        """Move the freshly built propagator onto the device.

        Separate from _send_arrays_to_gpu because the propagator is built
        after delta_z is settled, which is after the other arrays have gone.

        It used to take the field's dtype, and never looked at it: the
        propagator is built from that dtype a line or two earlier, so the
        argument restated a fact rather than deciding anything. Reading the
        call, it looked as though passing the wrong one would convert
        something.
        """
        # _build_propagator leaves it on the device; the RK4 operator comes
        # from the cache on the host, so only that still needs moving.
        if (
            self._backend.is_device_backend
            and self.propagator is not None
            and isinstance(self.propagator, np.ndarray)
        ):
            self.propagator = self._backend.from_numpy(self.propagator)
        # The fused CUPY/CL linear step prefers _propagator_fft, so it has to
        # follow every rebuild.
        self._update_propagator_fft()

    def _retrieve_arrays_from_gpu(self) -> None:
        """Retrieve arrays from device using backend."""
        if not self._backend.is_device_backend:
            return
        for attr in self._gpu_array_attrs:
            val = getattr(self, attr, None)
            if val is None:
                continue
            setattr(self, attr, self._host_value(attr, val))
        if self._backend.broadcasts_parameters_natively:
            for attr in self._gpu_param_attrs:
                val = getattr(self, attr)
                if not isinstance(val, (int, float)):
                    setattr(self, attr, self._host_value(attr, val))
        self._sent_arrays.clear()

    def _host_value(self, attr: str, value: Any) -> Any:
        """Return an attribute's value on the host after a run.

        The array the caller gave, if the attribute still holds the copy sent
        from it: converting that copy back returned V at the field's width,
        so a float64 potential came back float32.

        Parameters
        ----------
        attr : str
            Attribute name.
        value : Any
            Its value on the device.

        Returns
        -------
        Any
            The host array that was sent, or ``value`` converted to numpy.
        """
        host, device = self._sent_arrays.get(attr, (None, None))
        return host if value is device else self._backend.to_numpy(value)

    # The pieces of a step.
    def _apply_linear_step(
        self, A: np.ndarray, propagator: np.ndarray, plans: list
    ) -> np.ndarray:
        """Apply linear propagation: FFT, propagator multiply, IFFT.

        Parameters
        ----------
        A : np.ndarray
            Field to propagate (modified in-place).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        prop, unnorm = self._fused_propagator(propagator)
        return self._multiply_in_fourier_space(A, prop, plans, unnorm)

    def _multiply_in_fourier_space(
        self, A: Any, multiplier: Any, plans: list, unnorm: bool
    ) -> Any:
        """Transform, multiply by an operator diagonal in K, transform back.

        Parameters
        ----------
        A : Any
            Complex array to transform, modified in place where the backend
            allows it.
        multiplier : Any
            Operator on the Fourier grid: complex at A's width, with A's grid
            shape or A's shape.
        plans : list
            FFT plans for A, from the backend.
        unnorm : bool
            Whether ``multiplier`` carries the inverse transform's 1/N.

        Returns
        -------
        Any
            The result.
        """
        kernels = self._backend.kernels
        if self._backend.has_linear_step:
            return kernels.linear_step(A, multiplier, plans[0], unnorm_ifft=unnorm)
        # The pre-normalized propagator carries the inverse transform's 1/N,
        # so the transform itself can skip it. A backend without a fused
        # linear step reaches it here rather than through kernels.linear_step.
        A = self._backend.fft(A, plans)
        A = kernels.apply_propagator(A, multiplier)
        A = self._backend.ifft(A, plans, normalize=not unnorm)
        return A

    def _nonlocal_intensity(self, A: Any, A_sq: Any) -> Any:
        """Write the intensity, smoothed by the non-local response, into A_sq.

        Parameters
        ----------
        A : Any
            Field. Left unchanged.
        A_sq : Any
            Real scratch of A's shape and width, overwritten.

        Returns
        -------
        Any
            ``A_sq``.
        """
        response, unnorm = self._nonlocal_response_on_device(self._field_dtype(A))
        # |A|^2 as a complex array, so it goes through the field's own plans.
        intensity = self._multiply_in_fourier_space(
            A * A.conj(), response, self._build_fft_plan(A), unnorm
        )
        A_sq[:] = intensity.real
        return A_sq

    def _RK4_rhs(
        self,
        A_in: np.ndarray,
        k: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
    ) -> np.ndarray:
        """Compute the RHS of NLSE into pre-allocated buffer k.

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
            List of FFT plan objects from backend.
        """
        kernels = self._backend.kernels
        # Use pre-computed constants (set in out_field), with fallbacks
        alpha_half = self._constant("_alpha_half")
        g = self._constant("_g")
        Isat_conv = self._constant("_Isat_conv")
        k_half = self._constant("_k_half")
        V_scaled = self._V_scaled
        if V_scaled is None and V is not None:
            V_scaled = V * k_half

        # Fused fast path: out-of-place FFT eliminates the buffer copy
        if self._backend.has_fused_rk4_rhs and self.nl_length == 0:
            prop, unnorm = self._fused_propagator(propagator)
            return kernels.rk4_rhs_fused(
                A_in,
                k,
                V_scaled,
                prop,
                plans[0],
                alpha_half,
                g,
                Isat_conv,
                unnorm_ifft=unnorm,
            )

        k[:] = A_in
        k = self._apply_linear_step(k, propagator, plans)

        if self.nl_length > 0:
            A_sq = self._nonlocal_intensity(
                A_in,
                self._backend.allocate_real_field(
                    A_in.shape, self._real_dtype(self._field_dtype(A_in))
                ),
            )
            if V is None:
                k = kernels.rk4_nl_rhs(k, A_in, A_sq, alpha_half, g, Isat_conv)
            else:
                k = kernels.rk4_nl_rhs_v(
                    k, A_in, A_sq, V_scaled, alpha_half, g, Isat_conv
                )
        else:
            if V is None:
                k = kernels.square_mod_rk4_nl_rhs(k, A_in, alpha_half, g, Isat_conv)
            else:
                k = kernels.square_mod_rk4_nl_rhs_v(
                    k, A_in, V_scaled, alpha_half, g, Isat_conv
                )
        return k

    def _take_partial_step(self, step_factory, remainder, method, dtype):
        """Cover the distance left over after the whole steps.

        The propagator is built from the step, so a shorter one needs its own.
        Cache-backed, and a run takes at most one of these per segment: one
        landing on L where the run leaves the medium, one landing on z.

        Parameters
        ----------
        step_factory : callable
            Builds the per-step closure for a given step size.
        remainder : float
            Distance still to cover, signed like the step: negative in a
            backward run.
        method : str
            Integration method ("split_step" or "RK4").
        dtype : np.dtype
            Complex dtype of the field.

        The splitting is not passed: ``step_factory`` defaults it to the one
        the run asked for, which is what this wants. A merged Strang run
        overrides it for the loop body only, and its bracket is closed before
        the remainder is reached, so the leftover is a whole step of its own.
        """
        # Swapped rather than rebuilt afterwards: the solver should describe
        # the run it just did, not the sliver at the end of it.
        in_force = self._current_delta_z
        saved = self._linear_state()
        if method != "RK4":
            self.propagator = self._build_propagator(dtype, remainder)
            self._send_propagator_to_gpu()
        self._current_delta_z = remainder
        try:
            step_factory(remainder)()
        finally:
            self._current_delta_z = in_force
            self._restore_linear_state(saved)

    # _medium_attrs: everything the medium provides, zeroed once the propagation
    # leaves it and re-derived by _precompute_step_constants; subclasses extend it.
    # alpha is in it because nothing outside the medium absorbs.
    _medium_attrs = ("n2", "alpha")

    # Whether this solver's Lie step is exactly the inside of its Strang step.
    # DDGPE's is not: it drives and it adds noise, so its steps are not a
    # product of the same two operators. The coupled solvers opt out too.
    _lie_step_is_strang_body = True

    def _merges_strang_halves(self, splitting: str) -> bool:
        """Whether consecutive Strang steps may share their half steps.

        A Strang step is ``N(h/2) L(h) N(h/2)``, so a run of them is

            N(h/2) [L(h) N(h)] ... [L(h) N(h)] N(-h/2)

        and the bracketed body is precisely a Lie step. Merging costs one
        nonlinear application for the whole run instead of one per step, and
        the loop body becomes what ``splitting="lie"`` already runs, so
        there is no third stepper to keep in step with the other two.

        It is exact only where ``N(a) N(b) == N(a + b)``. ``N`` rotates the
        phase by an amount that depends on ``|A|^2``, so what it needs is for
        ``N`` to be the exact solution of the real-space equation over its
        step, and then composing two is solving over the sum.

        **Loss used to break that and no longer does.** The kernels froze
        ``|A|^2`` across the step, which is the exact solution only while the
        step preserves it, so a second half step saw an intensity the first
        had damped and the two were not one whole one. They now solve the step
        instead (see ``_loss_factor`` in kernels/cpu.py), and both halves of
        it telescope exactly: the phase over a step is
        ``(g/(2*alpha))*(y0 - y_end)`` and the amplitude ``sqrt(y_end/y0)``,
        and a product of two of those is the one over the sum.

        An absorbing potential still breaks it. That decay is a second channel,
        which the identity behind the solved step does not cover, so it is
        still applied frozen and two halves of it are still not one whole one.

        Parameters
        ----------
        splitting : str
            Splitting the caller asked for. Only Strang has halves to merge.

        Returns
        -------
        bool
            True if the run may be merged.
        """
        if splitting != "strang" or not self._lie_step_is_strang_body:
            return False
        if self.nl_length > 0:
            return False
        # A Rabi coupling wraps each Strang step in two half rotations, which
        # a Lie body between two nonlinear half steps does not reproduce.
        if getattr(self, "omega", None) is not None:
            return False
        V_scaled = self._as_host_array(self._V_scaled)
        return V_scaled is None or not np.iscomplexobj(V_scaled)

    def _open_strang_bracket(self, merged, A, A_sq, delta_z):
        """Take the half step that precedes a merged run."""
        if not merged:
            return A
        return self._nonlinear_step(A, A_sq, self._V_scaled, delta_z / 2)

    def _close_strang_bracket(self, merged, A, A_sq, delta_z):
        """Undo the half step the merged run's last body left over."""
        if not merged:
            return A
        return self._nonlinear_step(A, A_sq, self._V_scaled, -delta_z / 2)

    def _run_propagation(
        self,
        A,
        A_sq,
        V,
        z,
        delta_z,
        splitting,
        method,
        callback,
        callback_args,
        verbose,
        pbar,
    ):
        """Execute the propagation loop, using the backend's fast path if eligible.

        Returns
        -------
        np.ndarray
            The propagated field (may be a new object for functional backends).
        """
        # Use a mutable container so the closure can update A for functional
        # backends (MLX) where kernels return new arrays.
        state = [A]

        def _make_step(dz, body_precision=splitting):
            """Build the per-step closure for a given step size.

            Rebuilt when an adaptive callback changes the step, so the step
            and the propagator it was built from cannot come apart.

            ``body_precision`` is the splitting the loop body runs, which is
            not always the one the caller asked for: a merged Strang run
            takes Lie steps between two half steps of its own.
            """
            if method == "RK4":

                def _step():
                    state[0] = self.split_step_RK4(
                        state[0], V, self.propagator, self.plans, dz
                    )
            elif body_precision == "yoshida":
                # Built here rather than per step: the three sub-steps have
                # fixed lengths once dz is known, and this closure is remade
                # whenever dz changes.
                sub = self._yoshida_propagators(self._field_dtype(state[0]), dz)

                def _step():
                    state[0] = self.split_step_yoshida(
                        state[0], A_sq, V, sub, self.plans, dz
                    )
            else:

                def _step():
                    state[0] = self.split_step(
                        state[0],
                        A_sq,
                        V,
                        self.propagator,
                        self.plans,
                        dz,
                        body_precision,
                    )

            return _step

        is_scalar_n2 = isinstance(self.n2, (int, float, np.floating))
        can_use_fast_loop = callback is None and is_scalar_n2 and self.nl_length == 0
        # Zeroing the nonlinearity mutates self, so restore it afterwards.
        saved = {attr: getattr(self, attr) for attr in self._medium_attrs}
        try:
            if can_use_fast_loop:
                self._loop_without_callbacks(
                    _make_step, z, delta_z, splitting, method, state, A_sq, V
                )
            else:
                self._loop_with_callbacks(
                    _make_step,
                    z,
                    delta_z,
                    callback,
                    callback_args,
                    state,
                    verbose,
                    pbar,
                    V=V,
                    method=method,
                    dtype=self._field_dtype(A),
                )
        finally:
            for attr, value in saved.items():
                setattr(self, attr, value)
        return state[0]

    def _loop_without_callbacks(
        self, step_factory, z, delta_z, splitting, method, state, A_sq, V
    ):
        """Run the whole steps through the backend's loop, segment by segment.

        One ``execute_loop`` per segment rather than a mid-loop switch: the
        constants are baked into the CUDA graph it captures, so each segment
        needs its own capture. A segment the step does not divide ends on a
        cut step, so the medium acts over L exactly and the run stops at z.

        Parameters
        ----------
        step_factory : callable
            Builds the per-step closure for a step size and a splitting.
        z : float
            Total propagation distance.
        delta_z : float
            Step, negative for a backward run.
        splitting : str
            Splitting the caller asked for.
        method : str
            Integration method ("split_step" or "RK4").
        state : list
            One-element list holding the field, updated in place.
        A_sq : np.ndarray
            Intensity buffer the nonlinear steps use.
        V : np.ndarray or None
            Potential, needed to re-derive the constants outside the medium.
        """
        merged = self._merges_strang_halves(splitting)
        step = step_factory(delta_z, "lie" if merged else splitting)
        self._current_delta_z = delta_z
        start = 0.0
        for n, end in enumerate(self._medium_ends(z)):
            if n > 0:
                self._leave_medium(V, self._field_dtype(state[0]))
            n_steps, remainder = self._whole_steps(end - start, delta_z)
            state[0] = self._open_strang_bracket(merged, state[0], A_sq, delta_z)
            self._backend.execute_loop(step, n_steps)
            state[0] = self._close_strang_bracket(merged, state[0], A_sq, delta_z)
            if remainder:
                self._take_partial_step(
                    step_factory, remainder, method, self._field_dtype(state[0])
                )
            start = end

    def _loop_with_callbacks(
        self,
        step_factory,
        z,
        delta_z,
        callback,
        callback_args,
        state,
        verbose,
        pbar,
        V=None,
        method="split_step",
        dtype=np.complex64,
    ):
        """Run propagation loop with per-step callbacks.

        A callback may return a new step to use from here on. The propagator
        is rebuilt to match before the next step: it is built from delta_z, so
        changing one without the other silently propagates the linear part by
        the wrong distance.

        A step that would pass the end of a segment -- L where the run leaves
        the medium, then z -- is cut to land on it, whatever a callback has
        made of the step by then. A grown step would otherwise overshoot.
        The callbacks are shown the step just taken, cut or whole.

        Parameters
        ----------
        step_factory : callable
            Builds the per-step closure for a step size.
        z : float
            Total propagation distance.
        delta_z : float
            Initial step, negative for a backward run.
        callback : callable, list of callable, or None
            Called after every step as ``callback(self, A, z, i, *args)``.
        callback_args : tuple or list of tuples
            Extra arguments for the callback, or one tuple per callback.
        state : list
            One-element list holding the field, updated in place.
        verbose : bool
            Whether to update the progress bar.
        pbar : tqdm or None
            Progress bar, used when verbose.
        V : np.ndarray or None
            Potential, needed to re-derive the constants outside the medium.
        method : str
            Integration method ("split_step" or "RK4").
        dtype : np.dtype
            Complex dtype of the field.
        """
        step_fn = step_factory(delta_z)
        self._current_delta_z = delta_z
        self._step_in_force = delta_z
        z_prop = 0.0
        i = 0
        for n, end in enumerate(self._medium_ends(z)):
            if n > 0:
                self._leave_medium(V, dtype)
            while end - abs(z_prop) > STEP_ROUNDOFF * abs(delta_z):
                # Advanced before the callbacks, so the position they get is
                # the one the field they are handed is actually at.
                taken = self._advance(
                    step_fn, step_factory, delta_z, end - abs(z_prop), method, dtype
                )
                z_prop += taken
                if callback is not None:
                    # The callbacks see the step just taken, cut or whole:
                    # DDGPE's pump and noise integrate over it.
                    self._current_delta_z = taken
                    requested = self._run_callbacks(
                        callback, callback_args, state[0], z_prop, i
                    )
                    if requested is not None and abs(requested) != abs(delta_z):
                        delta_z = self._adopt_step(requested, method, dtype)
                        step_fn = step_factory(delta_z)
                    else:
                        self._current_delta_z = delta_z
                i += 1
                if verbose:
                    # Clamped: rounding can put z_prop a hair past z, which
                    # tqdm reports as a fraction outside [0, 1].
                    pbar.n = min(100.0, abs(z_prop) / z * 100)
                    pbar.refresh()

    def _advance(self, step_fn, step_factory, delta_z, left, method, dtype):
        """Take one step, or a cut one if a whole step would pass the end.

        Parameters
        ----------
        step_fn : callable
            The whole step, built for delta_z.
        step_factory : callable
            Builds a step of another size, for the cut one.
        delta_z : float
            Step in force, negative for a backward run.
        left : float
            Distance to the end of the segment, positive.
        method : str
            Integration method ("split_step" or "RK4").
        dtype : np.dtype
            Complex dtype of the field.

        Returns
        -------
        float
            Signed distance covered.
        """
        if abs(delta_z) > left + STEP_ROUNDOFF * abs(delta_z):
            cut = math.copysign(left, delta_z)
            self._take_partial_step(step_factory, cut, method, dtype)
            return cut
        step_fn()
        return delta_z

    def _adopt_step(self, requested, method, dtype):
        """Make the step a callback asked for the one in force.

        Parameters
        ----------
        requested : float
            Step the callback returned. Only its size is used.
        method : str
            Integration method ("split_step" or "RK4").
        dtype : np.dtype
            Complex dtype of the field.

        Returns
        -------
        float
            The step now in force, signed like the run.

        Raises
        ------
        TypeError
            If the step is complex.
        ValueError
            If the step is zero or not finite.
        """
        # The direction is the caller's: the adaptive callbacks return a size,
        # and a backward run that took it as given turned forward.
        self._check_step(requested)
        requested = math.copysign(abs(requested), self._current_delta_z)
        self._current_delta_z = requested
        self._step_in_force = requested
        # RK4's propagator does not depend on the step.
        if method != "RK4":
            self.propagator = self._build_propagator(dtype, requested)
            self._update_propagator_fft()
        return requested

    def _run_callbacks(self, callback, callback_args, A, z, i):
        """Dispatch user callbacks at each solver step.

        Returns
        -------
        float or None
            A new step, if a callback asked for one by returning it. Callbacks
            that return nothing leave the step alone, which is all of them
            except the adaptive ones.
        """
        requested = None
        if isinstance(callback, Callable):
            requested = callback(self, A, z, i, *callback_args)
        elif isinstance(callback, list) and isinstance(callback[0], Callable):
            # Left at its default, the arguments are none for every callback,
            # as for a single one; a list given must match in length.
            per_callback = callback_args if callback_args else [()] * len(callback)
            for c, ca in zip(callback, per_callback, strict=True):
                got = c(self, A, z, i, *ca)
                if got is not None:
                    requested = got
        else:
            raise ValueError("callbacks should be a callable or a list of callables")
        return requested

    def _medium_ends(self, z: float) -> tuple[float, ...]:
        """Return where each segment of a run over z ends.

        Propagation past the medium length L continues outside it, so a run
        longer than L is two segments, to L and on to z. Solvers that do not
        model a finite medium leave L at 0, which disables the cutoff entirely
        (GPE passes L=0, so every step stays in the medium; DDGPE passes its
        duration T, so a run longer than T is cut there like any other).

        Parameters
        ----------
        z : float
            Total propagation distance.

        Returns
        -------
        tuple of float
            ``(L, z)`` if the run leaves the medium, else ``(z,)``.
        """
        if 0 < self.L < z:
            return (self.L, z)
        return (z,)

    @staticmethod
    def _whole_steps(distance: float, delta_z: float) -> tuple[int, float]:
        """Split a distance into whole steps and a signed leftover.

        Whole steps, then the distance left over. Taking ceil(distance/dz)
        whole steps instead overshoots by up to a step, and the medium
        imprints its phase over the excess.

        Parameters
        ----------
        distance : float
            Distance to cover, positive.
        delta_z : float
            Step, negative for a backward run.

        Returns
        -------
        tuple of (int, float)
            Number of whole steps, and the leftover signed like delta_z, so a
            backward run takes its last partial step backward too. 0.0 when
            the leftover is round-off.
        """
        step = abs(delta_z)
        n_steps = int(distance / step + STEP_ROUNDOFF)
        remainder = distance - n_steps * step
        if remainder < STEP_ROUNDOFF * step:
            remainder = 0.0
        return n_steps, math.copysign(remainder, delta_z)

    def _leave_medium(self, V: np.ndarray | None, field_dtype: np.dtype) -> None:
        """Zero what the medium provides and re-derive the step constants.

        Parameters
        ----------
        V : np.ndarray or None
            Potential field, needed to re-derive the scaled potential.
        field_dtype : np.dtype
            Complex dtype of the field, which fixes the width of the constants
            being re-derived.
        """
        for attr in self._medium_attrs:
            setattr(self, attr, 0)
        self._precompute_step_constants(V, field_dtype)

    @staticmethod
    def _check_distance_and_step(z: float, delta_z: float | None) -> None:
        """Reject a distance or a step the propagation loop cannot take.

        Parameters
        ----------
        z : float
            Propagation distance.
        delta_z : float or None
            Step, or None for the solver to derive one.

        Raises
        ------
        TypeError
            If delta_z is complex.
        ValueError
            If delta_z is zero or not finite, or if z is negative or not
            finite. A backward run is a negative delta_z over a positive z.
        """
        if delta_z is not None:
            NLSE._check_step(delta_z)
        if not math.isfinite(z):
            raise ValueError(f"z={z!r} is not a finite distance.")
        if z < 0:
            raise ValueError(
                f"z={z!r} is negative: pass a positive distance and a negative "
                "delta_z to propagate backwards."
            )

    @staticmethod
    def _check_step(delta_z: float) -> None:
        """Reject a step the propagation loop cannot take.

        Parameters
        ----------
        delta_z : float
            Step, asked of out_field or returned by a callback.

        Raises
        ------
        TypeError
            If delta_z is complex. Imaginary time would need the step limits,
            the propagator cache and the kernels to take a complex step, and
            none of them do.
        ValueError
            If delta_z is zero or not finite. A zero step never reaches the
            end of the run, and a NaN one ends it at once.
        """
        if np.iscomplexobj(delta_z):
            raise TypeError(
                f"delta_z={delta_z!r} is complex: imaginary-time evolution is "
                "not supported. Pass a real step; a negative one propagates "
                "backwards."
            )
        if delta_z == 0 or not math.isfinite(delta_z):
            raise ValueError(
                f"delta_z={delta_z!r} is not a step: it must be finite and not zero."
            )

    # Plotting helpers.
    def _to_plot_array(self, A_plot: np.ndarray, target_ndim: int) -> np.ndarray:
        """Reduce dimensions and convert to numpy for plotting."""
        while A_plot.ndim > target_ndim:
            A_plot = A_plot[0]
        if not isinstance(A_plot, np.ndarray):
            A_plot = self._backend.to_numpy(A_plot)
        return A_plot

    def _plot_title(self, z: float) -> str:
        """Return the figure title, in this solver's axis and units."""
        return (
            rf"Field at ${self._plot_axis_symbol}$ = "
            rf"{z:{self._plot_axis_format}} {self._plot_axis_unit}"
        )
