#!/usr/bin/env python3
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import contextlib
import multiprocessing
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import tqdm
from scipy import special
from scipy.constants import c, epsilon_0

from ..backends import Backend, get_backend
from ..utils import (
    __BACKEND__,
    __CUPY_AVAILABLE__,
    __MLX_AVAILABLE__,
    __PYOPENCL_AVAILABLE__,
)

if __CUPY_AVAILABLE__:
    pass  # type: ignore[import-not-found]

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()

# Explicit RK4 is stable for |lambda * dz| up to ~2*sqrt(2) along the
# imaginary axis. Every solver's step limit is derived from it.
RK4_STABILITY_RADIUS = 2.83

# Phase in radians the default step imprints per step. The limits are
# ceilings (pi for split-step aliasing, 2.83 for RK4 stability); this is where
# a default sits under them. Measured: RK4 is at its accuracy floor by 0.15
# and gains nothing below, and split-step's discretisation error stays under
# the complex64 round-off floor across three decades of step size.
DEFAULT_PHASE_PER_STEP = 0.1

# Fewest steps a default may take over the requested distance, so that a run
# is something a callback can sample and a plot can show rather than one jump.
DEFAULT_MIN_STEPS = 10


class NLSE:
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

    # Step currently in force, for callbacks that need it -- callbacks receive
    # (simu, A, z, i, *args) and have no other way to see it. Written by the
    # propagation loop, meaningless outside one, and not a way to set the step:
    # a callback changes it by *returning* a new one.
    _current_delta_z: float | complex | None = None

    # Set per run rather than derived from the parameters, so they are None
    # until a run sets them up. Declared here so reading one before that is a
    # plain attribute access rather than a getattr with a default at each site.
    _V_scaled: Any = None
    _propagator_fft: Any = None

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
            Non local length in m.
            The non-local kernel is the instantiated as a Bessel function
            to model a diffusive non-locality stored in the nl_profile
            attribute.
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
        self.propagator = None
        self.plans = None
        self.nl_length = self._resolved_nl_length(nl_length)
        if self.nl_length > 0 and self._backend.convolution is None:
            raise NotImplementedError(
                f"Non-local interaction (nl_length > 0) is not supported "
                f"with the {self._backend.name} backend. Use CPU or CUPY instead."
            )
        if self.nl_length > 0:
            d = self.nl_length // self.delta_X
            x = np.arange(-3 * d, 3 * d + 1)
            y = np.arange(-3 * d, 3 * d + 1)
            XX, YY = np.meshgrid(x, y)
            R = np.hypot(XX, YY)
            self.nl_profile = special.kn(0, R / d)
            self.nl_profile[
                self.nl_profile.shape[0] // 2, self.nl_profile.shape[1] // 2
            ] = np.nanmax(self.nl_profile[np.logical_not(np.isinf(self.nl_profile))])
            self.nl_profile /= self.nl_profile.sum()
            self.nl_profile = self.nl_profile.astype(np.float32)
        else:
            self.nl_profile = np.ones((self.NY, self.NX), dtype=np.float32)

        # Pre-compute normalization factors to avoid runtime upcasting
        self._norm_grid_factor = np.float32(self.delta_X * self.delta_Y)
        self._norm_constant = np.float32(c * epsilon_0 / 2)

        # Propagator cache for repeated calls with same parameters
        self._propagator_cache: dict[tuple, np.ndarray] = {}

    # The public surface: a solver is built, asked to propagate, and plotted.
    # split_step and split_step_RK4 are the single-step primitives out_field
    # drives; they are public because a caller may want one step at a time.
    @property
    def backend(self) -> str:
        """Return the backend used for the simulation."""
        return self._backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the backend for the simulation."""
        self._backend = get_backend(value, grid_size=(self.NX, self.NY))

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
        delta_z: float | complex | None = None,
        plot: bool = False,
        precision: str = "single",
        method: str = "split_step",
        verbose: bool = True,
        normalize: bool = True,
        callback: list[Callable] | Callable | None = None,
        callback_args: list[tuple] | tuple = (),
    ) -> np.ndarray:
        """Propagate the field at a distance z.

        This function propagates the field E_in over a distance z by
        calling the split step function in a loop.

        This function supports imaginary time evolution provided you pass
        a complex delta_z.
        This allows to find the ground state of the system.
        Warning: this is still experimental !

        Parameters
        ----------
        E_in : np.ndarray
            Normalized input field (between 0 and 1).
        z : float
            Propagation distance in m.
        delta_z : float or complex, optional
            Step to propagate with. Defaults to None, meaning the solver
            derives one from the field: a step that imprints a fixed phase
            per step, against the same energy rates the stability and
            accuracy limits are built from. A step given here is used as
            given, capped only where it would leave the method's region of
            convergence. Pass a complex value for imaginary time evolution.
        plot : bool, optional
            Plots the results. Defaults to False.
        precision : str, optional
            Order of the split step, *not* the floating-point width. Does a
            "single" or a "double" application of the nonlinear term, giving
            O(dz) or O(dz^3) accuracy. Defaults to "single".

            The floating-point width comes from ``E_in``: pass a complex128
            field for float64 arithmetic, on a device that supports it. The
            propagator is built to match, because the kernels select their
            precision from the field and then read the propagator with it.
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
        callback_args : tuple, optional
            Additional arguments for the
            callback function.

        Returns
        -------
        np.ndarray
            Propagated field in proper units V/m.
        """
        # Backward compat: precision="RK4" maps to method="RK4"
        if precision == "RK4":
            method = "RK4"
            precision = "single"

        assert (
            E_in.shape[self._last_axes[0] :] == self.XX.shape[self._last_axes[0] :]
        ), "Shape mismatch"
        assert E_in.dtype in [
            np.complex64,
            np.complex128,
        ], "Type mismatch, E_in should be complex64 or complex128"
        self._check_batch_support(E_in)
        field_dtype = self._field_dtype(E_in)
        # Rebuilt below once delta_z is settled; drop the previous run's so
        # the transfer does not carry it.
        self.propagator = None
        with self._arrays_on_device(field_dtype):
            V = self.V
            A, A_sq = self._prepare_output_array(E_in, normalize)
            self.plans = self._build_fft_plan(A)
            self._allocate_rk4_buffers(A, method)
            self._precompute_step_constants(V, precision)

            # Settle the step before building anything that depends on it.
            if delta_z is None:
                delta_z = self._default_delta_z(A, method, z)
            delta_z = self._capped_delta_z(delta_z, A, method)

            # The propagator depends on delta_z, k, the grid and the precision,
            # any of which may have changed since the last run. _build_propagator
            # is cache-backed, so an unchanged configuration is not recomputed.
            if method == "RK4":
                self.propagator = self._build_propagator_rk4()
            else:
                self.propagator = self._build_propagator(field_dtype, delta_z)
            self._send_propagator_to_gpu(field_dtype)
            if verbose:
                pbar = tqdm.tqdm(
                    total=100,
                    position=4,
                    desc="Propagation",
                    leave=False,
                    unit="%",
                    unit_scale=True,
                )
            if type(delta_z) is complex:
                print("Warning: imaginary time evolution !")

            with self._backend.timed() as timing:
                A = self._run_propagation(
                    A,
                    A_sq,
                    V,
                    z,
                    delta_z,
                    precision,
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
        precision: str = "single",
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
        precision : str, optional
            Order of the split step: "single" applies the nonlinear step
            once, "double" splits it around the linear step. Not the
            floating-point width, which follows the field. Defaults to
            "single".

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

        # Fused fast path (CL, MLX). Not eligible with a nonlocal kernel,
        # which needs the convolution between |A|^2 and the nonlinear step.
        if self._backend.has_fused_split_step and self.nl_length == 0:
            dz = delta_z / 2 if precision == "double" else delta_z
            prop_fft = self._propagator_fft
            return kernels.split_step_fused(
                A,
                prop_fft if prop_fft is not None else propagator,
                V_scaled,
                dz,
                alpha_half,
                g,
                Isat_conv,
                precision,
                plans[0],
                unnorm_ifft=(prop_fft is not None),
            )

        # First half-step (only for precision == "double")
        if precision == "double":
            if self.nl_length > 0:
                # Need A_sq for convolution — must keep separate
                A_sq = kernels.square_mod(A, A_sq)
                A_sq[:] = self._backend.convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
                if V is None:
                    A = kernels.nl_prop_without_V(
                        A,
                        A_sq,
                        delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
                    )
                else:
                    A = kernels.nl_prop(
                        A,
                        A_sq,
                        delta_z / 2,
                        alpha_half,
                        V_scaled,
                        g,
                        Isat_conv,
                    )
            else:
                if V is None:
                    A = kernels.square_mod_nl_prop(
                        A,
                        delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
                    )
                else:
                    A = kernels.square_mod_nl_prop_v(
                        A,
                        V_scaled,
                        delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
                    )

        # Linear propagation in Fourier domain
        A = self._apply_linear_step(A, propagator, plans)

        # Second half-step (always executed)
        # Determine step size based on precision mode
        dz_step = delta_z / 2 if precision == "double" else delta_z

        if self.nl_length > 0:
            # Can't use fused kernel with convolution
            A_sq = kernels.square_mod(A, A_sq)
            A_sq[:] = self._backend.convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
            if V is None:
                A = kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    dz_step,
                    alpha_half,
                    g,
                    Isat_conv,
                )
            else:
                A = kernels.nl_prop(
                    A,
                    A_sq,
                    dz_step,
                    alpha_half,
                    V_scaled,
                    g,
                    Isat_conv,
                )
        else:
            if V is None:
                A = kernels.square_mod_nl_prop(
                    A,
                    dz_step,
                    alpha_half,
                    g,
                    Isat_conv,
                )
            else:
                A = kernels.square_mod_nl_prop_v(
                    A,
                    V_scaled,
                    dz_step,
                    alpha_half,
                    g,
                    Isat_conv,
                )
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

        # Whole-step fused fast path (MLX), single component only
        if self._backend.has_fused_rk4_step and self.nl_length == 0 and A.ndim == 2:
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
        acc = kernels.rk4_accumulate(acc, 1.0, k)  # acc = k1+2*k2+2*k3+k4

        # Final update: A += h/6 * acc
        A = kernels.rk4_accumulate(A, h / 6, acc)

        self._rk4_k = k
        self._rk4_A_tmp = A_tmp
        self._rk4_acc = acc
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
        im0 = ax[0].imshow(rho, extent=ext_real)
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label=self._plot_density_label)
        im1 = ax[1].imshow(
            phi,
            extent=ext_real,
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
            cmap="nipy_spectral",
        )
        ax[2].set_title("Fourier space")
        ax[2].set_xlabel(r"$k_x$ ($mm^{-1}$)")
        ax[2].set_ylabel(r"$k_y$ ($mm^{-1}$)")
        fig.colorbar(im2, ax=ax[2], shrink=0.6, label="Intensity (a.u.)")
        plt.show()

    # Construction.
    def _resolved_nl_length(self, nl_length: float) -> float:
        """Return the non-local length, or 0 where the grid cannot resolve it.

        The non-local kernel spans ``nl_length // delta_X`` cells, so on a grid
        coarser than ``nl_length`` it is a single point — the identity, which
        is a local run that still pays for a convolution on every step. Fall
        back to the local path, and say so, rather than quietly charge for a
        non-locality that is not there.

        Parameters
        ----------
        nl_length : float
            Non-local length in m, as given to the constructor.

        Returns
        -------
        float
            The same value, or 0 if the grid does not resolve it.
        """
        if nl_length <= 0 or nl_length // self.delta_X >= 1:
            return nl_length
        warnings.warn(
            f"nl_length={nl_length:.3g} m is below one grid cell "
            f"({self.delta_X:.3g} m), so the non-local kernel would be a "
            f"single point and the propagation local anyway. Running local. "
            f"Refine the grid or raise nl_length to model non-locality.",
            stacklevel=3,
        )
        return 0

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

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for the split-step propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.k),
        )

    def _compute_propagator(self, dtype: np.dtype, delta_z: float) -> np.ndarray:
        """Compute the linear propagation matrix (no caching)."""
        return np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * delta_z,
            dtype=dtype,
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for the RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw dispersion operator for RK4 (no caching)."""
        return (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k).astype(np.complex64)

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
            return self._propagator_cache[cache_key]
        propagator = self._compute_propagator(dtype, delta_z)
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _build_propagator_rk4(self) -> np.ndarray:
        """Build raw dispersion operator for RK4 with caching.

        Returns
        -------
        np.ndarray
            The raw dispersion operator.
        """
        cache_key = self._propagator_rk4_cache_key()
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        propagator = self._compute_propagator_rk4()
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _update_propagator_fft(self) -> None:
        """Derive the pre-normalized propagator used by fused linear steps.

        Absorbs the 1/N of the inverse FFT into the propagator so that
        ``linear_step`` can skip a separate normalization multiply. Only
        CUPY (cuFFT) and CL (VkFFT norm=0) expose an unnormalized IFFT.

        Must be called whenever ``self.propagator`` changes, otherwise the
        fused kernels keep using a propagator derived from the previous one.
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
        else:
            self._propagator_fft = None

    # Choosing the step. The rates are expectation values over the field, and
    # everything else here is derived from them: the default, the two limits,
    # and the cap applied to a step the caller passed.
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
        else:
            rate = rates["potential"] + rates["interaction"]
        # A rate of zero means nothing rotates the phase, so only the bound
        # below decides.
        dz = DEFAULT_PHASE_PER_STEP / rate if rate > 0 else np.inf

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
                E_00 = (self._norm_target / integral) ** 0.5
                result = (E_00.T * E_in_np.T).T.astype(E_in_np.dtype)
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
                E_00 = (target / integral) ** 0.5
                A[:] = (E_00.T * E_in.T).T
        else:
            A[:] = E_in
        return A, A_sq

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
        # Pass the actual array for in-place FFT optimization
        # CPU backend will handle wisdom loading/saving internally
        plan = self._backend.build_fft(A.shape, self._last_axes, A.dtype, array=A)
        return plan

    def _allocate_rk4_buffers(self, A: np.ndarray, method: str) -> None:
        """Pre-allocate scratch buffers for the RK4 stepper."""
        if method == "RK4":
            dtype = np.complex64
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

    def _precompute_step_constants(
        self, V: np.ndarray | None, precision: str = "single"
    ) -> None:
        """Pre-compute constants that are invariant across propagation steps.

        Parameters
        ----------
        V : np.ndarray or None
            Potential field.
        precision : str
            Order of the split step ("single" or "double"), used here only
            to pick the width of the precomputed scalar constants.
        """
        fp = np.float32 if precision == "single" else np.float64
        for name, value in self._step_constants().items():
            # A batched parameter is an array and stays one; only scalars are
            # narrowed, so the kernels get a value of the right width.
            setattr(
                self,
                name,
                fp(value) if isinstance(value, (int, float, np.floating)) else value,
            )
        self._V_scaled = None if V is None else V * self._k_half
        self._update_propagator_fft()

    # Moving arrays on and off the device.
    #
    # Arrays are always transferred; params only if they are np.ndarray, i.e.
    # batched. Subclasses extend these tuples with their own attributes.
    _gpu_array_attrs = ("V", "nl_profile", "propagator")
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

    @contextlib.contextmanager
    def _arrays_on_device(self, field_dtype: np.dtype) -> Iterator[None]:
        """Hold the solver's arrays on the device for the duration of a run.

        V, nl_profile, the propagator and any batched parameter are moved onto
        the device and put back afterwards, on the way out of a failed run as
        well as a finished one. Left there, they break the next run rather than
        the one that failed: from_numpy is handed a device array.

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
            if val is None:
                continue
            if attr == "V":
                val = np.ascontiguousarray(
                    val, dtype=self._potential_dtype(val, field_dtype)
                )
            setattr(self, attr, self._backend.from_numpy(val))
        if not self._backend.broadcasts_parameters_natively:
            # The kernels take one simulation's scalar value per launch, so a
            # batched parameter is picked apart on the host and never reaches
            # the device as an array.
            return
        for attr in self._gpu_param_attrs:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, self._backend.from_numpy(val))

    def _send_propagator_to_gpu(self, field_dtype: np.dtype) -> None:
        """Move the freshly built propagator onto the device.

        Separate from _send_arrays_to_gpu because the propagator is built
        after delta_z is settled, which is after the other arrays have gone.

        Parameters
        ----------
        field_dtype : np.dtype
            Complex dtype of the field, which the propagator already matches.
        """
        if self._backend.is_device_backend and self.propagator is not None:
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
            setattr(self, attr, self._backend.to_numpy(val))
        if not self._backend.broadcasts_parameters_natively:
            return
        for attr in self._gpu_param_attrs:
            val = getattr(self, attr)
            if not isinstance(val, (int, float)):
                setattr(self, attr, self._backend.to_numpy(val))

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
        kernels = self._backend.kernels
        if self._backend.has_linear_step:
            prop_fft = self._propagator_fft
            if prop_fft is not None:
                return kernels.linear_step(A, prop_fft, plans[0], unnorm_ifft=True)
            return kernels.linear_step(A, propagator, plans[0])
        A = self._backend.fft(A, plans)
        A = kernels.apply_propagator(A, propagator)
        A = self._backend.ifft(A, plans)
        return A

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
            prop_fft = self._propagator_fft
            return kernels.rk4_rhs_fused(
                A_in,
                k,
                V_scaled,
                prop_fft if prop_fft is not None else propagator,
                plans[0],
                alpha_half,
                g,
                Isat_conv,
                unnorm_ifft=(prop_fft is not None),
            )

        k[:] = A_in
        k = self._apply_linear_step(k, propagator, plans)

        if self.nl_length > 0:
            A_sq = (A_in * A_in.conj()).real
            A_sq[:] = self._backend.convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
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

    def _take_partial_step(self, step_factory, remainder, method, precision, dtype):
        """Cover the distance left over after the whole steps.

        The propagator is built from the step, so a shorter one needs its own.
        Cache-backed, and a run takes at most one of these.

        Parameters
        ----------
        step_factory : callable
            Builds the per-step closure for a given step size.
        remainder : float
            Distance still to cover.
        method : str
            Integration method ("split_step" or "RK4").
        precision : str
            Order of the split step.
        dtype : np.dtype
            Complex dtype of the field.
        """
        # Swapped rather than rebuilt afterwards: the solver should describe
        # the run it just did, not the sliver at the end of it.
        in_force = self._current_delta_z
        saved = (self.propagator, self._propagator_fft)
        if method != "RK4":
            self.propagator = self._build_propagator(dtype, remainder)
            self._send_propagator_to_gpu(dtype)
        self._current_delta_z = remainder
        try:
            step_factory(remainder)()
        finally:
            self._current_delta_z = in_force
            self.propagator, self._propagator_fft = saved

    # The propagation loop, and what it switches off along the way.
    #
    # _nonlinearity_attrs holds the nonlinear coupling: zeroed once the
    # propagation leaves the medium, then re-derived by
    # _precompute_step_constants. Subclasses extend it with their own.
    _nonlinearity_attrs = ("n2",)

    def _run_propagation(
        self,
        A,
        A_sq,
        V,
        z,
        delta_z,
        precision,
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
        # Whole steps, then whatever distance is left over. Taking ceil(z/dz)
        # whole steps instead propagates past z by up to a full step, and the
        # error that leaves is the phase the medium imprints over the excess --
        # which is not small: a step derived from the physics rarely divides z,
        # and floating point makes even an exact division fragile (z/dz for 237
        # steps comes to 237.00000000000003, so ceil asks for 238).
        n_steps = int(z / abs(delta_z))
        remainder = z - n_steps * abs(delta_z)
        # Anything below this is round-off in z/dz rather than real distance.
        if remainder < 1e-9 * abs(delta_z):
            remainder = 0.0
        n_steps_nl = self._nonlinear_step_count(z, n_steps, delta_z)

        # Use a mutable container so the closure can update A for functional
        # backends (MLX) where kernels return new arrays.
        state = [A]

        def _make_step(dz):
            """Build the per-step closure for a given step size.

            Rebuilt when an adaptive callback changes the step, so the step
            and the propagator it was built from cannot come apart.
            """
            if method == "RK4":

                def _step():
                    state[0] = self.split_step_RK4(
                        state[0], V, self.propagator, self.plans, dz
                    )
            else:

                def _step():
                    state[0] = self.split_step(
                        state[0], A_sq, V, self.propagator, self.plans, dz, precision
                    )

            return _step

        is_scalar_n2 = isinstance(self.n2, (int, float, np.floating))
        can_use_fast_loop = (
            callback is None
            and is_scalar_n2
            and self.nl_length == 0
            and not isinstance(delta_z, complex)
        )
        # Zeroing the nonlinearity mutates self, so restore it afterwards.
        saved = {attr: getattr(self, attr) for attr in self._nonlinearity_attrs}
        try:
            if can_use_fast_loop:
                # Two segments rather than a mid-loop switch: the constants are
                # baked into the CUDA graph that execute_loop captures, so each
                # segment needs its own capture.
                step = _make_step(delta_z)
                self._current_delta_z = delta_z
                self._backend.execute_loop(step, n_steps_nl)
                if n_steps > n_steps_nl:
                    self._disable_nonlinearity(V, precision)
                    self._backend.execute_loop(step, n_steps - n_steps_nl)
                if remainder:
                    self._take_partial_step(
                        _make_step,
                        remainder,
                        method,
                        precision,
                        self._field_dtype(state[0]),
                    )
            else:
                self._loop_with_callbacks(
                    _make_step,
                    z,
                    delta_z,
                    remainder,
                    callback,
                    callback_args,
                    state,
                    verbose,
                    pbar,
                    z_switch=self.L if n_steps_nl < n_steps else None,
                    V=V,
                    precision=precision,
                    method=method,
                    dtype=self._field_dtype(A),
                )
        finally:
            for attr, value in saved.items():
                setattr(self, attr, value)
        return state[0]

    def _loop_with_callbacks(
        self,
        step_factory,
        z,
        delta_z,
        remainder,
        callback,
        callback_args,
        state,
        verbose,
        pbar,
        z_switch=None,
        V=None,
        precision="single",
        method="split_step",
        dtype=np.complex64,
    ):
        """Run propagation loop with per-step callbacks.

        A callback may return a new step to use from here on. The propagator
        is rebuilt to match before the next step: it is built from delta_z, so
        changing one without the other silently propagates the linear part by
        the wrong distance.
        """
        step_fn = step_factory(delta_z)
        self._current_delta_z = delta_z
        z_prop = 0.0
        i = 0
        switched = False
        # Strictly less than the whole-step total: the leftover is taken below,
        # at its own size, so the run lands on z rather than past it.
        whole = z - remainder
        while abs(z_prop) < whole - 1e-12 * z:
            if z_switch is not None and not switched and abs(z_prop) >= z_switch:
                self._disable_nonlinearity(V, precision)
                switched = True
            step_fn()
            # Advance before the callbacks, so the position they get is the one
            # the field they are handed is actually at.
            z_prop += delta_z
            if callback is not None:
                requested = self._run_callbacks(
                    callback, callback_args, state[0], z_prop, i
                )
                if requested is not None and requested != delta_z:
                    delta_z = requested
                    self._current_delta_z = delta_z
                    if method != "RK4":
                        self.propagator = self._build_propagator(dtype, delta_z)
                        if self._backend.is_device_backend:
                            self.propagator = self._backend.from_numpy(self.propagator)
                        self._update_propagator_fft()
                    step_fn = step_factory(delta_z)
            i += 1
            if verbose:
                pbar.n = abs(z_prop) / z * 100
                pbar.refresh()
        if remainder:
            # An adaptive callback may have moved the step, so what is left is
            # measured from where the loop actually stopped, not from the
            # remainder computed before it ran.
            left = z - abs(z_prop)
            if left > 1e-12 * z:
                self._take_partial_step(step_factory, left, method, precision, dtype)
                z_prop += left
                if callback is not None:
                    self._run_callbacks(callback, callback_args, state[0], z_prop, i)

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
            for c, ca in zip(callback, callback_args, strict=True):
                got = c(self, A, z, i, *ca)
                if got is not None:
                    requested = got
        else:
            raise ValueError("callbacks should be a callable or a list of callables")
        return requested

    def _nonlinear_step_count(self, z: float, n_steps: int, delta_z: float) -> int:
        """Return how many of the n_steps fall inside the nonlinear medium.

        Propagation past the medium length L continues linearly. Solvers that
        do not model a finite medium leave L at 0, which disables the cutoff
        entirely (GPE passes L=0, so every step stays nonlinear).

        Parameters
        ----------
        z : float
            Total propagation distance.
        n_steps : int
            Total number of steps for this run.
        delta_z : float
            Step size.

        Returns
        -------
        int
            Number of leading steps to run with the nonlinearity enabled.
        """
        if not self.L > 0 or z <= self.L:
            return n_steps
        return min(n_steps, int(np.ceil(self.L / abs(delta_z))))

    def _disable_nonlinearity(self, V: np.ndarray | None, precision: str) -> None:
        """Zero the nonlinear coupling and re-derive the step constants.

        Parameters
        ----------
        V : np.ndarray or None
            Potential field, needed to re-derive the scaled potential.
        precision : str
            "single" or "double".
        """
        for attr in self._nonlinearity_attrs:
            setattr(self, attr, 0)
        self._precompute_step_constants(V, precision)

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
