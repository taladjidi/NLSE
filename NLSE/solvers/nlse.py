#!/usr/bin/env python3
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import multiprocessing
import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import tqdm
from scipy import signal, special
from scipy.constants import c, epsilon_0

from ..backends import Backend, get_backend
from ..utils import (
    __BACKEND__,
    __CUPY_AVAILABLE__,
    __PYOPENCL_AVAILABLE__,
)

if __CUPY_AVAILABLE__:
    import cupy as cp
    import cupyx.scipy.signal as signal_cp  # type: ignore[import-not-found]

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()


class NLSE:
    """A class to solve NLSE."""

    __CUPY_AVAILABLE__ = __CUPY_AVAILABLE__
    __PYOPENCL_AVAILABLE__ = __PYOPENCL_AVAILABLE__

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
        # Setup backend-specific convolution
        if self._backend.name == "CUPY":
            self._convolution = signal_cp.oaconvolve
        elif self._backend.name == "CPU":
            self._convolution = signal.oaconvolve
        # CL backend doesn't have convolution yet
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
        if isinstance(window, float) or isinstance(window, int):
            self.window = [window, window]
        elif isinstance(window, tuple) or isinstance(window, list):
            self.window = window
        Dn = self.n2 * self.power / min(self.window) ** 2
        z_nl = 1 / (self.k * abs(Dn))
        if not isinstance(z_nl, (int, float)):
            # z_nl might be an array - extract scalar
            z_nl = float(np.min(z_nl))
        self.delta_z = 5e-3 * z_nl
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
        self.nl_length = nl_length
        if self.nl_length > 0 and self._backend.name in ["CL", "MLX"]:
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

    @property
    def backend(self) -> str:
        """Return the backend used for the simulation."""
        return self._backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the backend for the simulation."""
        self._backend = get_backend(value, grid_size=(self.NX, self.NY))
        # Setup backend-specific convolution
        if self._backend.name == "CUPY":
            self._convolution = signal_cp.oaconvolve
        elif self._backend.name == "CPU":
            self._convolution = signal.oaconvolve
        # CL backend doesn't have convolution yet

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Uses caching to avoid recomputing propagators with identical parameters.

        Parameters
        ----------
        precision : str
            "single" or "double" precision for the split step propagator.

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        # Create cache key from parameters that affect propagator
        cache_key = (self.NX, self.NY, float(self.delta_z), precision, float(self.k))

        # Return cached propagator if available
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]

        # Use appropriate dtype based on precision
        dtype = np.complex128 if precision == "double" else np.complex64
        propagator = np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z,
            dtype=dtype,
        )

        # Cache for future use
        self._propagator_cache[cache_key] = propagator
        return propagator

    def _build_propagator_rk4(self) -> np.ndarray:
        """Build raw dispersion operator for RK4 (no exp, no delta_z).

        Returns
        -------
        np.ndarray
            The raw dispersion operator.
        """
        cache_key = (self.NX, self.NY, "RK4", float(self.k))
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        propagator = (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k).astype(
            np.complex64
        )
        self._propagator_cache[cache_key] = propagator
        return propagator

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
            if self._backend.name in ["CL", "MLX"]:
                # CL/MLX: compute normalization on numpy then convert back.
                arr_np = self._backend.to_numpy(arr)
                E_in_np = self._backend.to_numpy(E_in)
                integral = np.sum(arr_np, axis=self._last_axes)
                integral = integral * self._norm_constant
                E_00 = (self.power / integral) ** 0.5
                result = (E_00.T * E_in_np.T).T.astype(E_in_np.dtype)
                A = self._backend.from_numpy(result)
            else:
                integral = np.sum(arr, axis=self._last_axes)
                integral = integral * self._norm_constant
                E_00 = (self.power / integral) ** 0.5
                A[:] = (E_00.T * E_in.T).T
        else:
            if self._backend.name == "MLX":
                A = E_in
            else:
                A[:] = E_in
        return A, A_sq

    def _send_arrays_to_gpu(self) -> None:
        """Send arrays to device using backend."""
        if self._backend.name in ["CUPY", "CL", "MLX"]:
            if self.V is not None:
                # Ensure float32 dtype for GPU backends
                self.V = self._backend.from_numpy(
                    np.ascontiguousarray(self.V, dtype=np.float32)
                )
            self.nl_profile = self._backend.from_numpy(self.nl_profile)
            self.propagator = self._backend.from_numpy(self.propagator)
            # for broadcasting of parameters in case they are
            # not already on the device
            if isinstance(self.power, np.ndarray):
                self.power = self._backend.from_numpy(self.power)
            if isinstance(self.n2, np.ndarray):
                self.n2 = self._backend.from_numpy(self.n2)
            if isinstance(self.alpha, np.ndarray):
                self.alpha = self._backend.from_numpy(self.alpha)
            if isinstance(self.I_sat, np.ndarray):
                self.I_sat = self._backend.from_numpy(self.I_sat)

    def _retrieve_arrays_from_gpu(self) -> None:
        """Retrieve arrays from device using backend."""
        if self._backend.name in ["CUPY", "CL", "MLX"]:
            if self.V is not None:
                self.V = self._backend.to_numpy(self.V)
            self.nl_profile = self._backend.to_numpy(self.nl_profile)
            self.propagator = self._backend.to_numpy(self.propagator)
            # Retrieve parameters if they were sent to device
            if not isinstance(self.power, (int, float)):
                self.power = self._backend.to_numpy(self.power)
            if not isinstance(self.n2, (int, float)):
                self.n2 = self._backend.to_numpy(self.n2)
            if not isinstance(self.alpha, (int, float)):
                self.alpha = self._backend.to_numpy(self.alpha)
            if not isinstance(self.I_sat, (int, float)):
                self.I_sat = self._backend.to_numpy(self.I_sat)

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
        if hasattr(kernels, "linear_step"):
            return kernels.linear_step(A, propagator, plans[0])
        A = self._backend.fft(A, plans)
        A = kernels.apply_propagator(A, propagator)
        A = self._backend.ifft(A, plans)
        return A

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
            Field to propagate.
        A_sq : np.ndarray
            Field modulus squared.
        V : np.ndarray
            Potential field (can be None).
        propagator : np.ndarray
            Propagator matrix.
        plans : list
            List of FFT plan objects from backend.
        precision : str, optional
            Single or double application of
            the nonlinear propagation step. Defaults to "single".

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        kernels = self._backend.kernels

        # MLX fused fast path (nl_length == 0 only)
        if hasattr(kernels, "split_step") and self.nl_length == 0:
            dz = self.delta_z / 2 if precision == "double" else self.delta_z
            return kernels.split_step(
                A,
                propagator,
                V,
                dz,
                self.alpha / 2,
                self.k / 2 * self.n2 * c * epsilon_0,
                self.k / 2,
                2 * self.I_sat / (epsilon_0 * c),
                precision,
                plans[0],
            )

        # First half-step (only for precision == "double")
        if precision == "double":
            if self.nl_length > 0:
                # Need A_sq for convolution — must keep separate
                A_sq = kernels.square_mod(A, A_sq)
                A_sq[:] = self._convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
                if V is None:
                    A = kernels.nl_prop_without_V(
                        A,
                        A_sq,
                        self.delta_z / 2,
                        self.alpha / 2,
                        self.k / 2 * self.n2 * c * epsilon_0,
                        2 * self.I_sat / (epsilon_0 * c),
                    )
                else:
                    V_scaled = V * np.float32(self.k / 2)
                    A = kernels.nl_prop(
                        A,
                        A_sq,
                        self.delta_z / 2,
                        self.alpha / 2,
                        V_scaled,
                        self.k / 2 * self.n2 * c * epsilon_0,
                        2 * self.I_sat / (epsilon_0 * c),
                    )
            else:
                if V is None:
                    A = kernels.square_mod_nl_prop(
                        A,
                        self.delta_z / 2,
                        self.alpha / 2,
                        self.k / 2 * self.n2 * c * epsilon_0,
                        2 * self.I_sat / (epsilon_0 * c),
                    )
                else:
                    V_scaled = V * np.float32(self.k / 2)
                    A = kernels.square_mod_nl_prop_v(
                        A,
                        V_scaled,
                        self.delta_z / 2,
                        self.alpha / 2,
                        self.k / 2 * self.n2 * c * epsilon_0,
                        2 * self.I_sat / (epsilon_0 * c),
                    )

        # Linear propagation in Fourier domain
        A = self._apply_linear_step(A, propagator, plans)

        # Second half-step (always executed)
        # Determine step size based on precision mode
        dz_step = self.delta_z / 2 if precision == "double" else self.delta_z

        if self.nl_length > 0:
            # Can't use fused kernel with convolution
            A_sq = kernels.square_mod(A, A_sq)
            A_sq[:] = self._convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
            if V is None:
                A = kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    dz_step,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                A = kernels.nl_prop(
                    A,
                    A_sq,
                    dz_step,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        else:
            if V is None:
                A = kernels.square_mod_nl_prop(
                    A,
                    dz_step,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                V_scaled = V * np.float32(self.k / 2)
                A = kernels.square_mod_nl_prop_v(
                    A,
                    V_scaled,
                    dz_step,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
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
        if self._backend.name == "MLX":
            k = self._apply_linear_step(A_in, propagator, plans)
        else:
            k[:] = A_in
            k = self._apply_linear_step(k, propagator, plans)

        kernels = self._backend.kernels
        g = self.k / 2 * self.n2 * c * epsilon_0
        alpha_half = self.alpha / 2
        Isat_conv = 2 * self.I_sat / (epsilon_0 * c)

        if self.nl_length > 0:
            A_sq = (A_in * A_in.conj()).real
            A_sq[:] = self._convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
            if V is None:
                k = kernels.rk4_nl_rhs(k, A_in, A_sq, alpha_half, g, Isat_conv)
            else:
                V_scaled = V * np.float32(self.k / 2)
                k = kernels.rk4_nl_rhs_v(
                    k, A_in, A_sq, V_scaled, alpha_half, g, Isat_conv
                )
        else:
            if V is None:
                k = kernels.square_mod_rk4_nl_rhs(k, A_in, alpha_half, g, Isat_conv)
            else:
                V_scaled = V * np.float32(self.k / 2)
                k = kernels.square_mod_rk4_nl_rhs_v(
                    k, A_in, V_scaled, alpha_half, g, Isat_conv
                )
        return k

    def split_step_RK4(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
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

        Returns
        -------
        np.ndarray
            The propagated field.
        """
        kernels = self._backend.kernels

        # MLX fused fast path (nl_length == 0, single-component only)
        if hasattr(kernels, "split_step_rk4") and self.nl_length == 0 and A.ndim == 2:
            return kernels.split_step_rk4(
                A,
                propagator,
                V,
                self.delta_z,
                self.alpha / 2,
                self.k / 2 * self.n2 * c * epsilon_0,
                self.k / 2,
                2 * self.I_sat / (epsilon_0 * c),
                plans[0],
            )

        if not hasattr(self, "_rk4_k"):
            self._allocate_rk4_buffers(A, "RK4")
        k = self._rk4_k
        A_tmp = self._rk4_A_tmp
        acc = self._rk4_acc
        h = self.delta_z

        # Stage 1: k1 = f(A)
        k = self._RK4_rhs(A, k, V, propagator, plans)
        acc = kernels.rk4_axpy(acc, k, 0.0, k)  # acc = k (copy via axpy)
        A_tmp = kernels.rk4_axpy(A_tmp, A, h / 2, k)  # A_tmp = A + h/2*k1

        # Stage 2: k2 = f(A + h/2*k1)
        k = self._RK4_rhs(A_tmp, k, V, propagator, plans)
        acc = kernels.rk4_accumulate(acc, 2.0, k)  # acc = k1 + 2*k2
        A_tmp = kernels.rk4_axpy(A_tmp, A, h / 2, k)  # A_tmp = A + h/2*k2

        # Stage 3: k3 = f(A + h/2*k2)
        k = self._RK4_rhs(A_tmp, k, V, propagator, plans)
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

    def _allocate_rk4_buffers(self, A: np.ndarray, method: str) -> None:
        """Pre-allocate scratch buffers for the RK4 stepper."""
        if method == "RK4":
            dtype = np.complex64
            self._rk4_k = self._backend.allocate_field(A.shape, dtype)
            self._rk4_A_tmp = self._backend.allocate_field(A.shape, dtype)
            self._rk4_acc = self._backend.allocate_field(A.shape, dtype)

    def _run_callbacks(self, callback, callback_args, A, z, i):
        """Dispatch user callbacks at each solver step."""
        if isinstance(callback, Callable):
            callback(self, A, z, i, *callback_args)
        elif isinstance(callback, list) and isinstance(callback[0], Callable):
            for c, ca in zip(callback, callback_args, strict=True):
                c(self, A, z, i, *ca)
        else:
            raise ValueError("callbacks should be a callable or a list of callables")

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
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

        This function supports imaginary time evolution provided you set
        the delta_z to a complex number.
        This allows to find the ground state of the system.
        Warning: this is still experimental !

        Parameters
        ----------
        E_in : np.ndarray
            Normalized input field (between 0 and 1).
        z : float
            Propagation distance in m.
        plot : bool, optional
            Plots the results. Defaults to False.
        precision : str, optional
            Does a "double" or a "single" application
            of the nonlinear term. This leads to a dz (single) or dz^3
            (double) precision. Defaults to "single".
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
        # define propagator if not already done
        if self.propagator is None:
            if method == "RK4":
                self.propagator = self._build_propagator_rk4()
            else:
                self.propagator = self._build_propagator(precision=precision)
        if self._backend.name in ["CUPY", "CL", "MLX"]:
            self._send_arrays_to_gpu()
        V = self.V
        A, A_sq = self._prepare_output_array(E_in, normalize)
        self.plans = self._build_fft_plan(A)
        self._allocate_rk4_buffers(A, method)
        if verbose:
            pbar = tqdm.tqdm(
                total=100,
                position=4,
                desc="Propagation",
                leave=False,
                unit="%",
                unit_scale=True,
            )
        n2_old = self.n2
        if self._backend.name == "CUPY":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        z_prop = 0
        i = 0
        if type(self.delta_z) is complex:
            print("Warning: imaginary time evolution !")
        while abs(z_prop) < z:
            if z > self.L:
                self.n2 = 0
            if method == "RK4":
                A = self.split_step_RK4(A, V, self.propagator, self.plans)
            else:
                A = self.split_step(A, A_sq, V, self.propagator, self.plans, precision)

            if callback is not None:
                self._run_callbacks(callback, callback_args, A, z, i)
            z_prop += self.delta_z
            i += 1
            if verbose:
                pbar.n = abs(z_prop) / z * 100
                pbar.refresh()
        t_cpu = time.perf_counter() - t0
        if verbose:
            pbar.close()

        if self._backend.name == "CUPY":
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if self._backend.name == "CUPY":
                print(
                    f"\nTime spent to solve : {t_gpu * 1e-3} s (GPU) /"
                    f" {time.perf_counter() - t0} s (CPU)\n"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)\n")
        self.n2 = n2_old
        return_np_array = isinstance(E_in, np.ndarray)
        if self._backend.name in ["CUPY", "CL", "MLX"]:
            if return_np_array:
                A = self._backend.to_numpy(A)
            self._retrieve_arrays_from_gpu()

        if plot:
            self.plot_field(A, z)
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        # Convert to numpy if on device
        if not isinstance(A_plot, np.ndarray):
            A_plot = self._backend.to_numpy(A_plot)
        fig, ax = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
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
        rho = np.abs(A_plot) ** 2 * 1e-4 * c / 2 * epsilon_0
        phi = np.angle(A_plot)
        im_fft = np.abs(np.fft.fftshift(np.fft.fft2(A_plot)))
        im0 = ax[0].imshow(rho, extent=ext_real)
        ax[0].set_title("Intensity")
        ax[0].set_xlabel("x (mm)")
        ax[0].set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax[0], shrink=0.6, label=r"Intensity ($W/cm^2$)")
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
