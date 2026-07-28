#!/usr/bin/env python3
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import multiprocessing
import time
import warnings
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
        if isinstance(window, (list, tuple)):
            self.window = window
        else:
            self.window = [window, window]
        # Coerce through numpy so that n2 == 0 yields inf rather than raising,
        # and so batched (array) parameters work the same way.
        Dn = np.asarray(self.n2 * self.power / min(self.window) ** 2, dtype=float)
        with np.errstate(divide="ignore"):
            z_nl = float(np.min(1.0 / (self.k * np.abs(Dn))))
        if not np.isfinite(z_nl):
            # n2 == 0: the problem is linear and has no nonlinear length
            # scale. Fall back to the shorter of the diffraction length over
            # the window and the medium length, so the default step stays
            # finite and still resolves the propagation.
            z_nl = self.k * min(self.window) ** 2
            if self.L > 0:
                z_nl = min(z_nl, float(self.L))
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

    def _propagator_cache_key(self, precision: str) -> tuple:
        """Return cache key for the split-step propagator."""
        return (self.NX, self.NY, float(self.delta_z), precision, float(self.k))

    def _compute_propagator(self, precision: str) -> np.ndarray:
        """Compute the linear propagation matrix (no caching)."""
        dtype = np.complex128 if precision == "double" else np.complex64
        return np.exp(
            -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z,
            dtype=dtype,
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for the RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.k))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw dispersion operator for RK4 (no caching)."""
        return (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k).astype(np.complex64)

    def _rk4_max_dz(self) -> float:
        """Compute the maximum stable step size for explicit RK4.

        The RK4 stability region for purely imaginary eigenvalues has
        radius ~2.83. The largest eigenvalue of the dispersion operator
        is K_max^2 / (2*k). Returns the maximum dz that keeps the
        scheme within the stability region.
        """
        D_max = float(np.max(0.5 * (self.Kxx**2 + self.Kyy**2) / self.k))
        if D_max == 0:
            return np.inf
        return 2.83 / D_max

    def _split_step_max_dz(self, A: np.ndarray) -> float:
        """Compute the maximum step size for split-step accuracy.

        Ensure the intensity-dependent nonlinear phase per step stays
        below pi to avoid phase aliasing. Only the Kerr term
        ``g * |A|^2 * sat`` is considered because it varies with the
        field and causes splitting error. The potential V is applied
        exactly via the exponential and does not limit accuracy.

        Parameters
        ----------
        A : np.ndarray
            Normalized field (possibly on device).
        """
        A_np = self._backend.to_numpy(A)
        I_peak = float(np.max(np.abs(A_np) ** 2))
        g = abs(getattr(self, "_g", self.k / 2 * self.n2 * c * epsilon_0))
        Isat = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        nl_rate = g * I_peak / (1 + I_peak / Isat)
        if nl_rate == 0:
            return np.inf
        return np.pi / nl_rate

    def _enforce_step_limit(self, A: np.ndarray, method: str, precision: str) -> None:
        """Cap delta_z to the stability/accuracy limit for the chosen method.

        Parameters
        ----------
        A : np.ndarray
            Normalized field (possibly on device).
        method : str
            Integration method ("split_step" or "RK4").
        precision : str
            "single" or "double".
        """
        if method == "RK4":
            max_dz = self._rk4_max_dz()
            label = "RK4 stability"
        else:
            max_dz = self._split_step_max_dz(A)
            label = "split-step accuracy"
        if self.delta_z > max_dz:
            warnings.warn(
                f"delta_z={self.delta_z:.2e} exceeds {label} limit "
                f"({max_dz:.2e}). Reducing to {0.9 * max_dz:.2e}.",
                stacklevel=2,
            )
            self.delta_z = 0.9 * max_dz
            # Rebuild propagator (split_step propagator depends on dz)
            self.propagator = None
            if method == "RK4":
                self.propagator = self._build_propagator_rk4()
            else:
                self.propagator = self._build_propagator(precision=precision)
            # Send only the new propagator to device
            if self._backend.is_device_backend:
                self.propagator = self._backend.from_numpy(self.propagator)
            # The fused CUPY/CL linear step reads _propagator_fft in
            # preference to propagator, so it has to follow the rebuild.
            self._update_propagator_fft()

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix with caching.

        Parameters
        ----------
        precision : str
            "single" or "double" precision for the split step propagator.

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        cache_key = self._propagator_cache_key(precision)
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        propagator = self._compute_propagator(precision)
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

    # Attributes to send to GPU: arrays (always transferred) and params
    # (transferred only if they are np.ndarray, i.e. batched).
    # Subclasses extend these tuples to add their own attributes.
    _gpu_array_attrs = ("V", "nl_profile", "propagator")
    _gpu_param_attrs = ("power", "n2", "alpha", "I_sat")

    def _send_arrays_to_gpu(self) -> None:
        """Send arrays to device using backend."""
        if not self._backend.is_device_backend:
            return
        for attr in self._gpu_array_attrs:
            val = getattr(self, attr, None)
            if val is None:
                continue
            if attr == "V":
                val = np.ascontiguousarray(val, dtype=np.float32)
            setattr(self, attr, self._backend.from_numpy(val))
        for attr in self._gpu_param_attrs:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, self._backend.from_numpy(val))

    def _retrieve_arrays_from_gpu(self) -> None:
        """Retrieve arrays from device using backend."""
        if not self._backend.is_device_backend:
            return
        for attr in self._gpu_array_attrs:
            val = getattr(self, attr, None)
            if val is None:
                continue
            setattr(self, attr, self._backend.to_numpy(val))
        for attr in self._gpu_param_attrs:
            val = getattr(self, attr)
            if not isinstance(val, (int, float)):
                setattr(self, attr, self._backend.to_numpy(val))

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
            prop_fft = getattr(self, "_propagator_fft", None)
            if prop_fft is not None:
                return kernels.linear_step(A, prop_fft, plans[0], unnorm_ifft=True)
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
        # Use pre-computed constants (set in out_field), with fallbacks
        # for direct split_step calls outside out_field.
        alpha_half = getattr(self, "_alpha_half", self.alpha / 2)
        g = getattr(self, "_g", self.k / 2 * self.n2 * c * epsilon_0)
        Isat_conv = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        k_half = getattr(self, "_k_half", np.float32(self.k / 2))
        V_scaled = getattr(self, "_V_scaled", None)
        if V_scaled is None and V is not None:
            V_scaled = V * k_half

        # Fused fast path (CL, MLX). Not eligible with a nonlocal kernel,
        # which needs the convolution between |A|^2 and the nonlinear step.
        if self._backend.has_fused_split_step and self.nl_length == 0:
            dz = self.delta_z / 2 if precision == "double" else self.delta_z
            prop_fft = getattr(self, "_propagator_fft", None)
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
                A_sq[:] = self._convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
                if V is None:
                    A = kernels.nl_prop_without_V(
                        A,
                        A_sq,
                        self.delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
                    )
                else:
                    A = kernels.nl_prop(
                        A,
                        A_sq,
                        self.delta_z / 2,
                        alpha_half,
                        V_scaled,
                        g,
                        Isat_conv,
                    )
            else:
                if V is None:
                    A = kernels.square_mod_nl_prop(
                        A,
                        self.delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
                    )
                else:
                    A = kernels.square_mod_nl_prop_v(
                        A,
                        V_scaled,
                        self.delta_z / 2,
                        alpha_half,
                        g,
                        Isat_conv,
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
        alpha_half = getattr(self, "_alpha_half", self.alpha / 2)
        g = getattr(self, "_g", self.k / 2 * self.n2 * c * epsilon_0)
        Isat_conv = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
        k_half = getattr(self, "_k_half", np.float32(self.k / 2))
        V_scaled = getattr(self, "_V_scaled", None)
        if V_scaled is None and V is not None:
            V_scaled = V * k_half

        # Fused fast path: out-of-place FFT eliminates the buffer copy
        if self._backend.has_fused_rk4_rhs and self.nl_length == 0:
            prop_fft = getattr(self, "_propagator_fft", None)
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

        if self._backend.name == "MLX":
            k = self._apply_linear_step(A_in, propagator, plans)
        else:
            k[:] = A_in
            k = self._apply_linear_step(k, propagator, plans)

        if self.nl_length > 0:
            A_sq = (A_in * A_in.conj()).real
            A_sq[:] = self._convolution(
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

        # Whole-step fused fast path (MLX), single component only
        if self._backend.has_fused_rk4_step and self.nl_length == 0 and A.ndim == 2:
            alpha_half = getattr(self, "_alpha_half", self.alpha / 2)
            g = getattr(self, "_g", self.k / 2 * self.n2 * c * epsilon_0)
            k_half = getattr(self, "_k_half", np.float32(self.k / 2))
            Isat_conv = getattr(self, "_Isat_conv", 2 * self.I_sat / (epsilon_0 * c))
            V_scaled = getattr(self, "_V_scaled", None)
            if V_scaled is None and V is not None:
                V_scaled = V * k_half
            return kernels.split_step_rk4_fused(
                A,
                propagator,
                V_scaled,
                self.delta_z,
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
        h = self.delta_z

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

    def _run_propagation(
        self,
        A,
        A_sq,
        V,
        z,
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
        n_steps = int(np.ceil(z / abs(self.delta_z)))
        n_steps_nl = self._nonlinear_step_count(z, n_steps)

        # Use a mutable container so the closure can update A for functional
        # backends (MLX) where kernels return new arrays.
        state = [A]

        if method == "RK4":

            def _step():
                state[0] = self.split_step_RK4(state[0], V, self.propagator, self.plans)
        else:

            def _step():
                state[0] = self.split_step(
                    state[0], A_sq, V, self.propagator, self.plans, precision
                )

        is_scalar_n2 = isinstance(self.n2, (int, float, np.floating))
        can_use_fast_loop = (
            callback is None
            and is_scalar_n2
            and self.nl_length == 0
            and not isinstance(self.delta_z, complex)
        )
        # Zeroing the nonlinearity mutates self, so restore it afterwards.
        saved = {attr: getattr(self, attr) for attr in self._nonlinearity_attrs}
        try:
            if can_use_fast_loop:
                # Two segments rather than a mid-loop switch: the constants are
                # baked into the CUDA graph that execute_loop captures, so each
                # segment needs its own capture.
                self._backend.execute_loop(_step, n_steps_nl)
                if n_steps > n_steps_nl:
                    self._disable_nonlinearity(V, precision)
                    self._backend.execute_loop(_step, n_steps - n_steps_nl)
            else:
                self._loop_with_callbacks(
                    _step,
                    z,
                    callback,
                    callback_args,
                    state,
                    verbose,
                    pbar,
                    z_switch=self.L if n_steps_nl < n_steps else None,
                    V=V,
                    precision=precision,
                )
        finally:
            for attr, value in saved.items():
                setattr(self, attr, value)
        return state[0]

    # Attributes holding the nonlinear coupling. Zeroed once propagation
    # leaves the medium, then re-derived by _precompute_step_constants.
    # Subclasses extend this with their own coupling parameters.
    _nonlinearity_attrs = ("n2",)

    def _nonlinear_step_count(self, z: float, n_steps: int) -> int:
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

        Returns
        -------
        int
            Number of leading steps to run with the nonlinearity enabled.
        """
        if not self.L > 0 or z <= self.L:
            return n_steps
        return min(n_steps, int(np.ceil(self.L / abs(self.delta_z))))

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

    def _loop_with_callbacks(
        self,
        step_fn,
        z,
        callback,
        callback_args,
        state,
        verbose,
        pbar,
        z_switch=None,
        V=None,
        precision="single",
    ):
        """Run propagation loop with per-step callbacks."""
        z_prop = 0.0
        i = 0
        switched = False
        while abs(z_prop) < z:
            if z_switch is not None and not switched and abs(z_prop) >= z_switch:
                self._disable_nonlinearity(V, precision)
                switched = True
            step_fn()
            if callback is not None:
                self._run_callbacks(callback, callback_args, state[0], z, i)
            z_prop += self.delta_z
            i += 1
            if verbose:
                pbar.n = abs(z_prop) / z * 100
                pbar.refresh()

    def _precompute_step_constants(
        self, V: np.ndarray | None, precision: str = "single"
    ) -> None:
        """Pre-compute constants that are invariant across propagation steps.

        Parameters
        ----------
        V : np.ndarray or None
            Potential field.
        precision : str
            "single" or "double" — used to select scalar dtype.
        """
        fp = np.float32 if precision == "single" else np.float64
        alpha_half = self.alpha / 2
        g = self.k / 2 * self.n2 * c * epsilon_0
        Isat_conv = 2 * self.I_sat / (epsilon_0 * c)
        k_half = self.k / 2
        # Cast scalar constants to target precision once
        if isinstance(alpha_half, (int, float, np.floating)):
            self._alpha_half = fp(alpha_half)
        else:
            self._alpha_half = alpha_half
        if isinstance(g, (int, float, np.floating)):
            self._g = fp(g)
        else:
            self._g = g
        if isinstance(Isat_conv, (int, float, np.floating)):
            self._Isat_conv = fp(Isat_conv)
        else:
            self._Isat_conv = Isat_conv
        self._k_half = fp(k_half)
        if V is not None:
            self._V_scaled = V * self._k_half
        else:
            self._V_scaled = None
        self._update_propagator_fft()

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
        # Rebuild the propagator on every call. It depends on delta_z (and on
        # k, the grid and the precision), any of which the caller may have
        # changed since the last run, so it cannot be built once and kept.
        # _build_propagator is cache-backed, so an unchanged configuration
        # returns the previously computed array rather than recomputing it.
        if method == "RK4":
            self.propagator = self._build_propagator_rk4()
        else:
            self.propagator = self._build_propagator(precision=precision)
        if self._backend.is_device_backend:
            self._send_arrays_to_gpu()
        V = self.V
        A, A_sq = self._prepare_output_array(E_in, normalize)
        self.plans = self._build_fft_plan(A)
        self._allocate_rk4_buffers(A, method)
        self._precompute_step_constants(V, precision)
        self._enforce_step_limit(A, method, precision)
        if verbose:
            pbar = tqdm.tqdm(
                total=100,
                position=4,
                desc="Propagation",
                leave=False,
                unit="%",
                unit_scale=True,
            )
        if self._backend.name == "CUPY":
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        if type(self.delta_z) is complex:
            print("Warning: imaginary time evolution !")

        A = self._run_propagation(
            A,
            A_sq,
            V,
            z,
            precision,
            method,
            callback,
            callback_args,
            verbose,
            pbar if verbose else None,
        )

        if verbose:
            pbar.n = 100
            pbar.refresh()
        # Synchronize device backends before timing
        if self._backend.name == "CL":
            self._backend.queue.finish()
        elif self._backend.name == "MLX":
            import mlx.core as mx

            mx.eval(A)
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
        # _run_propagation restores the nonlinear coupling itself.
        return_np_array = isinstance(E_in, np.ndarray)
        if self._backend.is_device_backend:
            if return_np_array:
                A = self._backend.to_numpy(A)
            self._retrieve_arrays_from_gpu()

        if plot:
            self.plot_field(A, z)
        return A

    def _to_plot_array(self, A_plot: np.ndarray, target_ndim: int) -> np.ndarray:
        """Reduce dimensions and convert to numpy for plotting."""
        while A_plot.ndim > target_ndim:
            A_plot = A_plot[0]
        if not isinstance(A_plot, np.ndarray):
            A_plot = self._backend.to_numpy(A_plot)
        return A_plot

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Parameters
        ----------
        A_plot : np.ndarray
            Field to plot.
        z : float
            Propagation distance.
        """
        A_plot = self._to_plot_array(A_plot, 2)
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
