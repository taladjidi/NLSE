#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

from __future__ import annotations

import multiprocessing
import time
import types
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import tqdm
from scipy import special
from scipy.constants import c, epsilon_0

from ..backends import get_backend
from ..utils import (
    __BACKEND__,
    __CUPY_AVAILABLE__,
    __METAL_AVAILABLE__,
    __PYOPENCL_AVAILABLE__,
)

if __CUPY_AVAILABLE__:
    import cupy as cp

if __PYOPENCL_AVAILABLE__:
    pass

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()


class NLSE:
    """A class to solve the Nonlinear Schrodinger Equation (NLSE)."""

    __CUPY_AVAILABLE__ = __CUPY_AVAILABLE__
    __PYOPENCL_AVAILABLE__ = __PYOPENCL_AVAILABLE__
    __METAL_AVAILABLE__ = __METAL_AVAILABLE__

    def __init__(
        self,
        alpha: float | np.floating,
        power: float | np.floating,
        window: float | tuple[float, ...] | list[float],
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
            Absorption coefficient.
        power : float
            Power in W.
        window : float or tuple of float or list of float
            Computational window in the transverse plane in m.
            Can be different in x and y.
        n2 : float
            Non linear coefficient in m^2/W.
        V : np.ndarray or None
            Potential.
        L : float
            Length in m of the nonlinear medium.
        NX : int, optional
            Number of points in the x direction. Defaults to 1024.
        NY : int, optional
            Number of points in the y direction. Defaults to 1024.
        Isat : float, optional
            Saturation intensity in W/m^2. Defaults to np.inf.
        nl_length : float, optional
            Non local length in m. The non-local kernel is instantiated
            as a Bessel function to model a diffusive non-locality stored
            in the ``nl_profile`` attribute. Defaults to 0.
        wvl : float, optional
            Wavelength in m. Defaults to 780e-9.
        backend : str, optional
            Compute backend, one of ``"CPU"``, ``"CUPY"``, ``"CL"``,
            ``"Metal"``. Defaults to ``__BACKEND__``.
        """
        # list of physical parameters
        self.backend = backend
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
        if isinstance(window, (float, int)):
            self.window: list[float] = [window, window]
        elif isinstance(window, (tuple, list)):
            self.window = list(window)
        Dn = self.n2 * self.power / min(self.window) ** 2
        z_nl = 1 / (self.k * abs(Dn))
        if isinstance(z_nl, np.ndarray) or (
            self.__CUPY_AVAILABLE__ and isinstance(z_nl, cp.ndarray)
        ):
            z_nl = float(z_nl.min())
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
        self._last_axes: tuple[int, ...] = (-2, -1)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        # definition of the Fourier frequencies for the linear step
        self.Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        self.Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)
        self.Kxx, self.Kyy = np.meshgrid(self.Kx, self.Ky)
        self.propagator: np.ndarray | None = None
        self.plans: Any = None
        self.nl_length = nl_length
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
        else:
            self.nl_profile = np.ones((self.NY, self.NX), dtype=np.float32)

        # GPU transfer caching flag
        self._gpu_initialized = False

    @property
    def backend(self) -> str:
        """str : The backend name used for the simulation."""
        return self._backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        self._backend = get_backend(value)
        self._kernels: types.ModuleType = self._backend.kernels

    @property
    def _cl_queue(self) -> Any:
        """Backward-compatible access to the OpenCL command queue."""
        if hasattr(self._backend, "cl_queue"):
            return self._backend.cl_queue
        raise AttributeError("_cl_queue is only available on CL backend")

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Parameters
        ----------
        precision : str, optional
            Type of propagator to generate (``"single"``, ``"double"``,
            or ``"RK4"``). For split-step schemes the step is inside
            the propagator; for RK4 it is not.

        Returns
        -------
        np.ndarray
            The propagator matrix.
        """
        # Reset GPU initialization flag when propagator changes
        self._gpu_initialized = False

        match precision:
            case "single" | "double":
                propagator = np.exp(
                    -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z
                ).astype(np.complex64)
            case "RK4":
                propagator = (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k).astype(
                    np.complex64
                )
        return propagator  # type: ignore[no-any-return]

    def _build_fft_plan(self, A: Any) -> Any:
        """Build the FFT plan objects for propagation.

        Parameters
        ----------
        A : array-like
            Array to transform.

        Returns
        -------
        FFTPlan
            Plan object with ``.fft()`` and ``.ifft()`` methods.
        """
        plan = self._backend.build_fft_plan(A, self._last_axes)
        self._fft_plan = plan
        return plan

    def _linear_step(self, A: Any, propagator: Any) -> None:
        """Apply linear propagation in Fourier space."""
        self._backend.fft(self._fft_plan, A)
        A *= propagator
        self._backend.ifft(self._fft_plan, A)

    def _compute_norm_factor(self, E_in: Any) -> Any:
        """Compute the normalization factor for the input field.

        Override in subclasses for different physics.

        Parameters
        ----------
        E_in : array-like
            Input field (possibly on device).

        Returns
        -------
        array-like
            Normalization factor ``E_00``.
        """
        arr = E_in.real * E_in.real + E_in.imag * E_in.imag
        # forbid numpy systematically upcasting to double precision
        arr = (arr * self.delta_X * self.delta_Y).astype(E_in.real.dtype)
        integral = self._backend.sum(arr, axis=self._last_axes)
        integral = integral * c * epsilon_0 / 2
        E_00 = self._backend.sqrt(self.power / integral)
        return E_00

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> tuple[Any, Any]:
        """Prepare the output arrays depending on backend.

        Prepares the ``A`` and ``A_sq`` arrays to store the field and its
        modulus squared.

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
        E_dev = self._backend.to_device(E_in)
        if normalize:
            E_00 = self._compute_norm_factor(E_dev)
            A[:] = (E_00.T * E_dev.T).T
        else:
            A[:] = E_dev
        return A, A_sq

    def _send_arrays_to_gpu(self, force_refresh: bool = False) -> None:
        """Send arrays to device memory.

        Parameters
        ----------
        force_refresh : bool, optional
            Force re-uploading arrays even if already on GPU. Defaults to False.
        """
        if self.backend == "CPU":
            return

        # Lazy GPU transfer: skip if arrays are already on device
        if not force_refresh and hasattr(self, "_gpu_initialized") and self._gpu_initialized:
            return

        if self.V is not None:
            self.V = self._backend.to_device(self.V)
        assert self.propagator is not None
        self.propagator = self._backend.to_device(self.propagator)
        # Metal only needs V and propagator on device
        if self.backend != "Metal":
            self.nl_profile = self._backend.to_device(self.nl_profile)
            # for broadcasting of parameters in case they are arrays
            for attr in ("power", "n2", "alpha", "I_sat"):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    setattr(self, attr, self._backend.to_device(val))

        # Mark as initialized
        self._gpu_initialized = True

    def _retrieve_arrays_from_gpu(self) -> None:
        """Retrieve arrays from device memory."""
        if self.backend == "CPU":
            return
        if self.V is not None:
            self.V = self._backend.to_host(self.V)
        if self._backend.is_device_array(self.nl_profile):
            self.nl_profile = self._backend.to_host(self.nl_profile)
        self.propagator = self._backend.to_host(self.propagator)
        for attr in ("power", "n2", "alpha", "I_sat"):
            val = getattr(self, attr)
            if self._backend.is_device_array(val):
                setattr(self, attr, self._backend.to_host(val))

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
            Field to propagate.
        A_sq : np.ndarray
            Field modulus squared.
        V : np.ndarray or None
            Potential field (can be ``None``).
        propagator : np.ndarray
            Propagator matrix.
        plans : FFTPlan
            FFT plan object.
        precision : str, optional
            Single or double application of the nonlinear propagation
            step. Defaults to ``"single"``.
        """
        if precision == "double":
            self._kernels.square_mod(A, A_sq)
            if self.nl_length > 0:
                A_sq[:] = self._backend.convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        self._linear_step(A, propagator)
        self._kernels.square_mod(A, A_sq)
        if self.nl_length > 0:
            A_sq[:] = self._backend.convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        else:
            if V is None:
                self._kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                self._kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )

    def _RK4_rhs_non_mutating(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: Any,
    ) -> Any:
        """Compute the RHS of NLSE in a non-mutating manner for RK4.

        Parameters
        ----------
        A : np.ndarray
            Field to propagate.
        V : np.ndarray or None
            Potential field (can be ``None``).
        propagator : np.ndarray
            Propagator matrix.
        plans : FFTPlan
            FFT plan object.

        Returns
        -------
        np.ndarray
            The RHS evaluation.
        """
        # prepare output array, this kills performance but we need it
        A_prop = A.copy()
        A_sq = A.real * A.real + A.imag * A.imag
        self._linear_step(A_prop, propagator)
        if self.nl_length > 0:
            A_sq[:] = self._backend.convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
        # Linear prop
        arg = A_prop
        # saturation
        sat = 1 / (1 + A_sq / (2 * self.I_sat / (epsilon_0 * c)))
        # Interactions
        arg += 1j * self.k / 2 * self.n2 * c * epsilon_0 * A_sq * sat * A
        # Losses
        arg -= self.alpha / 2 * sat * A
        if V is not None:
            V_ = 1j * self.k / 2 * V * A
            arg += V_
        return arg

    def split_step_RK4(
        self,
        A: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: Any,
    ) -> None:
        """Perform one propagation step using the RK4 scheme.

        y_n+1 = y_n + dz/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        k_1 = rhs(A)
        k_2 = rhs(A+k_1/2)
        k_3 = rhs(A+k_2/2)
        k_4 = rhs(A+k_3)

        Parameters
        ----------
        A : np.ndarray
            Field to propagate.
        V : np.ndarray or None
            Potential field (can be ``None``).
        propagator : np.ndarray
            Propagator matrix.
        plans : FFTPlan
            FFT plan object.
        """
        k_1 = self._RK4_rhs_non_mutating(A, V, propagator, plans)
        k_2 = self._RK4_rhs_non_mutating(
            A + k_1 * self.delta_z / 3, V, propagator, plans
        )
        k_3 = self._RK4_rhs_non_mutating(
            A + (-k_1 / 3 + k_2) * self.delta_z, V, propagator, plans
        )
        k_4 = self._RK4_rhs_non_mutating(
            A + (k_1 - k_2 + k_3) * self.delta_z, V, propagator, plans
        )
        A += self.delta_z / 8 * (k_1 + 3 * k_2 + 3 * k_3 + k_4)

    def out_field(
        self,
        E_in: np.ndarray,
        z: float,
        plot: bool = False,
        precision: str = "single",
        verbose: bool = True,
        normalize: bool = True,
        callback: list[Callable[..., Any]] | Callable[..., Any] | None = None,
        callback_args: list[tuple[Any, ...]] | tuple[Any, ...] = (),
    ) -> Any:
        """Propagate the field over a distance *z*.

        Calls the split-step function in a loop. Supports imaginary time
        evolution when ``delta_z`` is set to a complex number (experimental).

        Parameters
        ----------
        E_in : np.ndarray
            Normalized input field (between 0 and 1).
        z : float
            Propagation distance in m.
        plot : bool, optional
            Plot the results. Defaults to ``False``.
        precision : str, optional
            ``"single"`` (O(dz)) or ``"double"`` (O(dz^3)) application of
            the nonlinear term. Defaults to ``"single"``.
        verbose : bool, optional
            Print progress and elapsed time. Defaults to ``True``.
        normalize : bool, optional
            Normalize the field to the total power. Defaults to ``True``.
        callback : callable or list of callable, optional
            Callback function(s) executed at every propagation step.
            Defaults to ``None``.
        callback_args : tuple or list of tuple, optional
            Additional arguments for the callback function(s).

        Returns
        -------
        np.ndarray
            Propagated field in proper units (V/m).
        """
        assert (
            E_in.shape[self._last_axes[0] :] == self.XX.shape[self._last_axes[0] :]
        ), "Shape mismatch"
        assert E_in.dtype in [
            np.complex64,
            np.complex128,
        ], "Type mismatch, E_in should be complex64 or complex128"
        # define propagator if not already done
        if self.propagator is None:
            self.propagator = self._build_propagator(precision=precision)
        assert self.propagator is not None
        if self.backend != "CPU":
            self._send_arrays_to_gpu()
        if self.V is None:
            V = self.V
        else:
            V = self.V.copy()
        A, A_sq = self._prepare_output_array(E_in, normalize)
        self._fft_plan = self._build_fft_plan(A)
        # Keep self.plans for backward compat (tests that check it)
        self.plans = self._fft_plan
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
        if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
        t0 = time.perf_counter()
        z_prop = 0.0
        i = 0
        if type(self.delta_z) is complex:
            print("Warning: imaginary time evolution !")
        while abs(z_prop) < z:
            if z > self.L:
                self.n2 = 0
            if precision == "RK4":
                self.split_step_RK4(A, V, self.propagator, self.plans)
            else:
                self.split_step(A, A_sq, V, self.propagator, self.plans, precision)

            if callback is not None:
                if callable(callback):
                    callback(self, A, z, i, *callback_args)
                elif isinstance(callback, list) and callable(callback[0]):
                    for c_, ca in zip(callback, callback_args):
                        c_(self, A, z, i, *ca)
                else:
                    raise ValueError(
                        "callbacks should be a callable or a list of callables"
                    )
            z_prop += self.delta_z
            i += 1
            if verbose:
                pbar.n = abs(z_prop) / z * 100
                pbar.refresh()
        t_cpu = time.perf_counter() - t0
        if verbose:
            pbar.close()

        if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        if verbose:
            if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
                print(
                    f"\nTime spent to solve : {t_gpu * 1e-3} s (GPU) /"
                    f" {time.perf_counter() - t0} s (CPU)\n"
                )
            else:
                print(f"\nTime spent to solve : {t_cpu} s (CPU)\n")
        self.n2 = n2_old
        return_np_array = isinstance(E_in, np.ndarray)
        if self.backend != "CPU":
            if return_np_array:
                A = self._backend.to_host(A)
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
            Propagation distance in m.
        """
        # if array is multi-dimensional, drop dims until the shape is 2D
        if A_plot.ndim > 2:
            while len(A_plot.shape) > 2:
                A_plot = A_plot[0]
        A_plot = self._backend.to_host(A_plot)
        if not isinstance(A_plot, np.ndarray):
            A_plot = np.asarray(A_plot)
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
