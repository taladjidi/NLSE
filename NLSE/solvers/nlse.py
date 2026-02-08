#!/usr/bin/env python3
# @author: Tangui Aladjidi / Clara Piekarski
"""NLSE Main module."""

import multiprocessing
import pickle
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
from ..utils import __BACKEND__, __CUPY_AVAILABLE__, __PYOPENCL_AVAILABLE__

if __CUPY_AVAILABLE__:
    import cupy as cp
    import cupyx.scipy.signal as signal_cp  # type: ignore[import-not-found]

if __PYOPENCL_AVAILABLE__:
    from pyopencl import array as cla
    from pyopencl import clmath

pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.interfaces.cache.enable()


class NLSE:
    """A class to solve NLSE"""

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

        Args:
            alpha (float): alpha
            power (float): Power in W
            window (float, list or tuple): Computational window in the
                transverse plane in m.
                Can be different in x and y.
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray): Potential.
            L (float): Length in m of the nonlinear medium
            NX (int, optional): Number of points in the x direction.
                Defaults to 1024.
            NY (int, optional): Number of points in the y direction.
                Defaults to 1024.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non local length in m.
                The non-local kernel is the instantiated as a Bessel function
                to model a diffusive non-locality stored in the nl_profile
                attribute.
            wvl (float): Wavelength in m
            backend (str, optional): Will run using the "GPU" or "CPU".
                Defaults to __BACKEND__.
        """
        # list of physical parameters
        self._backend: Backend = get_backend(backend)
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

    @property
    def backend(self) -> str:
        """Return the backend used for the simulation."""
        return self._backend.name

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the backend for the simulation."""
        self._backend = get_backend(value)
        # Setup backend-specific convolution
        if self._backend.name == "CUPY":
            self._convolution = signal_cp.oaconvolve
        elif self._backend.name == "CPU":
            self._convolution = signal.oaconvolve
        # CL backend doesn't have convolution yet

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Returns:
            propagator (np.ndarray): the propagator matrix
            precision (str): Type of propagator to generate. For split step schemes
            the step is inside the propagator, for RK4 it is not.
        """
        # Use appropriate dtype based on precision
        dtype = np.complex128 if precision == "double" else np.complex64

        match precision:
            case "single" | "double":
                propagator = np.exp(
                    -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z,
                    dtype=dtype,
                )
            case "RK4":
                propagator = (-1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k).astype(dtype)
        return propagator

    def _build_fft_plan(self, A: np.ndarray) -> list:
        """Build the FFT plan objects for propagation.

        Args:
            A (np.ndarray): Array to transform.
        Returns:
            list: List of FFT plan objects from the backend
        """
        # Load FFTW wisdom for CPU backend
        if self._backend.name == "CPU":
            try:
                with open("fft.wisdom", "rb") as file:
                    wisdom = pickle.load(file)
                    pyfftw.import_wisdom(wisdom)
            except FileNotFoundError:
                print("No FFT wisdom found, starting over ...")

        plan = self._backend.build_fft(A.shape, self._last_axes, A.dtype)

        # Save FFTW wisdom for CPU backend
        if self._backend.name == "CPU":
            with open("fft.wisdom", "wb") as file:
                wisdom = pyfftw.export_wisdom()
                pickle.dump(wisdom, file)

        return plan

    def _prepare_output_array(
        self, E_in: np.ndarray, normalize: bool
    ) -> tuple[np.ndarray | Any, np.ndarray | Any]:
        """Prepare the output arrays depending on backend.

        Prepares the A and A_sq arrays to store the field and its modulus.

        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            A (np.ndarray): Output field array
            A_sq (np.ndarray): Output field modulus squared array
        """
        # Allocate arrays on the backend
        A = self._backend.allocate_field(E_in.shape, E_in.dtype)
        A_sq = self._backend.allocate_real_field(E_in.shape, E_in.real.dtype)
        E_in = self._backend.from_numpy(E_in)

        if normalize:
            # normalization of the field
            arr = E_in.real * E_in.real + E_in.imag * E_in.imag
            # forbid numpy systematically upcasting to double precision
            arr = (arr * self.delta_X * self.delta_Y).astype(E_in.real.dtype)
            if self._backend.name == "CL":
                integral = cla.sum(
                    arr,
                    dtype=arr.dtype,
                    queue=self._backend.queue,
                )
                integral = integral * c * epsilon_0 / 2
                E_00 = clmath.sqrt(self.power / integral)
            else:
                integral = np.sum(arr, axis=self._last_axes)
                integral = integral * c * epsilon_0 / 2
                E_00 = (self.power / integral) ** 0.5
            A[:] = (E_00.T * E_in.T).T
        else:
            A[:] = E_in
        return A, A_sq

    def _send_arrays_to_gpu(self) -> None:
        """Send arrays to device using backend."""
        if self._backend.name in ["CUPY", "CL"]:
            if self.V is not None:
                self.V = self._backend.from_numpy(self.V)
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
        if self._backend.name in ["CUPY", "CL"]:
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

    def split_step(
        self,
        A: np.ndarray,
        A_sq: np.ndarray,
        V: np.ndarray | None,
        propagator: np.ndarray,
        plans: list,
        precision: str = "single",
    ) -> None:
        """Split step function for one propagation step.

        Args:
            A (np.ndarray): Field to propagate
            A_sq (np.ndarray): Field modulus squared.
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects from backend.
            precision (str, optional): Single or double application of
                the nonlinear propagation step. Defaults to "single".
        """
        kernels = self._backend.kernels
        if precision == "double":
            kernels.square_mod(A, A_sq)
            if self.nl_length > 0:
                A_sq[:] = self._convolution(
                    A_sq, self.nl_profile, mode="same", axes=self._last_axes
                )
            if V is None:
                kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                kernels.nl_prop(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * V,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
        # Linear propagation step in Fourier domain
        self._backend.fft(A, plans)
        A *= propagator
        self._backend.ifft(A, plans)

        kernels.square_mod(A, A_sq)
        if self.nl_length > 0:
            A_sq[:] = self._convolution(
                A_sq, self.nl_profile, mode="same", axes=self._last_axes
            )
        if precision == "double":
            if V is None:
                kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z / 2,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                kernels.nl_prop(
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
                kernels.nl_prop_without_V(
                    A,
                    A_sq,
                    self.delta_z,
                    self.alpha / 2,
                    self.k / 2 * self.n2 * c * epsilon_0,
                    2 * self.I_sat / (epsilon_0 * c),
                )
            else:
                kernels.nl_prop(
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
        plans: list,
    ) -> np.ndarray:
        """Compute the RHS of NLSE in a non-mutating manner for RK4.

        Split step function for one propagation step using a 4th order Runge-Kutta method (RK4).

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects from backend.
        """
        # prepare output array, this kills performance but we need it
        A_prop = A.copy()
        A_sq = (A * A.conj()).real

        # Linear propagation step in Fourier domain
        self._backend.fft(A_prop, plans)
        A_prop *= propagator
        self._backend.ifft(A_prop, plans)

        if self.nl_length > 0:
            A_sq[:] = self._convolution(
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
        plans: list,
    ) -> None:
        """Split step function for one propagation step using RK4 scheme.

        y_n+1 = y_n + dz/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        k_1 = rhs(A)
        k_2 = rhs(A+k_1/2)
        k_3 = rhs(A+k_2/2)
        k_4 = rhs(A+k_3)

        Args:
            A (np.ndarray): Field to propagate
            V (np.ndarray): Potential field (can be None).
            propagator (np.ndarray): Propagator matrix.
            plans (list): List of FFT plan objects from backend.
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

        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1).
            z (float): propagation distance in m.
            plot (bool, optional): Plots the results. Defaults to False.
            precision (str, optional): Does a "double" or a "single" application
                of the nonlinear term. This leads to a dz (single) or dz^3
                (double)precision. Defaults to "single".
            verbose (bool, optional): Prints progress and time.
                Defaults to True.
            normalize (bool, optional): Normalize the field to the total power.
                Defaults to True.
            callback (callable, optional): Callback function.
                Defaults to None.
            callback_args (tuple, optional): Additional arguments for the
                callback function.
        Returns:
            np.ndarray: Propagated field in proper units V/m
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
        if self._backend.name in ["CUPY", "CL"]:
            self._send_arrays_to_gpu()
        if self.V is None:
            V = self.V
        else:
            V = self.V.copy()
        A, A_sq = self._prepare_output_array(E_in, normalize)
        self.plans = self._build_fft_plan(A)
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
            if precision == "RK4":
                self.split_step_RK4(A, V, self.propagator, self.plans)
            else:
                self.split_step(A, A_sq, V, self.propagator, self.plans, precision)

            if callback is not None:
                if isinstance(callback, Callable):
                    callback(self, A, z, i, *callback_args)
                elif isinstance(callback, list) and isinstance(callback[0], Callable):
                    for c, ca in zip(callback, callback_args, strict=True):
                        c(self, A, z, i, *ca)
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
        if self._backend.name in ["CUPY", "CL"]:
            if return_np_array:
                A = self._backend.to_numpy(A)
            self._retrieve_arrays_from_gpu()

        if plot:
            self.plot_field(A, z)
        return A

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot.
            z (float): Propagation distance.
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
