import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0

from ..utils import __BACKEND__, __CUPY_AVAILABLE__, __PYOPENCL_AVAILABLE__
from .nlse import NLSE

if __CUPY_AVAILABLE__:
    import cupy as cp

if __PYOPENCL_AVAILABLE__:
    from pyopencl import array as cla


class NLSE_1d(NLSE):
    """A class to solve NLSE in 1d"""

    def __init__(
        self,
        alpha: float,
        power: float,
        window: float,
        n2: float,
        V: np.ndarray | None,
        L: float,
        NX: int = 1024,
        Isat: float = np.inf,
        nl_length: float = 0,
        wvl: float = 780e-9,
        backend: str = __BACKEND__,
    ) -> object:
        """Instantiate the simulation.

        Solves an equation : d/dz psi = -1/2k0(d2/dx2) psi + k0 dn psi +
          k0 n2 psi**2 psi

        Args:
            alpha (float): Transmission coeff
            power (float): Power in W
            n2 (float): Non linear coeff in m^2/W
            V (np.ndarray) : Potential
            L (float): Length of the medium.
            Isat (float): Saturation intensity in W/m^2
            nl_length (float): Non local length in m.
                The non-local kernel is the instantiated as a Bessel function
                to model a diffusive non-locality stored in the nl_profile
                attribute.
            wvl (float, optional): Wavelength in m. Defaults to 780 nm.
            backend (str, optional): "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=alpha,
            power=power,
            window=window,
            n2=n2,
            V=V,
            L=L,
            NX=NX,
            Isat=Isat,
            nl_length=nl_length,
            wvl=wvl,
            backend=backend,
        )
        self._last_axes = (-1,)
        self.nl_profile = self.nl_profile[0]
        self.nl_profile /= self.nl_profile.sum()

    def _prepare_output_array(self, E_in: np.ndarray, normalize: bool) -> np.ndarray:
        """Prepare the output array depending on __BACKEND__.

        Args:
            E_in (np.ndarray): Input array
            normalize (bool): Normalize the field to the total power.
        Returns:
            np.ndarray: Output array
        """
        if self.backend == "CUPY" and self.__CUPY_AVAILABLE__:
            A = cp.zeros_like(E_in)
            A_sq = cp.zeros_like(A, dtype=A.real.dtype)
            E_in = cp.asarray(E_in)
        elif self.backend == "CL" and self.__PYOPENCL_AVAILABLE__:
            A = cla.zeros(self._backend.queue, E_in.shape, E_in.dtype)
            A_sq = cla.zeros(self._backend.queue, E_in.shape, E_in.real.dtype)
        else:
            A = pyfftw.zeros_aligned(
                E_in.shape, dtype=E_in.dtype, n=pyfftw.simd_alignment
            )
            A_sq = np.zeros_like(A, dtype=A.real.dtype)
        if normalize:
            # normalization of the field
            if self.backend == "CL" and self.__PYOPENCL_AVAILABLE__:
                E_in_cl = cla.to_device(self._backend.queue, E_in)
                arr = E_in_cl.real * E_in_cl.real + E_in_cl.imag * E_in_cl.imag
                arr = arr * self.delta_X**2
                integral = cla.sum(arr, dtype=arr.dtype, queue=self._backend.queue).get()
                integral *= c * epsilon_0 / 2
                E_00 = (self.power / integral) ** 0.5
                A[:] = E_00 * E_in_cl
            else:
                integral = (
                    (E_in.real * E_in.real + E_in.imag * E_in.imag) * self.delta_X**2
                ).sum(axis=self._last_axes)
                integral *= c * epsilon_0 / 2
                E_00 = (self.power / integral) ** 0.5
                A[:] = (E_00.T * E_in.T).T
        else:
            if self.backend == "CL" and self.__PYOPENCL_AVAILABLE__:
                A[:] = cla.to_device(self._backend.queue, E_in)
            else:
                A[:] = E_in
        return A, A_sq

    def _build_propagator(self, precision: str = "single") -> np.ndarray:
        """Build the linear propagation matrix.

        Returns:
            propagator (np.ndarray): the propagator matrix
        """
        dtype = np.complex128 if precision == "double" else np.complex64
        propagator = np.exp(-1j * 0.5 * (self.Kx**2) / self.k * self.delta_z, dtype=dtype)
        return propagator

    def plot_field(self, A_plot: np.ndarray, z: float) -> None:
        """Plot a field for monitoring.

        Args:
            A_plot (np.ndarray): Field to plot
            z (float): Propagation distance in m.
        """
        fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
        fig.suptitle(rf"Field at $z$ = {z:.2e} m")
        if A_plot.ndim == 2:
            for i in range(A_plot.shape[0]):
                ax[0].plot(
                    self.X * 1e3,
                    1e-4 * c / 2 * epsilon_0 * np.abs(A_plot[i, :]) ** 2,
                )
                ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot[i, :])))
        elif A_plot.ndim == 1:
            ax[0].plot(self.X * 1e3, 1e-4 * c / 2 * epsilon_0 * np.abs(A_plot) ** 2)
            ax[1].plot(self.X * 1e3, np.unwrap(np.angle(A_plot)))
        ax[0].set_title(r"$|\psi|^2$")
        ax[0].set_ylabel(r"Intensity $\frac{\epsilon_0 c}{2}|\psi|^2$ in $W/cm^2$")
        ax[1].set_title(r"Phase $\mathrm{arg}(\psi)$")
        ax[1].set_ylabel(r"Phase arg$(\psi)$")
        for a in ax:
            a.set_xlabel("Position x in mm")
        plt.show()
