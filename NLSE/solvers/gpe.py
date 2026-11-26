from typing import Any

import numpy as np
from scipy.constants import atomic_mass, c, epsilon_0, hbar

from ..utils import __BACKEND__
from .nlse import NLSE
from .parameter import Parameter

# i hbar dpsi/dt = hbar (V + g|psi|^2) psi turns the phase at -(V + g|psi|^2),
# and the kernels rotate by +i times the rate they are given. Dimensionless.
GPE_PHASE_SIGN = -1.0


class GPE(NLSE):
    """A class to solve GPE."""

    # A density in atoms/m^2, not an optical intensity, and a time in seconds,
    # not a distance: no conversion, and labels to match.
    _plot_density_scale = 1.0
    _plot_density_label = "Density (at/m^2)"
    _plot_axis_symbol = "t"
    _plot_axis_unit = "s"

    # GPE integrates NLSE's equation with cold-gas physics on each term, so it
    # shares the storage rather than keeping a second copy of every value.
    gamma = Parameter("alpha", "Atom loss rate in 1/s: N(t) = N(0) exp(-gamma t).")
    N = Parameter("power", "Total number of atoms.")
    m = Parameter("k", "Mass of one atom in kg.")
    g = Parameter("n2", "Interaction energy over hbar, in rad/s*m^2.")
    sat = Parameter("I_sat", "Saturation density in at/m^2.", scale=epsilon_0 * c / 2)

    def __init__(
        self,
        gamma: float,
        N: float,
        window: float,
        g: float,
        V: np.ndarray | None,
        m: float = 87 * atomic_mass,
        NX: int = 1024,
        NY: int = 1024,
        sat: float = np.inf,
        nl_length: float = 0,
        backend: str = __BACKEND__,
    ) -> None:
        """Instantiate the simulation.

        Solves i hbar dpsi/dt = -hbar^2/(2m) (d2/dx2 + d2/dy2) psi
        + hbar V psi + hbar g |psi|^2 psi, so V and g|psi|^2 are rates in
        rad/s: the energies divided by hbar.

        Parameters
        ----------
        gamma : float
            Atom loss rate in 1/s: N(t) = N(0) exp(-gamma t).
        N : float
            Total number of atoms.
        window : float
            Window size in m.
        g : float
            Interaction energy divided by hbar, in rad/s*m^2.
            g > 0 is repulsive.
        V : np.ndarray
            Potential energy divided by hbar, in rad/s.
        m : float, optional
            Mass of one atom in kg.
            Defaults to 87*atomic_mass for Rubidium 87.
        NX : int, optional
            Number of points in x.
            Defaults to 1024.
        NY : int, optional
            Number of points in y.
            Defaults to 1024.
        sat : float
            Saturation density in atoms/m^2: the interaction and the
            loss are divided by 1 + |psi|^2 / sat.
        nl_length : float
            Non-local length in m, of a diffusive non-locality: the
            interaction reads the intensity smoothed by
            1 / (1 + nl_length^2 K^2) in Fourier space. 0 is local.
        backend : str, optional
            "GPU" or "CPU". Defaults to __BACKEND__.
        """
        super().__init__(
            alpha=gamma,
            power=N,
            window=window,
            n2=g,
            V=V,
            L=0,
            NX=NX,
            NY=NY,
            Isat=sat,
            nl_length=nl_length,
            wvl=2 * np.pi / m,
            backend=backend,
        )
        self.V = V
        # do some conversion for the units
        self.I_sat *= epsilon_0 * c / 2
        # |psi|^2 is a density in atoms/m^2, not an optical field, so the
        # norm_constant is 1.0
        self._norm_constant = np.float32(1.0)

    def _propagator_cache_key(self, dtype: np.dtype, delta_z: float) -> tuple:
        """Return cache key for GPE propagator."""
        return (
            self.NX,
            self.NY,
            float(delta_z),
            np.dtype(dtype).str,
            float(self.m),
        )

    def _propagator_rk4_cache_key(self) -> tuple:
        """Return cache key for GPE RK4 dispersion operator."""
        return (self.NX, self.NY, "RK4", float(self.m))

    def _compute_propagator_rk4(self) -> np.ndarray:
        """Compute the raw GPE dispersion operator for RK4."""
        return -1j * 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m

    def _dispersion_operator(self) -> np.ndarray:
        """Return the GPE dispersion eigenvalues.

        The GPE dispersion operator is hbar*K^2/(2m), unlike NLSE's K^2/(2k).
        """
        return 0.5 * hbar * (self.Kxx**2 + self.Kyy**2) / self.m

    def _step_constants(self) -> dict[str, Any]:
        """Return the per-step constants, with V and g|psi|^2 as rates.

        NLSE converts its couplings from optical units, multiplying the
        interaction by ``k / 2 * c * epsilon_0`` and the potential by ``k / 2``.
        GPE's ``k`` is the mass, so for rubidium 87 that conversion multiplied
        the potential by 7e-26 and the interaction by 2e-28. Here V and
        g|psi|^2 are already rad/s and only take the sign of the GPE.

        Returns
        -------
        dict
            Attribute name to value: ``_g`` in rad/s*m^2, ``_k_half`` the
            dimensionless factor the potential is multiplied by.
        """
        return {
            **super()._step_constants(),
            "_g": GPE_PHASE_SIGN * self.g,
            "_k_half": GPE_PHASE_SIGN,
        }
