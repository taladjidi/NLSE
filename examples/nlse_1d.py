"""
Propagation in one dimension
============================

The same physics on a line, which is where the splitting order is easiest to read.
"""

import numpy as np
from NLSE import NLSE_1d

PRECISION_COMPLEX = np.complex64
N = 2048
n2 = -1.6e-9
waist = 2.23e-3
window = 4 * waist
puiss = 376  # W/m, per unit length: gives the 2D example's peak intensity
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-3
alpha = 20


def main() -> None:
    simu = NLSE_1d(alpha, puiss, window, n2, None, L, NX=N, Isat=Isat, backend="CUPY")
    E_0 = np.exp(-(simu.X**2) / waist**2).astype(PRECISION_COMPLEX)
    simu.out_field(E_0, L, verbose=True, plot=True, splitting="lie")


if __name__ == "__main__":
    main()
