"""
Many simulations at once
========================

A batched parameter turns one run into a sweep, without a Python loop over simulations.
"""

import matplotlib.pyplot as plt
import numpy as np
from NLSE import NLSE

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


def main():
    N = 2048
    n2 = np.zeros((10, 1, 1))
    n2[:, 0, 0] = np.linspace(-1.6e-9, -1e-10, 10)
    waist = 2.23e-3
    window = 4 * waist
    puiss = 1.05
    Isat = 10e4  # saturation intensity in W/m^2
    L = 10e-3
    alpha = 20
    E_0 = np.ones((10, N, N), dtype=PRECISION_COMPLEX)
    simu = NLSE(alpha, puiss, window, n2, V=None, L=L, NX=N, NY=N, Isat=Isat)
    E_0 *= np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    E = simu.out_field(E_0, L, verbose=True, plot=False, splitting="lie")

    # One run, ten simulations. Show what the sweep bought: the same beam
    # through ten media, from the weakest nonlinearity to the strongest.
    E = np.asarray(simu._backend.to_numpy(E))
    intensity = np.abs(E) ** 2
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, index in zip(axes, (0, 3, 6, 9), strict=True):
        ax.imshow(
            intensity[index],
            extent=[
                simu.X.min() * 1e3,
                simu.X.max() * 1e3,
                simu.Y.min() * 1e3,
                simu.Y.max() * 1e3,
            ],
            cmap="inferno",
        )
        ax.set_title(rf"$n_2$ = {float(n2[index, 0, 0]):.2e} m$^2$/W", fontsize=9)
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.suptitle("Ten simulations from one batched run")
    plt.show()


if __name__ == "__main__":
    main()
