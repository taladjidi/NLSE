"""
Sampling a run, and adapting its step
=====================================

A callback that stores the field as it goes, and one that moves the step to track the physics.
"""

import numpy as np
from NLSE import NLSE, callbacks

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32

N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
puiss = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-2
alpha = 20


def main():
    import matplotlib.pyplot as plt
    from scipy.constants import c, epsilon_0

    simu = NLSE(
        alpha,
        puiss,
        window,
        n2,
        None,
        L,
        NX=N,
        NY=N,
        Isat=Isat,
        backend="CUPY",
    )
    # callbacks.norm writes into norms[i // save_every], so it needs the number
    # of steps up front. Letting the solver choose the step means not knowing
    # that until the run is over, so collect instead of preallocating, and keep
    # the distance the solver actually reached alongside each sample.
    zs = []
    norms = []

    def sample_norm(simu, A, z, i, save_every):
        if i % save_every == 0:
            A_host = simu._as_host_array(A)
            zs.append(z)
            norms.append((A_host.real**2 + A_host.imag**2).sum())

    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(PRECISION_COMPLEX)
    simu.V = -1e-4 * np.exp(-(simu.XX**2 + simu.YY**2) / waist2**2).astype(
        PRECISION_COMPLEX
    )
    simu.out_field(
        E_0,
        L,
        verbose=True,
        plot=True,
        splitting="lie",
        callback=sample_norm,
        callback_args=(1,),
    )
    norms = np.array(norms) * simu.delta_X * simu.delta_Y * c * epsilon_0 / 2
    plt.plot(np.array(zs) * 1e3, norms)
    plt.xlabel("Propagation distance in mm")
    plt.ylabel("Total power in W")
    plt.title("Total power of the field")
    plt.show()
    dzs = []
    simu.out_field(
        E_0,
        L,
        verbose=True,
        plot=True,
        splitting="lie",
        callback=callbacks.adapt_delta_z,
        callback_args=(10, dzs),
    )
    plt.plot(dzs)
    plt.xlabel("Iteration")
    plt.ylabel("Step size")
    plt.title("Adaptive step size")
    plt.show()


if __name__ == "__main__":
    main()
