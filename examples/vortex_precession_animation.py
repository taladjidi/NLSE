"""
A precessing vortex pair, animated
==================================

Animate a vortex pair precessing around each other.

One grid, sampled as it propagates. See vortex_precession_benchmark.py for
the timings across grid sizes.

The step is fixed rather than left to the solver, and the reason is the
sampling rather than the propagation: taking a frame every ``SAVE_EVERY``
steps only spaces the frames evenly in z if every step is the same length.
With an adaptive step the frames would land wherever the step happened to be,
which is not a movie of the precession.
"""

import numpy as np
from NLSE import NLSE


def vortex(x, y, xi=10e-6, ell=1):
    """Return the phase and density profile of a charge-``ell`` vortex."""
    r = np.hypot(x, y)
    theta = np.arctan2(x, y)
    return r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * ell * theta)


N = 256
n2 = -1.6e-10
waist = 750e-6
window = 3.5 * waist
power = 1.05
intensity = power / (np.pi * waist**2)
Isat = np.inf  # saturation intensity in W/m^2
L = 5e-2
alpha = 0
SAVE_EVERY = 2

simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend="MLX")
cs = np.sqrt(abs(n2) * intensity) / (1 + intensity / Isat)
delta_n = abs(n2) * intensity / (1 + intensity / Isat) ** 2
xi = 1 / (simu.k * cs)
z_nl = 1 / (simu.k * delta_n)
DELTA_Z = z_nl / 6
# Far enough for the pair to come back around.
simu.L = 48 * z_nl

# Appended rather than preallocated: the frames are evenly spaced because the
# step is fixed, and nothing here needs to know how many there will be.
E_samples = []


def callback_samples(sim, A, z, i):
    """Keep a copy of the field every SAVE_EVERY steps."""
    if i % SAVE_EVERY == 0:
        # Through the host conversion: on a GPU backend A is a device array.
        E_samples.append(sim._as_host_array(A).copy())


def main():
    """Propagate, sampling as it goes, then animate what was sampled."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # A vortex pair, separated by four healing lengths.
    d = 4 * xi
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
    E_0 *= vortex(simu.XX + d / 2, simu.YY + d / 2, xi=xi, ell=1)
    E_0 *= vortex(simu.XX - d / 2, simu.YY - d / 2, xi=xi, ell=1)
    # Hand tuned potential for Thomas-Fermi
    simu.V = 4.31e-4 * np.exp(-2 * (simu.XX**2 + simu.YY**2) / waist**2).astype(
        np.float32
    )
    simu.out_field(
        E_0,
        simu.L,
        verbose=True,
        plot=False,
        splitting="lie",
        callback=callback_samples,
        delta_z=DELTA_Z,
    )

    frames = np.array(E_samples)
    rho = np.abs(frames) ** 2
    phi = np.angle(frames)
    ext = [
        simu.X.min() * 1e3,
        simu.X.max() * 1e3,
        simu.Y.min() * 1e3,
        simu.Y.max() * 1e3,
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    im0 = ax[0].imshow(rho[0], cmap="viridis", interpolation="none", extent=ext)
    ax[0].set_title("Density")
    im1 = ax[1].imshow(
        phi[0], cmap="twilight_shifted", interpolation="none", extent=ext
    )
    ax[1].set_title("Phase")
    for a in ax:
        a.set_xlabel("x in mm")
        a.set_ylabel("y in mm")

    def animate(i):
        im0.set_data(rho[i])
        im0.set_clim(0, np.max(rho[i]))
        im1.set_data(phi[i])
        fig.suptitle(f"z = {i * SAVE_EVERY * DELTA_Z * 1e3:.2f} mm")
        return im0, im1

    # Held so it is not garbage collected before the window is drawn.
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=60, blit=True)
    plt.show()
    return anim


# Bound at module level rather than dropped: sphinx-gallery looks through the
# example's namespace for an Animation and embeds what it finds there, so an
# animation that only exists inside a function is a still picture on the page.
if __name__ == "__main__":
    anim = main()
