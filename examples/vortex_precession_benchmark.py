"""Time a vortex-pair propagation across grid sizes.

The timings are saved so they can be put next to the Julia implementation's,
so this measures propagation and nothing else -- no sampling, no plotting.
See vortex_precession_animation.py for the picture of what is propagating.

The step is fixed rather than left to the solver: a benchmark compares cost
per unit of work, and letting each grid size choose its own step would change
the work between the cells being compared.
"""

import time

import numpy as np
from _output import output_path
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
BACKEND = "MLX"

# A solver at the reference size, only to get the healing length and the
# nonlinear length the step and the vortex separation are written against.
reference = NLSE(
    alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat, backend=BACKEND
)
cs = np.sqrt(abs(n2) * intensity) / (1 + intensity / Isat)
delta_n = abs(n2) * intensity / (1 + intensity / Isat) ** 2
xi = 1 / (reference.k * cs)
z_nl = 1 / (reference.k * delta_n)
DELTA_Z = z_nl / 6

sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
navg = 10
ts = np.zeros((len(sizes), navg))
for i, n in enumerate(sizes):
    print(n)
    simu = NLSE(
        alpha, power, window, n2, None, L, NX=n, NY=n, Isat=Isat, backend=BACKEND
    )
    # A vortex pair, separated by four healing lengths.
    d = 4 * xi
    E_0 = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
    E_0 *= vortex(simu.XX + d / 2, simu.YY + d / 2, xi=xi, ell=1)
    E_0 *= vortex(simu.XX - d / 2, simu.YY - d / 2, xi=xi, ell=1)
    # Hand tuned potential for Thomas-Fermi
    simu.V = 4.31e-4 * np.exp(-2 * (simu.XX**2 + simu.YY**2) / waist**2).astype(
        np.float32
    )
    for rep in range(navg):
        t0 = time.perf_counter()
        simu.out_field(
            E_0,
            simu.L,
            verbose=False,
            plot=False,
            precision="single",
            delta_z=DELTA_Z,
        )
        ts[i, rep] = time.perf_counter() - t0
    timing_string = f"Average time: {np.mean(ts[i]):.2f} s "
    timing_string += f"(min: {np.min(ts[i]):.2f} s, max: {np.max(ts[i]):.2f} s)"
    print(timing_string)

np.save(output_path(f"python_vortex_precession_{BACKEND}_times.npy"), ts)
np.save(output_path(f"python_vortex_precession_{BACKEND}_sizes.npy"), sizes)
