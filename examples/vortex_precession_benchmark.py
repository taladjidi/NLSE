"""
Vortex precession, timed across grid sizes
==========================================

Time a vortex-pair propagation across grid sizes.

The timings are saved so they can be put next to the Julia implementation's,
so this measures propagation and nothing else -- no sampling, no plotting.
See vortex_precession_animation.py for the picture of what is propagating.

The step is fixed rather than left to the solver: a benchmark compares cost
per unit of work, and letting each grid size choose its own step would change
the work between the cells being compared.
"""

import os
import time

import matplotlib.pyplot as plt
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

# A documentation build sets NLSE_DOCS_BUILD and gets the small sweep, so the
# gallery shows a real graph made by this script rather than a placeholder.
# Run it yourself and you get the full range.
sizes = (
    [128, 256, 512]
    if os.environ.get("NLSE_DOCS_BUILD")
    else [128, 256, 512, 1024, 2048, 4096, 8192]
)
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
            splitting="lie",
            delta_z=DELTA_Z,
        )
        ts[i, rep] = time.perf_counter() - t0
    timing_string = f"Average time: {np.mean(ts[i]):.2f} s "
    timing_string += f"(min: {np.min(ts[i]):.2f} s, max: {np.max(ts[i]):.2f} s)"
    print(timing_string)

np.save(output_path(f"python_vortex_precession_{BACKEND}_times.npy"), ts)
np.save(output_path(f"python_vortex_precession_{BACKEND}_sizes.npy"), sizes)

# The timings are the point of the script, so draw them rather than leaving
# the reader to open a .npy. Best of the repeats, which is the figure least
# disturbed by whatever else the machine was doing.
best = ts.min(axis=1)
fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
ax.bar([str(size) for size in sizes], best, color="tab:blue")
for x, value in enumerate(best):
    ax.text(x, value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("grid size")
ax.set_ylabel("time for the propagation (s)")
ax.set_title(f"Vortex precession on {BACKEND}, best of {navg}")
ax.margins(y=0.15)
plt.show()
