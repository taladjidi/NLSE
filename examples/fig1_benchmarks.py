"""
Backends compared
=================

The backend comparison figure, drawn from a fresh run.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from _output import output_path
from NLSE import NLSE
from NLSE.backends import list_available_backends

PRECISION = "lie"
if PRECISION == "strang":
    PRECISION_REAL = np.float64
    PRECISION_COMPLEX = np.complex128
else:
    PRECISION_REAL = np.float32
    PRECISION_COMPLEX = np.complex64

n2 = -1.99e-9
n12 = -0.75e-10
waist = 2.29e-3
waist_d = 70e-6
nl_length = 0
d_real = 3.76e-6
d_fourier = 5.5e-6
f_fourier = 200e-3
window = 3008 * d_real
puiss = 1.05
Isat = 3.92e4  # saturation intensity in W/m^2
L = 2e-3
alpha = 22
N_avg = 2

BACKENDS = list_available_backends()
METHODS = ["split_step", "RK4"]
METHOD_MARKERS = {"split_step": "o", "RK4": "s"}
METHOD_LINESTYLES = {"split_step": "-", "RK4": "--"}

# A documentation build sets NLSE_DOCS_BUILD and gets the small sweep, so the
# gallery shows a real graph made by this script rather than a placeholder.
# Run it yourself and you get the full range.
sizes = (
    np.logspace(6, 9, 4, base=2, dtype=int)
    if os.environ.get("NLSE_DOCS_BUILD")
    else np.logspace(6, 13, 8, base=2, dtype=int)
)
times = np.zeros((len(sizes), len(BACKENDS), len(METHODS), N_avg))

COLORS = plt.cm.tab10.colors

pbar = tqdm.tqdm(
    total=len(sizes) * len(BACKENDS) * len(METHODS) * N_avg, desc="Benchmarks"
)
for i, size in enumerate(sizes):
    E_0 = None
    for j, backend in enumerate(BACKENDS):
        simu = NLSE(
            alpha,
            puiss,
            window,
            n2,
            None,
            L,
            NX=size,
            NY=size,
            nl_length=nl_length,
            backend=backend,
        )
        simu.I_sat = Isat
        if E_0 is None:
            E_0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(
                PRECISION_COMPLEX
            )
        for m, method in enumerate(METHODS):
            for k in range(N_avg):
                t0 = time.perf_counter()
                simu.out_field(
                    E_0.copy(), L, verbose=False, method=method, delta_z=1e-4
                )
                times[i, j, m, k] = time.perf_counter() - t0
                pbar.update(1)
pbar.close()

fig, ax = plt.subplots()
for j, backend in enumerate(BACKENDS):
    color = COLORS[j % len(COLORS)]
    for m, method in enumerate(METHODS):
        median = np.median(times[:, j, m, :], axis=-1)
        err = np.vstack(
            [
                median - np.min(times[:, j, m, :], axis=-1),
                np.max(times[:, j, m, :], axis=-1) - median,
            ]
        )
        ax.errorbar(
            np.log2(sizes).astype(int),
            median,
            yerr=err,
            label=f"{backend} ({method})",
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            color=color,
            capsize=4,
        )
ax.legend()
ax.set_xticks(np.log2(sizes).astype(int))
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Execution time in s")
ax.set_title("Execution time (lower is better)")
ax.set_yscale("log")
fig.savefig(output_path("benchmarks.pdf"), dpi=300)
plt.show()
