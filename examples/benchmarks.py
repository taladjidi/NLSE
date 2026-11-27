"""
Benchmarking every backend
==========================

Every backend available here against a hand-rolled numpy split step, over a
range of grid sizes.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from _output import output_path
from cycler import cycler
from NLSE import NLSE
from NLSE.backends import list_available_backends
from scipy.constants import c, epsilon_0

# Propagation step, passed to out_field and used by the plots below.
DELTA_Z = 1e-4
# Largest gap allowed between the numpy baseline and the CPU solver, relative to
# max |E|. Measured 9.7e-5 from 64^2 to 1024^2, since the baseline freezes |E|^2
# through a lossy step that the solver solves (_loss_factor, NLSE/kernels/cpu.py).
BASELINE_RTOL = 1e-3


def numpy_split_step(field, propagator, n_steps, g, alpha_half, isat_field):
    """Propagate with a Lie split step written in plain numpy.

    The same equation as ``NLSE.out_field(..., splitting="lie")``: the linear
    step in Fourier space, then the saturable interaction and loss in real
    space, with ``|E|^2`` read once per step.

    Parameters
    ----------
    field : np.ndarray
        Input field in V/m, at the power the solver normalizes to.
    propagator : np.ndarray
        ``exp(-i K^2 DELTA_Z / (2 k))`` on the unshifted Fourier grid.
    n_steps : int
        Number of steps of ``DELTA_Z``.
    g : float
        Interaction ``k n2 c epsilon_0 / 2``, in m/V^2.
    alpha_half : float
        Loss on the field, ``alpha / 2``, in m^-1.
    isat_field : float
        Saturation intensity as ``|E|^2``, ``2 I_sat / (c epsilon_0)``, in V^2/m^2.

    Returns
    -------
    np.ndarray
        The field after ``n_steps`` steps.
    """
    for _ in range(n_steps):
        field = np.fft.ifft2(np.fft.fft2(field) * propagator)
        intensity = np.abs(field) ** 2
        sat = 1 / (1 + intensity / isat_field)
        field = field * np.exp(DELTA_Z * (-alpha_half * sat + 1j * g * intensity * sat))
    return field


# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)

PRECISION_COMPLEX = np.complex64

n2 = -1.99e-9
waist = 2.29e-3
nl_length = 0
d_real = 3.76e-6
window = 3008 * d_real
puiss = 1.05
Isat = 3.92e4  # saturation intensity in W/m^2
L = 2e-3
alpha = 22
N_avg = 2
# A documentation build sets NLSE_DOCS_BUILD and gets the small sweep, so the
# gallery shows a real graph made by this script rather than a placeholder.
# Run it yourself and you get the full range.
sizes = (
    np.logspace(6, 9, 4, base=2, dtype=int)
    if os.environ.get("NLSE_DOCS_BUILD")
    else np.logspace(6, 14, 9, base=2, dtype=int)
)
# One column per backend that exists here, plus the hand-rolled numpy
# split step as the thing they are all being compared against. It used
# to ask for "CUPY" by name, which on a machine without it falls back to
# the CPU and draws two CPU curves under different labels.
BACKENDS = list_available_backends()
times = np.zeros((len(sizes), len(BACKENDS) + 1, N_avg))
pbar = tqdm.tqdm(total=np.prod(times.shape), desc="Benchmarks")
for i, size in enumerate(sizes):
    for j, backend in enumerate(BACKENDS):
        simu0 = NLSE(
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
        simu0.I_sat = Isat
        if j == 0:
            E_0 = np.exp(-(np.hypot(simu0.XX, simu0.YY) ** 2) / waist**2).astype(
                PRECISION_COMPLEX
            )
            # At the power out_field normalizes to, so the baseline starts there too.
            flux = c * epsilon_0 / 2 * np.sum(np.abs(E_0) ** 2)
            E_0 *= np.sqrt(puiss / (flux * simu0.delta_X * simu0.delta_Y))
        # Untimed: the first call compiles the kernels and plans the transforms.
        E_out = simu0.out_field(E_0, L, verbose=False, delta_z=DELTA_Z)
        if backend == "CPU":
            E_cpu = np.asarray(E_out)
        for k in range(N_avg):
            t0 = time.perf_counter()
            simu0.out_field(E_0, L, verbose=False, delta_z=DELTA_Z)
            times[i, j, k] = time.perf_counter() - t0
            pbar.update(1)
    # numpy split step, solving the same equation as the solver
    propagator = np.exp(
        -1j * DELTA_Z * (simu0.Kxx**2 + simu0.Kyy**2) / (2 * simu0.k)
    ).astype(PRECISION_COMPLEX)
    g = simu0.k / 2 * n2 * c * epsilon_0
    isat_field = 2 * Isat / (c * epsilon_0)
    for k in range(N_avg):
        E1 = E_0.copy()
        t0 = time.perf_counter()
        E1 = numpy_split_step(
            E1, propagator, round(L / DELTA_Z), g, alpha / 2, isat_field
        )
        times[i, len(BACKENDS), k] = time.perf_counter() - t0
        pbar.update(1)
    if i == 0:
        gap = np.max(np.abs(E1 - E_cpu)) / np.max(np.abs(E_cpu))
        if gap > BASELINE_RTOL:
            raise RuntimeError(
                f"The numpy baseline is {gap:.1e} of max |E| away from the CPU "
                f"solver, above {BASELINE_RTOL:.0e}: they no longer solve the "
                f"same equation, so the timings compare different work."
            )
        else:
            print(f"numpy baseline agrees with the CPU solver to {gap:.1e}")
pbar.close()
np.save(output_path("benchmarks_times.npy"), times)
np.save(output_path("benchmarks_sizes.npy"), sizes)
fig, ax = plt.subplots()
MARKERS = ["o", "s", "^", "v", "D"]
for j, label in enumerate([*BACKENDS, "Numpy"]):
    mean = np.mean(times[:, j, :], axis=-1)
    ax.errorbar(
        np.log2(sizes).astype(int),
        mean,
        yerr=[
            mean - np.min(times[:, j, :], axis=-1),
            np.max(times[:, j, :], axis=-1) - mean,
        ],
        label=label,
        marker=MARKERS[j % len(MARKERS)],
        capsize=4,
    )
ax.legend()
ax.set_xticks(np.log2(sizes).astype(int))
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Execution time in s")
ax.set_title(f"Execution time, lower is better -- {', '.join(BACKENDS)}")
# A backend comparison is only a comparison if there is more than one backend.
# The documentation runner has no GPU and no OpenCL, so the figure published
# with these docs has a single bar; say so on it rather than let a reader take
# CPU-only for the whole story.
if len(BACKENDS) == 1:
    ax.text(
        0.5,
        0.94,
        f"only {BACKENDS[0]} available here -- run this on a machine with a GPU",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )
ax.set_yscale("log")
fig.savefig(output_path("benchmarks.svg"), dpi=300)
plt.show()
