"""
Against FourierGPE.jl
=====================

The same condensate solved here and in Julia, for comparison.

It plots what two other runs write to ``examples/output``:
``vortex_precession_benchmark.py`` for every backend it found, and
``vortex_precession.jl`` for FourierGPE.jl. Both leave the first run of each
size, which compiles and plans the transform, out of their timings.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _output import output_path
from cycler import cycler

# What the two benchmark runs write, under examples/output.
PYTHON_TIMES = "python_vortex_precession_{}_times.npy"
PYTHON_SIZES = "python_vortex_precession_{}_sizes.npy"
JULIA_TIMES = "julia_vortex_precession_times.npy"
JULIA_SIZES = "julia_vortex_precession_sizes.npy"
# Every name vortex_precession_benchmark.py can write timings under.
BACKENDS = ("CPU", "CUPY", "CL", "MLX")
MARKERS = ("o", "s", "D", "v")

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


def plot_timings(ax, sizes, times, label, marker):
    """Draw the mean time per grid size, with the spread of the repeats.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    sizes : np.ndarray
        Grid side lengths, in points.
    times : np.ndarray
        Times in s, one row per size and one column per repeat.
    label : str
        Legend entry.
    marker : str
        Matplotlib marker.
    """
    mean = np.mean(times, axis=-1)
    ax.errorbar(
        np.log2(sizes),
        mean,
        yerr=[mean - np.min(times, axis=-1), np.max(times, axis=-1) - mean],
        label=label,
        marker=marker,
        capsize=4,
    )


found = [b for b in BACKENDS if output_path(PYTHON_TIMES.format(b)).exists()]
if not found or not output_path(JULIA_TIMES).exists():
    print(
        "This plots timings that two other runs produce, and they are not in "
        f"{output_path(JULIA_TIMES).parent} yet.\n"
        "Run examples/vortex_precession_benchmark.py for the Python side, and "
        "examples/vortex_precession.jl for the Julia one."
    )
    sys.exit(0)

fig, ax = plt.subplots()
for backend, marker in zip(BACKENDS, MARKERS, strict=True):
    if backend in found:
        plot_timings(
            ax,
            np.load(output_path(PYTHON_SIZES.format(backend))),
            np.load(output_path(PYTHON_TIMES.format(backend))),
            f"NLSE {backend}",
            marker,
        )
sizes_jl = np.load(output_path(JULIA_SIZES))
plot_timings(ax, sizes_jl, np.load(output_path(JULIA_TIMES)), "FourierGPE.jl", "^")
ax.legend()
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Time (s)")
ax.set_xticks(np.log2(sizes_jl).astype(int))
ax.set_yscale("log")
ax.set_title("Vortex precession")
fig.savefig(output_path("vortex_precession.svg"), dpi=300)
plt.show()
