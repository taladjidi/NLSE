"""
Against FourierGPE.jl
=====================

The same condensate solved here and in Julia, for comparison.
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from _output import output_path
from cycler import cycler

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

NEEDED = (
    "python_vortex_precession_CPU_times.npy",
    "python_vortex_precession_CPU_sizes.npy",
    "python_vortex_precession_GPU_times.npy",
    "python_vortex_precession_GPU_sizes.npy",
    "julia_vortex_precession_times.npy",
    "julia_vortex_precession_sizes.npy",
)
_missing = [name for name in NEEDED if not pathlib.Path(name).exists()]
if _missing:
    print(
        "This plots timings that two other runs produce; "
        f"{len(_missing)} of {len(NEEDED)} files are not here yet "
        f"(first: {_missing[0]}).\n"
        "Run examples/vortex_precession_benchmark.py for the Python side, and "
        "the Julia benchmark for the other."
    )
    sys.exit(0)

ts_py_CPU = np.load("python_vortex_precession_CPU_times.npy")
sizes_py_CPU = np.load("python_vortex_precession_CPU_sizes.npy")
ts_py_GPU = np.load("python_vortex_precession_GPU_times.npy")
sizes_py_GPU = np.load("python_vortex_precession_GPU_sizes.npy")
ts_jl_CPU = np.load("julia_vortex_precession_times.npy")
sizes_jl_CPU = np.load("julia_vortex_precession_sizes.npy")
fig, ax = plt.subplots()
err_low = np.mean(ts_py_CPU, axis=-1) - np.min(ts_py_CPU, axis=-1)
err_high = np.max(ts_py_CPU, axis=-1) - np.mean(ts_py_CPU, axis=-1)
ax.errorbar(
    np.log2(sizes_py_CPU),
    np.mean(ts_py_CPU, axis=-1),
    yerr=[err_low, err_high],
    label="NLSE CPU",
    marker="o",
    capsize=4,
)
err_low = np.mean(ts_py_GPU, axis=-1) - np.min(ts_py_GPU, axis=-1)
err_high = np.max(ts_py_GPU, axis=-1) - np.mean(ts_py_GPU, axis=-1)
ax.errorbar(
    np.log2(sizes_py_GPU),
    np.mean(ts_py_GPU, axis=-1),
    yerr=[err_low, err_high],
    label="NLSE GPU",
    marker="s",
    capsize=4,
)
err_low = np.mean(ts_jl_CPU, axis=-1) - np.min(ts_jl_CPU, axis=-1)
err_high = np.max(ts_jl_CPU, axis=-1) - np.mean(ts_jl_CPU, axis=-1)
ax.errorbar(
    np.log2(sizes_jl_CPU),
    np.mean(ts_jl_CPU, axis=-1),
    yerr=[err_low, err_high],
    label="JuliaGPE.jl",
    marker="^",
    capsize=4,
)
# ax.plot(
#     sizes_jl_CPU,
#     np.min(ts_jl_CPU, axis=-1),
#     label="Julia min",
#     marker="^",
# )
# ax.plot(
#     sizes_jl_CPU,
#     np.max(ts_jl_CPU, axis=-1),
#     label="Julia",
#     marker="^",
# )
ax.legend()
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Time (s)")
ax.set_xticks(np.log2(sizes_py_CPU).astype(int))

ax.set_yscale("log")
ax.set_title("Vortex precession")
fig.savefig(output_path("vortex_precession.svg"), dpi=300)
plt.show()
