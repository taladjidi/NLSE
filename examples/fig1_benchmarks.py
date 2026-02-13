import time

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from NLSE import NLSE
from NLSE.backends import list_available_backends

PRECISION = "single"
if PRECISION == "double":
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
# Add numpy naive as a special "backend"
ALL_LABELS = [*BACKENDS, "Numpy"]

sizes = np.logspace(6, 14, 9, base=2, dtype=int)
times = np.zeros((len(sizes), len(ALL_LABELS), N_avg))

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]
COLORS = plt.cm.tab10.colors

pbar = tqdm.tqdm(total=np.prod(times.shape), desc="Benchmarks")
for i, size in enumerate(sizes):
    E_0 = None
    simu_ref = None
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
        simu.delta_z = 1e-4
        if E_0 is None:
            E_0 = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / waist**2).astype(
                PRECISION_COMPLEX
            )
            simu_ref = simu
        for k in range(N_avg):
            t0 = time.perf_counter()
            simu.out_field(E_0.copy(), L, verbose=False)
            times[i, j, k] = time.perf_counter() - t0
            pbar.update(1)
    # numpy naive implementation
    j_np = len(BACKENDS)
    for k in range(N_avg):
        E1 = E_0.copy()
        t0 = time.perf_counter()
        for _ in range(int(L / simu_ref.delta_z)):
            E1 = np.fft.fft2(E1)
            E1 *= np.exp(1j * simu_ref.delta_z * simu_ref.propagator / (2 * simu_ref.k))
            E1 = np.fft.ifft2(E1)
            E1 *= np.exp(
                1j
                * simu_ref.delta_z
                * simu_ref.k
                * simu_ref.n2
                * np.abs(E1) ** 2
                / (1 + np.abs(E1) ** 2 / Isat)
            )
            E1 *= np.exp(-simu_ref.alpha * simu_ref.delta_z)
        times[i, j_np, k] = time.perf_counter() - t0
        pbar.update(1)
pbar.close()

fig, ax = plt.subplots()
for j, label in enumerate(ALL_LABELS):
    median = np.median(times[:, j, :], axis=-1)
    err = np.vstack(
        [
            median - np.min(times[:, j, :], axis=-1),
            np.max(times[:, j, :], axis=-1) - median,
        ]
    )
    ax.errorbar(
        np.log2(sizes).astype(int),
        median,
        yerr=err,
        label=label,
        marker=MARKERS[j % len(MARKERS)],
        color=COLORS[j % len(COLORS)],
        capsize=4,
    )
ax.legend()
ax.set_xticks(np.log2(sizes).astype(int))
ax.set_xlabel(r"Size of the system $2^N$")
ax.set_ylabel("Execution time in s")
ax.set_title("Execution time (lower is better)")
ax.set_yscale("log")
fig.savefig("benchmarks.pdf", dpi=300)
plt.show()
