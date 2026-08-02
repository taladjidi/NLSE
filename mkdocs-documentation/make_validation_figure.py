"""Draw the figure on docs/physical_validation.md, from the same closed forms.

Run from anywhere:

    python mkdocs-documentation/make_validation_figure.py

The curves are the analytic solutions and the markers are what the solver
returns; the point of the figure is that they lie on top of each other, so
nothing here fits anything. The tolerances the test suite asserts are tighter
than a plot can show -- see tests/physics/test_closed_forms.py -- which is why
the residuals go in the panel titles.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from NLSE import NLSE, NLSE_1d

OUT = pathlib.Path(__file__).resolve().parent / "docs" / "img" / "validation.png"

WINDOW = 4e-3
WAIST = 200e-6
N = 256


def linear_solver(**kwargs):
    """Return a lossless, linear 2D solver unless told otherwise."""
    base = {
        "alpha": 0,
        "power": 1,
        "window": WINDOW,
        "n2": 0,
        "V": None,
        "L": 1.0,
        "NX": N,
        "NY": N,
        "Isat": 1e30,
        "backend": "CPU",
    }
    return NLSE(**{**base, **kwargs})


def diffraction():
    """Return the beam radius beside w0 sqrt(1 + (z/zR)^2)."""
    simu = linear_solver()
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    rayleigh = simu.k * WAIST**2 / 2
    ratios = np.linspace(0, 2.5, 11)
    measured = []
    for ratio in ratios:
        out = np.asarray(
            simu.out_field(
                field.copy(),
                max(ratio, 1e-9) * rayleigh,
                verbose=False,
                plot=False,
                normalize=False,
            )
        )
        intensity = np.abs(out) ** 2
        measured.append(
            np.sqrt(
                2 * np.sum(intensity * (simu.XX**2 + simu.YY**2)) / np.sum(intensity)
            )
        )
    exact = WAIST * np.sqrt(1 + ratios**2)
    return ratios, np.array(measured) * 1e6, exact * 1e6


def beer():
    """Return transmitted intensity beside exp(-alpha z)."""
    alpha, distances = 20.0, np.linspace(0, 0.1, 11)
    simu = linear_solver(alpha=alpha, NX=64, NY=64)
    field = np.ones((64, 64), dtype=np.complex64)
    measured = [
        float(
            np.abs(
                np.asarray(
                    simu.out_field(
                        field.copy(),
                        max(z, 1e-9),
                        delta_z=1e-4,
                        verbose=False,
                        plot=False,
                        normalize=False,
                    )
                )[32, 32]
            )
            ** 2
        )
        for z in distances
    ]
    return distances, np.array(measured), np.exp(-alpha * distances)


def soliton():
    """Return a sech profile that diffraction and self-focusing hold in place."""
    simu = NLSE_1d(
        alpha=0,
        power=1,
        window=WINDOW,
        n2=+1e-9,
        V=None,
        L=1.0,
        NX=2048,
        Isat=1e30,
        backend="CPU",
    )
    amplitude = 4000.0
    width = 1 / (amplitude * np.sqrt(simu.k * simu._constant("_g")))
    field = (amplitude / np.cosh(simu.X / width)).astype(np.complex128)
    out = np.asarray(
        simu.out_field(
            field.copy(),
            10 * simu.k * width**2,
            delta_z=simu.k * width**2 / 500,
            verbose=False,
            plot=False,
            normalize=False,
            splitting="strang",
        )
    )
    # The same profile with no nonlinearity, to show what it is being held against.
    linear = NLSE_1d(
        alpha=0,
        power=1,
        window=WINDOW,
        n2=0,
        V=None,
        L=1.0,
        NX=2048,
        Isat=1e30,
        backend="CPU",
    )
    spread = np.asarray(
        linear.out_field(
            field.copy(),
            10 * simu.k * width**2,
            delta_z=simu.k * width**2 / 500,
            verbose=False,
            plot=False,
            normalize=False,
            splitting="strang",
        )
    )
    return simu.X * 1e6, np.abs(field), np.abs(out), np.abs(spread), width


def main():
    """Draw the three panels and write the figure."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ratios, measured, exact = diffraction()
    residual = np.max(np.abs(measured - exact) / exact)
    axes[0].plot(ratios, exact, "-", lw=2, label=r"$w_0\sqrt{1+(z/z_R)^2}$")
    axes[0].plot(ratios, measured, "o", ms=5, mfc="none", label="solver")
    axes[0].set_xlabel(r"$z / z_R$")
    axes[0].set_ylabel(r"beam radius ($\mu$m)")
    axes[0].set_title(f"Gaussian diffraction\nmax error {residual:.1e}")

    distances, transmitted, law = beer()
    residual = np.max(np.abs(transmitted - law))
    axes[1].semilogy(distances * 100, law, "-", lw=2, label=r"$e^{-\alpha z}$")
    axes[1].semilogy(
        distances * 100, transmitted, "o", ms=5, mfc="none", label="solver"
    )
    axes[1].set_xlabel("z (cm)")
    axes[1].set_ylabel(r"$I/I_0$")
    axes[1].set_title(
        f"Beer's law, $\\alpha$ = 20 m$^{{-1}}$\nmax error {residual:.1e}"
    )

    x, start, kept, spread, width = soliton()
    residual = np.max(np.abs(kept - start)) / start.max()
    window = np.abs(x) < 8 * width * 1e6
    axes[2].plot(x[window], start[window], "-", lw=2, label="input sech")
    axes[2].plot(x[window], kept[window], "o", ms=4, mfc="none", label="after 10 $z_R$")
    axes[2].plot(x[window], spread[window], "--", lw=1, label="same, without $n_2$")
    axes[2].set_xlabel(r"x ($\mu$m)")
    axes[2].set_ylabel(r"$|A|$")
    axes[2].set_title(f"Bright soliton\nshape change {residual:.1e}")

    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
