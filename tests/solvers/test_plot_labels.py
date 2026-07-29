"""Each solver plots its own quantity, in its own units.

``plot_field`` was written out in full for six solvers, ~50 lines of matplotlib
each, differing only in what the density means and what to call it: an optical
intensity in W/cm^2 for NLSE and CNLSE, a particle density for GPE and DDGPE,
propagation distance in metres against time in picoseconds, psi_1/psi_2 against
psi_x/psi_c. The plotting is shared now and the differences are declared.

Which means the declarations have to be checked. The tests that existed only
asserted plot_field ran, so a solver drawing the wrong quantity under the wrong
label would have passed.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from NLSE import CNLSE, DDGPE, GPE, NLSE
from scipy.constants import atomic_mass, c, epsilon_0

N = 32
H_BAR = 0.654


def nlse():
    """Return a small NLSE."""
    return NLSE(
        alpha=0,
        power=1.05,
        window=8.92e-3,
        n2=-1.6e-9,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        Isat=1e5,
        backend="CPU",
    )


def gpe():
    """Return a small GPE."""
    return GPE(
        gamma=0.1,
        N=1e5,
        window=100e-6,
        g=1e3,
        V=None,
        m=87 * atomic_mass,
        NX=N,
        NY=N,
        backend="CPU",
    )


def cnlse():
    """Return a small CNLSE."""
    return CNLSE(
        alpha=0,
        power=1.05,
        window=8.92e-3,
        n2=-1.6e-9,
        n12=-1e-10,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        Isat=1e5,
        backend="CPU",
    )


def ddgpe():
    """Return a small DDGPE."""
    return DDGPE(
        gamma=0.1,
        power=1.0,
        window=256,
        g=1e-2 / H_BAR,
        g12=0,
        omega=5.07 / H_BAR,
        T=1,
        omega_exc=1484.44 / H_BAR,
        omega_cav=1482.76 / H_BAR,
        detuning=0.17 / H_BAR,
        k_z=27,
        V=None,
        NX=N,
        NY=N,
        backend="CPU",
    )


def draw(simu, components):
    """Plot a flat field and return (suptitle, all axis titles, all labels)."""
    plt.close("all")
    shape = (components, N, N) if components > 1 else (N, N)
    simu.plot_field(np.ones(shape, dtype=np.complex64), 0.5)
    fig = plt.gcf()
    titles = [a.get_title() for a in fig.axes]
    labels = [a.get_ylabel() for a in fig.axes] + [a.get_xlabel() for a in fig.axes]
    return fig._suptitle.get_text(), titles, labels


@pytest.mark.parametrize(
    "build,components,symbol,unit",
    [
        (nlse, 1, "z", "m"),
        (gpe, 1, "z", "m"),
        (cnlse, 2, "z", "m"),
        (ddgpe, 2, "t", "ps"),
    ],
    ids=["NLSE", "GPE", "CNLSE", "DDGPE"],
)
def test_the_axis_is_named_and_given_units(build, components, symbol, unit):
    """A polariton run is a time evolution; the title has to say so."""
    suptitle, _, _ = draw(build(), components)
    assert f"${symbol}$" in suptitle, f"expected ${symbol}$ in {suptitle!r}"
    assert suptitle.rstrip().endswith(unit), f"expected {unit!r} in {suptitle!r}"


@pytest.mark.parametrize(
    "build,components,expected",
    [
        (nlse, 1, "Intensity"),
        (gpe, 1, "Density"),
        (cnlse, 2, "Intensity"),
        (ddgpe, 2, "Density"),
    ],
    ids=["NLSE", "GPE", "CNLSE", "DDGPE"],
)
def test_the_density_is_labelled_for_what_it_is(build, components, expected):
    """An optical intensity and a particle density are not the same quantity."""
    _, _, labels = draw(build(), components)
    assert any(expected in x for x in labels), (
        f"no axis labelled {expected!r}; got {[x for x in labels if x]}"
    )


@pytest.mark.parametrize(
    "build,expected",
    [(cnlse, (r"\psi_1", r"\psi_2")), (ddgpe, (r"\psi_x", r"\psi_c"))],
    ids=["CNLSE", "DDGPE"],
)
def test_coupled_components_are_named(build, expected):
    """DDGPE's two components are an exciton and a cavity, not 1 and 2."""
    _, titles, _ = draw(build(), 2)
    joined = " ".join(titles)
    for name in expected:
        assert name in joined, f"{name!r} missing from panel titles {titles}"


def test_the_optical_density_is_converted_and_the_quantum_one_is_not():
    """The scale is the substantive difference, not just the wording.

    NLSE draws |E|^2 in W/cm^2; GPE draws |psi|^2 as it stands. Getting this
    wrong misstates the plotted values by 1e-4 * c * eps0 / 2, about 1.3e-7.
    """
    assert nlse()._plot_density_scale == pytest.approx(c * epsilon_0 / 2 * 1e-4)
    assert gpe()._plot_density_scale == 1.0
    assert cnlse()._plot_density_scale == pytest.approx(c * epsilon_0 / 2 * 1e-4)
    assert ddgpe()._plot_density_scale == 1.0
