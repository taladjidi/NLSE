"""Plots draw a spot where it is."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from NLSE import CNLSE, NLSE, NLSE_3d
from scipy.constants import c

N = 32
NZ = 16  # time samples of NLSE_3d, a power of two for the FFT
WINDOW = 8.92e-3  # m
DURATION = 2e-6  # s, temporal window of NLSE_3d over 8
MM = 1e3  # mm per m, the plots' length unit
PER_MM = 1e-3  # 1/mm per 1/m, the plots' wavevector unit
US = 1e6  # us per s, the NLSE_3d xt panels' time unit
# Where the spot sits, in windows from the centre: (-1/4, +1/4) is on grid
# points and in the upper-left quadrant.
SPOT = (-0.25, 0.25)
# Where the tilt sits, in grid wavevectors: kx < 0, ky > 0.
TILT = (-N // 8, N // 8)
# Where the NLSE_3d pulse sits, as grid indices: transverse position above
# the centre, time before it.
PULSE = (3 * N // 4, NZ // 4)
# (NX, NY) of the NLSE_3d grids that are not square: fewer columns than
# rows, and more.
OBLONG = [(N // 2, N), (N, N // 2)]


def nlse(backend):
    """Return a small NLSE.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    NLSE
        The solver.
    """
    return NLSE(
        alpha=0,
        power=1.05,
        window=WINDOW,
        n2=-1.6e-9,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        Isat=1e5,
        backend=backend,
    )


def cnlse(backend):
    """Return a small CNLSE.

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    CNLSE
        The solver.
    """
    return CNLSE(
        alpha=0,
        power=1.05,
        window=WINDOW,
        n2=-1.6e-9,
        n12=-1e-10,
        V=None,
        L=1e-2,
        NX=N,
        NY=N,
        Isat=1e5,
        backend=backend,
    )


def nlse_3d(backend, NX=N, NY=N):
    """Return a small NLSE_3d.

    Parameters
    ----------
    backend : str
        Backend name.
    NX, NY : int
        Points along x and y.

    Returns
    -------
    NLSE_3d
        The solver.
    """
    return NLSE_3d(
        alpha=0,
        energy=1.05 * DURATION,
        window=np.array([WINDOW, 8 * DURATION]),
        n2=-1.6e-9,
        D0=1e-16,
        vg=1e-1 * c,
        V=None,
        L=1e-2,
        NX=NX,
        NY=NY,
        NZ=NZ,
        Isat=1e5,
        backend=backend,
    )


def spot(simu):
    """Return the spot's position in m and a field peaked there.

    Parameters
    ----------
    simu : NLSE
        The solver.

    Returns
    -------
    tuple
        ``(x0, y0, field)``.
    """
    x0, y0 = SPOT[0] * simu.window[0], SPOT[1] * simu.window[1]
    width = 2 * simu.delta_X
    blob = np.exp(-((simu.XX - x0) ** 2 + (simu.YY - y0) ** 2) / width**2)
    return x0, y0, (blob * np.exp(1j * blob)).astype(np.complex64)


def drawn_at(image, x, y):
    """Return the value ``image`` draws at data coordinates ``(x, y)``.

    Parameters
    ----------
    image : matplotlib.image.AxesImage
        The panel's image.
    x, y : float
        Data coordinates, in the panel's units.

    Returns
    -------
    float
        The array value drawn there.
    """
    canvas = image.axes.figure.canvas
    canvas.draw()
    px, py = image.axes.transData.transform((x, y))
    return image.get_cursor_data(MouseEvent("motion_notify_event", canvas, px, py))


def plotted_images(simu, field):
    """Plot ``field`` from the solver's backend and return the panels' images.

    Parameters
    ----------
    simu : NLSE
        The solver.
    field : np.ndarray
        Field on the host.

    Returns
    -------
    list
        One ``AxesImage`` per image panel, in drawing order.
    """
    plt.close("all")
    simu.plot_field(simu._backend.from_numpy(np.ascontiguousarray(field)), 0.0)
    return [axes.images[0] for axes in plt.gcf().axes if axes.images]


@pytest.mark.parametrize(
    "build,components,real_panels", [(nlse, 1, 2), (cnlse, 2, 4), (nlse_3d, 1, 2)]
)
def test_a_spot_is_drawn_where_it_is(backend, build, components, real_panels):
    """Every real-space panel shows its largest value at the spot."""
    simu = build(backend)
    x0, y0, field = spot(simu)
    if components > 1:
        field = np.stack([field] * components)
    images = plotted_images(simu, field)[:real_panels]
    for image in images:
        drawn = drawn_at(image, x0 * MM, y0 * MM)
        assert drawn == np.max(image.get_array()), (
            f"{backend}: {image.axes.get_title()} is not drawn at its spot"
        )
    plt.close("all")


def test_a_tilt_is_drawn_at_its_wavevector(backend):
    """The Fourier panel shows a tilted plane wave at its ``(kx, ky)``."""
    simu = nlse(backend)
    kx, ky = simu.Kx[TILT[0]], simu.Ky[TILT[1]]
    field = np.exp(1j * (kx * simu.XX + ky * simu.YY)).astype(np.complex64)
    fourier = plotted_images(simu, field)[2]
    drawn = drawn_at(fourier, kx * PER_MM, ky * PER_MM)
    assert drawn == np.max(fourier.get_array())
    plt.close("all")


def test_a_pulse_is_drawn_where_it_is_in_the_xt_panels(backend):
    """Both NLSE_3d ``xt`` panels show a pulse at its position and time."""
    simu = nlse_3d(backend)
    r0, t0 = simu.Y[PULSE[0]], simu.T[PULSE[1]]
    width, duration = 2 * simu.delta_X, 2 * simu.delta_T
    ridges = np.exp(-((simu.XX - r0) ** 2) / width**2) + np.exp(
        -((simu.YY - r0) ** 2) / width**2
    )
    amplitude = ridges * np.exp(-((simu.TT - t0) ** 2) / duration**2)
    field = (amplitude * np.exp(1j * amplitude)).astype(np.complex64)
    for image in plotted_images(simu, field)[2:4]:
        drawn = drawn_at(image, t0 * US, r0 * MM)
        assert drawn == np.max(image.get_array()), (
            f"{backend}: {image.axes.get_title()} is not drawn at its pulse"
        )
    plt.close("all")


@pytest.mark.parametrize("NX,NY", OBLONG)
def test_the_xt_panels_hold_x_and_t_on_a_grid_that_is_not_square(backend, NX, NY):
    """Both NLSE_3d ``xt`` panels are ``x`` by ``t``, cut at the middle row."""
    simu = nlse_3d(backend, NX, NY)
    x0, y0, t0 = simu.X[3 * NX // 4], simu.Y[NY // 2], simu.T[PULSE[1]]
    width, duration = 2 * simu.delta_X, 2 * simu.delta_T
    amplitude = np.exp(
        -((simu.XX - x0) ** 2) / width**2 - (simu.TT - t0) ** 2 / duration**2
    ) * (simu.YY == y0)
    field = (amplitude * np.exp(1j * amplitude)).astype(np.complex64)
    for image in plotted_images(simu, field)[2:4]:
        assert image.get_array().shape == (NX, NZ)
        drawn = drawn_at(image, t0 * US, x0 * MM)
        assert drawn > 0, f"{backend}: {image.axes.get_title()} misses the row"
        assert drawn == np.max(image.get_array())
    plt.close("all")
