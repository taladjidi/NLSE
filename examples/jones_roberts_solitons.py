r"""
Jones-Roberts solitons
======================

The soliton family of a 2D superfluid, and the trip along it, after
`Baker-Rasooli et al. <https://arxiv.org/abs/2501.08383>`_,
Phys. Rev. Lett. **134**, 233401 (2025).

A one-dimensional fluid has dark solitons. A two-dimensional one has a whole
family of solitary waves, found by Jones and Roberts in 1982, and the family
has two halves: a **vortex pair** at low speed, a **rarefaction pulse** -- a
density dip carrying no circulation at all -- at high speed, and a critical
speed where one becomes the other. What makes it a family rather than two
unrelated objects is that a single soliton can travel along it.

That is what this page shows, in a single propagation. Two counter-rotating
circulations are imprinted on the beam, as the experiment imprints them with a
spatial light modulator, and left alone. They drift together; the depletion
between them fills in; and the two phase singularities meet and
**annihilate**, leaving behind a rarefaction pulse -- a solitary wave that
still travels, still holds together, and has no circulation anywhere in it.
Nothing is done to it in between. The soliton walks from one half of the
family into the other on its own.

**The parameters are the experiment's**, from the paper's supplementary
material: 5 W in a 2.23 mm waist through 20 cm of rubidium at 150 °C, 780 nm
and about -10 GHz detuned, :math:`n_2 = -1.04\times10^{-10}\,\mathrm{m^2/W}`,
:math:`I_{sat} = 60\,\mathrm{W/cm^2}`, and the ~30% losses the paper quotes.
Two checks that they are the right ones: they put the beam at 71 healing
lengths across, against the fluid radius of :math:`63\,\xi` the paper states,
and they put the 20 cm cell at :math:`\tau = 52`, against the
:math:`\tau > 52` the paper gives for the onset of the transition -- the cell
ends just about where the circulation dies.

The experiment reaches other :math:`\tau` by scanning the fluid beam power,
since its cell has one length. A simulation has the whole propagation in hand,
so this one simply runs on past the cell to :math:`\tau = 100` and watches.

.. note::

   Two conventions travel with this paper and neither is the Bogoliubov one.
   Its nonlinear length is :math:`z_{NL} = 1/k_0\Delta n` and its healing
   length :math:`\xi = \sqrt{z_{NL}/k_0}`, both built from :math:`\Delta n`
   rather than from :math:`c_s^2 = I\,\partial\Delta n/\partial I`. In a
   saturable medium those differ by :math:`\sqrt{1+I/I_{sat}}`, which is 1.44
   here -- 21.9 µm against 31.4 µm. The paper's are used throughout, so that
   :math:`\tau` and :math:`\Delta r/\xi` mean what they mean in the paper.

.. warning::

   The step size for a run like this is a **stability** condition, not an
   accuracy one. Split-step applies the linear part exactly in Fourier space,
   so dispersion cannot limit accuracy on its own -- but on a finite-amplitude
   background the nonlinear step couples modes, and past about a radian of
   linear phase per step at the grid's highest wavenumber the resonant modes
   grow out of round-off. It does not look numerical when it happens: the
   background fills with density fluctuations of order the density itself and
   thousands of spurious vortices, which reads as the fluid going turbulent.
   The solver enforces the bound itself and warns when it reduces the step.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from NLSE import NLSE
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# %%
# The experiment's parameters, from the supplementary material.

N2 = -1.04e-10  # nonlinear index in m^2/W, defocusing
I_SAT = 60e4  # saturation intensity, 60 W/cm^2
POWER = 5.0  # fluid beam power in W
WAIST = 2.23e-3  # fluid beam waist in m
L_CELL = 20e-2  # cell length in m
TRANSMISSION = 0.7  # the paper's ~30% losses
WVL = 780e-9  # 87Rb D2, about -10 GHz detuned, cell at 150 C
N_PTS = 1024

K0 = 2 * np.pi / WVL
I0 = 2 * POWER / (np.pi * WAIST**2)  # peak intensity of the Gaussian
DN = abs(N2) * I0 / (1 + I0 / I_SAT)  # saturated index depression
Z_NL = 1 / (K0 * DN)  # the paper's nonlinear length
XI = np.sqrt(Z_NL / K0)  # the paper's healing length
XI_BOGOLIUBOV = (1 + I0 / I_SAT) / (K0 * np.sqrt(abs(N2) * I0))

# The window holds the beam and the dark surround it decays into. That the
# beam is Gaussian rather than uniform matters for more than realism: the
# intensity falls to nothing at the edges, so the periodic boundary has no
# fluid to join up. A dipole's phase is only periodic to order dr/W, and on a
# uniform background that mismatch radiates from the seam and fringes the
# whole picture.
WINDOW = 3.5 * WAIST
DX = WINDOW / N_PTS

# Past the cell, which is tau = 52 -- where the paper puts the onset of the
# transition -- so the rarefaction pulse can be watched for a while after it
# forms rather than only at the moment it does.
TAU_END = 100
LENGTH = TAU_END * Z_NL

print(
    f"medium: Dn = {DN:.2e}, z_NL = {Z_NL * 1e3:.2f} mm, xi = {XI * 1e6:.1f} um "
    f"(Bogoliubov xi would be {XI_BOGOLIUBOV * 1e6:.1f} um)"
)
print(
    f"beam:   {POWER} W, waist {WAIST * 1e3:.2f} mm = {WAIST / XI:.0f} xi, "
    f"I0 = {I0 / I_SAT:.2f} Isat"
)
print(
    f"cell:   {L_CELL * 1e2:.0f} cm = tau {L_CELL / Z_NL:.0f}; this run goes to "
    f"tau {TAU_END}"
)
print(f"grid:   {N_PTS}^2, dx = xi/{XI / DX:.1f}, window = {WINDOW / XI:.0f} xi")


def small_signal_alpha():
    """Return the alpha whose saturable absorption gives TRANSMISSION.

    The medium bleaches where the beam is bright, so the loss is not
    exponential and ``-log(T)/L`` is not the coefficient to hand the solver.
    Solved on the radial profile instead.

    Returns
    -------
    float
        Small-signal absorption coefficient in 1/m.
    """
    r = np.linspace(0, 3 * WAIST, 800)
    profile = I0 * np.exp(-2 * r**2 / WAIST**2)

    def transmitted(alpha):
        """Power fraction left after the cell at this alpha."""
        solution = solve_ivp(
            lambda z, intensity: -alpha * intensity / (1 + intensity / I_SAT),
            (0, L_CELL),
            profile,
            rtol=1e-9,
            atol=1e-12,
        )
        return np.trapezoid(solution.y[:, -1] * 2 * np.pi * r, r) / np.trapezoid(
            profile * 2 * np.pi * r, r
        )

    return brentq(lambda a: transmitted(a) - TRANSMISSION, 1e-4, 500, xtol=1e-9)


ALPHA = small_signal_alpha()
print(
    f"loss:   alpha = {ALPHA:.2f}/m for {TRANSMISSION:.0%} over the cell "
    f"(a plain exponential would say {-np.log(TRANSMISSION) / L_CELL:.2f})"
)


def vortex(simu, x0, y0, charge):
    """Return one vortex: a phase winding with a healed core.

    Parameters
    ----------
    simu : NLSE
        Solver, for its coordinate grids.
    x0, y0 : float
        Core position in m.
    charge : int
        Sign of the circulation.

    Returns
    -------
    ndarray
        Complex field of unit modulus far from the core.

    Notes
    -----
    The core profile only needs the right asymptotics; whatever is imprinted
    radiates its excess within a few nonlinear lengths and relaxes onto the
    true one. A Pade profile was tried and left no less sound behind, which
    says the relaxation is set by the dipole and not by the core.
    """
    x, y = simu.XX - x0, simu.YY - y0
    r = np.hypot(x, y)
    return r / np.sqrt(r**2 + XI**2) * np.exp(1j * charge * np.arctan2(y, x))


def singularities(field):
    """Return the pixel coordinates of the phase singularities.

    How many there are is what says which half of the family the soliton is
    in: two for a vortex pair, none for a rarefaction pulse.

    Parameters
    ----------
    field : ndarray
        Complex field.

    Returns
    -------
    tuple of ndarray
        Column and row indices.
    """
    phase = np.angle(field)
    dx = np.angle(np.exp(1j * (np.roll(phase, -1, axis=1) - phase)))
    dy = np.angle(np.exp(1j * (np.roll(phase, -1, axis=0) - phase)))
    curl = dx + np.roll(dy, -1, axis=1) - np.roll(dx, -1, axis=0) - dy
    rows, cols = np.where(np.abs(curl) > np.pi)
    return cols, rows


# %%
# The imprint, and the propagation. The pair goes in **off-centre**, as the
# paper puts it: the Gaussian carries a density gradient, so the healing
# length the pair sits in changes as it travels, and it is that changing
# :math:`\Delta r/\xi` which walks the soliton along the family.

# Just above the critical separation. Below about 2.5 xi the pair annihilates
# within a couple of nonlinear lengths, which is over before the animation
# starts; above 3 xi it survives the whole run. This sits in between, so the
# transition happens on screen.
SEPARATION = 3.0 * XI
simu = NLSE(
    ALPHA, POWER, WINDOW, N2, None, LENGTH, NX=N_PTS, NY=N_PTS, Isat=I_SAT, wvl=WVL
)
envelope = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2)
E_in = (
    envelope
    * vortex(simu, -0.25 * WAIST, +SEPARATION / 2, +1)
    * vortex(simu, -0.25 * WAIST, -SEPARATION / 2, -1)
).astype(np.complex64)

frames, taus, gaps, counts = [], [], [], []


def follow(sim, A, z, i):
    """Sample the field, and how far apart the two circulations are."""
    if i % 120:
        return
    field = np.asarray(sim._backend.to_numpy(A))
    cols, rows = singularities(field)
    frames.append(field.copy())
    taus.append(z / Z_NL)
    counts.append(len(cols))
    gap = 0.0
    if len(cols) == 2:
        gap = float(np.hypot(np.diff(cols)[0], np.diff(rows)[0])) * sim.delta_X / XI
    gaps.append(gap)


simu.out_field(
    E_in, LENGTH, delta_z=Z_NL / 20, verbose=False, plot=False, callback=follow
)
taus, gaps, counts = np.array(taus), np.array(gaps), np.array(counts)
paired = counts == 2
print(f"\nsampled {len(frames)} planes out to tau = {taus[-1]:.0f}")
if (~paired).any():
    print(
        f"circulation absent between tau = {taus[~paired][0]:.0f} and "
        f"{taus[~paired][-1]:.0f}"
    )
else:
    print("the circulation never annihilated")

# %%
# The separation, and where the circulation goes. The two singularities close
# in, meet and vanish -- the soliton crossing from the vortex-pair half of the
# family into the rarefaction-pulse half -- and then reappear. The shaded
# stretch is where the field carries no circulation at all.

fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
ax.plot(taus[paired], gaps[paired], "o", ms=4, label=r"vortex pair, $\Delta r$")
if (~paired).any():
    ax.axvspan(
        taus[~paired][0],
        taus[~paired][-1],
        color="tab:red",
        alpha=0.15,
        label="no circulation (rarefaction pulse)",
    )
ax.axvline(L_CELL / Z_NL, color="gray", ls="--", lw=1.1)
ax.annotate(
    "end of the\n20 cm cell",
    xy=(L_CELL / Z_NL, 0.82),
    xycoords=("data", "axes fraction"),
    xytext=(6, 0),
    textcoords="offset points",
    fontsize=9,
    color="gray",
)
ax.set_xlabel(r"$\tau = z / z_{NL}$")
ax.set_ylabel(r"separation $\Delta r / \xi$")
ax.set_title("The circulation dies, and comes back")
ax.legend(fontsize=9)
plt.show()

# %%
# And the thing itself. Density on the left, phase on the right, with any
# circulations circled: watch the two dark cores approach, merge into a single
# dip whose phase is smooth all the way round, and then separate again with
# the circulation restored.

extent = [simu.X[0] / XI, simu.X[-1] / XI, simu.Y[0] / XI, simu.Y[-1] / XI]


def normalised(field):
    """Return the density scaled to its own peak.

    Each plane is scaled to itself rather than to the launch, so the 30% the
    vapour absorbs over the cell does not slowly dim the picture. It costs the
    absolute scale, which is not what this figure is about -- the soliton is a
    hole in the fluid, and what matters is how deep it is relative to the
    fluid around it.

    Parameters
    ----------
    field : ndarray
        Complex field.

    Returns
    -------
    ndarray
        Density in units of its own maximum.
    """
    density = np.abs(field) ** 2
    return density / density.max()


fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8), layout="constrained")
density = axs[0].imshow(
    normalised(frames[0]),
    cmap="viridis",
    origin="lower",
    extent=extent,
    vmin=0,
    vmax=1.05,
)
phase = axs[1].imshow(
    np.angle(frames[0]),
    cmap="twilight_shifted",
    origin="lower",
    extent=extent,
    vmin=-np.pi,
    vmax=np.pi,
)
marks = [ax.plot([], [], "o", mfc="none", mec="red", ms=11, mew=1.6)[0] for ax in axs]
for ax, label in zip(axs, ("density", "phase")):
    ax.set_xlim(-45, 45), ax.set_ylim(-30, 30)
    ax.set_xlabel(r"$x/\xi$"), ax.set_ylabel(r"$y/\xi$")
    ax.set_title(label)


def show(frame):
    """Draw one sampled plane, circling the circulations if there are any."""
    field = frames[frame]
    density.set_data(normalised(field))
    phase.set_data(np.angle(field))
    cols, rows = singularities(field)
    for mark in marks:
        mark.set_data(simu.X[cols] / XI, simu.Y[rows] / XI)
    state = "vortex pair" if len(cols) == 2 else "rarefaction pulse"
    fig.suptitle(rf"$\tau = {taus[frame]:.0f}$   --   {state}")
    return (density, phase, *marks)


# Bound at module level rather than dropped: sphinx-gallery looks through the
# example's namespace for an Animation and embeds what it finds there, so one
# that only exists inside a function is a still picture on the page.
anim = FuncAnimation(fig, show, frames=len(frames), interval=100, blit=False)
plt.show()
