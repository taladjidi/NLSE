r"""
Jones-Roberts solitons
======================

The soliton family of a 2D superfluid, after
`Baker-Rasooli et al. <https://arxiv.org/abs/2501.08383>`_.

A one-dimensional fluid has dark solitons. A two-dimensional one has a whole
family of solitary waves instead, found by Jones and Roberts in 1982: at low
speed a pair of counter-rotating vortices, at high speed a density dip
carrying no circulation at all -- a *rarefaction pulse* -- and a critical
speed where one turns into the other. Every member travels without changing
shape, which is what the figures below are for.

The experiment makes them the way this script does: it imprints **two
counter-rotating phase circulations** on the beam with a spatial light
modulator and lets the medium do the rest. Nothing is relaxed beforehand and
no profile is fitted. The imprint is not a soliton -- it radiates a little
sound over the first few :math:`z_{nl}` and settles onto the nearest member of
the family -- so that the thing which emerges then holds its shape for the
rest of the cell is a result rather than an input.

**The speed is set by the separation, and by nothing else.** Two vortices of
opposite charge advect each other, so a dipole of separation :math:`\Delta r`
travels at

.. math:: v = c_s \frac{\xi}{\Delta r}

perpendicular to the line joining them. That is an *asymptotic* statement, and
the scan below shows it earning the name: the measured speed sits 2% above it
at :math:`\Delta r = 10\,\xi` and 20% above by :math:`3\,\xi`, where the cores
are close enough to stop being two separate objects. Squeeze them further and
the circulation annihilates altogether, leaving a rarefaction pulse -- still a
soliton, still travelling, but with no phase singularity to point at.

.. warning::

   **The step size is a stability condition here, not an accuracy one, and
   the solver does not impose it.** Split-step applies the linear part exactly
   in Fourier space, so dispersion cannot limit accuracy on its own, and
   ``_split_step_max_dz`` deliberately leaves the kinetic term out for that
   reason. The argument holds for a linear problem and fails for this one: on
   a finite-amplitude background the nonlinear step couples modes, and the
   linear phase at the grid's highest wavenumber resonates with it once
   :math:`k_{max}^2\,\delta z/2k_0` approaches :math:`\pi` -- the conditional
   instability of Weideman and Herbst. It does not announce itself as a
   numerical artefact: it looks like the fluid going turbulent. At 12.5 rad
   per step this example destroys its own background, density fluctuations of
   order the density itself and a thousand spurious vortices, and at 2.4 rad
   it still does. At 1.2 rad it is stable, and four further halvings of the
   step leave the answer unchanged to three digits. ``STEPS_PER_ZNL`` is set
   from that measurement, not from trial and error.
"""

import matplotlib.pyplot as plt
import numpy as np
from NLSE import NLSE

# %%
# The soliton family is a statement about a regime, not about a vapour cell,
# so everything below is written in the two scales that define the regime --
# the healing length and the sound speed -- and the medium is whatever
# produces them.

N2 = -1e-9  # nonlinear index in m^2/W, defocusing
I_SAT = 1e5  # saturation intensity in W/m^2
I0 = 2e4  # background intensity in W/m^2
WVL = 780e-9
N_PTS = 256

K0 = 2 * np.pi / WVL
# The Bogoliubov sound speed of a saturable medium carries the saturation
# denominator once more than the index depression does: c_s^2 = I dDn/dI.
DN = abs(N2) * I0 / (1 + I0 / I_SAT)
CS = np.sqrt(abs(N2) * I0) / (1 + I0 / I_SAT)
XI = 1 / (K0 * CS)  # healing length
Z_NL = K0 * XI**2  # nonlinear length, identically 1/(k0 c_s^2) and xi/c_s

# Sized in healing lengths, which is the only scale the physics knows about:
# four points across a core resolves it, and 64 of them leaves room for the
# soliton to cross a good fraction of the window without meeting its own wake.
WINDOW = 64 * XI
DX = WINDOW / N_PTS
POWER = I0 * WINDOW**2  # a uniform background over the whole window

# The stability criterion of the warning above, written out rather than
# tuned. The phase the linear step imprints on the shortest resolved wave is
# k_max^2 dz / 2k0 with k_max = pi/dx, and holding it near 1 rad bounds
#
#     dz / z_nl  <  2 / (pi^2 (xi/dx)^2)
#
# which is a condition on the grid as much as on the step: refining the grid
# demands a shorter step, quadratically. Derived here so that changing the
# resolution cannot silently invalidate it.
STEPS_PER_ZNL = int(np.ceil(np.pi**2 * (XI / DX) ** 2 / 2))
DELTA_Z = Z_NL / STEPS_PER_ZNL
NYQUIST_PHASE = (np.pi / DX) ** 2 * DELTA_Z / (2 * K0)
assert NYQUIST_PHASE < 1.2, "step too long: the background will destabilise"

TAU = 40  # cell length in units of z_nl
L_CELL = TAU * Z_NL
SETTLE = 4 * Z_NL  # the imprint's relaxation, excluded from every speed fit

print(
    f"medium: Dn = {DN:.2e}, c_s = {CS:.3e}, xi = {XI * 1e6:.1f} um, "
    f"z_nl = {Z_NL * 1e3:.2f} mm"
)
print(
    f"grid:   {N_PTS}^2, dx = xi/{XI / DX:.0f}, window = {WINDOW / XI:.0f} xi, "
    f"L = {TAU} z_nl = {L_CELL * 1e2:.1f} cm"
)
print(
    f"step:   z_nl/{STEPS_PER_ZNL}, {NYQUIST_PHASE:.2f} rad per step at the "
    f"Nyquist wavenumber, {int(L_CELL / DELTA_Z)} steps"
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
    The core profile only needs the right asymptotics. Whatever is imprinted
    radiates its excess over the first few nonlinear lengths and relaxes onto
    the true one, which is exactly what the SLM does in the experiment.
    """
    x, y = simu.XX - x0, simu.YY - y0
    r = np.hypot(x, y)
    return r / np.sqrt(r**2 + XI**2) * np.exp(1j * charge * np.arctan2(y, x))


def dipole(simu, separation, x0):
    """Return a uniform background carrying a vortex dipole.

    The circulations sit on a line along y, so the pair advects itself along
    x, which is the direction every trajectory here is read in. The pair
    carries **zero net circulation**, which is what makes it legal on a
    periodic grid at all: a single vortex is not, since its phase cannot be
    single-valued on a torus, and imprinting one seeds a phase discontinuity
    along a whole seam.

    Parameters
    ----------
    simu : NLSE
        Solver, for its coordinate grids.
    separation : float
        Distance between the cores in m.
    x0 : float
        Where to put the pair, in m.

    Returns
    -------
    ndarray
        The complex field to launch.
    """
    field = vortex(simu, x0, +separation / 2, +1)
    field *= vortex(simu, x0, -separation / 2, -1)
    return field.astype(np.complex64)


def singularities(field):
    """Return the pixel coordinates of the phase singularities.

    Their number is what separates the two halves of the family: two for a
    vortex pair, none for a rarefaction pulse.

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


def locate(field, x_axis):
    """Return the soliton's position along x, in m, and its winding count.

    Both halves of the family have to be tracked by one estimator, and the
    obvious one -- a centroid of the missing density over the window -- is
    wrong: the imprint radiates sound that stays behind while the soliton
    leaves, and weighting the whole window by depletion averages the two
    together. That reads a speed 0.23 c_s too low at the widest separation,
    enough to send the measured trend the wrong way and turn Biot-Savart
    upside down. So the cores are used where there are cores, and a *local*
    depletion centroid, taken within four healing lengths of the deepest
    point, where there are not.

    Parameters
    ----------
    field : ndarray
        Complex field.
    x_axis : ndarray
        The solver's x coordinates, so the answer comes back in the same
        frame the fields are plotted in rather than counted from the corner.

    Returns
    -------
    tuple of float and int
        Position along x in m, and the number of singularities.
    """
    delta_x = float(x_axis[1] - x_axis[0])
    cols, _ = singularities(field)
    if len(cols) == 2:
        return float(np.interp(cols.mean(), np.arange(x_axis.size), x_axis)), 2
    density = np.abs(field) ** 2
    row, col = np.unravel_index(np.argmin(density), density.shape)
    yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
    near = np.hypot((xx - col) * delta_x, (yy - row) * delta_x) < 4 * XI
    missing = np.clip(np.median(density) - density, 0, None) * near
    centre = float((missing * xx).sum() / missing.sum())
    return float(np.interp(centre, np.arange(x_axis.size), x_axis)), len(cols)


def launch(separation, keep_fields=False):
    """Imprint a dipole, propagate it, and follow it.

    Parameters
    ----------
    separation : float
        Distance between the imprinted cores in m.
    keep_fields : bool
        Whether to keep sampled fields for plotting.

    Returns
    -------
    dict
        Trajectory, speed, winding count and optionally the sampled fields.
    """
    simu = NLSE(
        0, POWER, WINDOW, N2, None, L_CELL, NX=N_PTS, NY=N_PTS, Isat=I_SAT, wvl=WVL
    )
    E_in = dipole(simu, separation, x0=-0.25 * WINDOW)
    xs, zs, counts, fields = [], [], [], []

    def follow(sim, A, z, i):
        """Record where the soliton is, every 40 steps."""
        if i % 40:
            return
        field = np.asarray(sim._backend.to_numpy(A))
        x, count = locate(field, sim.X)
        xs.append(x), zs.append(z), counts.append(count)
        if keep_fields:
            fields.append(field.copy())

    out = np.asarray(
        simu._backend.to_numpy(
            simu.out_field(
                E_in,
                L_CELL,
                delta_z=DELTA_Z,
                verbose=False,
                plot=False,
                callback=follow,
            )
        )
    )
    zs, xs = np.array(zs), np.array(xs)
    settled = zs > SETTLE
    speed = np.polyfit(zs[settled], xs[settled], 1)[0] / CS
    return {
        "separation": separation,
        "z": zs,
        "x": xs,
        "speed": speed,
        "windings": counts[-1],
        "fields": fields,
        "in": E_in,
        "out": out,
        "extent": [simu.X[0] / XI, simu.X[-1] / XI, simu.Y[0] / XI, simu.Y[-1] / XI],
    }


# %%
# One soliton, followed the length of the cell. Four healing lengths apart,
# the pair is inside the regime where Biot-Savart holds and fast enough to
# cross a visible stretch of the window.

tracked = launch(4 * XI, keep_fields=True)
print(
    f"\nfour healing lengths apart: v = {tracked['speed']:.3f} c_s measured, "
    f"{1 / 4:.3f} predicted, {tracked['windings']} singularities left"
)

# %%
# What it looks like on the way. The pair is imprinted on the left, sheds a
# ring of sound in the first few nonlinear lengths -- the faint circular wave
# -- and then simply translates. Nothing about the shape after that is
# imposed: the two cores hold their separation on their own, which is what
# makes this a soliton rather than a decaying imprint.

picks = np.linspace(0, len(tracked["fields"]) - 1, 4).astype(int)
rho0 = float(np.median(np.abs(tracked["fields"][0]) ** 2))
fig, axs = plt.subplots(
    2, len(picks), figsize=(3.0 * len(picks), 6.0), layout="constrained"
)
for col, k in enumerate(picks):
    field = tracked["fields"][k]
    axs[0][col].imshow(
        np.abs(field) ** 2 / rho0,
        cmap="viridis",
        origin="lower",
        extent=tracked["extent"],
        vmin=0,
        vmax=1.3,
    )
    axs[0][col].set_title(f"z = {tracked['z'][k] / Z_NL:.0f} $z_{{nl}}$", fontsize=10)
    axs[1][col].imshow(
        np.angle(field),
        cmap="twilight_shifted",
        origin="lower",
        extent=tracked["extent"],
        vmin=-np.pi,
        vmax=np.pi,
    )
    for ax in (axs[0][col], axs[1][col]):
        ax.set_xlim(-32, 32), ax.set_ylim(-16, 16)
        ax.set_xlabel(r"$x/\xi$")
axs[0][0].set_ylabel(r"density   $y/\xi$")
axs[1][0].set_ylabel(r"phase   $y/\xi$")
plt.show()

# %%
# The trajectory, which is the assertion that it travels at all: a straight
# line whose slope is the speed. The first four nonlinear lengths are the
# imprint relaxing, and are excluded from the fit.

fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
ax.plot(tracked["z"] / Z_NL, tracked["x"] / XI, "o", ms=4, label="tracked")
settled = tracked["z"] > SETTLE
fit = np.polyfit(tracked["z"][settled], tracked["x"][settled], 1)
ax.plot(
    tracked["z"] / Z_NL,
    np.polyval(fit, tracked["z"]) / XI,
    "-",
    label=f"fit: {tracked['speed']:.3f} $c_s$",
)
ax.axvspan(0, SETTLE / Z_NL, color="gray", alpha=0.15, label="imprint settling")
ax.set_xlabel(r"$z / z_{nl}$")
ax.set_ylabel(r"soliton position $x/\xi$")
ax.set_title("A Jones-Roberts soliton travels at constant speed")
ax.legend()
plt.show()

# %%
# The family. Walking the separation from ten healing lengths down to one and
# a half traces the whole of it: at wide separation the speed is the
# Biot-Savart value to a couple of percent, it climbs above it as the cores
# begin to overlap, and below three healing lengths the two circulations
# annihilate, leaving a rarefaction pulse. The threshold the paper measures,
# :math:`0.61\,c_s`, is where that happens.

separations = np.array([1.5, 2, 3, 4, 5, 6, 8, 10])
family = [launch(k * XI) for k in separations]
speeds = np.array([r["speed"] for r in family])
counts = np.array([r["windings"] for r in family])
print("\n dr/xi   v/c_s   xi/dr  singularities")
for k, v, n in zip(separations, speeds, counts):
    print(f"{k:6.1f} {v:7.3f} {1 / k:7.3f} {n:9d}")

fig, ax = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
pair = counts == 2
ax.plot(1 / separations[pair], speeds[pair], "o", ms=7, label="vortex pair")
ax.plot(1 / separations[~pair], speeds[~pair], "s", ms=7, label="rarefaction pulse")
grid = np.linspace(0, 0.75, 50)
ax.plot(grid, grid, "k:", lw=1.3, label=r"Biot-Savart, $v = c_s\,\xi/\Delta r$")
ax.axhline(0.61, color="gray", ls="--", lw=1.1)
ax.text(0.02, 0.63, r"$v_c = 0.61\,c_s$", fontsize=9, color="gray")
ax.set_xlabel(r"$\xi / \Delta r$")
ax.set_ylabel(r"$v / c_s$")
ax.set_title("The Jones-Roberts family, and where the circulation goes")
ax.legend(fontsize=9, loc="upper left")
plt.show()

# %%
# The two ends of it, side by side at the cell exit. Left, a wide pair: two
# cores, and a phase that winds around each. Right, a squeezed one: the
# density still carries a localised dip travelling with the fluid, and the
# phase has no singularity in it at all.

fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.2), layout="constrained")
for col, run in ((0, family[-1]), (1, family[0])):
    kind = "vortex pair" if run["windings"] == 2 else "rarefaction pulse"
    field = run["out"]
    axs[0][col].imshow(
        np.abs(field) ** 2 / rho0,
        cmap="viridis",
        origin="lower",
        extent=run["extent"],
        vmin=0,
        vmax=1.3,
    )
    axs[0][col].set_title(
        rf"$\Delta r = {run['separation'] / XI:.1f}\,\xi$, "
        rf"$v = {run['speed']:.2f}\,c_s$" + f"\n{kind}",
        fontsize=10,
    )
    axs[1][col].imshow(
        np.angle(field),
        cmap="twilight_shifted",
        origin="lower",
        extent=run["extent"],
        vmin=-np.pi,
        vmax=np.pi,
    )
    centre = run["x"][-1] / XI
    for ax in (axs[0][col], axs[1][col]):
        ax.set_xlim(centre - 16, centre + 16), ax.set_ylim(-12, 12)
        ax.set_xlabel(r"$x/\xi$")
axs[0][0].set_ylabel(r"density   $y/\xi$")
axs[1][0].set_ylabel(r"phase   $y/\xi$")
plt.show()
