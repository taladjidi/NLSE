r"""
Vortex shedding past a mobile impurity
======================================

A mobile impurity in a flowing superfluid of light, after
`Baker-Rasooli et al. <https://arxiv.org/abs/2512.09028>`_.

The impurity is the second component of a ``CNLSE``, not a potential
landscape: a beam on the other rubidium line, which digs its own density dip
in the fluid and sits in it. It is not a rigid particle -- it breathes and
deforms as it goes, exchanging energy with the fluid -- and a static ``V``
could do neither. Above a critical Mach number the flow past the dip
nucleates quantized vortex-antivortex pairs, the drag rises off its
superfluid floor, and the impurity starts to slip upstream relative to the
fluid carrying it.

Three things about this system are easy to get wrong, and each one is a
figure below.

**The medium saturates, and that clamps the obstacle.** The index depression
is :math:`\Delta n = n_2 I/(1+I/I_{sat})`, so the impurity -- a hundred times
brighter than the fluid -- does *not* present a hundred times the index step:
both are pinned near :math:`n_2 I_{sat}`. The obstacle settles a factor of two
or three above the fluid's own chemical potential and stays there, whatever
the impurity beam does. Nothing has to be tuned to make it hard. The sound
speed follows the same denominator twice: :math:`c_s^2 = I\,\partial\Delta
n/\partial I = \Delta n/(1+I/I_{sat})`, not :math:`\Delta n`. And the
*losses* saturate too, so the beam does not decay exponentially and the
transmission is not :math:`e^{-\alpha L}` -- the small-signal :math:`\alpha`
behind a measured transmission is solved for below.

**Read the impurity in the fluid's frame, never the lab's.** Tilting the
fluid beam to make it flow also walks it sideways, by :math:`L c_s \beta` --
here 1.8 mm at :math:`\beta = 1`, comparable to the beam's own waist. The
impurity is entrained and follows at about 60% of that, so in the lab it
moves *downstream*, and only against the fluid's own displacement does it
appear to swim against the stream. Subtract the walk-off, or measure the
transverse momentum at the exit and subtract :math:`\beta`; both give the
same answer.

**Below a critical cross coupling the impurity stops being an obstacle.**
The saturation denominator is shared between the two beams, so a bright
impurity bleaches the medium under itself; that lowers the index depression
there, and since the depression is negative it reads as an index *bump*.
Repulsion wins only above :math:`|n_{12}| = \mu/I_{sat}` -- here
:math:`0.62\,n_2`, derived below and independent of how bright the impurity
is. Underneath it the impurity is an attractive well, has no Thomas-Fermi
state a beam could be launched in, and comes apart as it goes; the debris
carries phase windings that no vortex detector can tell from shed pairs. The
second scan below is deliberately in that regime, and its vortex count should
be read as a symptom rather than as shedding.

**A weakly coupled impurity also looks like the best swimmer of all, and is
really doing nothing.** If it barely feels the fluid it simply stays put
while the fluid slides past, which in the fluid's frame is exactly
:math:`-\beta`: the dotted line in the velocity figure. Every measurement has
to be read against that line, not against zero. Real slip is the *departure*
from it, and it comes from strong coupling, not weak: with :math:`n_{12} < 0`
the depletion the impurity digs is an index *bump* underneath itself, so it
guides itself in the hole it makes. Stronger coupling gives a more compact
impurity that survives the crossing intact -- and binds it to the fluid,
which is why it is entrained at all. Weak coupling gives a diffuse, shredded
blob tracing the trivial line.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.animation import FuncAnimation
from NLSE import CNLSE
from scipy.constants import c, epsilon_0
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter
from scipy.optimize import brentq

# The group's house style for curves.
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
plt.rc(
    "axes",
    prop_cycle=cycler(color=tab_colors)
    + cycler(markeredgecolor=tab_colors)
    + cycler(markerfacecolor=fills),
)

# %%
# What the experiment hands you: the two beams, the cell, and the medium.
# The nonlinear index and the saturation intensity are measured, so the
# index depression is not a free parameter and there is nothing to fit.

POWER = 4.5  # fluid power in W
WAIST = 1.7e-3  # fluid beam waist in m
WVL = 780e-9  # fluid wavelength
POWER2 = 100e-3  # impurity power in W
WAIST2 = 80e-6  # impurity waist in m, at the widest point of its breathing
WVL2 = 795e-9  # impurity wavelength, the other rubidium line
L_CELL = 20e-2  # cell length in m

N2 = -6e-10  # nonlinear index in m^2/W
I_SAT = 60e4  # saturation intensity in W/m^2
# The cross coupling. One medium, so the index the impurity writes for the
# fluid is the fluid's own n2 and the two share the saturation denominator.
# The velocity figure below is what settles this: anything much weaker puts
# the impurity on the non-interacting line.
N12 = N2
# The comparison run, below the crossover derived below, where the impurity
# stops being an obstacle at all.
N12_WEAK = N2 / 6
# Where the two regimes divide, exactly. Ask what density profile would leave
# the fluid seeing a flat index -- its Thomas-Fermi state, quantum pressure
# dropped -- and require it to equal the ambient. With mu = |n2| I_env/(1 +
# I_env/Isat) that collapses to |n12| I2 = mu I2/Isat, so the crossover is
#
#     |n12| = mu / Isat
#
# whatever the impurity's intensity. Above it the impurity digs a hole and is
# a repulsive obstacle; below it the shared saturation denominator wins --
# a bright impurity bleaches the medium under itself, which lowers a negative
# index depression and so reads as an index BUMP -- and the impurity becomes
# an attractive well instead. Nothing here is imposed on the launch; the
# beams enter as bare Gaussians and find their own state.
#
# Note what the crossover is NOT: it is the fluid's response to the impurity,
# not the impurity's response to the fluid. Running exactly at it, the
# impurity is still entrained and still slips by 0.4 c_s, because the fluid's
# own index gradient still acts on it. Only the digging cancels.
# The impurity's self-focusing. Side fluorescence on a 10 cm cell has it
# breathing between 60 and 80 um, which is what pins this: propagated alone,
# n22 = 5e-11 goes 80 -> 60 um over 10 cm, while 1.5e-10 collapses it to
# 30 um inside the first half of the cell and 2e-11 never contracts at all.
# Free diffraction would put an 80 um beam at 640 um after 7.9 Rayleigh
# ranges, so the impurity is a self-trapped filament, not a passive Gaussian.
N22 = 5e-11
# Absorption. Saturable, so this is the small-signal value; the transmission
# it implies is printed below. It is constrained less by a power meter than
# by the impurity's width at the exit, which is set by how long the fluid
# stays strong enough to dig the hole the impurity sits in.
ALPHA = ALPHA2 = 11.0

K0, K2 = 2 * np.pi / WVL, 2 * np.pi / WVL2
I0 = 2 * POWER / (np.pi * WAIST**2)  # peak input intensity of the fluid
ID0 = 2 * POWER2 / (np.pi * WAIST2**2)  # peak input intensity of the impurity


def transmission_of(alpha, peak, waist):
    """Return the power transmission of a saturable absorber.

    A saturable absorber bleaches where the beam is bright, so the beam does
    not decay exponentially and its centre survives better than its wings.
    Reading ``alpha`` off ``-log(T)/L`` would hand the solver the *effective*
    attenuation as if it were the small-signal one, and under-absorb.

    Parameters
    ----------
    alpha : float
        Small-signal absorption coefficient in 1/m.
    peak : float
        Peak input intensity in W/m^2.
    waist : float
        Beam waist in m.

    Returns
    -------
    float
        Fraction of the input power leaving the cell.
    """
    r = np.linspace(0, 4 * waist, 2000)
    profile = peak * np.exp(-2 * r**2 / waist**2)
    solution = solve_ivp(
        lambda z, intensity: -alpha * intensity / (1 + intensity / I_SAT),
        (0, L_CELL),
        profile,
        rtol=1e-10,
        atol=1e-12,
    )

    def power(profile_r):
        """Return the power carried by a radial intensity profile."""
        return np.trapezoid(profile_r * 2 * np.pi * r, r)

    return power(solution.y[:, -1]) / power(profile)


def alpha_for(transmission, peak, waist):
    """Invert :func:`transmission_of` for the small-signal coefficient.

    Parameters
    ----------
    transmission : float
        Measured power transmission through the cell.
    peak : float
        Peak input intensity in W/m^2.
    waist : float
        Beam waist in m.

    Returns
    -------
    float
        The small-signal absorption coefficient in 1/m.
    """
    return brentq(
        lambda a: transmission_of(a, peak, waist) - transmission, 1e-3, 5e3, xtol=1e-8
    )


def delta_n(intensity):
    """Return the saturated index depression at this intensity.

    Parameters
    ----------
    intensity : float or ndarray
        Optical intensity in W/m^2.

    Returns
    -------
    float or ndarray
        The index depression, unsigned.
    """
    return abs(N2) * intensity / (1 + intensity / I_SAT)


# %%
# The scales follow. The total nonlinear dephasing is an output of all this,
# not an input to it, and neither is the healing length: both come out of the
# measured pair (n2, Isat) once the powers are fixed.

DN0 = delta_n(I0)
CS = np.sqrt(abs(N2) * I0 / (1 + I0 / I_SAT) ** 2)
XI = 1 / (K0 * CS)
# The impurity's phase rate is the faster of the two, so it sets the step.
Z_NL = 1 / (K2 * delta_n(ID0))
# Checked against a run at half this step and at Strang splitting, on what
# this page plots rather than on a field norm: the fluid-frame velocity,
# the pair count and the impurity's width all hold to the last digit, while
# twice this step blows up outright. Lie costs one FFT pair per step against
# Strang's two, and buys nothing in accuracy here.
DELTA_Z = Z_NL / 4
WINDOW = 4.5 * WAIST
# Not negotiable downwards: at 512 the pixel is 1.1 healing lengths, the
# vortex core is unresolved, and the winding count becomes noise -- 36 pairs
# where the converged run finds 2.
N_PTS = 1024
# The cell is not uniform. Its windows run colder than the body -- heated as
# they are, they still collect rubidium -- and the vapour is saturated, so the
# local density follows the local temperature through the Antoine law,
# log10 Pv = 2.881 + 4.312 - 4040/T with n = Pv/(kB T). That is about 6% per
# kelvin at 120 C, so cold windows mean exponentially less density at the ends
# and the nonlinearity ramps in rather than switching on at the glass.
#
# It is not a detail. Launched into a uniform cell the impurity's dip is
# quenched and rings at 0.097 of the local density; with the windows 5 K colder
# that falls to 0.018 and by 20 K it is 0.016 -- saturating, so the suppression
# does not depend on knowing the profile precisely. The measured images ring
# far less than a uniform simulation does, and this is why.
T_BODY = 393.15  # cell body, 120 C
DT_WINDOW = 20.0  # how much colder the windows run
Z_RAMP = 2e-2  # length over which the ends come up to the body value

# The transverse walk-off of a beam tilted to beta = 1. Everything below is
# read against this, and it is not a small correction.
WALKOFF = L_CELL * CS
# Where the impurity starts, on the fluid's axis: the envelope exerts no
# force of its own there, so the drag curve starts from zero.
X0 = 0.0
# A fixed window in the lab frame for every panel, wide enough downstream to
# hold the impurity's whole excursion. Fixed rather than centred on the
# impurity, so one frame can be compared with the next and the motion is
# something to see rather than something to read off an axis.
# Symmetric and centred, because the two couplings move the impurity in
# OPPOSITE directions: strongly coupled it is dragged downstream, weakly
# coupled it is left behind upstream. A window placed for one of them cuts
# the other in half.
VIEW_X = (-110 * XI, 110 * XI)
VIEW_Y = (-110 * XI, 110 * XI)

# An adaptive step was measured against this fixed one and is not worth it
# here. Sized to the local phase rate it shrinks to 25 um and costs four
# times as much; sized to a measured local error it does grow from 76 to
# 349 um as the fluid is absorbed, and needs a fifth fewer steps -- but each
# check propagates the same distance twice to measure the error, which more
# than eats the saving. All three agree on the velocity and the pair count.
RNG = np.random.default_rng(0)
# Launch noise, as a fraction of the peak. Only there to break the symmetry
# so a vortex has a side to shed to: 1e-3 puts 3.6% rms ripple on the input,
# which is the 2-4% measured in the data, and the impurity's velocity is flat
# to within 0.03 across four decades of it. Set to 0 to compare against a
# perfectly symmetric launch.
NOISE = 0.0

# The scan, as in the experiment. The documentation build takes a coarse one
# so the page stays cheap; run the script yourself for the full scan.
if os.environ.get("NLSE_DOCS_BUILD"):
    BETAS = [0.0, 0.4, 0.7, 1.0]
else:
    BETAS = [0.0, 0.15, 0.3, 0.45, 0.55, 0.63, 0.7, 0.8, 0.9, 1.0, 1.1]

print(
    f"fluid:    I0 = {I0 / 1e4:.0f} W/cm^2 = {I0 / I_SAT:.2f} Isat, "
    f"alpha = {ALPHA:.1f}/m -> T = {transmission_of(ALPHA, I0, WAIST):.2f} "
    f"(a plain exponential would say "
    f"{np.exp(-ALPHA * L_CELL):.3f})"
)
print(
    f"impurity: Id0 = {ID0 / 1e4:.0f} W/cm^2 = {ID0 / I_SAT:.0f} Isat, "
    f"alpha2 = {ALPHA2:.1f}/m -> T = {transmission_of(ALPHA2, ID0, WAIST2):.2f}"
)
print(
    f"scales:   Dn = {DN0:.2e}, c_s = {CS:.2e}, xi = {XI * 1e6:.1f} um "
    f"(sqrt(Dn) would say {1 / (K0 * np.sqrt(DN0)) * 1e6:.1f} um)"
)
print(
    f"grid:     dx = {WINDOW / N_PTS * 1e6:.1f} um = {WINDOW / N_PTS / XI:.2f} xi, "
    f"dz = {DELTA_Z * 1e6:.0f} um, {int(L_CELL / DELTA_Z)} steps"
)
print(
    f"obstacle: {WAIST2 / XI:.1f} xi wide, index step "
    f"{delta_n(ID0) / DN0:.2f} x the fluid's own, "
    f"predicted beta_c = sqrt(2) xi/sigma = {np.sqrt(2) * XI / WAIST2:.2f}"
)
print(
    f"walk-off: {WALKOFF * 1e3:.2f} mm at beta = 1, against a {WAIST * 1e3:.2f} mm waist"
)


def density_ratio(z):
    """Return the vapour density at z, relative to the cell body.

    Parameters
    ----------
    z : float or ndarray
        Distance into the cell in m.

    Returns
    -------
    float or ndarray
        Local density over the body's, from the saturated vapour pressure at
        the local temperature.
    """
    ends = np.exp(-z / Z_RAMP) + np.exp(-(L_CELL - z) / Z_RAMP)
    temperature = T_BODY - DT_WINDOW * np.clip(ends, 0, 1)
    return 10 ** (4040 * (1 / T_BODY - 1 / temperature)) * (T_BODY / temperature)


def vortices(field, mask):
    """Locate quantized vortices by the phase winding around each plaquette.

    Parameters
    ----------
    field : ndarray
        Complex fluid field.
    mask : ndarray
        Boolean array of pixels bright enough to trust; a winding only counts
        if all four corners of its plaquette are inside it, since where the
        field is dark the phase is noise.

    Returns
    -------
    tuple of ndarray
        Pixel coordinates ``(x, y)`` and charges ``(+1/-1)``.
    """
    ph = np.angle(field)
    dpx = np.angle(np.exp(1j * (np.roll(ph, -1, axis=1) - ph)))
    dpy = np.angle(np.exp(1j * (np.roll(ph, -1, axis=0) - ph)))
    w = dpx + np.roll(dpy, -1, axis=1) - np.roll(dpx, -1, axis=0) - dpy
    corners = (
        mask
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, -1, axis=1)
        & np.roll(np.roll(mask, -1, axis=0), -1, axis=1)
    )
    w[~corners] = 0
    ys_p, xs_p = np.where(w > np.pi)
    ys_m, xs_m = np.where(w < -np.pi)
    xs = np.concatenate([xs_p, xs_m])
    ys = np.concatenate([ys_p, ys_m])
    charge = np.concatenate([np.ones(len(xs_p)), -np.ones(len(xs_m))])
    return xs, ys, charge


def impurity_track(field, dx):
    """Locate the impurity and read its transverse momentum at the exit.

    Both are taken on a disc around the brightest point, never over the whole
    window: outside that disc sits the light the fluid scattered off the
    obstacle, which is more than half the impurity's power by the end and
    carries no information about where the impurity is. A whole-window
    centroid tracks that halo instead, and when the impurity is diffuse it
    can report the wrong sign entirely.

    Parameters
    ----------
    field : ndarray
        Complex impurity field at the cell exit.
    dx : float
        Transverse pixel size in m.

    Returns
    -------
    tuple of float
        Position in m, lab-frame velocity in units of ``c_s``, and the
        enclosed-power radius in m -- how compact the impurity stayed.
    """
    intensity = np.abs(field) ** 2
    smooth = uniform_filter(intensity, size=max(3, round(XI / dx)))
    j, i = np.unravel_index(np.argmax(smooth), smooth.shape)
    yy, xx = np.mgrid[0 : field.shape[0], 0 : field.shape[1]]
    r = np.hypot((xx - i) * dx, (yy - j) * dx)
    disc = r < 25 * XI
    # Flow runs along x, so the momentum of interest is d(phase)/dx, taken
    # as Im(conj(E) dE/dx)/|E|^2 so no phase has to be unwrapped.
    grad = np.gradient(field, dx, axis=1)
    kx = np.imag(np.conj(field) * grad)[disc].sum() / intensity[disc].sum()
    order = np.argsort(r[disc])
    cumulative = np.cumsum(intensity[disc][order])
    r50 = r[disc][order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]
    return (i - field.shape[1] / 2) * dx, kx * XI, r50


def run(beta_target, n12):
    """Propagate fluid and impurity at one flow speed and image the output.

    Parameters
    ----------
    beta_target : float
        Flow Mach number against the sound speed at the input peak; rounded
        so the phase ramp closes on the grid.
    n12 : float
        Cross-coupling index in m^2/W.

    Returns
    -------
    dict
        Output fields, vortex positions, impurity trajectory and drag record.
    """
    simu = CNLSE(
        ALPHA,
        POWER,
        WINDOW,
        N2,
        n12,
        None,
        L_CELL,
        NX=N_PTS,
        NY=N_PTS,
        Isat=I_SAT,
        wvl=WVL,
    )
    simu.power2 = POWER2
    simu.k2 = K2
    simu.n22 = N22
    simu.I_sat2 = I_SAT
    simu.alpha2 = ALPHA2

    m = round(beta_target * CS * K0 * WINDOW / (2 * np.pi))
    kx = 2 * np.pi * m / WINDOW
    beta = kx / (K0 * CS)

    def with_noise(field, amplitude):
        """Seed the launch with the experiment's own level of speckle."""
        sigma = np.sqrt(amplitude) / 2
        return (
            field
            + RNG.normal(0, sigma, field.shape)
            + 1j * RNG.normal(0, sigma, field.shape)
        )

    # Two bare Gaussians, as the beams enter the cell, with nothing
    # pre-relaxed and nothing tuned to the model being run. Pre-digging the
    # impurity's dip -- a Thomas-Fermi hole, rounded over a healing length --
    # was worth six times on the ringing when the cell was uniform, and only
    # 2.7 once the density ramp is in place. The ramp is physics the cell has;
    # the dug hole is a numerical convenience whose shape depends on which
    # model it was derived for, and it biases anything it is compared against:
    # switching nonlocality on with a locally-derived hole in place makes the
    # launch disagree with the medium and the ringing rise for reasons that
    # have nothing to do with nonlocality.
    r2 = (simu.XX - X0) ** 2 + simu.YY**2
    beam = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2)
    fluid0 = with_noise(beam, NOISE) * np.exp(1j * kx * simu.XX)
    E = np.array([fluid0, with_noise(np.exp(-r2 / WAIST2**2), NOISE / 10)])
    E = E.astype(np.complex64)

    to_intensity = 0.5 * c * epsilon_0
    # Ambient fluid: a band clear of the impurity and of its wake, which is
    # what the local chemical potential has to be read against once the beam
    # has spread and bleached its way through the cell.
    ambient = (np.abs(simu.Y)[:, None] > 400e-6) & (np.abs(simu.Y)[:, None] < 700e-6)
    ambient = ambient & (np.abs(simu.X)[None, :] < 300e-6)
    zs, force = [], []
    dx = simu.delta_X

    def track(sim, A, z, i):
        """Record the drag force on the impurity as the fluid flows past.

        Parameters
        ----------
        sim : CNLSE
            The running solver.
        A : ndarray
            Device field, both components.
        z : float
            Current propagation distance in m.
        i : int
            Step counter.
        """
        if i % 20 != 0:
            return
        host = sim._as_host_array(A)
        i1 = np.abs(host[0]) ** 2 * to_intensity
        i2 = np.abs(host[1]) ** 2 * to_intensity
        rho1 = i1 / i1[ambient].mean()
        # The index hill the impurity presents, in units of the fluid's own
        # local depression. The drag on the impurity is the fluid density
        # against the hill's gradient, by action and reaction.
        hill = delta_n(i2) / delta_n(i1[ambient].mean())
        grad = (np.roll(hill, -1, axis=1) - np.roll(hill, 1, axis=1)) / (2 * dx)
        zs.append(z)
        force.append((rho1 * grad).sum() * dx * dx / XI)

    # The nonlinearity has to vary along z, which the solver has no public
    # way to express, so the callback rewrites the per-step constants the
    # kernels read. n2, n12, n22 and both absorptions are all proportional to
    # the vapour density; Isat is not, being a per-atom property. Reading the
    # nominal values needs one preparatory call, since out_field computes them
    # itself at the start of a run.
    simu._precompute_step_constants(None, np.complex64)
    nominal = {
        name: getattr(simu, name)
        for name in (
            "_g",
            "_g11",
            "_g12",
            "_g21",
            "_g22",
            "_alpha_half",
            "_alpha2_half",
        )
    }

    def thermal(sim, A, z, i):
        """Scale every density-proportional constant to the local density.

        Parameters
        ----------
        sim : CNLSE
            The running solver.
        A : ndarray
            Device field, unused.
        z : float
            Current propagation distance in m.
        i : int
            Step counter, unused.
        """
        factor = np.float32(density_ratio(min(z, L_CELL)))
        for name, value in nominal.items():
            setattr(sim, name, value * factor)

    t0 = time.perf_counter()
    out = simu.out_field(
        E,
        L_CELL,
        delta_z=DELTA_Z,
        verbose=False,
        callback=[thermal, track],
        callback_args=[(), ()],
        splitting="lie",
        plot=False,
    )
    fluid, impurity = out[0], out[1]
    i_out = np.abs(fluid) ** 2 * to_intensity
    rho_out = i_out / i_out[ambient].mean()
    # "Inside the fluid" has to be judged on a scale wider than the holes
    # being looked for. A vortex core is dark by construction and so is the
    # rarefaction around it -- together five to ten healing lengths across --
    # so a threshold on the local density throws away exactly the plaquettes
    # that carry a winding, and all four corners of a plaquette drop out
    # together. Smoothing over ten healing lengths bridges a core and its
    # surroundings while still following the beam, and discards only what is
    # genuinely empty: outside the beam, and under the impurity itself.
    envelope = uniform_filter(rho_out, size=max(3, round(10 * XI / dx)))
    vx, vy, charge = vortices(fluid, (beam**2 > 0.1) & (envelope > 0.15))
    x_imp, v_lab, r50 = impurity_track(impurity, dx)
    sx = slice(*np.searchsorted(simu.X, VIEW_X))
    sy = slice(*np.searchsorted(simu.Y, VIEW_Y))
    # The impurity's density against the intensity it was launched with.
    # It needs its own scale -- it is two orders brighter than the fluid --
    # but the reference has to be a physical intensity: the solver rescales
    # the field internally to carry the beam's power, so dividing the output
    # by the peak of the array handed to it compares a W/m^2 against a 1 and
    # saturates the panel to a handful of pixels.
    imp_rho = np.abs(impurity) ** 2 * to_intensity / ID0
    panels = {
        "fluid_rho": rho_out[sy, sx].astype(np.float32),
        "fluid_phi": np.angle(fluid)[sy, sx].astype(np.float32),
        "imp_rho": imp_rho[sy, sx].astype(np.float32),
        "imp_phi": np.angle(impurity)[sy, sx].astype(np.float32),
    }
    print(
        f"n12/n2 = {n12 / N2:.2f}  beta = {beta:.2f}: {len(charge) // 2} pairs, "
        f"impurity at {x_imp * 1e3:+.2f} mm against a {WALKOFF * beta * 1e3:.2f} mm "
        f"walk-off, v_fluid = {v_lab - beta:+.2f} c_s, r50 = {r50 * 1e6:.0f} um, "
        f"{time.perf_counter() - t0:.1f} s"
    )
    return {
        "beta": beta,
        "n12": n12,
        "rho": rho_out.astype(np.float32),
        **panels,
        "vx": vx,
        "vy": vy,
        "charge": charge,
        "x_imp": x_imp,
        "v_lab": v_lab,
        "r50": r50,
        "z": np.array(zs),
        "force": np.array(force),
        "X": simu.X,
        "Y": simu.Y,
    }


# %%
# The scan itself: one propagation per flow speed, imaged at z = L, and the
# same scan again at a tenth of the cross coupling for the comparison.

runs = [run(b, N12) for b in BETAS]
weak = [run(b, N12_WEAK) for b in BETAS]

# %%
# Output images across the scan, as the experiment sees them: the fluid
# density in gray, the shed vortices circled by their circulation. Both
# couplings, on the same window, so the two are read against each other.
# Below the critical Mach number the flow closes smoothly around the dip;
# above it pairs appear in the wake. The rings are the start-up bow wave --
# the price of switching the flow on abruptly at z = 0 -- receding at
# :math:`(1-\beta)c_s`.

ext = [VIEW_X[0] / XI, VIEW_X[1] / XI, VIEW_Y[0] / XI, VIEW_Y[1] / XI]
# No per-panel limits: the extent above IS the view, identical for every
# frame and both couplings, so nothing can be cropped out of one of them.
picks = list(range(0, len(BETAS), max(1, len(BETAS) // 5)))[:5]
fig, axs = plt.subplots(
    2, len(picks), figsize=(3.0 * len(picks), 6.0), layout="constrained"
)
for row, (scan, label) in enumerate(((runs, "n_2"), (weak, "n_2/6"))):
    for ax, k in zip(axs[row], picks):
        r = scan[k]
        ax.imshow(
            r["fluid_rho"],
            cmap="gray",
            origin="lower",
            extent=ext,
            vmin=0,
            vmax=1.6,
        )
        sel = r["charge"] > 0
        for keep, colour in ((sel, "red"), (~sel, "cyan")):
            if keep.any():
                ax.scatter(
                    r["X"][r["vx"][keep]] / XI,
                    r["Y"][r["vy"][keep]] / XI,
                    s=90,
                    facecolors="none",
                    edgecolors=colour,
                    lw=1.5,
                )
        ax.set_title(rf"$n_{{12}} = {label}$,  $\beta = {r['beta']:.2f}$", fontsize=9)
        ax.set_xlabel(r"$x/\xi$")
    axs[row][0].set_ylabel(r"$y/\xi$")
plt.show()

# %%
# The two curves of the experiment: how many pairs the wake holds at the
# output, and the drag the impurity feels, against the Mach number. The drag
# is averaged over the second half of the cell, past the transient; below the
# critical Mach number the superfluid flows around the dip without
# resistance, and the shedding threshold is where the vortex count leaves
# zero.

betas = np.array([r["beta"] for r in runs])
n_v = np.array([len(r["charge"]) // 2 for r in runs])
f_mean = np.array([r["force"][len(r["force"]) // 2 :].mean() for r in runs])
f_std = np.array([r["force"][len(r["force"]) // 2 :].std() for r in runs])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax0.plot(betas, n_v, "o-")
ax0.axvline(np.sqrt(2) * XI / WAIST2, color="gray", ls=":", lw=1)
ax0.set_xlabel(r"Mach number $\beta$")
ax0.set_ylabel("vortex pairs at $z = L$")
ax0.set_title(r"vortex count ($\sqrt{2}\,\xi/\sigma$ dotted)")
ax1.errorbar(betas, f_mean, yerr=f_std, fmt="s-", capsize=3)
ax1.axhline(0, color="gray", lw=0.8)
ax1.set_xlabel(r"Mach number $\beta$")
ax1.set_ylabel("drag force on the impurity (a.u.)")
ax1.set_title("drag")
plt.show()

# %%
# And the curve the whole thing turns on. Left: the lab frame, where the
# impurity is unambiguously *dragged* -- it just does not keep up with its
# own fluid, and the gap between the two lines is the entire effect. Right:
# the fluid frame, where that gap becomes an upstream velocity that grows
# through the shedding threshold and then saturates.
#
# The dotted line is the impurity that does nothing at all: held still while
# the fluid walks off beneath it, which is :math:`-\beta` exactly. Weak
# coupling runs along it and past it -- not entrained, and pushed off the
# beam axis by an envelope whose brightest point is its lowest index -- and
# would be read as "swimming upstream fastest" by anyone measuring in the lab
# frame. Strong coupling departs from it the other way, saturating near
# :math:`-0.3\,c_s`, which is the measured behaviour.

v_fluid = np.array([r["v_lab"] - r["beta"] for r in runs])
v_fluid_weak = np.array([r["v_lab"] - r["beta"] for r in weak])
x_imp = np.array([r["x_imp"] for r in runs])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
ax0.plot(betas, WALKOFF * betas * 1e3, "--", color="gray", label="fluid walk-off")
ax0.plot(betas, x_imp * 1e3, "o-", label=r"impurity, $n_{12} = n_2$")
ax0.fill_between(betas, x_imp * 1e3, WALKOFF * betas * 1e3, alpha=0.15, color="tab:red")
ax0.set_xlabel(r"Mach number $\beta$")
ax0.set_ylabel("transverse displacement at $z = L$ (mm)")
ax0.set_title("lab frame: dragged, but lagging")
ax0.legend()

ax1.plot(betas, -betas, ":", color="k", lw=1.4, label=r"no coupling at all ($-\beta$)")
ax1.plot(betas, v_fluid, "o-", label=r"$n_{12} = n_2$")
ax1.plot(betas, v_fluid_weak, "s-", label=r"$n_{12} = n_2/6$")
ax1.axhline(0, color="gray", lw=0.8)
ax1.set_xlabel(r"Mach number $\beta$")
ax1.set_ylabel(r"impurity velocity in the fluid frame ($c_s$)")
ax1.set_title("fluid frame: real slip, and the trivial line")
ax1.legend(fontsize=9)
plt.show()

# %%
# The scan, played, for each coupling in turn: both fields, density and
# phase, at every flow speed. The fluid's density carries the wake and the
# phase carries the windings that make it quantized -- a shed vortex is a
# dark core in the left panel and a branch cut in the one beside it, and it
# is only the pair that identifies it. The impurity's own phase is the third
# thing to watch: it is flat while the impurity rides the fluid, and it tilts
# as the impurity starts to slip.
#
# The window is the lab's and stays put, on the same axes and the same
# colour scales for both couplings and every flow speed, so what moves
# between frames is the physics and nothing else.


def play(scan, title):
    """Animate both fields, density and phase, across the flow scan.

    Parameters
    ----------
    scan : list of dict
        The runs to play, in order of increasing Mach number.
    title : str
        Which coupling this is, for the figure heading.

    Returns
    -------
    FuncAnimation
        Held by the caller so it is not collected before it is drawn.
    """
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4), layout="constrained")
    keys = (("fluid_rho", "imp_rho"), ("fluid_phi", "imp_phi"))
    names = (
        (r"fluid density $\rho/\rho_\infty$", "impurity density"),
        ("fluid phase", "impurity phase"),
    )
    images = {}
    for col, (key_pair, name_pair) in enumerate(zip(keys, names)):
        for row, (key, name) in enumerate(zip(key_pair, name_pair)):
            ax = axs[row][col]
            images[key] = ax.imshow(
                scan[0][key],
                origin="lower",
                extent=ext,
                cmap="twilight_shifted" if key.endswith("phi") else "gray",
                vmin=-np.pi if key.endswith("phi") else 0,
                vmax=np.pi if key.endswith("phi") else (1.6 if "fluid" in key else 0.6),
            )
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(r"$x/\xi$")
            ax.set_ylabel(r"$y/\xi$")

    def step(i):
        """Draw scan point ``i``."""
        r = scan[i]
        for key, im in images.items():
            im.set_data(r[key])
        fig.suptitle(
            rf"{title}    $\beta = {r['beta']:.2f}$    "
            rf"{len(r['charge']) // 2} pairs    "
            rf"$v_\mathrm{{fluid\ frame}} = {r['v_lab'] - r['beta']:+.2f}\,c_s$"
        )
        return tuple(images.values())

    return FuncAnimation(fig, step, frames=len(scan), interval=700, blit=False)


# Bound at module level rather than dropped: sphinx-gallery looks through the
# example's namespace for an Animation and embeds what it finds there, so an
# animation that only exists inside a function is a still picture on the page.
anim_strong = play(runs, r"$n_{12} = n_2$")
plt.show()
anim_weak = play(weak, r"$n_{12} = n_2/6$")
plt.show()
