"""The answers, against solutions that are known rather than computed here.

Everything else in this suite is self-referential. The cross-backend tests say
the four implementations agree; the order tests say the error falls at the rate
the method promises; the propagator tests write the package's own formula out a
second time and compare. All of them pass if the formula is wrong and both
copies of it are wrong together, and none reads what the solver returns against
anything from outside the package.

These do. Each one isolates one term of the equation against a closed form:

- the propagator, against Gaussian diffraction, the Gouy phase and the Talbot
  distance -- the last at a transverse wavenumber the first two never reach;
- the loss, against Beer's law;
- the interaction, against the self-phase modulation of a plane wave, which
  has no diffraction to confuse it, with and without saturation;
- the two of them together, against the identity the solved lossy step is
  built on -- the one place the recent work can be checked as physics rather
  than as convergence;
- and finally all of them at once, against a bright soliton, which exists only
  because diffraction and self-focusing cancel and so cannot be passed by
  getting either one separately right.

The conventions these pin down, which nothing else did:

    i dA/dz = -(1/2k) grad^2 A - k*n2*I/(1 + I/Isat) * A,  I = c*eps0*|A|^2/2

so ``n2 > 0`` focuses, and ``alpha`` attenuates the *intensity*: I ~ exp(-alpha z).
"""

import numpy as np
import pytest
from NLSE import NLSE, NLSE_1d
from NLSE.backends import get_backend, list_available_backends
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq

AVAILABLE_BACKENDS = list_available_backends()

N = 256
WINDOW = 4e-3
WAIST = 200e-6


def solver(backend, **kwargs):
    """Return a 2D solver with the module's grid and no physics but what is asked.

    Parameters
    ----------
    backend : str
        Backend name.
    **kwargs
        Constructor arguments overriding the linear, lossless defaults.

    Returns
    -------
    NLSE
        The solver.
    """
    base = {
        "alpha": 0,
        "power": 1,
        "window": WINDOW,
        "n2": 0,
        "V": None,
        "L": 1e-2,
        "NX": N,
        "NY": N,
        "Isat": 1e30,
        "backend": backend,
    }
    return NLSE(**{**base, **kwargs}, **{})


def to_host(simu, array):
    """Return an array as numpy, whatever backend produced it."""
    return np.asarray(simu._backend.to_numpy(array))


def waist_of(intensity, xx, yy):
    """Return the 1/e^2 radius of a round beam from its second moment.

    For ``I = exp(-2 r^2 / w^2)`` in two dimensions ``<r^2> = w^2/2``.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity on the grid.
    xx : np.ndarray
        x coordinate of each point.
    yy : np.ndarray
        y coordinate of each point.

    Returns
    -------
    float
        The radius.
    """
    total = np.sum(intensity)
    return float(np.sqrt(2 * np.sum(intensity * (xx**2 + yy**2)) / total))


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("z_over_zr", [0.5, 1.0, 2.0])
def test_a_gaussian_beam_diffracts_at_the_rate_diffraction_theory_gives(
    backend, z_over_zr
):
    """``w(z) = w0 sqrt(1 + (z/zR)^2)``, exactly, for the paraxial equation.

    One assertion covering the transform pair, its normalization, the k grid
    and the wavelength together: any of them wrong by a factor moves the
    waist.

    Other tests do check the propagator, but by writing the same expression a
    second time and comparing -- which catches a typo in one of the two and
    cannot catch the expression itself being wrong. This is scored against
    diffraction theory instead, which is not a restatement of anything in the
    package.
    """
    simu = solver(backend)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    z_rayleigh = simu.k * WAIST**2 / 2

    out = to_host(
        simu,
        simu.out_field(
            field,
            z_over_zr * z_rayleigh,
            verbose=False,
            plot=False,
            normalize=False,
        ),
    )
    got = waist_of(np.abs(out) ** 2, simu.XX, simu.YY)
    expected = WAIST * np.sqrt(1 + z_over_zr**2)
    assert got == pytest.approx(expected, rel=2e-3), (
        f"{backend}: at z = {z_over_zr} zR the waist is {got * 1e6:.2f} um "
        f"against the {expected * 1e6:.2f} um diffraction theory gives"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_the_on_axis_phase_is_the_gouy_phase(backend):
    """``-arctan(z/zR)``, which the intensity above cannot see.

    A propagator with the wrong sign, or with the dispersion relation off by
    a factor, still diffracts a beam -- it diffracts it wrongly, and the
    second moment is a weak witness. The phase is the sharp one.
    """
    simu = solver(backend)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    z_rayleigh = simu.k * WAIST**2 / 2

    for z_over_zr in (0.5, 1.0, 2.0):
        out = to_host(
            simu,
            simu.out_field(
                field.copy(),
                z_over_zr * z_rayleigh,
                verbose=False,
                plot=False,
                normalize=False,
            ),
        )
        got = float(np.angle(out[N // 2, N // 2]))
        assert got == pytest.approx(-np.arctan(z_over_zr), abs=2e-3), (
            f"{backend}: on-axis phase at z = {z_over_zr} zR is {got:+.4f} rad "
            f"against a Gouy phase of {-np.arctan(z_over_zr):+.4f}"
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_grating_revives_at_the_talbot_distance(backend):
    """Self-imaging at ``z_T = k d^2 / pi``, and not before it.

    The Gaussian tests probe the propagator near ``K = 0``, where a wrong
    dispersion relation is hardest to see. A grating puts all its power at one
    finite ``K``, and the revival is periodic in the phase that mode accrues,
    so the distance is a direct reading of ``K^2/2k``.

    The half-distance is asserted too: at ``z_T/2`` the image is displaced by
    half a period, so a solver that simply left the field alone -- which would
    pass the revival on its own -- fails here.
    """
    simu = solver(backend)
    period = WINDOW / 16
    grating = (1 + 0.5 * np.cos(2 * np.pi * simu.XX / period)).astype(np.complex64)
    talbot = simu.k * period**2 / np.pi

    def mismatch(distance):
        out = to_host(
            simu,
            simu.out_field(
                grating.copy(), distance, verbose=False, plot=False, normalize=False
            ),
        )
        return float(
            np.max(np.abs(np.abs(out) - np.abs(grating))) / np.max(np.abs(grating))
        )

    assert mismatch(talbot) < 1e-3, (
        f"{backend}: the grating did not revive at z_T = {talbot * 1e3:.3f} mm"
    )
    assert mismatch(talbot / 2) > 0.1, (
        f"{backend}: the field is unchanged at z_T/2 as well, so the revival "
        f"above says nothing -- the propagator may not be propagating"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("alpha", [5.0, 20.0])
def test_the_intensity_follows_beer_law(backend, alpha):
    """``I(z) = I0 exp(-alpha z)``: alpha attenuates intensity, not amplitude.

    A uniform field has no transverse structure, so the propagator is the
    identity and this is the loss term alone. It also pins the convention,
    which a factor of two in either direction would otherwise leave open.
    """
    simu = solver(backend, alpha=alpha, Isat=1e30, NX=64, NY=64)
    field = np.ones((64, 64), dtype=np.complex64)

    for z in (1e-2, 5e-2):
        out = to_host(
            simu,
            simu.out_field(
                field.copy(),
                z,
                delta_z=1e-4,
                verbose=False,
                plot=False,
                normalize=False,
            ),
        )
        got = float(np.abs(out[32, 32]) ** 2)
        assert got == pytest.approx(np.exp(-alpha * z), rel=1e-4), (
            f"{backend}: alpha={alpha} over {z} m leaves {got:.6f} of the "
            f"intensity against Beer's law's {np.exp(-alpha * z):.6f}"
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("saturation", [1.0, 0.25], ids=["I_sat=I", "I_sat=I/4"])
def test_self_phase_modulation_saturates_as_the_model_says(backend, saturation):
    """``phi = k n2 I z / (1 + I/Isat)`` on a plane wave.

    The interaction term alone, again with no diffraction to hide in, and the
    only test that reads the saturation as a physical law rather than as an
    expression the kernels happen to share.
    """
    n2, z = -1e-9, 5e-3
    probe = solver(backend, NX=64, NY=64)
    intensity = 1.0 / (probe.k * abs(n2) * z)  # about one radian of phase
    amplitude = float(np.sqrt(2 * intensity / (c * epsilon_0)))
    isat = saturation * intensity

    simu = solver(backend, n2=n2, Isat=isat, NX=64, NY=64)
    field = (amplitude * np.ones((64, 64))).astype(np.complex64)
    out = to_host(
        simu,
        simu.out_field(
            field, z, delta_z=1e-5, verbose=False, plot=False, normalize=False
        ),
    )
    expected = simu.k * n2 * intensity * z / (1 + intensity / isat)
    assert float(np.angle(out[32, 32])) == pytest.approx(expected, rel=1e-3), (
        f"{backend}: Isat = {saturation} I gives a phase of "
        f"{float(np.angle(out[32, 32])):+.5f} rad against {expected:+.5f}"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_lossy_saturable_step_obeys_the_identity_it_is_solved_from(backend):
    """The physics behind the solved real-space step, not its convergence.

    With ``y = |A|^2``, ``s = 1/(1 + y/Isat)``, ``dy/dz = -alpha s y`` and
    ``dphi/dz = k n2 y s`` give ``dphi/dy = -k n2 / alpha``, so over any
    distance::

        phi = (k n2 / alpha) (y0 - y_end)

    whatever the saturation does in between -- and ``y_end`` is fixed by
    ``ln y + y/Isat`` falling by ``alpha z``. Both are asserted. This is what
    ``test_lossy_substep.py`` measures the *order* of; here it is measured
    against the closed form, so a step that converged beautifully to the wrong
    answer would fail.

    ``L`` is set to the whole distance deliberately. Past the medium length
    the solver zeroes ``n2`` and keeps ``alpha``, so a run that leaves the
    medium goes on losing intensity while it stops accruing phase -- and the
    identity, which ties the two together, no longer holds. Written first with
    the default ``L`` an eighth of the distance, this asserted the intensity
    correctly and missed the phase by 3.2 radians.
    """
    n2, alpha, z = -1e-9, 20.0, 5e-2
    probe = solver(backend, NX=64, NY=64)
    intensity = 1.0 / (probe.k * abs(n2) * 5e-3)
    amplitude = float(np.sqrt(2 * intensity / (c * epsilon_0)))
    isat = 0.8 * intensity

    simu = solver(backend, n2=n2, alpha=alpha, Isat=isat, NX=64, NY=64, L=z)
    field = (amplitude * np.ones((64, 64))).astype(np.complex64)
    out = to_host(
        simu,
        simu.out_field(
            field, z, delta_z=1e-5, verbose=False, plot=False, normalize=False
        ),
    )

    target = np.log(intensity) + intensity / isat - alpha * z
    exact_end = brentq(
        lambda y: np.log(y) + y / isat - target, 1e-12 * intensity, intensity
    )
    got_end = c * epsilon_0 / 2 * float(np.abs(out[32, 32])) ** 2
    assert got_end == pytest.approx(exact_end, rel=1e-3), (
        f"{backend}: the intensity ends at {got_end:.6e} W/m^2 against the "
        f"{exact_end:.6e} that ln(I) + I/Isat falling by alpha*z gives"
    )

    expected_phase = simu.k * n2 / alpha * (intensity - exact_end)
    got_phase = float(np.angle(out[32, 32]))
    wrapped = float(np.angle(np.exp(1j * expected_phase)))
    assert got_phase == pytest.approx(wrapped, abs=2e-3), (
        f"{backend}: the phase is {got_phase:+.5f} rad against the "
        f"{wrapped:+.5f} the identity gives from the intensity drop alone"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_bright_soliton_keeps_its_shape(backend):
    """Diffraction and self-focusing cancel, or they do not.

    ``A0 sech(x/x0)`` propagates unchanged when ``x0^2 = 1/(k g A0^2)``, with
    ``g = k n2 c eps0 / 2`` and ``n2 > 0``. Both terms are pinned separately
    above; this is the one that fails if they are individually right and
    jointly wrong -- a sign, or a factor of two between them, leaves every
    other test here passing.

    A detuned width is propagated as a control, because "the shape barely
    changed" is only worth something next to an initial condition that is not
    a soliton and does change.
    """
    if not get_backend(backend).supports_double_precision():
        pytest.skip(f"{backend} has no double precision to hold the balance in")

    simu = NLSE_1d(
        alpha=0,
        power=1,
        window=WINDOW,
        n2=+1e-9,
        V=None,
        L=1e-2,
        NX=2048,
        Isat=1e30,
        backend=backend,
    )
    amplitude = 4000.0
    width = 1 / (amplitude * np.sqrt(simu.k * simu._constant("_g")))
    distance = simu.k * width**2  # one diffraction length of that width

    def spread(x0):
        field = (amplitude / np.cosh(simu.X / x0)).astype(np.complex128)
        out = to_host(
            simu,
            simu.out_field(
                field.copy(),
                distance,
                delta_z=distance / 500,
                verbose=False,
                plot=False,
                normalize=False,
                splitting="strang",
            ),
        )
        return float(np.max(np.abs(np.abs(out) - np.abs(field))) / amplitude)

    soliton = spread(width)
    detuned = spread(1.5 * width)
    assert soliton < 1e-4, (
        f"{backend}: the soliton changed shape by {soliton:.2e}, so diffraction "
        f"and self-focusing are not cancelling"
    )
    assert detuned > 100 * soliton, (
        f"{backend}: a profile 1.5x too wide changed by only {detuned:.2e} "
        f"against the soliton's {soliton:.2e}, so this grid would not have "
        f"noticed the balance being wrong"
    )
