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
from helpers import gaussian_beam
from NLSE import NLSE, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend, list_available_backends
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq

AVAILABLE_BACKENDS = list_available_backends()

N = 256
WINDOW = 4e-3
WAIST = 200e-6

# The plane-wave tests have no transverse structure to resolve.
PLANE_N = 64
# Two cells of that grid. A plane wave is its own local average, so the
# answer does not change: only the kernels that compute it do.
NONLOCAL_LENGTH = 2 * WINDOW / PLANE_N  # m
# Each integrator, and RK4 again through its non-local branch, which calls
# kernels of its own (NLSE._RK4_rhs).
PATHS = {
    "split_step": ("split_step", 0),
    "RK4": ("RK4", 0),
    "RK4_nonlocal": ("RK4", NONLOCAL_LENGTH),
}
# Steps of the Beer and self-phase runs, in m. RK4 takes the linear part
# explicitly, so each step costs four right-hand sides: these keep CPU fast.
BEER_STEP = 1e-3
SPM_STEP = 1e-4

# A uniform potential over POTENTIAL_DISTANCE: Re V turns the phase by
# POTENTIAL_PHASE, Im V divides the intensity by exp(POTENTIAL_ABSORPTION).
POTENTIAL_DISTANCE = 5e-3  # m
POTENTIAL_STEP = 1e-4  # m, as the self-phase test below
POTENTIAL_PHASE = 1.0  # rad
POTENTIAL_ABSORPTION = 0.5  # e-folds of intensity
# Relative tolerances. Measured worst cases over CPU and MLX and the three
# paths: phase 3.6e-7, intensity 1.04e-5.
POTENTIAL_PHASE_RTOL = 1e-3
POTENTIAL_INTENSITY_RTOL = 1e-4

# Factor the wavenumber is changed by between two runs of one solver.
WAVENUMBER_CHANGE = 0.5
CACHE_TEST_N = 64
CACHE_TEST_DISTANCE = 0.5  # Rayleigh lengths at the first k
# Steps per run, the same for both runs so both ask the cache for one step.
# 50 exceed the split-step cap at half k (StepSize._capped_delta_z), which
# then shortens the step and so changes the key by itself.
CACHE_TEST_STEPS = 64
# Relative tolerance on the waist; measured worst case 7.3e-7.
WAIST_RTOL = 2e-3
# Distance of the RK4 run that chooses its own step, in Rayleigh lengths. On
# this grid the shortest wave turns 1600 times faster than the beam does.
RK4_DEFAULT_STEP_DISTANCE = 0.5

# A pulse in time, uniform across: dispersion alone. Its 1/e half-width
# spans 8 samples and the window holds eight of them each side, so after
# PULSE_DISTANCE the field at the edge of the window is below 1e-5.
PULSE_N = 4  # transverse points; the field is uniform there
PULSE_NZ = 128
PULSE_WINDOW = 1e-9  # s
PULSE_DURATION = PULSE_WINDOW / 16  # s, 1/e half-width of the field
PULSE_D0 = 1e-18  # s^2/m, puts the dispersion length at 1.95 mm
PULSE_VG = c  # m/s, stored by NLSE_3d and not used in the step
PULSE_DISTANCE = 2  # dispersion lengths
# Absolute tolerance on the field, unit peak at the start. Measured: 8.1e-6
# on CPU, 2.5e-5 on MLX, single-precision round-off over the capped steps; a
# flipped D0 sign misses by 0.70.
PULSE_ATOL = 1e-4


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
    return NLSE(**{**base, **kwargs})


def plane_wave_solver(backend, path, **kwargs):
    """Return a PLANE_N-point solver for one integration path, and its method.

    Parameters
    ----------
    backend : str
        Backend name.
    path : str
        Key of PATHS.
    **kwargs
        Constructor arguments, as for ``solver``.

    Returns
    -------
    tuple
        The solver and the ``method`` to pass to ``out_field``.
    """
    method, nl_length = PATHS[path]
    simu = solver(backend, NX=PLANE_N, NY=PLANE_N, nl_length=nl_length, **kwargs)
    return simu, method


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
@pytest.mark.parametrize("path", sorted(PATHS))
def test_the_intensity_follows_beer_law(backend, alpha, path):
    """``I(z) = I0 exp(-alpha z)``: alpha attenuates intensity, not amplitude.

    A uniform field has no transverse structure, so the propagator is the
    identity and this is the loss term alone. It also pins the convention,
    which a factor of two in either direction would otherwise leave open.

    ``L`` covers the whole distance, as everywhere in this file: absorption
    is a property of the medium and stops at the end of it, so a run that
    leaves the medium follows Beer's law only as far as ``L``.
    """
    distances = (1e-2, 5e-2)
    simu, method = plane_wave_solver(backend, path, alpha=alpha, L=max(distances))
    field = np.ones((PLANE_N, PLANE_N), dtype=np.complex64)

    for z in distances:
        out = to_host(
            simu,
            simu.out_field(
                field.copy(),
                z,
                delta_z=BEER_STEP,
                method=method,
                verbose=False,
                plot=False,
                normalize=False,
            ),
        )
        got = float(np.abs(out[PLANE_N // 2, PLANE_N // 2]) ** 2)
        assert got == pytest.approx(np.exp(-alpha * z), rel=1e-4), (
            f"{backend}, {path}: alpha={alpha} over {z} m leaves {got:.6f} of the "
            f"intensity against Beer's law's {np.exp(-alpha * z):.6f}"
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("saturation", [1.0, 0.25], ids=["I_sat=I", "I_sat=I/4"])
@pytest.mark.parametrize("path", sorted(PATHS))
def test_self_phase_modulation_saturates_as_the_model_says(backend, saturation, path):
    """``phi = k n2 I z / (1 + I/Isat)`` on a plane wave.

    The interaction term alone, again with no diffraction to hide in, and the
    only test that reads the saturation as a physical law rather than as an
    expression the kernels happen to share.
    """
    n2, z = -1e-9, 5e-3
    probe = solver(backend, NX=PLANE_N, NY=PLANE_N)
    intensity = 1.0 / (probe.k * abs(n2) * z)  # about one radian of phase
    amplitude = float(np.sqrt(2 * intensity / (c * epsilon_0)))
    isat = saturation * intensity

    simu, method = plane_wave_solver(backend, path, n2=n2, Isat=isat)
    field = (amplitude * np.ones((PLANE_N, PLANE_N))).astype(np.complex64)
    out = to_host(
        simu,
        simu.out_field(
            field,
            z,
            delta_z=SPM_STEP,
            method=method,
            verbose=False,
            plot=False,
            normalize=False,
        ),
    )
    expected = simu.k * n2 * intensity * z / (1 + intensity / isat)
    phase = float(np.angle(out[PLANE_N // 2, PLANE_N // 2]))
    assert phase == pytest.approx(expected, rel=1e-3), (
        f"{backend}, {path}: Isat = {saturation} I gives a phase of "
        f"{phase:+.5f} rad against {expected:+.5f}"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("path", sorted(PATHS))
def test_a_uniform_potential_turns_and_absorbs_a_plane_wave(backend, path):
    """``A(z) = exp(i k V z / 2)``: the potential enters as ``-(k/2) V``.

    So ``Re V`` turns the phase by ``k Re(V) z / 2`` and ``Im V`` absorbs,
    ``I(z) = exp(-k Im(V) z)`` (docs/physics_problem.md:17). A plane wave
    leaves the potential alone to act, and both parts are read at once: a
    factor on ``V`` moves both, a sign turns the loss into gain.
    """
    z = POTENTIAL_DISTANCE
    probe = solver(backend, NX=PLANE_N, NY=PLANE_N)
    potential = (2 * POTENTIAL_PHASE + 1j * POTENTIAL_ABSORPTION) / (probe.k * z)
    simu, method = plane_wave_solver(
        backend,
        path,
        V=np.full((PLANE_N, PLANE_N), potential, dtype=np.complex64),
        L=z,
    )
    out = to_host(
        simu,
        simu.out_field(
            np.ones((PLANE_N, PLANE_N), dtype=np.complex64),
            z,
            delta_z=POTENTIAL_STEP,
            method=method,
            verbose=False,
            plot=False,
            normalize=False,
        ),
    )
    centre = out[PLANE_N // 2, PLANE_N // 2]
    phase = float(np.angle(centre))
    assert phase == pytest.approx(POTENTIAL_PHASE, rel=POTENTIAL_PHASE_RTOL), (
        f"{backend}, {path}: Re V turned the phase by {phase:+.5f} rad "
        f"against k Re(V) z / 2 = {POTENTIAL_PHASE:+.5f}"
    )
    kept = float(np.abs(centre) ** 2)
    assert kept == pytest.approx(
        np.exp(-POTENTIAL_ABSORPTION), rel=POTENTIAL_INTENSITY_RTOL
    ), (
        f"{backend}, {path}: Im V left {kept:.6f} of the intensity against "
        f"exp(-k Im(V) z) = {np.exp(-POTENTIAL_ABSORPTION):.6f}"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_a_new_wavenumber_diffracts_at_its_own_rate(backend, method):
    """A solver keeps its propagators, so the key has to hold ``k``.

    The second run asks for the same step on the same grid as the first, at
    another wavenumber, and must diffract as that wavenumber says,
    ``w0 sqrt(1 + (z/zR)^2)`` with ``zR = k w0^2 / 2``. A key without ``k``
    hands it the first run's propagator, and the waist of the first run.
    """
    simu = solver(backend, NX=CACHE_TEST_N, NY=CACHE_TEST_N)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    z = CACHE_TEST_DISTANCE * simu.k * WAIST**2 / 2
    for k in (simu.k, WAVENUMBER_CHANGE * simu.k):
        simu.k = k
        out = to_host(
            simu,
            simu.out_field(
                field.copy(),
                z,
                delta_z=z / CACHE_TEST_STEPS,
                method=method,
                verbose=False,
                plot=False,
                normalize=False,
            ),
        )
        got = waist_of(np.abs(out) ** 2, simu.XX, simu.YY)
        expected = WAIST * np.sqrt(1 + (z / (k * WAIST**2 / 2)) ** 2)
        assert got == pytest.approx(expected, rel=WAIST_RTOL), (
            f"{backend}, {method}: at k = {k:.4e} /m the waist is "
            f"{got * 1e6:.2f} um against the {expected * 1e6:.2f} um that k gives"
        )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_rk4_left_to_choose_its_step_diffracts_at_the_closed_form_rate(backend):
    """RK4 bounds its step by the grid's fastest mode, not by the beam's.

    Outside its stability region RK4 amplifies whatever a mode holds, and
    round-off puts something in every mode. A step sized by the rates the
    field weighs is far too long for the grid's shortest wave: that run
    returned NaN. Here it has to return ``w0 sqrt(1 + (z/zR)^2)``.
    """
    simu = solver(backend)
    field = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)
    z = RK4_DEFAULT_STEP_DISTANCE * simu.k * WAIST**2 / 2
    # The run lowers the default step and says so.
    with pytest.warns(UserWarning, match="RK4 stability"):
        out = to_host(
            simu,
            simu.out_field(
                field, z, method="RK4", verbose=False, plot=False, normalize=False
            ),
        )
    got = waist_of(np.abs(out) ** 2, simu.XX, simu.YY)
    expected = WAIST * np.sqrt(1 + RK4_DEFAULT_STEP_DISTANCE**2)
    assert got == pytest.approx(expected, rel=WAIST_RTOL), (
        f"{backend}: RK4 at its own step gives a waist of {got * 1e6:.2f} um "
        f"against the {expected * 1e6:.2f} um diffraction theory gives"
    )


@pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
def test_a_pulse_disperses_as_the_chirped_gaussian(backend):
    """The documented ``i dA/dz = (D0/2) d^2A/dt^2``, as a chirped Gaussian.

    In Fourier space it reads ``dA/dz = +i (D0/2) w^2 A``, diffraction's
    ``-i w^2 / (2k)`` with ``1/k`` replaced by ``-D0``. So the pulse is the
    Gaussian beam of wavenumber ``-1/D0`` along time: width
    ``tau0 sqrt(1 + (z/zD)^2)`` with ``zD = tau0^2 / (2 |D0|)``, and a chirp
    whose sign is the sign of ``D0``, which the complex field carries.
    """
    simu = NLSE_3d(
        alpha=0,
        energy=1,
        window=[WINDOW, PULSE_WINDOW],
        n2=0,
        D0=PULSE_D0,
        vg=PULSE_VG,
        V=None,
        L=1e-2,
        NX=PULSE_N,
        NY=PULSE_N,
        NZ=PULSE_NZ,
        backend=backend,
    )
    field = np.exp(-(simu.TT**2) / PULSE_DURATION**2).astype(np.complex64)
    z = PULSE_DISTANCE * PULSE_DURATION**2 / (2 * abs(PULSE_D0))
    out = to_host(
        simu,
        simu.out_field(field, z, verbose=False, plot=False, normalize=False),
    )
    expected = gaussian_beam(simu.TT**2, PULSE_DURATION, -1 / PULSE_D0, z, 1)
    np.testing.assert_allclose(
        out,
        expected,
        rtol=0,
        atol=PULSE_ATOL,
        err_msg=f"{backend}: the pulse is not the chirped Gaussian D0 gives",
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
