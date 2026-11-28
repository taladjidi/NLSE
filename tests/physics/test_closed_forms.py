"""The answers, against solutions that are known rather than computed here."""

import numpy as np
import pytest
from helpers import gaussian_beam
from NLSE import NLSE, NLSE_1d, NLSE_3d
from NLSE.backends import get_backend
from scipy.constants import c, epsilon_0
from scipy.optimize import brentq

N = 256
WINDOW = 4e-3
WAIST = 200e-6

# The plane-wave tests have no transverse structure to resolve.
PLANE_N = 64
# Two cells of that grid. A plane wave is its own local average, so the
# answer does not change: only the kernels that compute it do.
NONLOCAL_LENGTH = 2 * WINDOW / PLANE_N  # m
# Each integrator, and each again through its non-local branch, which calls
# kernels of its own: NLSE._RK4_rhs, and nl_prop with |A|^2 passed in.
PATHS = {
    "split_step": ("split_step", 0),
    "split_step_nonlocal": ("split_step", NONLOCAL_LENGTH),
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
# Relative tolerances. Measured worst cases over CPU and MLX and the four
# paths, both on MLX split-step, local and non-local alike: phase 3.6e-7,
# intensity 1.06e-5.
POTENTIAL_PHASE_RTOL = 1e-3
POTENTIAL_INTENSITY_RTOL = 1e-4

# Loss, interaction and an absorbing potential at once, over POTENTIAL_DISTANCE
# at POTENTIAL_STEP, Re V turning the phase by POTENTIAL_PHASE as above.
MIXED_N2 = -1e-9  # m^2/W, the self-phase test's
MIXED_KERR_PHASE = 1.0  # rad, k |n2| I0 z at the entrance intensity I0
MIXED_LOSS = 0.5  # e-folds of intensity alpha takes out
MIXED_ABSORPTION = 0.1  # e-folds of intensity Im V takes out
# Measured worst cases over CPU and MLX and the four paths: phase 7.5e-4 rad on
# split-step, which freezes the Im V decay across a step, and intensity 8.6e-6
# on MLX. Smallest effects guarded: Im V dropped, 10% of the intensity, and
# the potential or the self-phase dropped or flipped, 0.75 rad and more.
MIXED_PHASE_ATOL = 2e-3  # rad
MIXED_INTENSITY_RTOL = 1e-4

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

# The lossy saturable run: its step (m), and what its end intensity
# (relative) and phase (rad) may be off by. Measured 2.0e-5 and 2.1e-5 (MLX,
# the worst); 1.1e-4 (MLX) to 1.4e-4 (CPU) and 1.1e-3 with the step frozen.
# At 1e-5 m the two are both inside the float32 round-off of 5000 steps.
LOSSY_STEP = 1e-4
LOSSY_INTENSITY_RTOL = 6e-5
LOSSY_PHASE_ATOL = 2e-4
# Points per side for the Gouy phase: 28 to 112 steps where N has 449 to
# 1797, and the same answer (9.9e-6 rad off at worst, 5.7e-5 at N).
GOUY_N = 64


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


@pytest.mark.parametrize("z_over_zr", [0.5, 1.0, 2.0])
def test_a_gaussian_beam_diffracts_at_the_rate_diffraction_theory_gives(
    backend, z_over_zr
):
    """A Gaussian beam's radius is ``w(z) = w0 sqrt(1 + (z/zR)^2)``."""
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


def test_the_on_axis_phase_is_the_gouy_phase(backend):
    """The on-axis phase is the Gouy phase ``-arctan(z/zR)``."""
    simu = solver(backend, NX=GOUY_N, NY=GOUY_N)
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
        got = float(np.angle(out[GOUY_N // 2, GOUY_N // 2]))
        assert got == pytest.approx(-np.arctan(z_over_zr), abs=2e-3), (
            f"{backend}: on-axis phase at z = {z_over_zr} zR is {got:+.4f} rad "
            f"against a Gouy phase of {-np.arctan(z_over_zr):+.4f}"
        )


def test_a_grating_revives_at_the_talbot_distance(backend):
    """A grating images itself at ``z_T = k d^2 / pi``, and not before."""
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


@pytest.mark.parametrize("alpha", [5.0, 20.0])
@pytest.mark.parametrize("path", sorted(PATHS))
def test_the_intensity_follows_beer_law(backend, alpha, path):
    """Intensity decays as ``I0 exp(-alpha z)``: alpha is an intensity rate."""
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


@pytest.mark.parametrize("saturation", [1.0, 0.25], ids=["I_sat=I", "I_sat=I/4"])
@pytest.mark.parametrize("path", sorted(PATHS))
def test_self_phase_modulation_saturates_as_the_model_says(backend, saturation, path):
    """A plane wave's self-phase is ``phi = k n2 I z / (1 + I/Isat)``."""
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


@pytest.mark.parametrize("path", sorted(PATHS))
def test_a_uniform_potential_turns_and_absorbs_a_plane_wave(backend, path):
    """A uniform potential gives a plane wave ``A(z) = exp(i k V z / 2)``."""
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


@pytest.mark.parametrize("path", sorted(PATHS))
def test_loss_interaction_and_potential_act_together_on_a_plane_wave(backend, path):
    """Loss, interaction and potential together match their closed form."""
    z = POTENTIAL_DISTANCE
    probe = solver(backend, NX=PLANE_N, NY=PLANE_N)
    intensity = MIXED_KERR_PHASE / (probe.k * abs(MIXED_N2) * z)
    amplitude = float(np.sqrt(2 * intensity / (c * epsilon_0)))
    potential = (2 * POTENTIAL_PHASE + 1j * MIXED_ABSORPTION) / (probe.k * z)
    simu, method = plane_wave_solver(
        backend,
        path,
        alpha=MIXED_LOSS / z,
        n2=MIXED_N2,
        V=np.full((PLANE_N, PLANE_N), potential, dtype=np.complex64),
        L=z,
    )
    out = to_host(
        simu,
        simu.out_field(
            (amplitude * np.ones((PLANE_N, PLANE_N))).astype(np.complex64),
            z,
            delta_z=POTENTIAL_STEP,
            method=method,
            verbose=False,
            plot=False,
            normalize=False,
        ),
    )
    centre = out[PLANE_N // 2, PLANE_N // 2]
    decay = MIXED_LOSS + MIXED_ABSORPTION  # gamma z
    kept = float(np.abs(centre / amplitude) ** 2)
    assert kept == pytest.approx(np.exp(-decay), rel=MIXED_INTENSITY_RTOL), (
        f"{backend}, {path}: {kept:.6f} of the intensity is left against "
        f"exp(-(alpha + k Im V) z) = {np.exp(-decay):.6f}"
    )
    expected = POTENTIAL_PHASE - MIXED_KERR_PHASE * (1 - np.exp(-decay)) / decay
    phase = float(np.angle(centre))
    assert phase == pytest.approx(expected, abs=MIXED_PHASE_ATOL), (
        f"{backend}, {path}: the phase is {phase:+.5f} rad against "
        f"{expected:+.5f} from Re V and the self-phase of the decaying intensity"
    )


@pytest.mark.parametrize("method", ["split_step", "RK4"])
def test_a_new_wavenumber_diffracts_at_its_own_rate(backend, method):
    """A new wavenumber diffracts at its own rate: the cache key holds ``k``."""
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


def test_rk4_left_to_choose_its_step_diffracts_at_the_closed_form_rate(backend):
    """RK4 bounds its step by the grid's fastest mode, not by the beam's."""
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


def test_a_pulse_disperses_as_the_chirped_gaussian(backend):
    """A pulse disperses as the chirped Gaussian of the documented equation."""
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


def test_a_lossy_saturable_step_obeys_the_identity_it_is_solved_from(backend):
    """A lossy saturable step obeys the identity it is solved from."""
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
            field, z, delta_z=LOSSY_STEP, verbose=False, plot=False, normalize=False
        ),
    )

    target = np.log(intensity) + intensity / isat - alpha * z
    exact_end = brentq(
        lambda y: np.log(y) + y / isat - target, 1e-12 * intensity, intensity
    )
    got_end = c * epsilon_0 / 2 * float(np.abs(out[32, 32])) ** 2
    assert got_end == pytest.approx(exact_end, rel=LOSSY_INTENSITY_RTOL), (
        f"{backend}: the intensity ends at {got_end:.6e} W/m^2 against the "
        f"{exact_end:.6e} that ln(I) + I/Isat falling by alpha*z gives"
    )

    expected_phase = simu.k * n2 / alpha * (intensity - exact_end)
    got_phase = float(np.angle(out[32, 32]))
    wrapped = float(np.angle(np.exp(1j * expected_phase)))
    assert got_phase == pytest.approx(wrapped, abs=LOSSY_PHASE_ATOL), (
        f"{backend}: the phase is {got_phase:+.5f} rad against the "
        f"{wrapped:+.5f} the identity gives from the intensity drop alone"
    )


def test_a_bright_soliton_keeps_its_shape(backend):
    """A bright soliton keeps its shape: diffraction and self-focusing cancel."""
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
