"""The non-local response, against the closed form of a diffusive non-locality.

The interaction reads the intensity smoothed by the Green function of a
diffusion equation. On a periodic grid that is a multiplication by
``R(K) = 1 / (1 + l^2 K^2)`` in Fourier space, so two answers are known
without running anything else:

- a uniform intensity has only ``K = 0``, where ``R = 1``: the non-local phase
  is the local one, the same at every point;
- ``I = I0 (1 + eps cos(q x))`` is smoothed to ``I0 (1 + eps R(q) cos(q x))``,
  so the modulation of the nonlinear phase shrinks by exactly ``R(q)``.

Both integrators are checked, as the rate of nonlinear phase each one puts on
the field: the split step's nonlinear step, and RK4's right-hand side with the
dispersion left out. They reach the response through different call sites.

The grids have ``NX != NY`` and ``delta_X != delta_Y`` on purpose, so a length
measured in the wrong axis's cells, or in cells at all, shows up as a wrong
factor along one of the two axes.
"""

import numpy as np
import pytest
from NLSE import CNLSE, NLSE, CNLSE_1d, NLSE_1d, NLSE_3d
from scipy.constants import c, epsilon_0

WVL = 780e-9  # m, the solvers' default wavelength
DZ = 1e-3  # m, the step the nonlinear phase is taken over
PHASE = 1.0  # rad, nonlinear phase of a unit field over DZ: well inside (-pi, pi)
N2 = PHASE / (2 * np.pi / WVL * c * epsilon_0 / 2 * DZ)  # m^2/W, gives PHASE
ISAT = 1e30  # W/m^2, saturation out of reach of a unit field
EPS = 0.5  # depth of the intensity modulation
PERIODS = 3  # periods of the modulation across the window
NX, NY, NZ = 64, 48, 16
WINDOW = (1.28e-3, 1.44e-3)  # m, so delta_X = 20 um and delta_Y = 30 um
WINDOW_3D = (6.4e-4, 7.2e-4, 1e-9)  # m, m, s: delta_X = 20 um, delta_Y = 30 um
NL_LENGTH = 40e-6  # m, two cells along x and a cell and a third along y
RTOL = 1e-4  # float32 phase of order 1 rad, measured error below 1e-5
METHODS = ["split_step", "RK4"]  # the integrators, as out_field names them


def response(q, nl_length=NL_LENGTH):
    """Return the closed-form response ``1 / (1 + l^2 q^2)`` at wavenumber q."""
    return 1 / (1 + nl_length**2 * q**2)


def to_host(simu, array):
    """Return an array as numpy, whatever backend produced it."""
    return np.asarray(simu._backend.to_numpy(array))


def nonlinear_phase(simu, field, method):
    """Return the nonlinear phase one integrator puts on a real field over DZ.

    The split step returns the phase of one nonlinear step of length DZ. RK4
    returns ``DZ Im(f / A)``, f its right-hand side with a zero propagator:
    the same phase, to first order, read off the rate.

    Parameters
    ----------
    simu : NLSE
        Solver, of any of the single-component classes.
    field : np.ndarray
        Real, positive field, complex64.
    method : str
        ``"split_step"`` or ``"RK4"``.

    Returns
    -------
    np.ndarray
        The phase at every point, in rad.
    """
    A, A_sq = simu._prepare_output_array(field, normalize=False)
    simu._precompute_step_constants(None, field.dtype)
    if method == "split_step":
        return np.angle(to_host(simu, simu._nonlinear_step(A, A_sq, None, DZ)))
    zero = np.zeros(simu._compute_propagator_rk4().shape, dtype=field.dtype)
    rhs = simu._RK4_rhs(
        A,
        simu._backend.from_numpy(np.zeros_like(field)),
        None,
        simu._backend.from_numpy(zero),
        simu._build_fft_plan(A),
    )
    return DZ * (to_host(simu, rhs) / field).imag


def modulated(coordinate, window):
    """Return ``sqrt(1 + EPS cos(q x))`` and q, with PERIODS periods in window."""
    q = 2 * np.pi * PERIODS / window
    return np.sqrt(1 + EPS * np.cos(q * coordinate)).astype(np.complex64), q


def make_2d(backend, nl_length):
    """Return the 2D solver on the module's anisotropic grid."""
    return NLSE(0, 1, WINDOW, N2, None, 1, NX, NY, ISAT, nl_length, WVL, backend)


def make_1d(backend, nl_length):
    """Return the 1D solver on the module's x grid."""
    return NLSE_1d(0, 1, WINDOW[0], N2, None, 1, NX, ISAT, nl_length, WVL, backend)


def make_3d(backend, nl_length):
    """Return the 3D solver on the module's anisotropic grid."""
    return NLSE_3d(
        0, 1, WINDOW_3D, N2, 0, 3e8, None, 1, NX, NY, NZ, ISAT, nl_length, WVL, backend
    )


@pytest.mark.parametrize("method", METHODS)
def test_a_local_step_gives_the_self_phase_of_its_intensity(backend, method):
    """At nl_length = 0 the phase is ``k n2 I dz``, ``I = c eps0 |A|^2 / 2``."""
    simu = make_2d(backend, 0)
    field, _ = modulated(simu.XX, WINDOW[0])
    intensity = c * epsilon_0 / 2 * np.abs(field) ** 2
    np.testing.assert_allclose(
        nonlinear_phase(simu, field, method), simu.k * N2 * intensity * DZ, rtol=RTOL
    )


@pytest.mark.parametrize("make", [make_1d, make_2d, make_3d], ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("method", METHODS)
def test_a_uniform_field_takes_the_local_phase_everywhere(backend, make, method):
    """``R(0) = 1``: a uniform intensity is its own smoothing, edges included."""
    local = make(backend, 0)
    field = np.ones(local.XX.shape, dtype=np.complex64)
    expected = nonlinear_phase(local, field, method)
    phase = nonlinear_phase(make(backend, NL_LENGTH), field, method)
    np.testing.assert_allclose(phase, expected, rtol=RTOL)
    assert np.ptp(phase) <= RTOL * PHASE, "the phase is not the same everywhere"


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("method", METHODS)
def test_a_modulation_is_reduced_by_the_response_2d(backend, axis, method):
    """``I0 (1 + eps cos(q r))`` keeps ``R(q)`` of its phase modulation."""
    local, nonlocal_ = make_2d(backend, 0), make_2d(backend, NL_LENGTH)
    grid, window = (local.XX, WINDOW[0]) if axis == "x" else (local.YY, WINDOW[1])
    field, q = modulated(grid, window)
    ratio = np.ptp(nonlinear_phase(nonlocal_, field, method)) / np.ptp(
        nonlinear_phase(local, field, method)
    )
    assert ratio == pytest.approx(response(q), rel=RTOL)


@pytest.mark.parametrize("method", METHODS)
def test_a_modulation_is_reduced_by_the_response_1d(backend, method):
    """The 1D solver smooths with the 1D form of the same response."""
    local, nonlocal_ = make_1d(backend, 0), make_1d(backend, NL_LENGTH)
    field, q = modulated(local.X, WINDOW[0])
    ratio = np.ptp(nonlinear_phase(nonlocal_, field, method)) / np.ptp(
        nonlinear_phase(local, field, method)
    )
    assert ratio == pytest.approx(response(q), rel=RTOL)


@pytest.mark.parametrize("axis", ["x", "y", "t"])
@pytest.mark.parametrize("method", METHODS)
def test_a_modulation_is_reduced_by_the_transverse_response_3d(backend, axis, method):
    """Across the beam the 3D response is the 2D one; along the pulse it is 1."""
    local, nonlocal_ = make_3d(backend, 0), make_3d(backend, NL_LENGTH)
    grid, window = {
        "x": (local.XX, WINDOW_3D[0]),
        "y": (local.YY, WINDOW_3D[1]),
        "t": (local.TT, WINDOW_3D[2]),
    }[axis]
    field, q = modulated(grid, window)
    ratio = np.ptp(nonlinear_phase(nonlocal_, field, method)) / np.ptp(
        nonlinear_phase(local, field, method)
    )
    expected = 1.0 if axis == "t" else response(q)
    assert ratio == pytest.approx(expected, rel=RTOL)


@pytest.mark.parametrize("method", METHODS)
def test_a_batch_is_smoothed_member_by_member(backend, method):
    """A batch shares one response, and each member gets its own smoothing."""
    simu = make_2d(backend, NL_LENGTH)
    along_x, _ = modulated(simu.XX, WINDOW[0])
    along_y, _ = modulated(simu.YY, WINDOW[1])
    batched = nonlinear_phase(simu, np.array([along_x, along_y]), method)
    np.testing.assert_allclose(
        batched[0], nonlinear_phase(simu, along_x, method), rtol=RTOL
    )
    np.testing.assert_allclose(
        batched[1], nonlinear_phase(simu, along_y, method), rtol=RTOL
    )


def coupled_case(backend, cls):
    """Return a coupled solver, its two modulated fields and their closed forms.

    Parameters
    ----------
    backend : str
        Backend name.
    cls : type
        ``CNLSE`` (modulations along x and along y) or ``CNLSE_1d`` (both
        along x).

    Returns
    -------
    tuple
        The solver, the fields as one (2, ...) array, and the two smoothed
        intensities ``1 + EPS R(q) cos(q x)`` the response leaves of them.
    """
    if cls is CNLSE:
        simu = CNLSE(
            0, 1, WINDOW, N2, N2, None, 1, NX, NY, ISAT, NL_LENGTH, WVL, None, backend
        )
        first, q1 = modulated(simu.XX, WINDOW[0])
        second, q2 = modulated(simu.YY, WINDOW[1])
        coordinates = (simu.XX, simu.YY)
    else:
        simu = CNLSE_1d(
            0, 1, WINDOW[0], N2, N2, None, 1, NX, ISAT, NL_LENGTH, WVL, None, backend
        )
        first, q1 = modulated(simu.X, WINDOW[0])
        second, q2 = first, q1
        coordinates = (simu.X, simu.X)
    smoothed = [
        1 + EPS * response(q) * np.cos(q * coordinate)
        for q, coordinate in zip((q1, q2), coordinates)
    ]
    return simu, np.array([first, second]), smoothed


@pytest.mark.parametrize("cls", [CNLSE, CNLSE_1d], ids=lambda cls: cls.__name__)
def test_each_component_of_a_coupled_field_is_smoothed(backend, cls):
    """The coupled solvers smooth both intensities with the same response."""
    simu, fields, expected = coupled_case(backend, cls)
    A, A_sq = simu._prepare_output_array(fields, normalize=False)
    _, smoothed_1, smoothed_2 = simu._compute_A_sq_components(A, A_sq)
    for smoothed, closed_form in zip((smoothed_1, smoothed_2), expected):
        np.testing.assert_allclose(to_host(simu, smoothed), closed_form, atol=RTOL)


@pytest.mark.parametrize("cls", [CNLSE, CNLSE_1d], ids=lambda cls: cls.__name__)
def test_the_coupled_rk4_rate_takes_both_smoothed_intensities(backend, cls):
    """With n12 = n2, each component's RK4 phase is PHASE (I1 + I2), smoothed."""
    simu, fields, expected = coupled_case(backend, cls)
    A, _ = simu._prepare_output_array(fields, normalize=False)
    simu._precompute_step_constants(None, fields.dtype)
    simu._allocate_rk4_buffers(A, "RK4")
    zero = np.zeros(simu._compute_propagator_rk4().shape, dtype=fields.dtype)
    rhs = simu._RK4_rhs(
        A,
        simu._backend.from_numpy(np.zeros_like(fields)),
        None,
        simu._backend.from_numpy(zero),
        simu._build_fft_plan(A),
    )
    phase = DZ * (to_host(simu, rhs) / fields).imag
    for component in phase:
        np.testing.assert_allclose(
            component, PHASE * (expected[0] + expected[1]), rtol=RTOL
        )
