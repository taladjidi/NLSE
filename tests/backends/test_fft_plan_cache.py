"""FFT plans are built once per transform, not once per propagation.

A plan depends only on the shape, the axes and the dtype, so rebuilding one
per ``out_field`` call bought nothing and cost a great deal: VkFFT compiles
its own kernels, and cuFFT allocates its work area.

Caching is only safe because a plan is not bound to the array it was built
with: FFTW, cuFFT and VkFFT all take their arrays at call time. The tests
below pin that, since a plan that did capture its array would couple two
solvers silently rather than fail.
"""

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends

AVAILABLE_BACKENDS = list_available_backends()

N = 32
WAIST = 2.23e-3
BASE = {
    "alpha": 0.0,
    "power": 1.05,
    "window": 4 * WAIST,
    "n2": -1.6e-9,
    "V": None,
    "L": 10e-3,
    "Isat": 10e4,
}


def solver(backend_name, n=N):
    """Return a solver on an n x n grid."""
    return NLSE(NX=n, NY=n, backend=backend_name, **BASE)


def gaussian(simu):
    """Return a Gaussian input field for this solver's grid."""
    return np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2).astype(np.complex64)


def host(simu, array):
    """Return an array as numpy, whatever backend produced it."""
    if isinstance(array, np.ndarray):
        return array
    return simu._backend.to_numpy(array)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_same_transform_is_planned_once(backend_name):
    """Repeated builds of one transform must return the same plan."""
    backend = get_backend(backend_name)
    backend.clear_fft_plans()
    first = backend.build_fft((N, N), (-2, -1), np.complex64)
    second = backend.build_fft((N, N), (-2, -1), np.complex64)
    assert first is second, (
        f"{backend_name} rebuilt a plan for a transform it had already planned"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_different_transforms_get_different_plans(backend_name):
    """Shape, axes and dtype must each separate cache entries."""
    backend = get_backend(backend_name)
    backend.clear_fft_plans()
    base = backend.build_fft((N, N), (-2, -1), np.complex64)
    assert backend.build_fft((2 * N, N), (-2, -1), np.complex64) is not base
    assert backend.build_fft((N, N), (-1,), np.complex64) is not base
    if backend.supports_double_precision():
        assert backend.build_fft((N, N), (-2, -1), np.complex128) is not base


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_propagation_reuses_the_plan(backend_name):
    """A second out_field must not plan again."""
    simu = solver(backend_name)
    E = gaussian(simu)
    simu.out_field(E.copy(), 1e-3, verbose=False, plot=False, delta_z=1e-4)
    first = simu.plans
    simu.out_field(E.copy(), 1e-3, verbose=False, plot=False, delta_z=1e-4)
    assert simu.plans is first, (
        f"{backend_name} planned again on the second propagation"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_solvers_on_one_grid_share_a_plan(backend_name):
    """A parameter sweep must plan once, not once per point."""
    a, b = solver(backend_name), solver(backend_name)
    E = gaussian(a)
    for simu in (a, b):
        simu.out_field(E.copy(), 1e-3, verbose=False, plot=False, delta_z=1e-4)
    assert a.plans is b.plans, (
        f"{backend_name}: two solvers on the same grid hold different plans, "
        f"so a sweep pays the planning cost per point"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_shared_plan_does_not_couple_solvers(backend_name):
    """Sharing a plan must not let one solver's field reach another.

    This is what makes the cache safe. A plan that captured the array it was
    built with would silently mix two runs rather than fail.
    """
    strong = solver(backend_name)
    weak = NLSE(NX=N, NY=N, backend=backend_name, **{**BASE, "n2": -1e-12})
    E = gaussian(strong)

    together_strong = host(
        strong,
        strong.out_field(E.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-4),
    )
    together_weak = host(
        weak, weak.out_field(E.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-4)
    )

    alone = NLSE(NX=N, NY=N, backend=backend_name, **BASE)
    get_backend(backend_name).clear_fft_plans()
    expected = host(
        alone, alone.out_field(E.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-4)
    )

    np.testing.assert_allclose(
        np.asarray(together_strong),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-6 * float(np.max(np.abs(expected))),
        err_msg=f"{backend_name}: sharing a cached plan changed the result",
    )
    assert not np.allclose(together_strong, together_weak), (
        "precondition: the two solvers should disagree, so the check above "
        "is not trivially satisfied"
    )


@pytest.mark.skipif("CL" not in AVAILABLE_BACKENDS, reason="OpenCL not available")
def test_vkfft_apps_are_built_on_demand():
    """Only the apps a run actually uses may be compiled.

    _VkFFTPlan wraps three VkFFTApps and each compiles its own kernels. A
    split-step run needs the in-place one, and the unnormalized one only
    where the 1/N is folded into the propagator; the out-of-place one is
    for RK4.
    """
    backend = get_backend("CL")
    backend.clear_fft_plans()
    plan = backend.build_fft((N, N), (-2, -1), np.complex64)[0]

    assert plan._app is None and plan._app_oop is None, (
        "a freshly built plan should have compiled nothing yet"
    )

    from pyopencl import array as cla

    A = cla.zeros(backend.queue, (N, N), np.complex64)
    plan.fft(A, A)
    assert plan._app is not None, "the in-place app should now exist"
    assert plan._app_oop is None, (
        "a forward transform must not have compiled the out-of-place app"
    )
