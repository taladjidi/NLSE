"""Tests that backend capability flags match what the kernels actually provide.

The solvers pick their fused fast paths from the flags declared on the
Backend subclasses. A flag set to True with no matching kernel method is an
AttributeError at propagation time, and a method present with the flag left
False is silently dead code. Both are caught here rather than on hardware.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from NLSE import NLSE
from NLSE.backends import Backend, get_backend, list_available_backends
from NLSE.backends.backend import Timing
from scipy.constants import c, epsilon_0

# Capability flag -> kernel methods it promises.
CAPABILITY_METHODS = {
    "has_linear_step": ["linear_step"],
    "has_fused_split_step": ["split_step_fused"],
    "has_fused_rk4_rhs": ["rk4_rhs_fused"],
    "has_fused_rk4_step": ["split_step_rk4_fused"],
    "has_fused_rk4_stage_update": ["rk4_set_and_axpy", "rk4_acc_and_axpy"],
    "has_fused_rk4_final_update": ["rk4_final_update"],
    "has_fused_rk4_stage": ["rk4_stage_fused", "rk4_stage_coupled_fused"],
    "has_fused_coupled_split_step": ["split_step_coupled_fused"],
    "has_fused_coupled_rk4_rhs": ["rk4_rhs_coupled_fused"],
}

# Methods every backend's kernels must provide, whatever it declares.
REQUIRED_METHODS = [
    "nl_prop",
    "nl_prop_without_V",
    "nl_prop_c",
    "nl_prop_without_V_c",
    "square_mod",
    "square_mod_nl_prop",
    "square_mod_nl_prop_v",
    "apply_propagator",
    "rabi_coupling",
    "rk4_axpy",
    "rk4_accumulate",
    "rk4_nl_rhs",
    "rk4_nl_rhs_v",
    "square_mod_rk4_nl_rhs",
    "square_mod_rk4_nl_rhs_v",
    "rk4_nl_rhs_c",
    "rk4_nl_rhs_c_v",
]

AVAILABLE_BACKENDS = list_available_backends()


def test_every_capability_is_declared_on_the_base_class():
    """The flags the tests know about must exist on Backend with a default."""
    for flag in CAPABILITY_METHODS:
        assert hasattr(Backend, flag), f"Backend does not declare {flag}"
        assert getattr(Backend, flag) is False, f"{flag} must default to False"
    assert hasattr(Backend, "supports_unnormalized_ifft")


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_declared_capabilities_are_backed_by_kernels(backend_name):
    """Every capability a backend claims must have its kernel method."""
    backend = get_backend(backend_name)
    kernels = backend.kernels
    for flag, methods in CAPABILITY_METHODS.items():
        if not getattr(backend, flag):
            continue
        for method in methods:
            assert hasattr(kernels, method), (
                f"{backend_name} declares {flag}=True but its kernels have no "
                f"{method}(). The solver will raise AttributeError on that path."
            )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_required_kernels_are_present(backend_name):
    """The non-optional part of the kernel interface must be complete."""
    kernels = get_backend(backend_name).kernels
    missing = [m for m in REQUIRED_METHODS if not hasattr(kernels, m)]
    assert not missing, f"{backend_name} kernels are missing: {missing}"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_an_unnormalized_ifft_really_skips_the_factor(backend_name):
    """A backend claiming the capability must actually honour it.

    The caller folds 1/N into the propagator and then asks the transform to
    skip it, so a backend that accepts ``normalize=False`` and normalizes
    anyway divides the field by N twice -- which no correctness test would
    attribute to the transform. Round-tripping is the direct check: with the
    factor skipped the field comes back N times larger.

    This replaces an assertion that the capability implied has_linear_step.
    That was true only while the fused kernels were the sole consumer; the
    generic linear step reads the pre-normalized propagator too, which is
    how the CPU backend gets the saving without a fused entry point.
    """
    backend = get_backend(backend_name)
    if not backend.supports_unnormalized_ifft:
        return
    n = 16
    field = np.zeros((n, n), dtype=np.complex64)
    field[n // 4, n // 4] = 1.0 + 0.5j
    plans = backend.build_fft(field.shape, (-2, -1), np.complex64, field)

    def round_trip(normalize):
        A = backend.from_numpy(field.copy())
        A = backend.fft(A, plans)
        A = backend.ifft(A, plans, normalize=normalize)
        return backend.to_numpy(A)

    normalized = round_trip(True)
    assert np.allclose(normalized, field, atol=1e-5), (
        f"{backend_name} does not round-trip a normalized transform"
    )
    assert np.allclose(round_trip(False), field * (n * n), atol=1e-3), (
        f"{backend_name} declares supports_unnormalized_ifft but its ifft "
        f"normalized anyway, so the propagator's 1/N would be applied twice"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_fused_signatures_accept_the_documented_arguments(backend_name):
    """Fused entry points must share one signature across backends.

    The solvers call them with a single argument list, so a backend that
    renames or reorders a parameter breaks silently on that backend only.
    """
    backend = get_backend(backend_name)
    kernels = backend.kernels
    expected = {
        # Every entry point the base class documents a signature for, not the
        # three that used to be here. That gap is how MLX's linear_step came
        # to take (A, propagator, axes) with no unnorm_ifft while CUPY and CL
        # took the documented four: nothing compared them, and the solvers
        # passed the argument only when it was True, which on MLX is never.
        "linear_step": ["A", "propagator", "plan", "unnorm_ifft"],
        "split_step_fused": [
            "A",
            "propagator",
            "V_scaled",
            "dz",
            "alpha",
            "g",
            "Isat",
            "splitting",
            "plan",
            "unnorm_ifft",
        ],
        "rk4_rhs_fused": [
            "A_in",
            "k",
            "V_scaled",
            "propagator",
            "plan",
            "alpha",
            "g",
            "Isat",
            "unnorm_ifft",
        ],
        "split_step_rk4_fused": [
            "A",
            "propagator",
            "V_scaled",
            "dz",
            "alpha",
            "g",
            "Isat",
            "plan",
        ],
        "rk4_set_and_axpy": ["acc", "out", "A", "k", "c"],
        "rk4_acc_and_axpy": ["acc", "out", "A", "k", "w", "c"],
        "split_step_coupled_fused": [
            "A",
            "propagator",
            "V1_scaled",
            "V2_scaled",
            "dz",
            "alpha1",
            "alpha2",
            "g11",
            "g12",
            "g22",
            "Isat1",
            "Isat2",
            "splitting",
            "plan",
            "omega",
            "unnorm_ifft",
        ],
        "rk4_rhs_coupled_fused": [
            "A_in",
            "k",
            "V1_scaled",
            "V2_scaled",
            "propagator",
            "plan",
            "alpha1",
            "alpha2",
            "g11",
            "g12",
            "g22",
            "Isat1",
            "Isat2",
            "unnorm_ifft",
        ],
    }
    for method, params in expected.items():
        if not hasattr(kernels, method):
            continue
        sig = inspect.signature(getattr(kernels, method))
        got = [p for p in sig.parameters if p != "self"]
        assert got == params, (
            f"{backend_name}.{method} signature is {got}, expected {params}"
        )


# ---------------------------------------------------------------------------
# Capabilities that replaced a check on the backend's name
# ---------------------------------------------------------------------------

SOLVERS = Path(__file__).resolve().parent.parent.parent / "NLSE" / "solvers"


def name_checks_in(path):
    """Return (line, source) for each comparison against a backend's name."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Compare):
            continue
        text = ast.unparse(node)
        if "_backend.name" in text and isinstance(node.ops[0], (ast.Eq, ast.In)):
            found.append((node.lineno, text))
    return found


def test_there_are_solvers_to_check():
    """A glob matching nothing would make the test below vacuous."""
    assert len(list(SOLVERS.glob("*.py"))) > 5


@pytest.mark.parametrize("path", sorted(SOLVERS.glob("*.py")), ids=lambda p: p.name)
def test_no_solver_branches_on_the_backend_name(path):
    """Ask what a backend can do. Which one it is is not a capability."""
    offenders = [f"{path.name}:{line}  {text}" for line, text in name_checks_in(path)]
    assert not offenders, (
        "branch on a capability rather than on the backend's identity:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_a_backend_claiming_a_convolution_has_a_working_one(backend_name):
    """``convolution`` doubles as the non-locality capability, so it must run."""
    backend = get_backend(backend_name)
    if backend.convolution is None:
        pytest.skip("declares no convolution")

    signal = backend.from_numpy(np.ones((8, 8), dtype=np.float32))
    kernel = backend.from_numpy(np.ones((3, 3), dtype=np.float32))
    out = np.asarray(backend.to_numpy(backend.convolution(signal, kernel, mode="same")))

    assert out.shape == (8, 8)
    assert out[4, 4] == pytest.approx(9.0), "a 3x3 box over ones is 9 in the interior"


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_synchronize_takes_an_array_or_nothing(backend_name):
    """The solver calls it with the field; the contract allows neither."""
    backend = get_backend(backend_name)
    backend.synchronize()
    backend.synchronize(backend.from_numpy(np.ones((4, 4), dtype=np.complex64)))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_timed_reports_a_positive_wall_time(backend_name):
    """And a device time only where the backend can measure one."""
    backend = get_backend(backend_name)
    with backend.timed() as timing:
        arr = backend.from_numpy(np.ones((64, 64), dtype=np.complex64))
        backend.synchronize(arr)

    assert isinstance(timing, Timing)
    assert timing.wall > 0, "wall time was not filled in on exit"
    if timing.device is not None:
        assert timing.device >= 0, "a reported device time cannot be negative"


def test_the_timing_line_names_a_device_only_when_there_is_one():
    """out_field prints this straight, so the wording lives with the data."""
    assert str(Timing(wall=1.5)) == "Time spent to solve : 1.5 s (CPU)"
    assert "GPU" in str(Timing(wall=1.5, device=0.5))
    assert "GPU" not in str(Timing(wall=1.5))


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_the_buffer_route_carries_the_field_through(backend_name):
    """Every backend stages through its pre-allocated buffer.

    MLX used to skip that and let each operation allocate. Measured, the two
    are within noise on MLX at every size and method, and staging keeps the
    buffer a run owns identifiable across a step. So there is one route, and
    this checks it.
    """
    waist = 2.23e-3
    simu = NLSE(
        alpha=0,
        power=1.05,
        window=4 * waist,
        n2=-1e-9,
        V=None,
        L=1e-3,
        NX=32,
        NY=32,
        Isat=1e5,
        backend=backend_name,
    )
    field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)

    out, _ = simu._prepare_output_array(field.copy(), normalize=False)
    np.testing.assert_allclose(
        np.asarray(simu._backend.to_numpy(out)),
        field,
        rtol=1e-6,
        err_msg=f"{backend_name} did not carry the field through unnormalized",
    )

    propagated = np.asarray(
        simu.out_field(
            field.copy(), 2e-3, verbose=False, plot=False, delta_z=1e-4, method="RK4"
        )
    )
    assert np.all(np.isfinite(propagated)), (
        f"{backend_name} produced non-finite values on its buffer route"
    )


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_normalizing_gives_the_same_answer_by_either_route(backend_name):
    """``normalizes_on_host`` picks the route; the result cannot depend on it."""
    waist = 2.23e-3
    simu = NLSE(
        alpha=0,
        power=1.05,
        window=4 * waist,
        n2=-1e-9,
        V=None,
        L=1e-3,
        NX=32,
        NY=32,
        Isat=1e5,
        backend=backend_name,
    )
    field = np.exp(-(simu.XX**2 + simu.YY**2) / waist**2).astype(np.complex64)
    out = np.asarray(
        simu._backend.to_numpy(simu._prepare_output_array(field, normalize=True)[0])
    )
    integral = float(
        np.sum(np.abs(out) ** 2) * simu.delta_X * simu.delta_Y * c * epsilon_0 / 2
    )
    assert integral == pytest.approx(simu.power, rel=1e-4)


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_exp_stays_on_the_backend(backend_name):
    """Every backend must exponentiate a complex array where it lives.

    The propagator is exp(theta * dz), and building it on the host costs the
    exponential over the grid plus a transfer -- 9.7 ms at 512x512 against a
    0.16 ms step. The base class falls back to that round trip, so a backend
    that does not override this is correct but slow, and one whose override is
    wrong is neither: the check is against numpy on the same input.
    """
    backend = get_backend(backend_name)
    operator = (-1j * np.linspace(0, 5, 4096).reshape(64, 64)).astype(np.complex64)
    got = np.asarray(backend.to_numpy(backend.exp(backend.from_numpy(operator))))
    assert np.allclose(got, np.exp(operator), atol=1e-6), (
        f"{backend_name}.exp disagrees with numpy"
    )
    assert got.shape == operator.shape


@pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
def test_norm_agrees_with_numpy(backend_name):
    """Reducing on the device must give what reducing on the host gives.

    Only the scalar should cross, but the answer has to be the same one. This
    was missing, and the OpenCL override shipped broken: pyopencl builds its
    reductions from a mako template, mako is not one of its hard dependencies,
    and without it the reduction raises the first time it is asked for --
    which was inside a callback, partway through a propagation.
    """
    backend = get_backend(backend_name)
    field = (
        np.linspace(-1, 1, 4096).reshape(64, 64)
        + 1j * np.linspace(2, -2, 4096).reshape(64, 64)
    ).astype(np.complex64)
    got = backend.norm(backend.from_numpy(field))
    assert got == pytest.approx(float(np.linalg.norm(field)), rel=1e-5), (
        f"{backend_name}.norm disagrees with numpy"
    )


def test_a_backend_without_its_reduction_falls_back(monkeypatch):
    """A missing optional dependency must cost speed, not the run.

    pyopencl builds its reductions from a mako template and does not require
    mako, so on a machine without it every reduction raises the first time one
    is asked for -- inside a callback, partway through a propagation. With
    mako installed the fallback is unreachable, so the failure is forced here:
    otherwise this passes whether the fallback exists or not, which is how the
    override shipped broken in the first place.
    """
    if "CL" not in AVAILABLE_BACKENDS:
        pytest.skip("no OpenCL backend")
    from NLSE.backends import opencl as opencl_backend

    backend = get_backend("CL")
    field = (np.linspace(-1, 1, 1024).reshape(32, 32) + 0.5j).astype(np.complex64)
    on_device = backend.from_numpy(field)

    def no_mako(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'mako'")

    monkeypatch.setattr(opencl_backend.cla, "vdot", no_mako)
    assert backend.norm(on_device) == pytest.approx(
        float(np.linalg.norm(field)), rel=1e-5
    ), "the fallback did not produce the norm"
