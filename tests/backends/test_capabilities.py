"""Tests that backend capability flags match what the kernels actually provide.

The solvers pick their fused fast paths from the flags declared on the
Backend subclasses. A flag set to True with no matching kernel method is an
AttributeError at propagation time, and a method present with the flag left
False is silently dead code. Both are caught here rather than on hardware.
"""

import inspect

import pytest
from NLSE.backends import Backend, get_backend, list_available_backends

# Capability flag -> kernel methods it promises.
CAPABILITY_METHODS = {
    "has_linear_step": ["linear_step"],
    "has_fused_split_step": ["split_step_fused"],
    "has_fused_rk4_rhs": ["rk4_rhs_fused"],
    "has_fused_rk4_step": ["split_step_rk4_fused"],
    "has_fused_rk4_stage_update": ["rk4_set_and_axpy", "rk4_acc_and_axpy"],
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
def test_unnormalized_ifft_implies_linear_step(backend_name):
    """Folding 1/N into the propagator only makes sense with linear_step.

    _update_propagator_fft builds the pre-normalized propagator whenever
    supports_unnormalized_ifft is set, and only linear_step consumes it.
    """
    backend = get_backend(backend_name)
    if backend.supports_unnormalized_ifft:
        assert backend.has_linear_step, (
            f"{backend_name} sets supports_unnormalized_ifft without "
            f"has_linear_step, so the pre-normalized propagator is unused."
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
        "split_step_fused": [
            "A",
            "propagator",
            "V_scaled",
            "dz",
            "alpha",
            "g",
            "Isat",
            "precision",
            "plan",
            "unnorm_ifft",
        ],
        "rk4_rhs_coupled_fused": [
            "A_in",
            "k",
            "V1",
            "V2",
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
