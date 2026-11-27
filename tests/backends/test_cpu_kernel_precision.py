"""The complex64 CPU kernels compute in float32, vectorized, and round like double."""

import re

import numba
import numpy as np
import pytest
from NLSE.kernels import cpu
from NLSE.solvers.step_size import DEFAULT_PHASE_PER_STEP

# One rounding of a float32, relative to the value: half an ulp just above 1.
UNIT = 2.0**-24
# Points per side. 2**18 points put the statistical error of a mean rounding
# error (about 0.6 UNIT a point) near 0.001 UNIT.
GRID = 512
# Points per side when only the generated code is read, not the result.
IR_GRID = 16
# The value every scalar takes when the IR is read: a Python float, which is
# what the solver passes and the worst case for promotion.
IR_SCALAR = 0.1
# Step (m) and saturation (W/m^2) of the rotation tests: the phase is then
# g * |A|^2, and |A| <= 1 holds it to the largest the solver takes by default.
DELTA_Z = 1.0
I_SAT = np.inf
G = float(np.float32(DEFAULT_PHASE_PER_STEP))
# Smallest |A| sampled, so that an error relative to |A| stays meaningful.
MODULUS_MIN = 0.1
# Largest error of a rotation, in UNIT * |A|. Rounding A once is 1; the phase's
# own rounding adds its relative error times the phase. Measured 1.02.
ROTATION_ERROR_MAX = 1.5
# Largest mean of |out|^2 / |in|^2 - 1 over the grid, in UNIT. Measured 0.003;
# multiplying by a float32 cosine instead gives -0.04.
MODULUS_BIAS_MAX = 0.01
# Rabi angle (rad) of the power test, and the largest mean change of the total
# power it may leave, in UNIT. Measured 0.002; cos and sin rounded to float32
# move it by up to 1.
RABI_ANGLE = 1.11
RABI_POWER_BIAS_MAX = 0.01

# Kernel, then its arguments: c a complex field, r a real one, s a scalar; f and
# g a flat complex and real field, i an index into them.
KERNELS = (
    ("_nl_prop", "crssrss"),
    ("_nl_prop", "crsscss"),
    ("_nl_prop_without_V", "crssss"),
    ("_nl_prop_c", "crrssrssss"),
    ("_nl_prop_without_V_c", "crrssssss"),
    ("_rabi_coupling", "ccss"),
    ("square_mod", "cr"),
    ("_square_mod_nl_prop_span", "fiissss"),
    ("_square_mod_nl_prop_v_span", "fgiissss"),
    ("_square_mod_nl_prop_v_span", "ffiissss"),
    ("_apply_propagator", "cc"),
    ("rk4_axpy", "ccsc"),
    ("rk4_accumulate", "csc"),
    ("_rk4_nl_rhs", "ccrsss"),
    ("_rk4_nl_rhs_v", "ccrrsss"),
    ("_square_mod_rk4_nl_rhs", "ccsss"),
    ("_square_mod_rk4_nl_rhs_v", "ccrsss"),
    ("_rk4_nl_rhs_c", "ccrrsssss"),
    ("_rk4_nl_rhs_c_v", "ccrrrsssss"),
)
# Kernels whose complex64 loop is double arithmetic by design: see the comment
# in _rabi_coupling, and test_rabi_coupling_keeps_the_total_power below.
DOUBLE_ON_PURPOSE = frozenset({"_rabi_coupling"})

LLVM_FUNCTION = re.compile(r"^define [^@]*@\"?([^\"(]+)\"?\(.*?^}", re.M | re.S)
DOUBLE_ARITHMETIC = re.compile(r"= f(?:mul|add|sub|div|neg)[^\n]*double")


def _ir_argument(kind):
    """Return one complex64-width argument of the kind a KERNELS letter names."""
    if kind == "c":
        return np.ones((IR_GRID, IR_GRID), dtype=np.complex64)
    if kind == "r":
        return np.ones((IR_GRID, IR_GRID), dtype=np.float32)
    if kind == "f":
        return np.ones(IR_GRID**2, dtype=np.complex64)
    if kind == "g":
        return np.ones(IR_GRID**2, dtype=np.float32)
    if kind == "i":
        return IR_GRID
    return IR_SCALAR


def _is_loop_body(function, parallel):
    """Whether an LLVM function is a loop body: a parfor's, or a serial kernel's own."""
    if parallel:
        return "parfor_gufunc" in function
    return function.startswith("_ZN4NLSE")  # not its cpython or cfunc wrapper


def _loop_bodies(name, kinds):
    """Return the IR of each loop body of a kernel compiled for complex64.

    A twin is compiled without the cache: code loaded from it carries no IR.
    """
    kernel = getattr(cpu, name)
    options = {k: v for k, v in kernel.targetoptions.items() if k != "nopython"}
    twin = numba.njit(**options)(kernel.py_func)
    twin(*[_ir_argument(kind) for kind in kinds])
    (signature,) = twin.signatures
    functions = LLVM_FUNCTION.finditer(twin.inspect_llvm(signature))
    return [
        match.group(0)
        for match in functions
        if _is_loop_body(match.group(1), "parallel" in options)
        and "numba_gil_ensure" not in match.group(0)
    ]


def _unit_disc_field(seed):
    """Return a complex64 field with MODULUS_MIN <= |A| <= 1 and any phase."""
    rng = np.random.default_rng(seed)
    modulus = rng.uniform(MODULUS_MIN, 1.0, (GRID, GRID))
    direction = rng.standard_normal((GRID, GRID, 2)).view(np.complex128)[..., 0]
    return (modulus * direction / np.abs(direction)).astype(np.complex64)


def _power(field):
    """Return |field|^2 in float64."""
    return np.abs(field.astype(np.complex128)) ** 2


def _rotated_fused(field):
    """Turn a field by its Kerr phase with the kernel that reads |A|^2 itself."""
    return cpu._square_mod_nl_prop(field.copy(), DELTA_Z, 0.0, G, I_SAT)


def _rotated_split(field):
    """Turn a field by its Kerr phase with the kernel that is handed |A|^2."""
    intensity = cpu.square_mod(field, np.empty(field.shape, dtype=np.float32))
    return cpu._nl_prop_without_V(field.copy(), intensity, DELTA_Z, 0.0, G, I_SAT)


@pytest.mark.parametrize(("name", "kinds"), KERNELS)
def test_complex64_kernel_is_float32_and_vectorized(name, kinds):
    """Every loop body is taken by the vectorizer and stays in float32."""
    bodies = _loop_bodies(name, kinds)
    assert bodies, "no parallel loop found: the IR pattern no longer matches"
    for body in bodies:
        assert "vector.body" in body
        if name not in DOUBLE_ON_PURPOSE:
            assert not DOUBLE_ARITHMETIC.search(body)


@pytest.mark.parametrize("rotated", [_rotated_fused, _rotated_split])
def test_lossless_rotation_rounds_once_and_keeps_the_modulus(rotated):
    """A * exp(i g |A|^2 dz) in float32 is the double result, rounded once."""
    field = _unit_disc_field(seed=0)
    out = rotated(field)
    assert out.dtype == np.complex64
    exact = field * np.exp(1j * G * DELTA_Z * _power(field))
    error = np.abs(out - exact) / np.abs(exact) / UNIT
    assert error.max() <= ROTATION_ERROR_MAX
    bias = np.mean(_power(out) / _power(field) - 1) / UNIT
    assert abs(bias) <= MODULUS_BIAS_MAX


def test_rabi_coupling_keeps_the_total_power():
    """The Rabi rotation of two complex64 fields leaves |A1|^2 + |A2|^2 alone."""
    first, second = _unit_disc_field(seed=1), _unit_disc_field(seed=2)
    before = _power(first) + _power(second)
    out_1, out_2 = cpu._rabi_coupling(first, second, DELTA_Z, RABI_ANGLE)
    bias = np.mean((_power(out_1) + _power(out_2)) / before - 1) / UNIT
    assert abs(bias) <= RABI_POWER_BIAS_MAX
