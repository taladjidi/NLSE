"""The CUDA nonlinear step and store callbacks, compiled for the host.

No CI machine has a CUDA device, so the source the device compiles is built
here as C++ behind a few definitions and held to the CPU kernels.
"""

import ctypes
import shutil
import subprocess
import sys

import numpy as np
import pytest
from NLSE.kernels import cpu as cpu_kernels, templating

COMPILER = shutil.which("c++")

pytestmark = pytest.mark.skipif(
    COMPILER is None or sys.platform == "win32",
    reason="needs a C++ compiler that builds a shared library ctypes can load",
)

# What CUDA defines and a host compiler does not. The headers come first:
# libstdc++ uses __noinline__ as an attribute name.
CUDA_ON_HOST = """
#include <cmath>
#define __device__
#define __noinline__
struct float2 { float x, y; };
struct double2 { double x, y; };
static inline float2 make_float2(float x, float y) { return {x, y}; }
static inline double2 make_double2(double x, double y) { return {x, y}; }
static inline void host_sincosf(float a, float *s, float *c) {
    *s = sinf(a);
    *c = cosf(a);
}
static inline void host_sincos(double a, double *s, double *c) {
    *s = sin(a);
    *c = cos(a);
}
#define sincosf host_sincosf
#define sincos host_sincos
"""

# Calls one store callback over a whole array, as cuFFT does per element.
DRIVER = """
extern "C" void drive_{name}(void *out, unsigned long long n, void *info,
                             const {{{{FP2_TYPE}}}} *in) {{
    for (unsigned long long i = 0; i < n; ++i) {name}(out, i, in[i], info, 0);
}}
"""
SIZEOF_NONLINEAR = """
extern "C" unsigned long long sizeof_nonlinear() { return sizeof(NlseNonlinear); }
"""

NONLINEAR_TWINS = {
    kind: templating.NONLINEAR_CALLBACK + suffix
    for kind, suffix in (
        ("none", ""),
        ("real", templating.REAL_V_SUFFIX),
        ("complex", templating.COMPLEX_V_SUFFIX),
    )
}
CALLBACKS = (
    *templating.STORE_CALLBACKS.values(),
    templating.SEPARABLE_CALLBACK,
    *NONLINEAR_TWINS.values(),
)

NY, NX = 12, 20  # points; unequal, so a swapped row and column index shows
SEED = 0

# (dz, alpha, g, Isat): no loss, loss under and over the limit that picks how
# it is solved (2 * alpha * dz against _LOSS_SOLVED_LIMIT), gain, a backward
# step, and each of them with and without saturation.
STEPS = [
    (0.3, 0.0, 0.5, np.inf),
    (0.3, 0.05, 0.5, np.inf),
    (0.3, 0.05, 0.5, 2.0),
    (0.3, -0.05, 0.5, 2.0),
    (0.3, 1.0, 0.5, np.inf),
    (0.3, 1.0, 0.5, 2.0),
    (-0.3, 0.05, 0.5, 2.0),
    (-0.3, 1.0, 0.5, 2.0),
]

# Relative to the largest output, in units of the width's epsilon. Measured
# worst over every case here: 1.6 (Apple clang 21, arm64). A pass less in the
# loss, the smallest error in the step looked for, is 1.0e3 or more in every
# lossy case at complex128.
STEP_EPS = 50
# One or two complex products, against numpy's: measured 0.75.
PRODUCT_EPS = 8


@pytest.fixture(scope="module", params=["complex64", "complex128"])
def host(request, tmp_path_factory):
    """Build the callbacks at one width and return the library and the width."""
    dtype = np.dtype(request.param)
    drivers = "".join(DRIVER.format(name=name) for name in CALLBACKS)
    folder = tmp_path_factory.mktemp(f"cuda_{dtype.name}")
    source = folder / "callbacks.cpp"
    source.write_text(
        CUDA_ON_HOST
        + templating.store_callbacks_source(dtype)
        + templating.fill_cuda_types(drivers, dtype)
        + SIZEOF_NONLINEAR
    )
    library = folder / "callbacks.so"
    subprocess.run(
        [COMPILER, "-std=c++17", "-O1", "-shared", "-fPIC", "-o", library, source],
        check=True,
        capture_output=True,
        text=True,
    )
    return ctypes.CDLL(str(library)), dtype


def drive(library, name, info, field):
    """Run one store callback over a field and return what it stored."""
    out = np.zeros_like(field)
    function = getattr(library, f"drive_{name}")
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    function.restype = None
    function(out.ctypes.data, field.size, ctypes.addressof(info), field.ctypes.data)
    return out


def block(*words):
    """Return a callback's block of words, as the backend's kernel writes it."""
    return (ctypes.c_uint64 * templating.INFO_WORDS)(*words)


def random_complex(rng, shape, dtype):
    """Return a complex array of unit-variance parts."""
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)


def worst_eps(got, want):
    """Return the largest error over the largest value, in epsilons."""
    return np.abs(got - want).max() / np.abs(want).max() / np.finfo(got.dtype).eps


def potential(rng, kind, dtype):
    """Return a potential of one kind for a field of one width, or None."""
    real = np.finfo(dtype).dtype
    if kind == "none":
        return None
    if kind == "real":
        return (3 * rng.normal(size=(NY, NX))).astype(real)
    return (3 * rng.normal(size=(NY, NX)) + 0.3j * rng.normal(size=(NY, NX))).astype(
        dtype
    )


def test_the_block_is_the_size_of_the_struct(host):
    """The words the host packs fill the struct the device reads, exactly."""
    library, dtype = host
    library.sizeof_nonlinear.restype = ctypes.c_ulonglong
    words = templating.nonlinear_words(0, dtype, dz=0.0, alpha=0.0, g=0.0, Isat=1.0)
    assert library.sizeof_nonlinear() == templating.WORD_BYTES * len(words)


@pytest.mark.parametrize("kind", list(NONLINEAR_TWINS))
@pytest.mark.parametrize(("dz", "alpha", "g", "Isat"), STEPS)
def test_the_nonlinear_callback_is_the_cpu_step(host, kind, dz, alpha, g, Isat):
    """Each twin of the CUDA step gives what the CPU kernel gives."""
    library, dtype = host
    rng = np.random.default_rng(SEED)
    field = 1.5 * random_complex(rng, (NY, NX), dtype)
    V = potential(rng, kind, dtype)
    words = templating.nonlinear_words(
        0 if V is None else V.ctypes.data, dtype, dz=dz, alpha=alpha, g=g, Isat=Isat
    )
    got = drive(library, NONLINEAR_TWINS[kind], block(*words), field)
    want = field.copy()
    if V is None:
        cpu_kernels.square_mod_nl_prop(want, dz, alpha, g, Isat)
    else:
        cpu_kernels.square_mod_nl_prop_v(want, V, dz, alpha, g, Isat)
    assert worst_eps(got, want) < STEP_EPS


@pytest.mark.parametrize("batch", [1, 3])
def test_the_propagator_callbacks_multiply_by_the_propagator(host, batch):
    """The plain callback indexes the field's shape, the batched one a member's."""
    library, dtype = host
    rng = np.random.default_rng(SEED)
    field = random_complex(rng, (batch, NY, NX), dtype)
    propagator = random_complex(rng, (NY, NX), dtype)
    info = block(propagator.ctypes.data, propagator.size)
    got = drive(library, templating.STORE_CALLBACKS[batch > 1], info, field)
    assert worst_eps(got, field * propagator) < PRODUCT_EPS


@pytest.mark.parametrize("batch", [1, 3])
def test_the_separable_callback_is_the_outer_product(host, batch):
    """Rows run over the batch and the first axis, columns over the last."""
    library, dtype = host
    rng = np.random.default_rng(SEED)
    field = random_complex(rng, (batch, NY, NX), dtype)
    rows = random_complex(rng, (batch, NY), dtype)
    columns = random_complex(rng, NX, dtype)
    info = block(rows.ctypes.data, columns.ctypes.data, NX)
    got = drive(library, templating.SEPARABLE_CALLBACK, info, field)
    assert worst_eps(got, field * rows[:, :, None] * columns) < PRODUCT_EPS
