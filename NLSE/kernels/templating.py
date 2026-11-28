"""Expansion of the kernel templates the CUDA and OpenCL backends share.

Both GPU backends keep their kernels as C source with ``{{...}}``
placeholders, and compile several variants of each: one per float width, and
one per kind of potential. The two source files are different dialects but the
variant machinery is the same, so it lives here rather than twice over.

Nothing here imports cupy or pyopencl, and nothing here touches a device: a
template is text until a backend compiles it. That is deliberate — it means
the CUDA variants can be generated and checked on a machine with no CUDA,
which is otherwise the half of this code that no local test can reach.
"""

import enum
import re
import struct
from pathlib import Path

import numpy as np

CALLBACK_SOURCE = Path(__file__).parent / "cuda_source" / "fft_callbacks.cu"
KERNEL_SOURCE = Path(__file__).parent / "cuda_source" / "kernels.cu"
NONLINEAR_SOURCE = Path(__file__).parent / "cuda_source" / "nonlinear_step.cuh"

# How fft_callbacks.cu opens a callback. cuFFT calls a __device__ function,
# so this is what expand_v_blocks renames in each callback twin.
CALLBACK_DECL = "__device__ void"

# Suffixes of the twins that take a potential. The twin that takes none keeps
# the bare name.
REAL_V_SUFFIX = "_v"
COMPLEX_V_SUFFIX = "_cv"

VBLOCK = re.compile(r"// \{\{VBLOCK\}\}\n(.*?)\n// \{\{END_VBLOCK\}\}", re.DOTALL)

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

# The vocabulary a template may draw on. A backend substitutes every one of
# these that its dialect uses, so a placeholder outside this set is one nobody
# will fill in: it reaches the device compiler verbatim, where it is a syntax
# error with no indication of where it came from.
PLACEHOLDERS = frozenset(
    {
        "FP_TYPE",  # the real type: float or double
        "FP2_TYPE",  # its two-component form, used for complex values
        "FP_SUFFIX",  # what a literal of that type is suffixed with
        "SINCOS_FUNC",  # the combined sine/cosine, which CUDA names per width
        "NONLINEAR_FIELDS",  # NlseNonlinear's scalars, from NONLINEAR_FIELDS
        "INFO_WORDS",  # the bound on its size, from INFO_WORDS
        "VBLOCK",  # markers, consumed by expand_v_blocks rather than
        "END_VBLOCK",  # substituted
    }
)

# Rabi coupling a backend passes when a run has none, in rad/m: the coupled
# kernels then add a zero slope and rotate by a zero angle.
NO_RABI_COUPLING = 0.0


class RabiPlacement(enum.IntEnum):
    """Where coupled_nl_prop_c rotates the Rabi coupling (kernels.cu)."""

    NONE = 0  # no coupling
    BEFORE = 1  # rotate, then the nonlinear step: opens a Strang step
    AFTER = 2  # the nonlinear step, then rotate: closes a Lie or Strang step


# Prefix -> the modes a kernel takes as an int, rendered as #defines ahead of
# the source so the device code names a mode as the Python caller does.
SOURCE_MODES: dict[str, type[enum.IntEnum]] = {"RABI": RabiPlacement}

_V_UNDEF = "#undef V_ARG\n#undef V_PHASE\n#undef V_LOSS\n#undef V_PASS\n"


def _v_variants(address_space: str) -> tuple:
    """Return the three spellings of a potential, as (suffix, macros).

    Absent, real and complex, in that order.

    ``V_ARG`` is the whole parameter, comma included, so it can vanish
    entirely. ``V_PHASE`` and ``V_LOSS`` are additive *terms* rather than
    values, for the same reason: with no potential they leave nothing behind
    rather than an added zero, so each twin is the instruction stream it would
    have been written by hand. ``V_LOSS`` is empty for a real V too — only a
    complex potential has a gain/loss part. ``V_PASS`` hands the potential on
    to ``nlse_nonlinear_element``, cast to the twin's type, comma included.

    Parameters
    ----------
    address_space : str
        Qualifier a pointer parameter needs in this dialect, with its trailing
        space: ``"__global "`` for OpenCL, empty for CUDA.

    Returns
    -------
    tuple
        Three ``(suffix, macro_definitions)`` pairs.
    """
    return (
        (
            "",
            "#define V_ARG(v)\n#define V_PHASE(v, i)\n#define V_LOSS(v, i)\n"
            "#define V_PASS(v)\n",
        ),
        (
            REAL_V_SUFFIX,
            f"#define V_ARG(v) {address_space}const {{{{FP_TYPE}}}}* v,\n"
            "#define V_PHASE(v, i) + (v)[i]\n"
            "#define V_LOSS(v, i)\n"
            f"#define V_PASS(v) ({address_space}const {{{{FP_TYPE}}}}*)(v),\n",
        ),
        (
            COMPLEX_V_SUFFIX,
            f"#define V_ARG(v) {address_space}const {{{{FP2_TYPE}}}}* v,\n"
            "#define V_PHASE(v, i) + (v)[i].x\n"
            "#define V_LOSS(v, i) + (v)[i].y\n"
            f"#define V_PASS(v) ({address_space}const {{{{FP2_TYPE}}}}*)(v),\n",
        ),
    )


def expand_v_blocks(source: str, decl: str, address_space: str = "") -> str:
    """Emit a no-V, a real-V and a complex-V twin of each VBLOCK kernel.

    V is a bare pointer, so the three cases cannot share one entry point:
    whether V is there at all changes the signature, and a real and a complex V
    differ in width. Each is compiled separately, so no run pays for a case it
    is not in — no wider load, no extra register, no branch.

    They are generated rather than written out because the three bodies are
    otherwise the same arithmetic three times over, and the copies drift: what
    a potential does to a step is stated once, in one kernel, and the variants
    follow from it.

    Parameters
    ----------
    source : str
        Kernel source containing VBLOCK-marked kernels.
    decl : str
        How this dialect opens a kernel, e.g. ``"__kernel void"``.
    address_space : str
        Qualifier a pointer parameter needs, with its trailing space.

    Returns
    -------
    str
        Source with each marked block replaced by its three twins.
    """
    variants = _v_variants(address_space)
    named = re.compile(r"(" + re.escape(decl) + r" )(\w+)\(")

    def expand(match: re.Match) -> str:
        block = match.group(1)
        return "".join(
            macros + named.sub(r"\1\2" + suffix + "(", block, count=1) + "\n" + _V_UNDEF
            for suffix, macros in variants
        )

    return VBLOCK.sub(expand, source)


def kernel_names(source: str, decl: str) -> list:
    """Return every kernel name a source declares, in order.

    A backend builds its host-side kernel dictionary from this rather than
    listing the names by hand, so a kernel cannot exist on one side and not the
    other, and the generated twins need no second list to keep in step.

    Parameters
    ----------
    source : str
        Kernel source, already expanded.
    decl : str
        How this dialect opens a kernel, e.g. ``"__kernel void"``.

    Returns
    -------
    list
        The declared kernel names.
    """
    return re.findall(re.escape(decl) + r" (\w+)\(", source)


# What the two propagator store callbacks in cuda_source/fft_callbacks.cu are
# called, by whether the propagator is shared across a batch of fields.
STORE_CALLBACKS = {
    False: "nlse_store_propagator",
    True: "nlse_store_propagator_batched",
}

# What each placeholder becomes at each field width. The complex element is
# spelled as a CUDA vector type rather than cufftComplex or cufftDoubleComplex,
# which are the same types under names nvrtc cannot reach: it has no cuFFT
# headers.
CUDA_TYPES = {
    "complex64": {
        "FP_TYPE": "float",
        "FP2_TYPE": "float2",
        "FP_SUFFIX": "f",
        "SINCOS_FUNC": "sincosf",
    },
    "complex128": {
        "FP_TYPE": "double",
        "FP2_TYPE": "double2",
        "FP_SUFFIX": "",
        "SINCOS_FUNC": "sincos",
    },
}

# The base name of the nonlinear store callback in fft_callbacks.cu. Its twins
# carry the suffix ``v_suffix`` gives the potential.
NONLINEAR_CALLBACK = "nlse_store_nonlinear"

# The store callback that builds the propagator from one factor per axis.
SEPARABLE_CALLBACK = "nlse_store_propagator_separable"

# NlseNonlinear's scalars after the potential's address, in order. The struct
# in fft_callbacks.cu is rendered from this and nonlinear_words packs from it,
# so the device and the host cannot read the block in different orders.
NONLINEAR_FIELDS = ("dz", "alpha", "g", "Isat")

# Bytes in one word of a store callback's block: a uint64, as _WRITE_INFO in
# backends/cupy_backend.py writes it.
WORD_BYTES = struct.calcsize("<Q")

# Words in that block. The largest it holds is NlseNonlinear at complex128:
# the potential's address, then one double per field, a word each.
INFO_WORDS = 1 + len(NONLINEAR_FIELDS)

# What fft_callbacks.cu's layout placeholders become, at either width.
CALLBACK_LAYOUT = {
    "NONLINEAR_FIELDS": "\n    ".join(
        f"{{{{FP_TYPE}}}} {name};" for name in NONLINEAR_FIELDS
    ),
    "INFO_WORDS": str(INFO_WORDS),
}

# The one-thread kernel that rewrites a block, one parameter per word: words
# go by value, so that a CUDA graph records them with the launch.
WRITE_INFO_KERNEL = "nlse_write_info"
WRITE_INFO_SOURCE = (
    f'extern "C" __global__ void {WRITE_INFO_KERNEL}(unsigned long long* info, '
    + ", ".join(f"const unsigned long long w{i}" for i in range(INFO_WORDS))
    + ") {\n"
    + "".join(f"    info[{i}] = w{i};\n" for i in range(INFO_WORDS))
    + "}\n"
)


def fill_cuda_types(source: str, dtype) -> str:
    """Substitute the float types of a field's width into CUDA source.

    Parameters
    ----------
    source : str
        CUDA source with ``{{FP_TYPE}}``-style placeholders.
    dtype : npt.DTypeLike
        Complex type of the field.

    Returns
    -------
    str
        The source at that width.

    Raises
    ------
    ValueError
        If the field is of a width no CUDA source is written for.

    """
    name = np.dtype(dtype).name
    if name not in CUDA_TYPES:
        raise ValueError(
            f"no CUDA source for a {name} field: the kernels and cuFFT take "
            f"complex64 and complex128"
        )
    for placeholder, value in CUDA_TYPES[name].items():
        source = source.replace("{{" + placeholder + "}}", value)
    return source


def mode_defines() -> str:
    """Return a ``#define`` per mode in ``SOURCE_MODES``.

    Returns
    -------
    str
        One line per mode, ``#define RABI_BEFORE 1`` and so on.

    """
    return "".join(
        f"#define {prefix}_{mode.name} {int(mode)}\n"
        for prefix, modes in SOURCE_MODES.items()
        for mode in modes
    )


def cuda_kernel_template() -> str:
    """Return the CUDA kernel template, with the step it shares prepended.

    Returns
    -------
    str
        The mode #defines, nonlinear_step.cuh, then kernels.cu, placeholders
        unfilled.

    """
    return mode_defines() + NONLINEAR_SOURCE.read_text() + KERNEL_SOURCE.read_text()


def store_callbacks_source(dtype) -> str:
    """Return the cuFFT store callbacks, at a field's width.

    Compiled by cuFFT and not by any backend: the source goes to
    ``cupy.fft.config.set_cufft_callbacks``, which hands it to nvrtc and links
    the callback it is asked for into an FFT plan. Built here so it can be
    checked on a machine with no CUDA.

    Parameters
    ----------
    dtype : npt.DTypeLike
        Complex type of the field, which fixes the callbacks' element type.

    Returns
    -------
    str
        The CUDA C defining every callback in ``STORE_CALLBACKS``,
        ``SEPARABLE_CALLBACK`` and the three twins of ``NONLINEAR_CALLBACK``.

    """
    source = NONLINEAR_SOURCE.read_text() + CALLBACK_SOURCE.read_text()
    for placeholder, value in CALLBACK_LAYOUT.items():
        source = source.replace("{{" + placeholder + "}}", value)
    return fill_cuda_types(expand_v_blocks(source, CALLBACK_DECL), dtype)


def nonlinear_words(pointer: int, dtype, **scalars) -> tuple:
    """Pack NlseNonlinear into the words of a store callback's block.

    Parameters
    ----------
    pointer : int
        Device address of the potential, 0 for none.
    dtype : npt.DTypeLike
        Complex type of the field, whose real width the scalars are stored at.
    **scalars : float
        One value per name in ``NONLINEAR_FIELDS``. Read with ``float``, so a
        0-d device array is taken as the number it holds.

    Returns
    -------
    tuple
        The block's words as unsigned integers, the last zero-padded: three at
        complex64, ``INFO_WORDS`` at complex128.

    Raises
    ------
    ValueError
        If the scalars are not exactly the ones ``NONLINEAR_FIELDS`` names.

    """
    if set(scalars) != set(NONLINEAR_FIELDS):
        raise ValueError(
            f"NlseNonlinear holds {', '.join(NONLINEAR_FIELDS)}, not "
            f"{', '.join(sorted(scalars))}"
        )
    code = np.finfo(dtype).dtype.char
    # Little-endian, as every CUDA device is.
    packed = struct.pack(
        f"<Q{len(NONLINEAR_FIELDS)}{code}",
        pointer,
        *(float(scalars[name]) for name in NONLINEAR_FIELDS),
    )
    packed += bytes(-len(packed) % WORD_BYTES)
    return struct.unpack(f"<{len(packed) // WORD_BYTES}Q", packed)


def v_suffix(V) -> str:
    """Return the suffix of the twin that takes this potential.

    Parameters
    ----------
    V : array or None
        The potential.

    Returns
    -------
    str
        Empty with no potential, ``REAL_V_SUFFIX`` for a real one and
        ``COMPLEX_V_SUFFIX`` for a complex one.

    """
    if V is None:
        return ""
    if V.dtype.kind == "c":
        return COMPLEX_V_SUFFIX
    return REAL_V_SUFFIX
