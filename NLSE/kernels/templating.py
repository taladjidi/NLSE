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

import re

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
        "VBLOCK",  # markers, consumed by expand_v_blocks rather than
        "END_VBLOCK",  # substituted
    }
)

_V_UNDEF = "#undef V_ARG\n#undef V_PHASE\n#undef V_LOSS\n"


def _v_variants(address_space: str) -> tuple:
    """Return the three spellings of a potential, as (suffix, macros).

    Absent, real and complex, in that order.

    ``V_ARG`` is the whole parameter, comma included, so it can vanish
    entirely. ``V_PHASE`` and ``V_LOSS`` are additive *terms* rather than
    values, for the same reason: with no potential they leave nothing behind
    rather than an added zero, so each twin is the instruction stream it would
    have been written by hand. ``V_LOSS`` is empty for a real V too — only a
    complex potential has a gain/loss part.

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
            "#define V_ARG(v)\n#define V_PHASE(v, i)\n#define V_LOSS(v, i)\n",
        ),
        (
            REAL_V_SUFFIX,
            f"#define V_ARG(v) {address_space}const {{{{FP_TYPE}}}}* v,\n"
            "#define V_PHASE(v, i) + (v)[i]\n"
            "#define V_LOSS(v, i)\n",
        ),
        (
            COMPLEX_V_SUFFIX,
            f"#define V_ARG(v) {address_space}const {{{{FP2_TYPE}}}}* v,\n"
            "#define V_PHASE(v, i) + (v)[i].x\n"
            "#define V_LOSS(v, i) + (v)[i].y\n",
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
