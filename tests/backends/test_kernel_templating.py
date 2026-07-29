"""The generated kernel variants must be well-formed, on every dialect.

Each V-reading kernel is written once and compiled three times: with no
potential, with a real one and with a complex one. The generation is plain
text substitution, so it can be checked without a device — which is the point
of testing it here. The CUDA half otherwise has no local coverage at all: it
compiles on a machine with an NVIDIA card and nowhere else, so a mistake in
the template reaches a person rather than a test.
"""

import re
from pathlib import Path

import pytest
from NLSE.kernels import templating

SOURCES = Path(__file__).parent.parent.parent / "NLSE" / "kernels"

# Each dialect, as (template, how it opens a kernel, pointer qualifier).
DIALECTS = {
    "OpenCL": (SOURCES / "cl_source" / "kernels.cl", "__kernel void", "__global "),
    "CUDA": (SOURCES / "cuda_source" / "kernels.cu", "__global__ void", ""),
}

SUFFIXES = ("", templating.REAL_V_SUFFIX, templating.COMPLEX_V_SUFFIX)


def expanded(dialect, fp="float", fp2="float2"):
    """Return a dialect's expanded source with its float types filled in."""
    path, decl, space = DIALECTS[dialect]
    source = templating.expand_v_blocks(path.read_text(), decl, space)
    return source.replace("{{FP_TYPE}}", fp).replace("{{FP2_TYPE}}", fp2)


# The V macros are resolved by the device's C preprocessor, not by the Python
# expansion, so anything that inspects a signature has to resolve them first.
# The definitions are read back out of the generated source rather than
# restated here, so this cannot drift from what the compiler will see.
_DEFINE = re.compile(r"#define (V_ARG|V_PHASE|V_LOSS)\((.*?)\)(.*)")
_UNDEF = re.compile(r"#undef (V_ARG|V_PHASE|V_LOSS)")


def preprocessed(dialect, **kwargs):
    """Return a dialect's source with the V macros applied, as a compiler would."""
    out, macros = [], {}
    for line in expanded(dialect, **kwargs).splitlines():
        if definition := _DEFINE.match(line):
            name, params, body = definition.groups()
            macros[name] = ([p.strip() for p in params.split(",")], body.strip())
            continue
        if _UNDEF.match(line):
            continue
        for name, (params, body) in macros.items():
            for call in re.finditer(rf"\b{name}\((.*?)\)", line):
                args = [a.strip() for a in call.group(1).split(",")]
                text = body
                for param, arg in zip(params, args):
                    text = re.sub(rf"\b{param}\b", arg, text)
                line = line.replace(call.group(0), text)
        out.append(line)
    return "\n".join(out)


def marked_bases(dialect):
    """Return the base name of every VBLOCK-marked kernel in a template."""
    path, decl, _ = DIALECTS[dialect]
    return [
        templating.kernel_names(block, decl)[0]
        for block in templating.VBLOCK.findall(path.read_text())
    ]


def parts(dialect, pattern):
    """Return {kernel name: the text pattern captures} for each kernel."""
    decl = DIALECTS[dialect][1]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            re.escape(decl) + pattern, preprocessed(dialect), re.DOTALL
        )
    }


def signatures(dialect):
    """Return {kernel name: its parameter list}."""
    return parts(dialect, r" (\w+)\((.*?)\)\s*\{")


def bodies(dialect):
    """Return {kernel name: its body}."""
    return parts(dialect, r" (\w+)\((.*?)\n\}")


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_every_marked_kernel_gets_three_twins(dialect):
    """A marked kernel must yield exactly its no-V, real-V and complex-V forms."""
    names = templating.kernel_names(expanded(dialect), DIALECTS[dialect][1])
    bases = marked_bases(dialect)
    assert bases, f"{dialect}: no VBLOCK-marked kernels found at all"

    for base in bases:
        for suffix in SUFFIXES:
            assert base + suffix in names, (
                f"{dialect}: {base} did not produce its {suffix or 'no-potential'} twin"
            )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_generated_names_are_unique(dialect):
    """Two kernels sharing a name would silently shadow one another.

    A base whose name already ends in _v or _cv would collide with another
    base's twin, which the compiler reports only as a redefinition.
    """
    names = templating.kernel_names(expanded(dialect), DIALECTS[dialect][1])
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"{dialect}: kernels declared more than once: {duplicates}"


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_no_potential_twin_takes_no_potential(dialect):
    """Its signature must lose the parameter, not merely stop reading it."""
    sigs = signatures(dialect)
    for base in marked_bases(dialect):
        assert "V" not in sigs[base], (
            f"{dialect}: the no-potential twin of {base} still takes a V "
            f"parameter, so every caller has to pass one anyway:\n{sigs[base]}"
        )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_complex_twin_widens_the_potential(dialect):
    """A complex V must arrive as the two-component type, not the scalar one.

    This is what the absorbing part rides on: read at the real width, the
    imaginary component is not merely ignored, it is misread.
    """
    sigs = signatures(dialect)
    for base in marked_bases(dialect):
        real = sigs[base + templating.REAL_V_SUFFIX]
        complex_ = sigs[base + templating.COMPLEX_V_SUFFIX]
        assert "float* V" in real and "float2* V" not in real, (
            f"{dialect}: the real twin of {base} does not take a real V:\n{real}"
        )
        assert "float2* V" in complex_, (
            f"{dialect}: the complex twin of {base} takes a real V, so the "
            f"gain/loss part would be read at the wrong width:\n{complex_}"
        )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_no_potential_twin_drops_the_potential_terms(dialect):
    """With no potential, nothing of it may survive into the arithmetic.

    An added zero would be correct but not free, which is why V_PHASE and
    V_LOSS are additive terms rather than values.
    """
    for base, body in ((b, bodies(dialect)[b]) for b in marked_bases(dialect)):
        assert not re.search(r"\bV\d?\[", body), (
            f"{dialect}: {base}'s no-potential twin still reads a potential"
        )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_macros_do_not_leak_past_their_block(dialect):
    """Every definition must be matched by an undefine.

    A macro left defined would silently change the next kernel in the file.
    """
    source = expanded(dialect)
    for macro in ("V_ARG", "V_PHASE", "V_LOSS"):
        assert source.count(f"#define {macro}") == source.count(f"#undef {macro}"), (
            f"{dialect}: {macro} is defined more often than it is undefined, so "
            f"it leaks past the block that wanted it"
        )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_templates_use_only_known_placeholders(dialect):
    """A placeholder nobody substitutes reaches the device compiler verbatim.

    It surfaces there as a syntax error with no hint of where it came from, so
    the vocabulary is fixed and checked here instead.
    """
    path = DIALECTS[dialect][0]
    unknown = set(templating.PLACEHOLDER.findall(path.read_text()))
    unknown -= templating.PLACEHOLDERS
    assert not unknown, (
        f"{dialect}: the template uses placeholders no backend fills in: "
        f"{sorted(unknown)}"
    )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_block_markers_are_consumed(dialect):
    """Expansion must leave no marker behind.

    An unbalanced marker would leave one in the emitted source, where it is a
    stray token rather than a comment.
    """
    source = expanded(dialect)
    assert "VBLOCK}}" not in source, (
        f"{dialect}: a VBLOCK marker survived expansion, so its block was "
        f"never matched -- check that every marker is paired"
    )
