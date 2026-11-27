"""The generated kernel variants must be well-formed, on every dialect."""

import re
import struct
from pathlib import Path

import numpy as np
import pytest
from NLSE.kernels import templating

SOURCES = Path(__file__).parent.parent.parent / "NLSE" / "kernels"

# Each dialect, as (template text, how it opens a kernel, pointer qualifier).
DIALECTS = {
    "OpenCL": (
        (SOURCES / "cl_source" / "kernels.cl").read_text(),
        "__kernel void",
        "__global ",
    ),
    "CUDA": (templating.cuda_kernel_template(), "__global__ void", ""),
}

SUFFIXES = ("", templating.REAL_V_SUFFIX, templating.COMPLEX_V_SUFFIX)

# kernels.cu's callers of the shared step, nlse_nonlinear_element.
STEP_KERNELS = ("square_mod_nl_prop", "nl_prop")
# fft_callbacks.cu's, which reach it through OUT_OF_LINE.
STEP_CALLBACKS = (templating.NONLINEAR_CALLBACK,)
STEP = "nlse_nonlinear_element"
OUT_OF_LINE = "nlse_nonlinear_out_of_line"

# CUDA's math functions end in f at float and in nothing at double (CUDA Math
# API), so a double field calling the float form computes at float, silently.
MATH_SUFFIX = {"float": "f", "double": ""}
# The combined sine/cosine at double, which SINCOS_FUNC stands in for.
SINCOS = "sincos"
# A literal suffixed as a float, which a double expression would round through.
FLOAT_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d*)?(?:[eE][-+]?\d+)?[fF]\b")

# struct's format code for each C scalar type (Python docs, struct module).
STRUCT_CODE = {"float": "f", "double": "d"}
# Scalars for the packing tests: distinct powers of two, exact at either width,
# so a field read in another's place shows as a different word.
SCALARS = {"dz": 0.25, "alpha": 0.5, "g": 2.0, "Isat": 8.0}
# A device address, in bytes, with bits set in both halves of its word.
POINTER = 0x7F00DEADBEE0


def expanded(dialect, fp="float", fp2="float2"):
    """Return a dialect's expanded source with its float types filled in."""
    template, decl, space = DIALECTS[dialect]
    source = templating.expand_v_blocks(template, decl, space)
    return source.replace("{{FP_TYPE}}", fp).replace("{{FP2_TYPE}}", fp2)


# The V macros are resolved by the device's C preprocessor, not by the Python
# expansion, so anything that inspects a signature has to resolve them first.
# The definitions are read back out of the generated source rather than
# restated here, so this cannot drift from what the compiler will see.
_DEFINE = re.compile(r"#define (V_ARG|V_PHASE|V_LOSS|V_PASS)\((.*?)\)(.*)")
_UNDEF = re.compile(r"#undef (V_ARG|V_PHASE|V_LOSS|V_PASS)")


def with_v_macros(source):
    """Return expanded source with the V macros applied, as a compiler would."""
    out, macros = [], {}
    for line in source.splitlines():
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


def preprocessed(dialect, **kwargs):
    """Return a dialect's source with the V macros applied."""
    return with_v_macros(expanded(dialect, **kwargs))


def marked_bases(dialect):
    """Return the base name of every VBLOCK-marked kernel in a template."""
    template, decl, _ = DIALECTS[dialect]
    blocks = templating.VBLOCK.findall(template)
    return [
        names[0]
        for names in (templating.kernel_names(b, decl) for b in blocks)
        if names
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
    """Two kernels sharing a name would silently shadow one another."""
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
    """A complex V must arrive as the two-component type, not the scalar one."""
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
    """With no potential, nothing of it may survive into the arithmetic."""
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
    for macro in ("V_ARG", "V_PHASE", "V_LOSS", "V_PASS"):
        assert source.count(f"#define {macro}") == source.count(f"#undef {macro}"), (
            f"{dialect}: {macro} is defined more often than it is undefined, so "
            f"it leaks past the block that wanted it"
        )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_templates_use_only_known_placeholders(dialect):
    """A placeholder nobody substitutes reaches the device compiler verbatim."""
    unknown = set(templating.PLACEHOLDER.findall(DIALECTS[dialect][0]))
    unknown -= templating.PLACEHOLDERS
    assert not unknown, (
        f"{dialect}: the template uses placeholders no backend fills in: "
        f"{sorted(unknown)}"
    )


@pytest.mark.parametrize("dialect", list(DIALECTS))
def test_the_block_markers_are_consumed(dialect):
    """Expansion must leave no marker behind."""
    source = expanded(dialect)
    assert "VBLOCK}}" not in source, (
        f"{dialect}: a VBLOCK marker survived expansion, so its block was "
        f"never matched -- check that every marker is paired"
    )


def element_overloads(source):
    """Return {signature: body} of each nlse_nonlinear_element overload."""
    pattern = r"__device__ inline \w+ nlse_nonlinear_element\((.*?)\)\s*\{(.*?)\n\}"
    return dict(re.findall(pattern, source, re.DOTALL))


def test_the_shared_step_has_one_overload_per_potential():
    """The step's three forms are told apart by V alone, as overloads."""
    overloads = element_overloads(preprocessed("CUDA"))
    assert len(overloads) == len(SUFFIXES), (
        f"expected {len(SUFFIXES)} overloads, found {len(overloads)}"
    )
    none = [s for s in overloads if not re.search(r"\bV\b", s)]
    real = [s for s in overloads if "float* V" in s]
    complex_ = [s for s in overloads if "float2* V" in s]
    assert len(none) == len(real) == len(complex_) == 1, (
        f"the overloads do not take no V, a real V and a complex V once each: "
        f"{list(overloads)}"
    )
    assert not re.search(r"\bV\d?\[", overloads[none[0]]), (
        "the no-potential overload still reads a potential"
    )


def test_the_step_is_written_once():
    """kernels.cu and fft_callbacks.cu call the step rather than restating it."""
    kernels = code_of((SOURCES / "cuda_source" / "kernels.cu").read_text())
    callbacks = code_of(templating.CALLBACK_SOURCE.read_text())
    assert "nlse_loss_factor(" not in kernels, "kernels.cu restates the step"
    assert "nlse_loss_factor(" not in callbacks, "fft_callbacks.cu restates the step"
    assert kernels.count("nlse_nonlinear_element(") == len(STEP_KERNELS), (
        f"{' and '.join(STEP_KERNELS)} should each call the shared step once"
    )
    assert callbacks.count("nlse_nonlinear_element(") == len(STEP_CALLBACKS), (
        "the nonlinear store callback should call the shared step once"
    )


def step_calls(source, decl, callee):
    """Return {function name: the arguments it hands ``callee``}, per caller."""
    calls = {}
    pattern = re.escape(decl) + r" (\w+)\((.*?)\n\}"
    for name, text in re.findall(pattern, source, re.DOTALL):
        call = re.search(callee + r"\((.*?)\);", text, re.DOTALL)
        if call:
            calls[name] = call.group(1)
    return calls


@pytest.mark.parametrize(
    "source,decl,bases,callee",
    [
        (preprocessed("CUDA"), DIALECTS["CUDA"][1], STEP_KERNELS, STEP),
        (
            with_v_macros(templating.store_callbacks_source("complex64")),
            templating.CALLBACK_DECL,
            STEP_CALLBACKS,
            OUT_OF_LINE,
        ),
    ],
    ids=["kernels", "callbacks"],
)
def test_each_twin_hands_the_step_its_potential(source, decl, bases, callee):
    """A twin that takes a potential must pass it on, and one without, none."""
    calls = step_calls(source, decl, callee)
    assert set(calls) == {base + suffix for base in bases for suffix in SUFFIXES}
    for base in bases:
        assert not re.search(r"\bV\b", calls[base]), (
            f"{base} hands the step a potential it does not take: {calls[base]}"
        )
        real = calls[base + templating.REAL_V_SUFFIX]
        assert "(const float*)(" in real, (
            f"the real twin of {base} does not hand the step its potential: {real}"
        )
        complex_ = calls[base + templating.COMPLEX_V_SUFFIX]
        assert "(const float2*)(" in complex_, (
            f"the complex twin of {base} does not hand the step its potential "
            f"as a complex one: {complex_}"
        )


def test_the_out_of_line_step_hands_on_its_potential():
    """Between a callback and the step: each overload passes on the V it takes."""
    source = with_v_macros(templating.store_callbacks_source("complex64"))
    pattern = rf"__device__ __noinline__ \w+ {OUT_OF_LINE}\((.*?)\)\s*\{{(.*?)\n\}}"
    overloads = re.findall(pattern, source, re.DOTALL)
    assert len(overloads) == len(SUFFIXES), f"found {len(overloads)} overloads"
    for params, body in overloads:
        call = re.search(STEP + r"\((.*?)\);", body, re.DOTALL).group(1)
        taken = re.search(r"const (\w+)\* V", params)
        if taken is None:
            assert not re.search(r"\bV\b", call), f"no V taken, one passed: {call}"
        else:
            assert f"(const {taken.group(1)}*)(V)" in call, (
                f"takes a {taken.group(1)} potential and does not pass it on: {call}"
            )


def cuda_source(dtype):
    """Return every CUDA source at a width, kernels and callbacks, sans comments."""
    kernels = templating.expand_v_blocks(
        templating.cuda_kernel_template(), DIALECTS["CUDA"][1]
    )
    return code_of(
        templating.fill_cuda_types(kernels, dtype)
        + templating.store_callbacks_source(dtype)
    )


@pytest.mark.parametrize("dtype", list(templating.CUDA_TYPES))
def test_each_width_calls_its_own_math(dtype):
    """A double field must call no float function, and a float one no double."""
    fp = templating.CUDA_TYPES[dtype]["FP_TYPE"]
    other = next(suffix for name, suffix in MATH_SUFFIX.items() if name != fp)
    templates = (
        templating.cuda_kernel_template() + templating.CALLBACK_SOURCE.read_text()
    )
    functions = {*re.findall(r"(\w+)\{\{FP_SUFFIX\}\}", templates), SINCOS}
    code = cuda_source(dtype)
    for function in sorted(functions):
        assert re.search(rf"\b{function}{MATH_SUFFIX[fp]}\(", code), (
            f"{dtype}: {function} is never called in its {fp} form"
        )
        assert not re.search(rf"\b{function}{other}\(", code), (
            f"{dtype}: {function}{other} computes a {fp} field at the other width"
        )


def test_a_double_field_has_no_float_literal():
    """A float literal in a double expression rounds it through float."""
    dtype = next(
        n for n, t in templating.CUDA_TYPES.items() if t["FP_TYPE"] == "double"
    )
    literal = FLOAT_LITERAL.search(cuda_source(dtype))
    assert literal is None, f"{dtype} source holds the float literal {literal}"


# The cuFFT store callbacks are the same idea one step further out: text
# substituted here, compiled by nvrtc inside cuFFT, and so reachable by no
# local test on a machine without an NVIDIA card. They are not in DIALECTS
# because they declare no kernel -- a callback is a __device__ function.


def code_of(source):
    """Return the source without its comments."""
    return re.sub(r"//[^\n]*", "", source)


# Every symbol cuFFT may be asked to link, from the source.
CALLBACK_SYMBOLS = [
    *templating.STORE_CALLBACKS.values(),
    templating.SEPARABLE_CALLBACK,
    *(templating.NONLINEAR_CALLBACK + suffix for suffix in SUFFIXES),
]


@pytest.mark.parametrize(
    "dtype,element", [("complex64", "float2"), ("complex128", "double2")]
)
def test_the_callbacks_are_generated_at_both_widths(dtype, element):
    """A field's width must reach every callback's element type."""
    code = code_of(templating.store_callbacks_source(dtype))
    assert element in code, f"{dtype} did not substitute {element}"
    other = "double2" if element == "float2" else "float2"
    assert other not in code, (
        f"{dtype} left {other} in the source, so one width's callback carries "
        f"the other's arithmetic"
    )
    for symbol in CALLBACK_SYMBOLS:
        assert f"__device__ void {symbol}(" in code, (
            f"the source does not define {symbol}, which is a name cuFFT is "
            f"asked to link"
        )


def test_the_callback_names_are_distinct():
    """Two callbacks sharing a name would leave one unreachable."""
    assert len(set(CALLBACK_SYMBOLS)) == len(CALLBACK_SYMBOLS), CALLBACK_SYMBOLS
    names = templating.kernel_names(
        templating.store_callbacks_source("complex64"), templating.CALLBACK_DECL
    )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"callbacks declared more than once: {duplicates}"


def test_only_the_batched_callback_wraps_the_propagator():
    """Only the batched store callback takes the propagator index modulo."""
    source = templating.store_callbacks_source("complex64")
    bodies = {
        b.split("(", 1)[0]: b for b in code_of(source).split("__device__ void ")[1:]
    }
    direct = bodies[templating.STORE_CALLBACKS[False]]
    wrapped = bodies[templating.STORE_CALLBACKS[True]]
    assert "%" not in direct, (
        "the direct callback wraps the propagator, which costs a remainder per "
        "element to compute an index it already had"
    )
    assert "%" in wrapped, "the batched callback does not wrap the propagator"


@pytest.mark.parametrize(
    "path", [templating.CALLBACK_SOURCE, templating.NONLINEAR_SOURCE]
)
def test_the_callback_source_uses_only_known_placeholders(path):
    """A placeholder nobody fills in reaches nvrtc verbatim."""
    unknown = set(templating.PLACEHOLDER.findall(path.read_text()))
    unknown -= templating.PLACEHOLDERS
    assert not unknown, (
        f"{path.name} uses placeholders no backend fills in: {sorted(unknown)}"
    )


def nonlinear_struct(dtype):
    """Return NlseNonlinear's fields as rendered at a width, as (type, name)."""
    code = code_of(templating.store_callbacks_source(dtype))
    body = re.search(r"struct NlseNonlinear \{(.*?)\};", code, re.DOTALL).group(1)
    return re.findall(r"(\w+) \*?(\w+);", body)


@pytest.mark.parametrize("dtype", list(templating.CUDA_TYPES))
def test_the_words_are_the_struct_the_callback_reads(dtype):
    """nonlinear_words packs the bytes of the rendered NlseNonlinear struct."""
    pointer, *scalars = nonlinear_struct(dtype)
    fp = templating.CUDA_TYPES[dtype]["FP_TYPE"]
    assert pointer == ("void", "V"), f"NlseNonlinear opens with {pointer}"
    assert {kind for kind, _ in scalars} == {fp}, f"{dtype}: {scalars}"
    expected = struct.pack(
        f"<Q{len(scalars)}{STRUCT_CODE[fp]}",
        POINTER,
        *(SCALARS[name] for _, name in scalars),
    )
    expected += bytes(-len(expected) % templating.WORD_BYTES)
    words = templating.nonlinear_words(POINTER, dtype, **SCALARS)
    assert struct.pack(f"<{len(words)}Q", *words) == expected, (
        f"{dtype}: the words do not hold {[name for _, name in scalars]} in order"
    )


def test_info_words_bounds_the_block_everywhere():
    """The static_assert, _WRITE_INFO and the widest struct agree on its size."""
    words = templating.INFO_WORDS
    code = code_of(templating.store_callbacks_source("complex128"))
    bound = re.search(r"sizeof\(NlseNonlinear\) <= (\d+) \*", code)
    assert bound is not None and int(bound.group(1)) == words, (
        f"the static_assert bounds NlseNonlinear by {bound and bound.group(1)} "
        f"words, not INFO_WORDS = {words}"
    )
    source = templating.WRITE_INFO_SOURCE
    assert re.findall(r"const unsigned long long (w\d+)", source) == [
        f"w{i}" for i in range(words)
    ]
    assert re.findall(r"info\[(\d+)\] = w(\d+);", source) == [
        (str(i), str(i)) for i in range(words)
    ]
    widest = templating.nonlinear_words(POINTER, "complex128", **SCALARS)
    assert len(widest) == words, "INFO_WORDS is not the widest struct's size"


class _DeviceScalar:
    """Stand-in for a 0-d CuPy array: a number to float(), refused by NumPy."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        """Return the number, as CuPy does by copying it to the host."""
        return self.value

    def __array__(self, *args, **kwargs):
        """Refuse, as CuPy does."""
        raise TypeError("Implicit conversion to a NumPy array is not allowed")


def test_a_device_scalar_packs_as_its_value():
    """n2 given as an array reaches the step as a 0-d device array."""
    device = {name: _DeviceScalar(value) for name, value in SCALARS.items()}
    assert templating.nonlinear_words(
        POINTER, "complex64", **device
    ) == templating.nonlinear_words(POINTER, "complex64", **SCALARS)


def test_the_words_take_exactly_the_fields():
    """A scalar missing, or one the struct has no field for, is an error."""
    missing = {name: SCALARS[name] for name in templating.NONLINEAR_FIELDS[1:]}
    with pytest.raises(ValueError, match="NlseNonlinear holds"):
        templating.nonlinear_words(POINTER, "complex64", **missing)


def test_an_unsupported_width_has_no_callback():
    """A real field is not something cuFFT transforms here; say so."""
    with pytest.raises(ValueError, match="float32"):
        templating.store_callbacks_source("float32")


class _Potential:
    """Stand-in for an array: v_suffix reads its dtype and nothing else."""

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)


@pytest.mark.parametrize(
    "V,suffix",
    [
        (None, ""),
        (_Potential("float32"), templating.REAL_V_SUFFIX),
        (_Potential("complex64"), templating.COMPLEX_V_SUFFIX),
    ],
)
def test_v_suffix_names_the_twin_a_potential_needs(V, suffix):
    """The kernels and the callbacks pick their twin by the same rule."""
    assert templating.v_suffix(V) == suffix
