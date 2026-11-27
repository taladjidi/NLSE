"""Example scripts must write through ``examples/_output.output_path``."""

import ast
import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT_DIR / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))

# Calls that put a file on disk at a caller-chosen path.
WRITERS = {"save", "savez", "savez_compressed", "savefig", "to_csv", "write_image"}
# The keywords those calls name their path by, when it is not passed first.
PATH_KEYWORDS = {"fname", "file", "path_or_buf"}


def writes_in_module(tree: ast.Module) -> list[tuple[int, str, ast.expr]]:
    """Return ``(line, func_name, path_argument)`` for each writing call."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        paths = node.args[:1] + [
            k.value for k in node.keywords if k.arg in PATH_KEYWORDS
        ]
        if name in WRITERS and paths:
            found.append((node.lineno, name, paths[0]))
    return found


def test_there_are_scripts_to_check():
    """A glob that matches nothing would make every test below vacuous."""
    assert len(SCRIPTS) > 5, f"only found {[p.name for p in SCRIPTS]}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_output_goes_through_output_path(script):
    """A literal filename is written to the caller's working directory."""
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    offenders = [
        f"{script.name}:{line} {func}({ast.unparse(arg)})"
        for line, func, arg in writes_in_module(tree)
        if isinstance(arg, (ast.Constant, ast.JoinedStr))
    ]
    assert not offenders, "writes a bare filename to the cwd:\n  " + "\n  ".join(
        offenders
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_script_that_writes_imports_the_helper(script):
    """And it must be that helper, not a local of the same name."""
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))
    if not writes_in_module(tree):
        pytest.skip("writes nothing")
    assert "from _output import output_path" in source, (
        f"{script.name} writes files without importing output_path"
    )


def test_the_helper_stays_out_of_the_working_directory(tmp_path, monkeypatch):
    """The resolved path does not depend on the working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import _output

    assert _output.OUTPUT_DIR == EXAMPLES / "output"
    assert not _output.OUTPUT_DIR.is_relative_to(tmp_path)
    monkeypatch.setattr(_output, "OUTPUT_DIR", tmp_path / "output")
    assert _output.output_path("probe.npy") == tmp_path / "output" / "probe.npy"
    assert (tmp_path / "output").is_dir(), "output_path did not create the directory"


# sphinx-gallery builds the gallery from examples/, so only the header can be
# wrong: an example with no reStructuredText title renders unnamed.

# The docstring may be raw: a page whose text carries LaTeX needs it.
TITLE_RULE = re.compile(r'^(?:#![^\n]*\n)?r?"""\n(.+)\n(=+)\n', re.M)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_example_has_a_gallery_title(script):
    """A title and its underline, or the thumbnail comes out unlabelled."""
    if script.name == "_output.py":
        pytest.skip("a helper, excluded from the gallery")
    found = TITLE_RULE.match(script.read_text(encoding="utf-8"))
    assert found, (
        f"{script.name} does not open with a docstring whose first line is a "
        f"title underlined by ===, which is what sphinx-gallery reads as the "
        f"name of the page"
    )
    title, underline = found.group(1), found.group(2)
    assert len(underline) == len(title), (
        f"{script.name}: the underline is {len(underline)} characters against "
        f"a {len(title)}-character title, which reStructuredText rejects"
    )


def test_the_gallery_header_exists():
    """sphinx-gallery needs it, and it introduces the grid."""
    readme = ROOT_DIR / "examples" / "README.rst"
    assert readme.exists(), f"{readme} is missing; the gallery will not build"
