"""Example scripts must write through ``examples/_output.output_path``.

A bare filename lands in whatever directory the script was run from, which
is what these check. Whether the scripts still *run* is the Examples job in
CI, which executes every one of them bar the two benchmark sweeps.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT_DIR / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*.py"))

# Calls that put a file on disk at a caller-chosen path.
WRITERS = {"save", "savez", "savez_compressed", "savefig", "to_csv", "write_image"}


def writes_in_module(tree: ast.Module) -> list[tuple[int, str, ast.expr]]:
    """Return ``(line, func_name, path_argument)`` for each writing call."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name in WRITERS:
            found.append((node.lineno, name, node.args[0]))
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
    from _output import output_path

    target = output_path("probe.npy")
    try:
        assert target.parent == EXAMPLES / "output"
        assert not target.is_relative_to(tmp_path)
        assert target.parent.is_dir(), "output_path did not create the directory"
    finally:
        if not any((EXAMPLES / "output").iterdir()):
            (EXAMPLES / "output").rmdir()


# sphinx-gallery builds the gallery from examples/ itself, so a script cannot
# be missing from it and a link cannot rot: both were real faults under the
# hand-written page this replaced, which listed a script that had been renamed
# and left four others out. What it can still get wrong is the header, because
# an example with no reStructuredText title renders in the grid unnamed.

TITLE_RULE = re.compile(r'^(?:#![^\n]*\n)?"""\n(.+)\n(=+)\n', re.M)


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
