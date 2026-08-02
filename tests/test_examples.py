"""Example scripts must write through ``examples/_output.output_path``.

A bare filename lands in whatever directory the script was run from, which
is what these check. Whether the scripts still *run* is the Examples job in
CI, which executes every one of them bar the two benchmark sweeps.
"""

import ast
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
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
    tree = ast.parse(script.read_text(), filename=str(script))
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
    source = script.read_text()
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


GALLERY = Path(__file__).resolve().parent.parent / "docs" / "examples.md"


def gallery_links() -> list[str]:
    """Return the repository paths the gallery links to."""
    return sorted(
        set(
            re.findall(
                r"/tree/main/((?:examples|docs)/\S+?\.(?:py|ipynb))",
                GALLERY.read_text(),
            )
        )
    )


def test_the_gallery_links_files_that_exist():
    """A dead link in the gallery is a reader following it to a 404.

    It listed ``examples/vortex_precession.py`` after the script was split in
    two and renamed, and pointed at the tutorial notebook in ``examples/``
    after it moved in with the docs. Nothing noticed either.
    """
    root = Path(__file__).resolve().parent.parent
    links = gallery_links()
    assert len(links) > 5, f"only found {links}; the gallery format changed"
    missing = [link for link in links if not (root / link).exists()]
    assert not missing, f"{GALLERY.name} links files that do not exist: {missing}"


def test_every_example_is_in_the_gallery():
    """A script nobody links is a script nobody runs."""
    listed = {Path(link).name for link in gallery_links()}
    unlisted = sorted(
        p.name for p in SCRIPTS if p.name != "_output.py" and p.name not in listed
    )
    assert not unlisted, (
        f"{unlisted} are in examples/ and not in {GALLERY.name}, so the gallery "
        f"is no longer a map of what is there"
    )
