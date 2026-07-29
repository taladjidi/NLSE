"""The example scripts must not write into whatever directory they are run from.

They all used to: ``np.save("benchmarks_times.npy", ...)``,
``fig.savefig("benchmarks.svg")``. Run from the repository root, as the README
tells you to, they dropped their output among the sources. The .gitignore had
grown a pattern per file to hide the result, including a repo-wide ``*.npy``
that would have hidden real data as readily as a stray timing array.

They go through ``examples/_output.output_path`` now. This test is what keeps
the next script from reintroducing the bare filename, since nothing else runs
these files.
"""

import ast
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
    """A literal filename here is a file dropped in the user's directory."""
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
    """And it has to be *that* helper, not a local of the same name."""
    source = script.read_text()
    tree = ast.parse(source, filename=str(script))
    if not writes_in_module(tree):
        pytest.skip("writes nothing")
    assert "from _output import output_path" in source, (
        f"{script.name} writes files without importing output_path"
    )


def test_the_helper_stays_out_of_the_working_directory(tmp_path, monkeypatch):
    """The point of the helper: the path does not depend on the cwd."""
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
