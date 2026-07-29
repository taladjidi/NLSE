"""The documentation's code examples have to work.

Every python block in the README and in mkdocs-documentation/ was broken or
stale at some point: ``backend="GPU"``, which is not a backend name and raises;
float64 fields where ``out_field`` asserts complex; ``simu.delta_z`` after the
step became an argument. None of it was caught, because nothing ran the docs.

Two checks, because the blocks are not all the same kind. One a reader could
copy and run is executed. One that illustrates an API shape, and leans on names
defined around it, is only read for API that no longer exists.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    *sorted((ROOT / "mkdocs-documentation" / "docs").glob("*.md")),
    ROOT / "README.md",
]

SOLVER_CALLS = (
    "NLSE(",
    "CNLSE(",
    "GPE(",
    "NLSE_1d(",
    "CNLSE_1d(",
    "NLSE_3d(",
    "DDGPE(",
)

# API that no longer exists. A doc mentioning any of it is telling a reader to
# write something that will not run.
REMOVED_API = {
    r"\.delta_z\s*=(?!=)": "assigns delta_z, which is an out_field argument now",
    r"\.delta_t\s*=(?!=)": "assigns delta_t, which no longer exists",
    r"\w+\.delta_z(?!\s*=)": "reads delta_z, which is not an attribute",
    r"""backend\s*=\s*["']GPU["']""": 'backend="GPU" is not a backend name',
}

# Shrunk so the suite stays quick; the docs quote sizes worth running for real.
SMALLER = [
    ("N = 2048", "N = 64"),
    ("N = 1024", "N = 64"),
    ("N = 4096", "N = 64"),
    ("NX = 1024", "NX = 64"),
    ("NY = 1024", "NY = 64"),
    ("NX=512", "NX=64"),
    ("NY=512", "NY=64"),
    ("(2, 512, 512)", "(2, 64, 64)"),
    ("plot=True", "plot=False"),
    ("verbose=True", "verbose=False"),
]


def blocks(path):
    """Return the python code blocks in a markdown file."""
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.S)


def self_contained(block):
    """Whether a block builds its own solver and propagates it."""
    return "out_field" in block and any(s in block for s in SOLVER_CALLS)


def _cases(predicate):
    out = []
    for path in DOC_FILES:
        for i, block in enumerate(blocks(path)):
            if predicate(block):
                out.append(pytest.param(block, id=f"{path.name}:{i}"))
    return out


def test_the_docs_were_found():
    """Guard against the checks below passing because they looked nowhere."""
    assert len(DOC_FILES) > 5, f"only found {DOC_FILES}"
    assert sum(len(blocks(p)) for p in DOC_FILES) > 20, "no code blocks found"


@pytest.mark.parametrize("block", _cases(lambda b: True))
def test_blocks_parse(block):
    """Every block must at least be Python."""
    ast.parse(block)


@pytest.mark.parametrize("block", _cases(lambda b: True))
def test_blocks_use_current_api(block):
    """No block may reference API that has been removed."""
    for pattern, why in REMOVED_API.items():
        found = re.search(pattern, block)
        assert found is None, f"{found.group(0).strip()!r}: {why}"


@pytest.mark.parametrize("block", _cases(self_contained))
def test_self_contained_examples_run(block):
    """A block a reader could copy and run has to run."""
    import matplotlib

    matplotlib.use("Agg")
    for old, new in SMALLER:
        block = block.replace(old, new)
    exec(compile(block, "<doc example>", "exec"), {})
