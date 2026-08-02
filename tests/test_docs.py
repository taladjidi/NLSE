"""The documentation's code examples have to work.

Two checks, because the blocks are not all the same kind. A block a reader
could copy and run is executed. A block that illustrates an API shape, and
leans on names defined around it, is only scanned for API that no longer
exists.
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "README.md",
]
# Notebooks are documentation too, and the tutorial is a page in the nav.
NOTEBOOKS = [
    p
    for p in sorted(ROOT.rglob("*.ipynb"))
    if ".ipynb_checkpoints" not in str(p) and ".cache" not in str(p)
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
    r"\bpuiss\b": "puiss was renamed to power",
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


def notebook_sources(path):
    """Return every cell's text from a notebook, code and markdown alike.

    Markdown cells included: the tutorial quotes source in prose, and a quoted
    signature that no longer exists misleads exactly as much as a code cell
    that no longer runs.
    """
    cells = json.loads(path.read_text()).get("cells", [])
    out = []
    for cell in cells:
        src = cell.get("source")
        out.append("".join(src) if isinstance(src, list) else (src or ""))
    return out


def self_contained(block):
    """Whether a block builds its own solver and propagates it."""
    return "out_field" in block and any(s in block for s in SOLVER_CALLS)


# The one page whose job is to show the API that was removed. Every other
# page mentioning it is out of date; this one is out of date the day it stops.
MIGRATION_GUIDE = ROOT / "docs" / "migration.md"


def _cases(predicate, skip=()):
    out = []
    for path in DOC_FILES:
        if path in skip:
            continue
        for i, block in enumerate(blocks(path)):
            if predicate(block):
                out.append(pytest.param(block, id=f"{path.name}:{i}"))
    return out


def test_the_docs_were_found():
    """Guard against the checks below passing because they looked nowhere."""
    assert len(DOC_FILES) > 5, f"only found {DOC_FILES}"
    assert sum(len(blocks(p)) for p in DOC_FILES) > 20, "no code blocks found"
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_notebooks_use_current_api(path):
    """No notebook cell may reference API that has been removed."""
    for i, text in enumerate(notebook_sources(path)):
        for pattern, why in REMOVED_API.items():
            found = re.search(pattern, text)
            assert found is None, (
                f"{path.name} cell {i}: {found.group(0).strip()!r}: {why}"
            )


@pytest.mark.parametrize("block", _cases(lambda b: True))
def test_blocks_parse(block):
    """Every block must at least be Python."""
    ast.parse(block)


@pytest.mark.parametrize("block", _cases(lambda b: True, skip=(MIGRATION_GUIDE,)))
def test_blocks_use_current_api(block):
    """No block may reference API that has been removed."""
    for pattern, why in REMOVED_API.items():
        found = re.search(pattern, block)
        assert found is None, f"{found.group(0).strip()!r}: {why}"


def test_the_migration_guide_shows_the_api_it_migrates_from():
    """The exemption above has to be earned, or it is just a hole.

    migration.md is skipped by the check because showing the removed call
    beside its replacement is the whole point of the page. That is only
    defensible while it actually shows them, so this asserts the same patterns
    the other pages are forbidden.
    """
    text = MIGRATION_GUIDE.read_text()
    unmentioned = [
        why for pattern, why in REMOVED_API.items() if re.search(pattern, text) is None
    ]
    assert not unmentioned, (
        f"{MIGRATION_GUIDE.name} does not show, and so does not help anyone "
        f"past: {unmentioned}"
    )


@pytest.mark.parametrize("block", _cases(self_contained))
def test_self_contained_examples_run(block):
    """A block a reader could copy and run has to run."""
    import matplotlib

    matplotlib.use("Agg")
    for old, new in SMALLER:
        block = block.replace(old, new)
    exec(compile(block, "<doc example>", "exec"), {})


# ── Maths, and where MathJax comes from ─────────────────────────────────────
#
# Under mkdocs this needed real machinery. Two renderers fed the site and
# handed MathJax different things: .md went through pymdownx.arithmatex, the
# tutorial through nbconvert, and the configuration served only the first, so
# every formula in the tutorial was published as its own LaTeX source for as
# long as the page existed. A first fix changed nothing while a test said it
# had, because the failure was in how the classes nested.
#
# Sphinx renders both through MyST, so the delimiters and the markup are the
# same on every page and there is no class to get wrong. What is left to hold
# is the pair of settings that keeps it that way.

CONF = ROOT / "docs" / "conf.py"


def test_dollar_maths_is_enabled():
    """``dollarmath`` is what makes $...$ work in .md *and* in the notebook."""
    text = CONF.read_text(encoding="utf-8")
    assert '"dollarmath"' in text, (
        "myst_enable_extensions does not list dollarmath, so $...$ is plain "
        "text again -- on the tutorial and on every markdown page at once"
    )


def test_mathjax_is_served_from_here():
    """Not from a CDN, which is the whole reason it is vendored.

    Sphinx loads MathJax from jsdelivr unless told otherwise. A script tag
    pointed at someone else's host runs whatever that host serves in the
    browser of every reader; the entry this replaced was polyfill.io, which
    was sold in 2024 and began serving malware.
    """
    text = CONF.read_text(encoding="utf-8")
    assert 'mathjax_path = "mathjax/' in text, (
        "mathjax_path does not point at the vendored copy, so Sphinx will "
        "fetch MathJax from a CDN"
    )
    bundle = ROOT / "docs" / "_static" / "mathjax" / "tex-mml-chtml.js"
    assert bundle.exists(), f"{bundle} is missing, so that path 404s"


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_notebook_images_exist(path):
    """A notebook's images have to be files, or the page shows a broken icon.

    ``mkdocs build --strict`` fails on a broken *link* and says nothing about
    an ``<img>`` inside rendered notebook HTML, so the tutorial asked for
    assets/equations.png and assets/tur.png -- neither of which has ever
    existed in this repository -- and every build passed. The 404s only showed
    up in the server log of someone who happened to be looking.
    """
    referenced = [
        target
        for source in notebook_sources(path)
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
        if not target.startswith(("http://", "https://", "data:"))
    ]
    missing = [t for t in referenced if not (path.parent / t).exists()]
    assert not missing, f"{path.name} references images that do not exist: {missing}"
