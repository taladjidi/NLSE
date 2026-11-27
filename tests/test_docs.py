"""The documentation's code examples have to work."""

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
# Notebooks are documentation too. Generated trees (sphinx-gallery, myst-nb,
# autoapi) are skipped: a finding there repeats one in a file nobody edits.
GENERATED = (
    ".ipynb_checkpoints",
    ".cache",
    "auto_examples",
    "_build",
    "jupyter_execute",
)

NOTEBOOKS = [
    p
    for p in sorted(ROOT.rglob("*.ipynb"))
    if not any(part in GENERATED for part in p.parts)
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
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def notebook_sources(path):
    """Return every cell's text from a notebook, code and markdown alike."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
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


def inline_code(path):
    """Return the inline code spans of a markdown file, fenced blocks left out.

    Parameters
    ----------
    path : Path
        Markdown file.

    Returns
    -------
    list of str
        The text between each pair of single backticks.
    """
    prose = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    return re.findall(r"`([^`\n]+)`", prose)


@pytest.mark.parametrize(
    "path", [p for p in DOC_FILES if p != MIGRATION_GUIDE], ids=lambda p: p.name
)
def test_inline_code_uses_current_api(path):
    """Prose quotes code too, and a removed call quoted there misleads as much."""
    stale = [
        f"{span!r}: {why}"
        for span in inline_code(path)
        for pattern, why in REMOVED_API.items()
        if re.search(pattern, span)
    ]
    assert not stale, f"{path.name}: " + "; ".join(stale)


def test_the_migration_guide_shows_the_api_it_migrates_from():
    """The migration guide still shows the removed calls it migrates from."""
    text = MIGRATION_GUIDE.read_text(encoding="utf-8")
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
# Sphinx renders .md and notebooks alike through MyST (under mkdocs the
# tutorial's formulas shipped as raw LaTeX); these hold the two settings.

CONF = ROOT / "docs" / "conf.py"


def test_dollar_maths_is_enabled():
    """``dollarmath`` is what makes $...$ work in .md *and* in the notebook."""
    text = CONF.read_text(encoding="utf-8")
    assert '"dollarmath"' in text, (
        "myst_enable_extensions does not list dollarmath, so $...$ is plain "
        "text again -- on the tutorial and on every markdown page at once"
    )


def test_mathjax_is_served_from_here():
    """MathJax is served from the vendored copy, not from a CDN."""
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
    """A notebook's images have to be files, or the page shows a broken icon."""
    referenced = [
        target
        for source in notebook_sources(path)
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
        if not target.startswith(("http://", "https://", "data:"))
    ]
    missing = [t for t in referenced if not (path.parent / t).exists()]
    assert not missing, f"{path.name} references images that do not exist: {missing}"
