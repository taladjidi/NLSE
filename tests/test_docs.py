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
    *sorted((ROOT / "mkdocs-documentation" / "docs").glob("*.md")),
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
MIGRATION_GUIDE = ROOT / "mkdocs-documentation" / "docs" / "migration.md"


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


# ── Maths has to reach MathJax, not just reach the page ──────────────────────
#
# Two renderers feed this site and they hand MathJax different things.
#
# A .md page goes through pymdownx.arithmatex, which rewrites $...$ into
# \(...\) and wraps it in class="arithmatex". The notebook does not: it is
# rendered by mkdocs-jupyter through nbconvert, which is not the mkdocs
# markdown pipeline, so its maths arrives as literal $...$ inside
# class="jp-MarkdownCell".
#
# The configuration served only the first of those, in both respects at once
# -- it processed the arithmatex class alone, and declared only the \(...\)
# delimiters -- so every formula in the tutorial was published as its own
# source: six display and twenty-two inline, for as long as the page existed.
# Nothing failed, nothing warned, and the release went out with it.
#
# So this pins the join: for each renderer the docs actually use, the config
# has to process the class that renderer emits and accept the delimiters it
# leaves behind. It reads the sources rather than the built site, so it runs
# without mkdocs installed; the docs CI job checks the built HTML as well.

MATHJAX_JS = ROOT / "mkdocs-documentation" / "docs" / "javascripts" / "mathjax.js"


def mathjax_config() -> tuple[str, str, str]:
    """Return (ignoreHtmlClass, inlineMath source, displayMath source)."""
    text = MATHJAX_JS.read_text(encoding="utf-8")
    options = re.search(r"options:\s*\{(.*?)\}", text, re.S)
    inline = re.search(r"inlineMath:\s*\[(.*?)\],\s*\n\s*displayMath", text, re.S)
    display = re.search(r"displayMath:\s*\[(.*?)\],\s*\n\s*processEscapes", text, re.S)
    assert options and inline and display, (
        f"could not read the MathJax configuration out of {MATHJAX_JS.name}; "
        f"if its shape changed, this test has to change with it rather than "
        f"quietly stop checking"
    )
    ignore = re.search(r'ignoreHtmlClass:\s*"([^"]*)"', options.group(1))
    assert ignore, f"no ignoreHtmlClass in {MATHJAX_JS.name}"
    return ignore.group(1), inline.group(1), display.group(1)


def notebook_math(path: Path) -> tuple[int, int]:
    """Return (display, inline) maths counts in a notebook's markdown cells."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    markdown = ["".join(c["source"]) for c in cells if c["cell_type"] == "markdown"]
    display = sum(len(re.findall(r"\$\$.+?\$\$", text, re.S)) for text in markdown)
    inline = sum(
        len(re.findall(r"\$[^$\n]+?\$", re.sub(r"\$\$.+?\$\$", " ", text, flags=re.S)))
        for text in markdown
    )
    return display, inline


def test_the_ignore_pattern_is_not_a_catch_all():
    """``ignoreHtmlClass`` must not match every element, or nesting kills it.

    MathJax re-evaluates the flag at each element::

        ignore = (ignore || ignoreHtmlClass.test(class)) && !processHtmlClass.test(class)

    so a pattern matching everything -- ``".*|"``, the recipe for a site whose
    maths is only ever arithmatex spans -- revokes any exemption one level
    below where it was granted. Only an element whose *own* text is the maths
    can then be processed, and nbconvert nests the tutorial's four divs deep.

    This is the assertion the first fix needed and did not have: naming the
    notebook's class in processHtmlClass passed a test while the page stayed
    exactly as broken.
    """
    ignore, _, _ = mathjax_config()
    pattern = re.compile(r"(?:^| )(?:" + ignore + r")(?: |$)")
    assert not pattern.search(""), (
        f"ignoreHtmlClass={ignore!r} matches an element with no class at all, "
        f"so it matches every element, so nothing nested inside a processed "
        f"container is ever reached. The built site is checked properly by "
        f"mkdocs-documentation/checks/check_rendered_maths.py"
    )


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_notebook_maths_is_configured_to_render(path):
    """Notebook maths stays as $...$, so those delimiters must be live."""
    display, inline = notebook_math(path)
    if not (display or inline):
        pytest.skip("no maths in this notebook's markdown cells")

    _, inline_delims, display_delims = mathjax_config()
    if display:
        assert '"$$", "$$"' in display_delims.replace("'", '"'), (
            f"{path.name} has {display} $$...$$ formulas and displayMath does "
            f"not list that delimiter, so they stay as written"
        )
    if inline:
        assert '"$", "$"' in inline_delims.replace("'", '"'), (
            f"{path.name} has {inline} $...$ formulas and inlineMath does not "
            f"list that delimiter, so they stay as written"
        )


def test_markdown_maths_is_configured_to_render():
    r"""Arithmatex rewrites .md maths to \(...\), which must be accepted."""
    _, inline_delims, _ = mathjax_config()
    assert "\\\\(" in inline_delims, (
        f"inlineMath ({inline_delims.strip()!r}) does not list the \\(...\\) "
        f"delimiters arithmatex rewrites $...$ into"
    )


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
