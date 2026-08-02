"""The documentation must not claim things the code does not do.

`test_docs.py` runs the self-contained code blocks, which catches a doc whose
*example* stopped working. It does not catch prose, and prose is where the
documentation had drifted furthest: a capability table said broadcasting was
CUPY-only four paragraphs below a section describing how CPU and OpenCL do it,
and both had been served to readers.

So the checkable claims are checked here. Not every sentence can be, but names,
keywords, capabilities and quoted constants can, and those are what go stale
when the code moves under them.
"""

import inspect
import pathlib
import re

import pytest
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends
from NLSE.solvers.nlse import SPLITTINGS
from NLSE.solvers.step_size import (
    DEFAULT_MIN_STEPS,
    DEFAULT_PHASE_PER_STEP,
    RK4_PHASE_PER_STEP,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = [
    *sorted((ROOT / "mkdocs-documentation" / "docs").rglob("*.md")),
    ROOT / "README.md",
]
IDS = [str(p.relative_to(ROOT)) for p in PAGES]

BACKEND_NAMES = {"CPU", "CUPY", "CL", "MLX"}
METHODS = {"split_step", "RK4"}


# The release the migration guide migrates *from*, which stays quoted in
# the prose after the current version moves on.
PREVIOUS_RELEASE = "3.0.0"


def test_there_are_pages_to_check():
    """A glob that matched nothing would make every test below vacuous."""
    assert len(PAGES) > 5, f"only found {IDS}"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_imported_symbols_exist(page):
    """`from NLSE import X` in a doc must be an X that exists."""
    import NLSE as package

    text = page.read_text()
    missing = []
    for group in re.findall(r"from NLSE import ([\w, ]+)", text):
        for symbol in (s.strip() for s in group.split(",")):
            if symbol and not hasattr(package, symbol):
                missing.append(symbol)
    assert not missing, f"{page.name} imports {missing} from NLSE, which has no such"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_out_field_keywords_exist(page):
    """A keyword shown in a call has to be one `out_field` takes.

    Except on the migration guide, whose left-hand column is keywords it no
    longer takes. Kept honest by the test below.
    """
    if page.name == "migration.md":
        pytest.skip("shows removed keywords on purpose; see the test below")
    allowed = set(inspect.signature(NLSE.out_field).parameters)
    text = page.read_text()
    bad = {
        kw
        for call in re.findall(r"out_field\((.*?)\)", text, re.S)
        for kw in re.findall(r"(\w+)=", call)
        if kw not in allowed
    }
    assert not bad, f"{page.name} passes {sorted(bad)} to out_field, which takes none"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_named_splittings_and_methods_are_real(page):
    """The names that select a scheme are the ones the solver accepts."""
    text = page.read_text()
    bad_split = set(re.findall(r'splitting="(\w+)"', text)) - set(SPLITTINGS)
    bad_method = set(re.findall(r'method="(\w+)"', text)) - METHODS
    assert not bad_split, (
        f"{page.name}: splitting {sorted(bad_split)} not in {SPLITTINGS}"
    )
    assert not bad_method, f"{page.name}: method {sorted(bad_method)} not in {METHODS}"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_named_backends_are_real(page):
    """A page naming a backend must name one that exists.

    The migration guide is the exception, and has to be: it exists to show
    the names that stopped working beside the ones that replaced them. The
    test below keeps that exemption honest.
    """
    if page.name == "migration.md":
        pytest.skip("names dead backends on purpose; see the test below")
    text = page.read_text()
    bad = set(re.findall(r'backend="(\w+)"', text)) - BACKEND_NAMES - {"auto"}
    assert not bad, f"{page.name} names backends {sorted(bad)}"


def test_the_migration_guide_shows_a_removed_out_field_keyword():
    """And shows what to write instead, or it is a list of complaints."""
    text = (ROOT / "mkdocs-documentation" / "docs" / "migration.md").read_text()
    allowed = set(inspect.signature(NLSE.out_field).parameters)
    shown = {
        kw
        for call in re.findall(r"out_field\((.*?)\)", text, re.S)
        for kw in re.findall(r"(\w+)=", call)
    }
    assert shown - allowed, (
        "migration.md shows no removed out_field keyword, so it is skipped "
        "above for nothing -- drop the skip or write the section"
    )
    assert shown & allowed, (
        f"migration.md shows {sorted(shown - allowed)} and no keyword that "
        f"still works, so it names the breakage and not the fix"
    )


def test_the_migration_guide_maps_dead_backends_to_live_ones():
    """Every dead backend it names must appear beside a real one."""
    text = (ROOT / "mkdocs-documentation" / "docs" / "migration.md").read_text()
    named = set(re.findall(r'backend="(\w+)"', text))
    dead = named - BACKEND_NAMES - {"auto"}
    assert dead, (
        "migration.md names no dead backend, so it is skipped above for "
        "nothing -- drop the skip or write the row"
    )
    assert named & BACKEND_NAMES, (
        f"migration.md names {sorted(dead)} and no backend that exists, so it "
        f"tells a reader what broke and not what to write instead"
    )


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_quoted_constants_match_the_code(page):
    """A constant quoted in prose drifts silently when the code moves."""
    facts = {
        "DEFAULT_PHASE_PER_STEP": DEFAULT_PHASE_PER_STEP,
        "RK4_PHASE_PER_STEP": RK4_PHASE_PER_STEP,
        "DEFAULT_MIN_STEPS": DEFAULT_MIN_STEPS,
    }
    text = page.read_text()
    wrong = []
    for name, value in facts.items():
        # The number on the same line as the name, if there is one.
        for line in (ln for ln in text.splitlines() if name in ln):
            for quoted in re.findall(r"(?<![\w.])(\d+\.?\d*)(?![\w])", line):
                if abs(float(quoted) - value) > 1e-12:
                    wrong.append(f"{name} quoted as {quoted}, is {value}")
    assert not wrong, f"{page.name}: " + "; ".join(wrong)


def test_the_capability_table_matches_the_backends():
    """backends.md's table is the thing readers plan around."""
    table = (ROOT / "mkdocs-documentation" / "docs" / "backends.md").read_text()
    rows = {
        m.group(1).strip(): [c.strip() for c in m.group(2).split("|")]
        for m in re.finditer(r"^\| ([^|]+?) \|([^\n]+)\|$", table, re.M)
    }
    order = rows.get("Feature")
    assert order, "no capability table found in backends.md"
    columns = {name: i for i, name in enumerate(order) if name in BACKEND_NAMES}
    assert columns, f"table header does not name backends: {order}"

    wrong = []
    for name in list_available_backends():
        backend = get_backend(name, (64, 64))
        claimed = rows["Convolution"][columns[name]]
        actual = backend.convolution is not None
        if (claimed.lower() == "yes") != actual:
            wrong.append(f"{name} convolution: table says {claimed!r}, is {actual}")

        claimed = rows["Double precision"][columns[name]]
        actual = bool(backend.supports_double_precision())
        # "Device-dependent" is an honest answer for OpenCL and not a claim.
        if claimed.lower() in {"yes", "no"} and (claimed.lower() == "yes") != actual:
            wrong.append(f"{name} double: table says {claimed!r}, is {actual}")
    assert not wrong, "; ".join(wrong)


def test_every_backend_is_claimed_to_broadcast():
    """It does, on all four, and the table said otherwise for three of them."""
    table = (ROOT / "mkdocs-documentation" / "docs" / "backends.md").read_text()
    row = re.search(r"^\| Broadcasting \|([^\n]+)\|$", table, re.M)
    assert row, "no Broadcasting row in the capability table"
    cells = [c.strip().lower() for c in row.group(1).split("|") if c.strip()]
    assert all(c == "yes" for c in cells), (
        f"broadcasting works on every backend; table says {cells}"
    )


def test_the_documented_version_is_the_real_one():
    """A version quoted on a page goes stale the moment one is released.

    index.md tells the reader which version the site documents, because the
    site is built from main and describes code they may not have installed,
    and migration.md is named for the release it migrates to. Both are prose,
    so nothing but this stops them naming 4.0.0 forever.
    """
    from NLSE import __version__

    pages = {
        "index.md": ROOT / "mkdocs-documentation" / "docs" / "index.md",
        "migration.md": ROOT / "mkdocs-documentation" / "docs" / "migration.md",
    }
    wrong = []
    for name, path in pages.items():
        text = path.read_text()
        # Any x.y.z in the prose that is not the current version and not the
        # previous release being migrated from.
        for quoted in set(re.findall(r"\b(\d+\.\d+\.\d+)\b", text)):
            if quoted not in {__version__, PREVIOUS_RELEASE}:
                wrong.append(f"{name} names {quoted}")
        if __version__ not in text:
            wrong.append(f"{name} never names the current version {__version__}")
    assert not wrong, (
        f"the docs and the package disagree about the version: "
        f"{'; '.join(wrong)}. NLSE.__version__ is {__version__}"
    )
