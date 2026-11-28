"""The documentation must not claim things the code does not do."""

import functools
import inspect
import pathlib
import re
import warnings
from unittest import mock

import numpy as np
import pytest
from helpers import ORDERS
from NLSE import NLSE
from NLSE.backends import get_backend, list_available_backends
from NLSE.solvers.nlse import PROPAGATOR_CACHE_SIZE, SPLITTINGS
from NLSE.solvers.step_size import (
    CAPPED_STEP_FRACTION,
    DEFAULT_MIN_STEPS,
    DEFAULT_PHASE_PER_STEP,
    LOSS_PER_STEP_LIMIT,
    RK4_CUBIC_RADIUS,
    RK4_PHASE_PER_STEP,
    RK4_STABILITY_RADIUS,
    SPLIT_STEP_DISPERSION_PHASE_LIMIT,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILT_DOCS = "_build"  # Sphinx's output, ignored by git: not the pages' source
PAGES = [
    *sorted(p for p in (ROOT / "docs").rglob("*.md") if BUILT_DOCS not in p.parts),
    ROOT / "README.md",
]
IDS = [str(p.relative_to(ROOT)) for p in PAGES]

BACKEND_NAMES = {"CPU", "CUPY", "CL", "MLX"}
METHODS = {"split_step", "RK4"}


# The release the migration guide migrates *from*, which stays quoted in
# the prose after the current version moves on.
PREVIOUS_RELEASE = "4.0.0"

# The migration guide describes one historical transition and keeps naming it
# forever. Those two numbers are not staleness, unlike a version quoted as if
# it were the current one.
MIGRATION_FROM, MIGRATION_TO = "3.0.0", "4.0.0"

# The README's splitting table: orders as in helpers.ORDERS, transform pairs
# per step as transform_pairs_per_step counts them.
# A row: `"name"` ... | $\mathcal{O}(\delta z^order)$ | pairs |
SPLITTING_ROW = (
    r'^\| `"(\w+)"`[^|]*\| \$\\mathcal\{O\}\(\\delta z(?:\^(\d))?\)\$ \| (\d+) \|'
)
# numerical_methods.md, a splitting's block: its heading, its error, its cost.
SPLITTING_BLOCK = (
    r'\*\*`splitting="(\w+)"`\*\*.*?'
    r"Error: \$\\mathcal\{O\}\(\\delta z(?:\^(\d))?\)\$.*?"
    r"Cost: (\d+) transform pair"
)
# And its sentence giving RK4's cost beside Lie and Strang's, then Yoshida's.
RK4_COST = (
    r"costs (\d+) transform pairs, against (\d+) for Lie\s+and\s+Strang"
    r"\s+and\s+(\d+)\s+for\s+Yoshida"
)

# Constants the pages quote as `NAME` = value, against the value the code holds.
QUOTED_CONSTANTS = {
    "DEFAULT_PHASE_PER_STEP": DEFAULT_PHASE_PER_STEP,
    "RK4_PHASE_PER_STEP": RK4_PHASE_PER_STEP,
    "DEFAULT_MIN_STEPS": DEFAULT_MIN_STEPS,
    "SPLIT_STEP_DISPERSION_PHASE_LIMIT": SPLIT_STEP_DISPERSION_PHASE_LIMIT,
    "RK4_STABILITY_RADIUS": RK4_STABILITY_RADIUS,
    "RK4_CUBIC_RADIUS": RK4_CUBIC_RADIUS,
    "LOSS_PER_STEP_LIMIT": LOSS_PER_STEP_LIMIT,
    "CAPPED_STEP_FRACTION": CAPPED_STEP_FRACTION,
    "PROPAGATOR_CACHE_SIZE": PROPAGATOR_CACHE_SIZE,
}
# A quote: the name in backticks, an equals sign, the number. A line break may
# fall anywhere between them, since prose is rewrapped.
QUOTE = r"`{name}`\s*=\s*(\d+\.?\d*)"

# The run that counts transforms: no step limit binds on it (a warning there
# fails the count), so it takes the steps it is asked for.
COUNT_GRID = 32  # points per axis
COUNT_WINDOW = 1e-3  # m
COUNT_STEP = 1e-5  # m
COUNTED_STEPS = 4  # steps the longer run takes over the shorter one
# Rows of the capability table every backend has to claim.
CLAIMED_BY_ALL = ["Broadcasting", "Non-local interactions"]


def transforms_in_a_run(splitting, method, steps):
    """Count the forward transforms a CPU run of ``steps`` steps takes.

    Parameters
    ----------
    splitting : str
        Splitting to run.
    method : str
        Integration method ("split_step" or "RK4").
    steps : int
        Steps of ``COUNT_STEP`` to propagate over.

    Returns
    -------
    int
        Calls the run made to the backend's forward transform.
    """
    simu = NLSE(
        0.0,
        1.0,
        COUNT_WINDOW,
        -1e-9,
        V=None,
        L=1.0,
        NX=COUNT_GRID,
        NY=COUNT_GRID,
        backend="CPU",
    )
    # The width each splitting is meant for, so that the run warns of nothing.
    dtype = np.complex128 if splitting == "yoshida" else np.complex64
    field = np.ones((COUNT_GRID, COUNT_GRID), dtype=dtype)
    with (
        warnings.catch_warnings(),
        mock.patch.object(simu._backend, "fft", wraps=simu._backend.fft) as fft,
    ):
        warnings.simplefilter("error")
        simu.out_field(
            field,
            steps * COUNT_STEP,
            delta_z=COUNT_STEP,
            splitting=splitting,
            method=method,
            verbose=False,
        )
    return fft.call_count


@functools.cache
def transform_pairs_per_step(splitting, method="split_step"):
    """Measure the transform pairs one more step costs.

    The difference of two runs, so that a transform a run takes once (the one
    its step limit is computed from) is not charged to the steps.

    Parameters
    ----------
    splitting : str
        Splitting to run.
    method : str
        Integration method ("split_step" or "RK4").

    Returns
    -------
    int
        Forward transforms per step, each of which has its inverse.
    """
    extra = transforms_in_a_run(
        splitting, method, 2 * COUNTED_STEPS
    ) - transforms_in_a_run(splitting, method, COUNTED_STEPS)
    pairs, rest = divmod(extra, COUNTED_STEPS)
    assert rest == 0, f"{extra} transforms over {COUNTED_STEPS} steps of {splitting}"
    return pairs


def test_there_are_pages_to_check():
    """A glob that matched nothing would make every test below vacuous."""
    assert len(PAGES) > 5, f"only found {IDS}"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_imported_symbols_exist(page):
    """`from NLSE import X` in a doc must be an X that exists."""
    import NLSE as package

    text = page.read_text(encoding="utf-8")
    missing = []
    for group in re.findall(r"from NLSE import ([\w, ]+)", text):
        for symbol in (s.strip() for s in group.split(",")):
            if symbol and not hasattr(package, symbol):
                missing.append(symbol)
    assert not missing, f"{page.name} imports {missing} from NLSE, which has no such"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_out_field_keywords_exist(page):
    """A keyword shown in a call has to be one `out_field` takes."""
    if page.name == "migration.md":
        pytest.skip("shows removed keywords on purpose; see the test below")
    allowed = set(inspect.signature(NLSE.out_field).parameters)
    text = page.read_text(encoding="utf-8")
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
    text = page.read_text(encoding="utf-8")
    bad_split = set(re.findall(r'splitting="(\w+)"', text)) - set(SPLITTINGS)
    bad_method = set(re.findall(r'method="(\w+)"', text)) - METHODS
    assert not bad_split, (
        f"{page.name}: splitting {sorted(bad_split)} not in {SPLITTINGS}"
    )
    assert not bad_method, f"{page.name}: method {sorted(bad_method)} not in {METHODS}"


@pytest.mark.parametrize("page", PAGES, ids=IDS)
def test_named_backends_are_real(page):
    """A page naming a backend must name one that exists."""
    if page.name == "migration.md":
        pytest.skip("names dead backends on purpose; see the test below")
    text = page.read_text(encoding="utf-8")
    bad = set(re.findall(r'backend="(\w+)"', text)) - BACKEND_NAMES - {"auto"}
    assert not bad, f"{page.name} names backends {sorted(bad)}"


def test_the_migration_guide_shows_a_removed_out_field_keyword():
    """And shows what to write instead, or it is a list of complaints."""
    text = (ROOT / "docs" / "migration.md").read_text(encoding="utf-8")
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
    text = (ROOT / "docs" / "migration.md").read_text(encoding="utf-8")
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
    text = page.read_text(encoding="utf-8")
    wrong = [
        f"{name} quoted as {quoted}, is {value}"
        for name, value in QUOTED_CONSTANTS.items()
        for quoted in re.findall(QUOTE.format(name=name), text)
        if abs(float(quoted) - value) > 1e-12
    ]
    assert not wrong, f"{page.name}: " + "; ".join(wrong)


def test_every_pinned_constant_is_quoted_somewhere():
    """A quote reworded out of the form the test reads would pin nothing."""
    text = "\n".join(page.read_text(encoding="utf-8") for page in PAGES)
    unquoted = [
        name
        for name in QUOTED_CONSTANTS
        if not re.search(QUOTE.format(name=name), text)
    ]
    assert not unquoted, f"no page quotes {unquoted} as `NAME` = value"


def test_the_capability_table_matches_the_backends():
    """backends.md's table is the thing readers plan around."""
    table = (ROOT / "docs" / "backends.md").read_text(encoding="utf-8")
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
        claimed = rows["Double precision"][columns[name]]
        actual = bool(backend.supports_double_precision())
        # "Device-dependent" is an honest answer for OpenCL and not a claim.
        if claimed.lower() in {"yes", "no"} and (claimed.lower() == "yes") != actual:
            wrong.append(f"{name} double: table says {claimed!r}, is {actual}")
    assert not wrong, "; ".join(wrong)


@pytest.mark.parametrize("feature", CLAIMED_BY_ALL)
def test_every_backend_is_claimed_to_have(feature):
    """docs/backends.md marks each of these features yes for every backend."""
    table = (ROOT / "docs" / "backends.md").read_text(encoding="utf-8")
    row = re.search(rf"^\| {feature} \|([^\n]+)\|$", table, re.M)
    assert row, f"no {feature} row in the capability table"
    cells = [c.strip().lower() for c in row.group(1).split("|") if c.strip()]
    assert all(c == "yes" for c in cells), (
        f"{feature} works on every backend; table says {cells}"
    )


def test_the_splitting_table_matches_the_code():
    """The README's order and cost per splitting are claims the tests measure."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(SPLITTING_ROW, text, re.M)
    assert [name for name, _, _ in rows] == list(SPLITTINGS), (
        f"the README's splitting table lists {rows}, the solver {SPLITTINGS}"
    )
    orders = {name: int(order or 1) for name, order, _ in rows}
    pairs = {name: int(count) for name, _, count in rows}
    assert orders == {name: ORDERS[name] for name in SPLITTINGS}
    assert pairs == {name: transform_pairs_per_step(name) for name in SPLITTINGS}


def test_the_methods_page_states_the_measured_order_and_cost():
    """numerical_methods.md gives each splitting's order and transform pairs."""
    text = (ROOT / "docs" / "numerical_methods.md").read_text(encoding="utf-8")
    blocks = re.findall(SPLITTING_BLOCK, text, re.S)
    assert [name for name, _, _ in blocks] == list(SPLITTINGS), (
        f"the page's splitting blocks read as {blocks}, the solver has {SPLITTINGS}"
    )
    measured = {name: transform_pairs_per_step(name) for name in SPLITTINGS}
    orders = {name: int(order or 1) for name, order, _ in blocks}
    assert orders == {name: ORDERS[name] for name in SPLITTINGS}
    assert {name: int(count) for name, _, count in blocks} == measured

    rk4 = re.search(RK4_COST, text)
    assert rk4, "the page no longer gives RK4's cost beside the splittings'"
    assert measured["lie"] == measured["strang"], "the sentence gives them one cost"
    assert [int(count) for count in rk4.groups()] == [
        transform_pairs_per_step("lie", "RK4"),
        measured["lie"],
        measured["yoshida"],
    ]


def test_the_documented_version_is_the_real_one():
    """index.md names the current version; no page quotes an unknown one."""
    from NLSE import __version__

    allowed = {__version__, PREVIOUS_RELEASE, MIGRATION_FROM, MIGRATION_TO}
    wrong = []
    for name in ("index.md", "migration.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        for quoted in set(re.findall(r"\b(\d+\.\d+\.\d+)\b", text)):
            if quoted not in allowed:
                wrong.append(f"{name} names {quoted}, which is no release we know")
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    if __version__ not in index:
        wrong.append(f"index.md never names the current version {__version__}")
    assert not wrong, (
        f"the docs and the package disagree about the version: "
        f"{'; '.join(wrong)}. NLSE.__version__ is {__version__}"
    )
