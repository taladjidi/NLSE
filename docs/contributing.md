# Contributing

Contributions are welcome! You can open a pull request or create an issue on the [GitHub repository](https://github.com/taladjidi/NLSE).

## Development Setup

Clone the repository and create the development environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/taladjidi/NLSE.git
cd NLSE
uv sync
```

This creates `.venv` on the Python named in `.python-version`, with the
package in editable mode and the `dev` dependency group (pytest, ruff, ty,
pre-commit, hatchling), at the versions pinned in `uv.lock`. Add
`--extra opencl`, `--extra mlx` or `--extra gpu` for a backend, and repeat
them on every later `uv sync`: it removes what it was not asked for. Every
command below, and every git hook, runs through `uv run`, which uses that
environment whatever else is on your PATH, so uv is required.

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock` with it; CI and the hooks refuse a stale lock.

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/solvers/test_nlse.py

# Run a single test
uv run pytest tests/solvers/test_nlse.py::test_name

# Verbose output with short tracebacks
uv run pytest tests/ -v --tb=short

# With coverage report
uv run pytest tests/ -v --cov=NLSE --cov-report=term

# Skip benchmarks
uv run pytest tests/ -m "not benchmark"
```

## Linting and Formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [ty](https://docs.astral.sh/ty/) for type checking:

```bash
# Lint
uv run ruff check NLSE/ tests/ examples/

# Format check
uv run ruff format --check NLSE/ tests/ examples/

# Auto-fix lint issues
uv run ruff check --fix NLSE/ tests/ examples/

# Auto-format
uv run ruff format NLSE/ tests/ examples/

# Type check
uv run ty check
```

All checks must pass before merging.

### Running them automatically

`pre-commit` wires them into git, so a mistake surfaces in the second before
the commit rather than in the minutes after the push:

```bash
uv run pre-commit install
```

That installs two hooks. On **commit**, ruff and ty run, well under a second
together, and ruff fixes what it can. On **push**, the test suite runs. Use
`git commit --no-verify` to skip them on work in progress.

Every hook runs its tool through `uv run --locked`, with the same commands
listed above: the versions in `uv.lock`, from `.venv`, so another Python or
an older install of the package earlier on your PATH cannot stand in for
them. Run them by hand over the whole repository with:

```bash
uv run pre-commit run --all-files
uv run pre-commit run --all-files --hook-stage pre-push   # includes the tests
```

ty checks `NLSE/` only, and fails only on diagnostics that are not already
recorded in `ty-baseline.json`. A recorded diagnostic is matched by file,
rule and columns, not by line number: moving it is fine, but renaming or
reformatting the line it is on reports it again. Then either fix it, or, once
you have checked it is the same diagnostic, run
`uv run ty check --update-baseline` and commit the file. Do the same after
fixing recorded ones, which shrinks it. Never update it to absorb a new
diagnostic: review reads the diff of `ty-baseline.json`, where a moved entry
changes only its columns and a new one adds an entry.

## Code Style

- **Formatter**: ruff (88-character line length)
- **Docstrings**: NumPy style (`Parameters`, `Returns`, `Raises` sections)
- **Type checking**: ty, on `NLSE/`, against the baseline in `ty-baseline.json`

## Project Structure

```
NLSE/
├── __init__.py          # Public API exports
├── callbacks.py         # Built-in callback functions
├── utils.py             # Backend detection utilities
├── solvers/             # Solver classes (NLSE, GPE, CNLSE, etc.)
├── kernels/             # Backend-specific numerical kernels
└── backends/            # Backend abstraction layer
tests/
├── solvers/             # Per-solver tests
├── backends/            # Backend-specific tests
├── integration/         # Integration tests (broadcasting, nonlocality)
└── benchmarks/          # Performance benchmarks
examples/                # Example scripts
```

## Contact

For questions or issues, contact tangui.aladjidi[at]lkb.upmc.fr or open a GitHub issue.

## Building the documentation

```bash
uv sync --group docs
uv run python -m sphinx -b html docs docs/_build/html -W
uv run python -m http.server -d docs/_build/html      # then open http://localhost:8000
```

`-W` turns warnings into errors, which is what CI does: a broken
cross-reference or a page missing from a toctree fails the build rather than
shipping. The API section is generated from the source by sphinx-autoapi, so
there is nothing to write by hand for a new class -- give it a numpy
docstring and it appears.

The gallery runs every script in `examples/` but two, 57 minutes on a GitHub
runner. sphinx-gallery keeps what it ran in `docs/auto_examples/` and runs a
script again only when its source changes, so a second build runs none, and
neither does a change to `NLSE/` alone. A change to `NLSE/` that breaks an
example therefore passes CI, and the weekly run is the first to fail on it;
the same holds for a change to `sphinx_gallery_conf` or `examples/_output.py`,
since neither re-runs an example. CI keeps that directory between runs, so a
push runs only the examples it changed. The weekly CI run, and one started
with **Run workflow** on the Actions page, run every example against the
current code and fail if one raises. To do that locally:

```bash
uv run python -m sphinx -b html docs docs/_build/html -W -D sphinx_gallery_conf.run_stale_examples=True
```

After deleting or renaming an example, delete `docs/auto_examples/` too: the
old page is otherwise still there, in no toctree, and `-W` fails on it.
