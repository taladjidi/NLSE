# Contributing

Contributions are welcome! You can open a pull request or create an issue on the [GitHub repository](https://github.com/Quantum-Optics-LKB/NLSE).

## Development Setup

Clone the repository and install in development mode:

```bash
git clone https://github.com/Quantum-Optics-LKB/NLSE.git
cd NLSE
pip install -e ".[dev]"
```

This installs all development dependencies: pytest, ruff, mypy, etc.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/solvers/test_nlse.py

# Run a single test
pytest tests/solvers/test_nlse.py::test_name

# Verbose output with short tracebacks
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ -v --cov=NLSE --cov-report=term

# Skip benchmarks
pytest tests/ -m "not benchmark"
```

## Linting and Formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
# Lint
ruff check NLSE/ tests/ examples/

# Format check
ruff format --check NLSE/ tests/ examples/

# Auto-fix lint issues
ruff check --fix NLSE/ tests/ examples/

# Auto-format
ruff format NLSE/ tests/ examples/

# Type check
mypy NLSE/ --non-interactive
```

All checks must pass before merging.

### Running them automatically

`pre-commit` wires them into git, so a mistake surfaces in the second before
the commit rather than in the minutes after the push:

```bash
pre-commit install
```

That installs two hooks. On **commit**, ruff and mypy run — about a third of a
second together, and ruff fixes what it can. On **push**, the test suite runs,
about 25 seconds. Use `git commit --no-verify` to skip them on work in
progress.

Every hook runs the tools from your own environment, with the same commands
listed above, so there is no second set of versions to drift from these. Run
them by hand over the whole repository with:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push   # includes the tests
```

Note that mypy is configured to skip `NLSE.solvers.*` and `NLSE.kernels.*`,
which is most of the package; see the type-checking note in the repository's
handoff notes.

## Code Style

- **Formatter**: ruff (88-character line length)
- **Docstrings**: NumPy style (`Parameters`, `Returns`, `Raises` sections)
- **Type checking**: mypy (solvers and kernels modules are excluded via `ignore_errors = true`)

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
