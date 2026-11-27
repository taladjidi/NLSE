"""Shared test configuration and fixtures."""

import pytest

try:
    import pytest_benchmark  # noqa: F401

    _HAS_BENCHMARK = True
except ImportError:
    _HAS_BENCHMARK = False


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests when pytest-benchmark is not installed."""
    if _HAS_BENCHMARK:
        return
    skip_benchmark = pytest.mark.skip(reason="pytest-benchmark not installed")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture(autouse=True)
def private_benchmark_cache(tmp_path, monkeypatch):
    """Give every test a benchmark cache of its own, empty at the start.

    Tests that benchmark or clear the cache otherwise rewrite the user's real
    one, as the audit's runs did with a 64x64 grid.
    """
    monkeypatch.setenv("NLSE_CACHE_DIR", str(tmp_path / "nlse_cache"))


@pytest.fixture(autouse=True)
def unblocked_cpu_step(monkeypatch):
    """Keep the CPU step unblocked, so that no test measures a grid by accident.

    The tests of the blocked step, in tests/backends/test_cpu_blocked_step.py,
    set NLSE_CPU_BLOCKING back themselves.
    """
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "0")
