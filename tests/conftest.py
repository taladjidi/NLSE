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
