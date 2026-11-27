"""Shared test configuration and fixtures."""

import pytest
from helpers import AVAILABLE_BACKENDS


@pytest.fixture(autouse=True)
def private_benchmark_cache(tmp_path, monkeypatch):
    """Give every test a benchmark cache of its own, empty at the start."""
    monkeypatch.setenv("NLSE_CACHE_DIR", str(tmp_path / "nlse_cache"))


@pytest.fixture(autouse=True)
def unblocked_cpu_step(monkeypatch):
    """Keep the CPU step unblocked, so that no test measures a grid by accident."""
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "0")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend(request):
    """Yield each available backend as a test parameter."""
    return request.param
