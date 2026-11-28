"""Shared test configuration and fixtures."""

import os

import pytest
from helpers import AVAILABLE_BACKENDS

# Dropped before any test module is imported, since several build their
# parameters from get_backend: an exported NLSE_BACKEND would run every
# backend's tests on one.
for _variable in ("NLSE_BACKEND", "NLSE_FORCE_BENCHMARK"):
    os.environ.pop(_variable, None)


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
