"""Tests for FFT auto-benchmarking system."""

import json
import platform
from datetime import datetime, timedelta, timezone

import pytest
from NLSE.backends import (
    _ASSUMED_SPEED_ORDER,
    backends_by_speed,
    benchmark,
    list_available_backends,
)
from NLSE.utils import get_benchmark_cache_path, get_cache_dir

# A cache record written by hand, so each rejection below spoils one field of
# a record that is otherwise accepted.
CACHED_GRID = [256, 256]
CACHED_TIME_MS = 1.0  # ms, any positive time
# Past load_benchmark_cache's 30-day limit (NLSE/backends/benchmark.py:259).
STALE_AGE_DAYS = 31
OLD_CACHE_VERSION = "2"
CPU_SPEEDUP = 1.0  # the CPU against itself, as benchmark_all_backends computes it
OTHER_SYSTEM = "Plan9"
OTHER_PROCESSOR = "pdp11"
OTHER_GRID = (128, 128)  # not CACHED_GRID
# Available backends listed against the assumed speed order, slowest first, so
# that an order kept from this list rather than from the prior shows.
AGAINST_THE_PRIOR = ["CPU", "CL", "MLX", "CUPY"]
# The same backends in the prior's order, NLSE/backends/__init__.py:186.
BY_THE_PRIOR = ["CUPY", "MLX", "CL", "CPU"]
# A cache timing only the CPU: the CPU first, the others in the prior's order.
CPU_TIMED_FIRST = ["CPU", "CUPY", "MLX", "CL"]
NOT_A_BACKEND = "NO_SUCH_BACKEND"
# The stand-in backend below allocates nothing, so any grid does.
LAZY_GRID = (8, 8)
LAZY_TRIALS = 3  # timed trials, each of which must wait on its own result


def valid_cache_record():
    """Return a cache record that load_benchmark_cache accepts.

    Returns
    -------
    dict
        Today's record, for this machine and the current cache version.
    """
    return {
        "version": benchmark.CACHE_VERSION,
        "timestamp": datetime.now().isoformat(),
        "grid_size": CACHED_GRID,
        "platform": {
            "system": platform.system(),
            "processor": platform.machine(),
            "python_version": platform.python_version(),
        },
        "results": {
            "CPU": {
                "time_ms": CACHED_TIME_MS,
                "speedup": CPU_SPEEDUP,
                "available": True,
            }
        },
        "fastest": "CPU",
    }


def write_cache(text):
    """Write raw text where the benchmark cache is read from.

    Parameters
    ----------
    text : str
        File contents.
    """
    get_benchmark_cache_path().write_text(text)


class LazyBackend:
    """Stand-in backend whose transforms return a new array, as MLX's do."""

    def __init__(self):
        self.latest = object()  # the array the last transform returned
        self.stale = 0  # calls handed any other array
        self.waits = 0  # calls to synchronize

    def _seen(self, array):
        """Count the array if it is not the one last returned."""
        self.stale += array is not self.latest

    def allocate_field(self, shape, dtype):
        """Return the first array."""
        return self.latest

    def build_fft(self, shape, axes, dtype):
        """Return no plan: the transform below needs none."""
        return None

    def fft(self, array, plan):
        """Return a new array, leaving the one passed in as it was."""
        self._seen(array)
        self.latest = object()
        return self.latest

    ifft = fft

    def synchronize(self, array=None):
        """Record a wait and the array it was asked to wait on."""
        self._seen(array)
        self.waits += 1


class TestCacheDir:
    """Test that the cache directory lives outside the installed package."""

    def test_cache_dir_created(self, tmp_path, monkeypatch):
        """get_cache_dir() should create the directory if needed."""
        missing = tmp_path / "missing" / "cache"
        monkeypatch.setenv("NLSE_CACHE_DIR", str(missing))
        assert get_cache_dir() == missing
        assert missing.is_dir()

    def test_benchmark_cache_in_cache_dir(self):
        """Benchmark cache file should be inside the cache dir."""
        cache_dir = get_cache_dir()
        bench_path = get_benchmark_cache_path()
        assert bench_path.parent == cache_dir
        assert bench_path.name == "fft_benchmark.json"


class TestBenchmarkBackend:
    """Test benchmarking individual backends."""

    def test_benchmark_cpu(self):
        """CPU backend should always be available and benchmarkable."""
        time_ms = benchmark.benchmark_backend("CPU", (256, 256), num_trials=3)
        assert time_ms is not None
        assert time_ms > 0
        assert time_ms < 1000  # Sanity check - should be fast on small grid

    @pytest.mark.parametrize("backend_name", list_available_backends())
    def test_benchmark_each_available_backend(self, backend_name):
        """Test benchmarking each available backend."""
        time_ms = benchmark.benchmark_backend(backend_name, (256, 256), num_trials=3)
        assert time_ms is not None
        assert time_ms > 0

    def test_benchmark_times_what_the_transforms_return(self, monkeypatch):
        """Each transform is fed, and each trial waits on, the last result."""
        lazy = LazyBackend()
        monkeypatch.setattr("NLSE.backends.get_backend", lambda name: lazy)
        time_ms = benchmark.benchmark_backend("LAZY", LAZY_GRID, LAZY_TRIALS)
        assert time_ms is not None
        assert lazy.stale == 0
        assert lazy.waits >= LAZY_TRIALS

    def test_benchmark_invalid_backend(self):
        """Test that invalid backend returns None."""
        time_ms = benchmark.benchmark_backend("INVALID", (256, 256), num_trials=3)
        assert time_ms is None


class TestBenchmarkAll:
    """Test benchmarking all backends at once."""

    def test_benchmark_all_backends(self):
        """Test benchmarking all available backends."""
        results = benchmark.benchmark_all_backends((256, 256))

        # Check structure
        assert "version" in results
        assert "timestamp" in results
        assert "grid_size" in results
        assert "platform" in results
        assert "results" in results
        assert "fastest" in results

        # Check CPU is present and working
        assert "CPU" in results["results"]
        assert results["results"]["CPU"]["available"] is True
        assert results["results"]["CPU"]["time_ms"] > 0

        # Check fastest is valid
        assert results["fastest"] in list_available_backends()

    def test_benchmark_speedup_calculation(self):
        """Test that speedup is calculated correctly."""
        results = benchmark.benchmark_all_backends((256, 256))

        cpu_speedup = results["results"]["CPU"]["speedup"]

        # CPU should have speedup of 1.0
        assert cpu_speedup == 1.0

        # Other backends should have speedup relative to CPU
        for backend_name, data in results["results"].items():
            if data["available"] and backend_name != "CPU":
                assert data["speedup"] is not None
                assert data["speedup"] > 0


class TestGetFastestBackend:
    """Test automatic selection of fastest backend."""

    def test_get_fastest_backend(self):
        """Test fastest backend selection."""
        # Force benchmark to ensure clean state
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest in list_available_backends()

    def test_get_fastest_returns_valid_backend(self):
        """Test that fastest backend is always valid."""
        fastest = benchmark.get_fastest_backend((512, 512), force_benchmark=True)

        # Should be one of the available backends
        available = list_available_backends()
        assert fastest in available


class TestCachePersistence:
    """Test benchmark caching functionality."""

    def test_cache_creation(self):
        """Test that benchmark results are cached."""
        cache_path = get_benchmark_cache_path()

        # Empty at the start: conftest gives each test a cache of its own
        assert not cache_path.exists()

        # Run benchmark
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        # Check cache was created
        assert cache_path.exists()

        # Verify cache content
        with open(cache_path) as f:
            cache = json.load(f)

        assert cache["fastest"] == fastest
        assert cache["grid_size"] == [256, 256]

    def test_cache_reuse(self, monkeypatch):
        """A cached call answers without benchmarking again."""
        fastest1 = benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        ran = []
        measure = benchmark.benchmark_all_backends

        def spy(*args, **kwargs):
            ran.append(args)
            return measure(*args, **kwargs)

        monkeypatch.setattr(benchmark, "benchmark_all_backends", spy)

        fastest2 = benchmark.get_fastest_backend((256, 256), force_benchmark=False)
        assert fastest2 == fastest1
        assert not ran, "the cached call re-benchmarked instead of reading the cache"

        # And the spy does fire when a benchmark is genuinely asked for, so
        # the assertion above cannot pass by having patched the wrong name.
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert ran, "patched a name get_fastest_backend does not call"

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache_path = get_benchmark_cache_path()

        # Create cache
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert cache_path.exists()

        # Invalidate
        benchmark.invalidate_cache()
        assert not cache_path.exists()

    def test_cache_grid_size_mismatch(self):
        """Test that cache is invalidated on grid size mismatch."""
        # Create cache with one size
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        cache1 = benchmark.load_benchmark_cache()

        # Request different size (should re-benchmark)
        benchmark.get_fastest_backend((512, 512), force_benchmark=False)
        cache2 = benchmark.load_benchmark_cache()

        # Cache should have been updated
        assert cache1["grid_size"] != cache2["grid_size"]
        assert cache2["grid_size"] == [512, 512]


class TestCacheLoading:
    """Test cache loading and validation."""

    def test_load_nonexistent_cache(self):
        """Test loading cache when it doesn't exist."""
        cache = benchmark.load_benchmark_cache()
        assert cache is None

    def test_load_valid_cache(self):
        """Test loading valid cache."""
        # Create cache
        benchmark.get_fastest_backend((256, 256), force_benchmark=True)

        # Load it
        cache = benchmark.load_benchmark_cache()
        assert cache is not None
        assert "version" in cache
        assert "timestamp" in cache

    def test_load_corrupted_cache(self):
        """Test that corrupted cache is handled gracefully."""
        cache_path = get_benchmark_cache_path()

        # Write invalid JSON
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("invalid json {{{")

        # Should return None
        cache = benchmark.load_benchmark_cache()
        assert cache is None

    def test_load_hand_written_cache(self):
        """The record the rejections below each spoil one field of is accepted."""
        record = valid_cache_record()
        write_cache(json.dumps(record))
        assert benchmark.load_benchmark_cache() == record

    @pytest.mark.parametrize(
        "written",
        [
            {"system": OTHER_SYSTEM, "processor": platform.machine()},
            {"system": platform.system(), "processor": OTHER_PROCESSOR},
            OTHER_SYSTEM,
        ],
        ids=["another_machine", "another_processor", "not_an_object"],
    )
    def test_load_rejects_another_platform(self, written):
        """A cache is a statement about the machine that wrote it."""
        record = valid_cache_record()
        record["platform"] = written
        write_cache(json.dumps(record))
        assert benchmark.load_benchmark_cache() is None

    def test_load_rejects_a_stale_cache(self):
        """Older than 30 days is measured again."""
        record = valid_cache_record()
        written = datetime.now() - timedelta(days=STALE_AGE_DAYS)
        record["timestamp"] = written.isoformat()
        write_cache(json.dumps(record))
        assert benchmark.load_benchmark_cache() is None

    def test_load_rejects_an_older_version(self):
        """A cache from before a change to what it timed is not believed."""
        record = valid_cache_record()
        record["version"] = OLD_CACHE_VERSION
        write_cache(json.dumps(record))
        assert benchmark.load_benchmark_cache() is None

    @pytest.mark.parametrize(
        "text",
        ["null", "42", '"version timestamp"', '["version", "timestamp"]', "{}"],
        ids=["null", "number", "string", "list", "no_keys"],
    )
    def test_load_rejects_valid_json_of_the_wrong_shape(self, text):
        """Valid JSON that is not a record is refused, not raised on."""
        write_cache(text)
        assert benchmark.load_benchmark_cache() is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("grid_size", 256),
            ("grid_size", None),
            ("grid_size", ["256", "256"]),
            ("fastest", 5),
            ("fastest", ["CPU"]),
            ("results", ["CPU"]),
            ("results", {"CPU": "fast"}),
            ("results", {"CPU": {"time_ms": "1.0"}}),
            ("timestamp", 0),
            ("timestamp", "yesterday"),
            ("timestamp", datetime.now(timezone.utc).isoformat()),
            ("version", 2),
        ],
        ids=lambda value: json.dumps(value)[:24],
    )
    def test_a_wrong_typed_field_is_no_cache(self, field, value):
        """One field of another type: no cache, and backend selection goes on."""
        record = valid_cache_record()
        record[field] = value
        write_cache(json.dumps(record))
        assert benchmark.load_benchmark_cache() is None
        assert backends_by_speed(tuple(CACHED_GRID)) == sorted(
            list_available_backends(), key=_ASSUMED_SPEED_ORDER.index
        )

    def test_a_cache_at_another_grid_is_not_believed(self, monkeypatch):
        """A cache entry measured at another grid is not believed."""
        monkeypatch.setattr(
            "NLSE.backends.list_available_backends", lambda: AGAINST_THE_PRIOR
        )
        write_cache(json.dumps(valid_cache_record()))
        assert backends_by_speed(OTHER_GRID) == BY_THE_PRIOR

    def test_untimed_backends_follow_the_prior(self, monkeypatch):
        """Backends the cache has no time for come after it, in the prior's order."""
        monkeypatch.setattr(
            "NLSE.backends.list_available_backends", lambda: AGAINST_THE_PRIOR
        )
        write_cache(json.dumps(valid_cache_record()))
        assert backends_by_speed(tuple(CACHED_GRID)) == CPU_TIMED_FIRST

    def test_a_cached_name_that_is_no_backend_is_measured_again(self):
        """A fastest backend this install does not have is not handed back."""
        record = valid_cache_record()
        record["fastest"] = NOT_A_BACKEND
        write_cache(json.dumps(record))
        fastest = benchmark.get_fastest_backend(tuple(CACHED_GRID))
        assert fastest in list_available_backends()

    def test_an_unusable_cache_directory_is_no_cache(self, tmp_path, monkeypatch):
        """A cache directory that cannot be made stops neither a read nor a write."""
        blocker = tmp_path / "a_file"
        blocker.write_text("")
        monkeypatch.setenv("NLSE_CACHE_DIR", str(blocker))
        assert benchmark.load_benchmark_cache() is None
        benchmark.save_benchmark_cache(valid_cache_record())
        assert blocker.read_text() == ""

    def test_a_failed_write_keeps_the_previous_cache(self):
        """The cache is replaced whole or not at all."""
        record = valid_cache_record()
        write_cache(json.dumps(record))
        benchmark.save_benchmark_cache({**record, "fastest": object()})
        assert benchmark.load_benchmark_cache() == record
        assert list(get_cache_dir().iterdir()) == [get_benchmark_cache_path()]


class TestFallbackBehavior:
    """Test graceful fallback on benchmark failures."""

    def test_fallback_on_all_backends_fail(self):
        """Test fallback when all backends fail."""
        # This is hard to test without mocking, but we can verify
        # that get_fastest_backend always returns a valid backend
        fastest = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest is not None
        assert fastest in list_available_backends()


class TestIntegration:
    """Integration tests for the benchmarking system."""

    def test_full_workflow(self):
        """Test complete workflow from benchmark to cache to reuse."""
        cache_path = get_benchmark_cache_path()

        # Step 1: Empty at the start: conftest gives each test a cache of its own
        assert not cache_path.exists()

        # Step 2: Get fastest backend (should benchmark)
        fastest1 = benchmark.get_fastest_backend((256, 256))
        assert cache_path.exists()

        # Step 3: Get again (should use cache)
        fastest2 = benchmark.get_fastest_backend((256, 256))
        assert fastest1 == fastest2

        # Step 4: Force re-benchmark
        fastest3 = benchmark.get_fastest_backend((256, 256), force_benchmark=True)
        assert fastest3 in list_available_backends()  # Valid backend returned

        # Step 5: Invalidate
        benchmark.invalidate_cache()
        assert not cache_path.exists()

    def test_benchmark_results_structure(self):
        """Test that benchmark results have correct structure."""
        results = benchmark.benchmark_all_backends((256, 256))

        # Validate structure
        assert isinstance(results, dict)
        assert "version" in results
        assert "timestamp" in results
        assert "grid_size" in results
        assert "platform" in results
        assert "results" in results
        assert "fastest" in results

        # Validate platform info
        assert "system" in results["platform"]
        assert "processor" in results["platform"]
        assert "python_version" in results["platform"]

        # Validate results for each backend
        for _backend_name, data in results["results"].items():
            assert "time_ms" in data
            assert "speedup" in data
            assert "available" in data

            if data["available"]:
                assert data["time_ms"] is not None
                assert data["time_ms"] > 0
                assert data["speedup"] is not None
                assert data["speedup"] > 0
