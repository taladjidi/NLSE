"""Measure, once per machine and grid, whether the CPU step should be blocked.

The blocked step (``NLSE/kernels/cpu_blocked.py``) wins or loses by how its
blocks sit in this processor's caches: on a Ryzen 3 4100 the best block was
256-512 KiB and 1 MiB lost, on an M3 Max 1 MiB was best, and at 1024x1024 it
lost to the unblocked step on the Mac at every block. So neither the block nor
whether to block is a constant. Both are measured against the unblocked step
the first time a grid runs, as the "auto" backend is, and kept in
``cpu_blocking.json`` in the cache directory, one entry per processor, grid,
dtype, thread count and scipy and numba release.

``NLSE_CPU_BLOCKING=0`` never blocks and never measures; ``NLSE_FORCE_TUNING=1``
measures again, once per grid in the process, whatever the cache holds. One
process measures at a time, through a lock file beside the cache; the others
take the unblocked step meanwhile rather than time each other's load.
"""

import json
import os
import socket
import threading
import time
import warnings
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy
from llvmlite import binding as llvm

from ..kernels.cpu_blocked import BlockedStep
from ..utils import (
    get_blocking_cache_path,
    machine_identity,
    say,
    write_json_atomically,
)

# Whether a grid no entry is cached for is measured when it first runs (True),
# or runs unblocked until tune_cpu_step measures it (False). Settable at run
# time: NLSE.backends.cpu_tuning.TUNE_ON_FIRST_RUN = False.
TUNE_ON_FIRST_RUN = True

# Bumped whenever a cached entry stops meaning what it meant.
CACHE_VERSION = "1"

# Age past which an entry is measured again, as the backend benchmark's cache.
CACHE_LIFETIME = timedelta(days=30)

# Bytes of one block, the candidates measured: around the best of the Ryzen
# 3 4100 (256-512 KiB, its L2; 1 MiB 0.58x at 4096^2 complex64) and of the M3
# Max (1 MiB), docs/optimization-log.md, "Three-pass blocked split step".
BLOCK_BYTES = (2**17, 2**18, 2**19, 2**20, 2**21)

# Thread counts measured, as divisors of the usable ones: all, and half, which
# won by 1.17-1.30x on the Ryzen 3 4100, two logical threads sharing one L2
# (same log entry).
THREAD_DIVISORS = (1, 2)

# Rounds of each candidate, interleaved, and steps per round; the best round
# counts. Measured: 4096x4096 complex128 tunes in 12.1-12.5 s on an M3 Max,
# the duel included.
ROUNDS = 3
STEPS = 4

# Rounds the candidate ranked first then takes against the unblocked step,
# alternating which goes first: the best of ~30 windows against the best of 3
# read 1.13x on a loaded M3 Max (1024x1024 complex128) where blocking loses.
DUEL_ROUNDS = 5

# Fraction the blocked step has to win by: what a timing here still says in
# another process. On an M3 Max the plain 1024x1024 complex128 step moves from
# 2.45 to 2.90 ms with what the process ran before, and a 1.08x read there
# was a 10-18% loss in fresh processes.
MARGIN = 0.10

# The measured step: Kerr, lossless, unsaturated, 0.1 rad a step where |A| = 1.
TUNING_DZ = 1.0
TUNING_G = 0.1
TUNING_ALPHA = 0.0
TUNING_ISAT = np.inf

# Phase, in rad, the tuning field and propagator wind through across y and x:
# any phases of modulus 1 keep the field's norm over the steps timed.
TUNING_PHASE_Y = np.pi
TUNING_PHASE_X = 2 * np.pi

MS_PER_S = 1e3

# Parts of a release in the key, major.minor: a release that changes the
# transforms or the kernels' code is measured afresh.
RELEASE_PARTS = 2

# The lock one tuning process holds, beside the cache, and its age in s past
# which it was left by a process that died: 24 times the longest tuning
# measured (4096x4096 complex128, 12.5 s on an M3 Max).
LOCK_NAME = "cpu_blocking.lock"
LOCK_STALE_AFTER = 300.0

# Seconds before a grid another process was tuning is asked for again: the
# tuning of a 2048x2048 grid, 1.7-3.1 s on an M3 Max. Asking reads the cache
# and the lock, 33 us there.
BUSY_RETRY = 3.0

# Steps settled in this process, by cache directory, grid, dtype and threads,
# and when to ask again for those another process was tuning (time.monotonic).
_SETTLED: dict = {}
_BUSY_UNTIL: dict = {}


def _forget_settled() -> None:
    """Forget the steps settled, and the grids deferred, in this process."""
    _SETTLED.clear()
    _BUSY_UNTIL.clear()


# A forked child may inherit a step a thread of the parent was inside, its
# lock held for good, and the parent's deadlines to ask again: it settles its own.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_settled)
else:
    pass  # no fork on this platform (Windows), nothing to forget


def _usable_cpus() -> int:
    """Return the CPUs this process may run on.

    Returns
    -------
    int
        ``os.process_cpu_count()`` where Python has it (3.13), else the
        affinity mask's size where the OS has one, else ``os.cpu_count()``.
    """
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def available_threads() -> int:
    """Return the threads the blocked step may use.

    Returns
    -------
    int
        The CPUs this process may run on, capped by numba's thread count
        (``NUMBA_NUM_THREADS``, ``numba.set_num_threads``).
    """
    return min(_usable_cpus(), numba.get_num_threads())


def _release(version: str) -> str:
    """Return ``major.minor`` of a version string."""
    return ".".join(version.split(".")[:RELEASE_PARTS])


@cache
def _host() -> tuple[str, str, str, str, str]:
    """Return what a timing is a statement about, besides the grid.

    Returns
    -------
    tuple of str
        OS, processor type, LLVM's name for the processor (``apple-m3``,
        ``znver3``), and the scipy and numba releases.
    """
    system, machine = machine_identity()
    return (
        system,
        machine,
        llvm.get_host_cpu_name(),
        _release(scipy.__version__),
        _release(numba.__version__),
    )


def _cache_key(shape: tuple, dtype: np.dtype, threads: int) -> str:
    """Return the entry a grid's measurement is kept under.

    Parameters
    ----------
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``.

    Returns
    -------
    str
        ``system/machine/cpu/NYxNX/dtype/threads/scipy-x.y/numba-x.y``.
    """
    system, machine, cpu, scipy_release, numba_release = _host()
    return (
        f"{system}/{machine}/{cpu}/{shape[0]}x{shape[1]}/{dtype.name}/{threads}"
        f"/scipy-{scipy_release}/numba-{numba_release}"
    )


def _candidates(shape: tuple, dtype: np.dtype, threads: int) -> list[tuple[int, int]]:
    """Return the ``(rows, threads)`` measured: each block size, each thread count.

    A block too large for every thread to get one is left out.

    Parameters
    ----------
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``: no candidate uses more.

    Returns
    -------
    list of tuple of int
        Distinct candidates, in order of thread count then block size.
    """
    ny, nx = shape
    counts = dict.fromkeys(threads // d for d in THREAD_DIVISORS if threads // d > 0)
    found = []
    for count in counts:
        for size in BLOCK_BYTES:
            rows = max(1, size // (ny * dtype.itemsize))
            if rows * count <= min(ny, nx) and (rows, count) not in found:
                found.append((rows, count))
    return found


def _tuning_problem(shape: tuple, dtype: np.dtype) -> tuple:
    """Return a field and a separable propagator to time steps on.

    Unit-modulus phases throughout: the propagator is unitary once the rows
    carry 1/N, so the field keeps its norm however many steps it takes.

    Parameters
    ----------
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    tuple
        ``(field, row_factor, column_factor, propagator)``.
    """
    ny, nx = shape
    along_y = np.exp(1j * np.linspace(0, TUNING_PHASE_Y, ny))
    along_x = np.exp(1j * np.linspace(0, TUNING_PHASE_X, nx))
    field = (along_y[:, None] * along_x[None, :]).astype(dtype)
    row_factor = (along_y / (ny * nx)).astype(dtype)
    column_factor = along_x.astype(dtype)
    propagator = row_factor[:, None] * column_factor[None, :]
    return field, row_factor, column_factor, propagator


def _unblocked_steps(
    backend: Any, field: np.ndarray, propagator: np.ndarray, steps: int
) -> None:
    """Take ``steps`` steps as the solver does where the blocked step declines.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    field : np.ndarray
        Field, overwritten in place.
    propagator : np.ndarray
        Propagator carrying 1/N, of the field's shape.
    steps : int
        Steps to take.
    """
    plans = backend.build_fft(field.shape, axes=(-2, -1), dtype=field.dtype)
    kernels = backend.kernels
    # Rebound, not copied back: scipy works in place, through a new view.
    for _ in range(steps):
        field = backend.fft(field, plans)
        field = kernels.apply_propagator(field, propagator)
        field = backend.ifft(field, plans, normalize=False)
        field = kernels.square_mod_nl_prop(
            field, TUNING_DZ, TUNING_ALPHA, TUNING_G, TUNING_ISAT
        )


def _blocked_steps(
    step: BlockedStep,
    field: np.ndarray,
    row_factor: np.ndarray,
    column_factor: np.ndarray,
    steps: int,
) -> None:
    """Take ``steps`` blocked steps.

    Parameters
    ----------
    step : BlockedStep
        The candidate.
    field : np.ndarray
        Field, overwritten.
    row_factor : np.ndarray
        ``(NY,)``, carrying 1/N.
    column_factor : np.ndarray
        ``(NX,)``.
    steps : int
        Steps to take.
    """
    nonlinear = (TUNING_DZ, TUNING_ALPHA, TUNING_G, TUNING_ISAT)
    for _ in range(steps):
        step(field, row_factor, column_factor, None, nonlinear)


def _time_per_step(run: Any, arguments: tuple) -> float:
    """Return the seconds one step of ``run`` took, over ``STEPS`` steps.

    Parameters
    ----------
    run : callable
        ``_unblocked_steps`` or ``_blocked_steps``.
    arguments : tuple
        Their arguments but the step count.

    Returns
    -------
    float
        Wall time per step, in s.
    """
    start = time.perf_counter()
    run(*arguments, STEPS)
    return (time.perf_counter() - start) / STEPS


def _fastest(runs: list[tuple[Any, tuple]]) -> int:
    """Return the index of the run with the best of ``ROUNDS`` interleaved rounds.

    Parameters
    ----------
    runs : list of tuple
        ``(run, arguments)`` as ``_time_per_step`` takes them; not empty.

    Returns
    -------
    int
        Index of the fastest.
    """
    best = [float("inf")] * len(runs)
    for _ in range(ROUNDS):
        for i, (run, arguments) in enumerate(runs):
            best[i] = min(best[i], _time_per_step(run, arguments))
    return min(range(len(runs)), key=best.__getitem__)


def _duel(first: tuple[Any, tuple], second: tuple[Any, tuple]) -> tuple[float, float]:
    """Return the best time of each of two runs over ``DUEL_ROUNDS`` rounds.

    Both get as many windows, side by side, and take turns going first: what
    else the machine is doing weighs on both alike.

    Parameters
    ----------
    first : tuple
        ``(run, arguments)`` as ``_time_per_step`` takes them.
    second : tuple
        The other.

    Returns
    -------
    tuple of float
        Best wall time per step of each, in s.
    """
    pair = (first, second)
    best = [float("inf"), float("inf")]
    for i in range(DUEL_ROUNDS):
        for side in (0, 1) if i % 2 == 0 else (1, 0):
            best[side] = min(best[side], _time_per_step(*pair[side]))
    return best[0], best[1]


def _measure(backend: Any, shape: tuple, dtype: np.dtype, threads: int) -> dict:
    """Rank the candidates, time the first against the unblocked step, return the entry.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``.

    Returns
    -------
    dict
        ``rows`` and ``threads`` of the step to take, both None for the
        unblocked one; ``unblocked_ms`` and ``blocked_ms``, the best per-step
        times measured; ``timestamp``.
    """
    field, row_factor, column_factor, propagator = _tuning_problem(shape, dtype)
    unblocked: tuple[Any, tuple] = (_unblocked_steps, (backend, field, propagator))
    blocked: list[tuple[Any, tuple]] = [
        (
            _blocked_steps,
            (BlockedStep(shape, dtype, rows, count), field, row_factor, column_factor),
        )
        for rows, count in _candidates(shape, dtype, threads)
    ]
    for run, arguments in [unblocked, *blocked]:
        run(*arguments, 1)  # compiles, starts the pools
    if not blocked:
        plain = min(_time_per_step(*unblocked) for _ in range(ROUNDS))
        return _entry(None, plain, None)
    first = blocked[_fastest(blocked)]
    plain, best = _duel(unblocked, first)
    wins = best * (1 + MARGIN) < plain
    return _entry(first[1][0] if wins else None, plain, best)


def _entry(chosen: BlockedStep | None, plain: float, best: float | None) -> dict:
    """Return the cache entry of one measurement.

    Parameters
    ----------
    chosen : BlockedStep or None
        The step to take; None for the unblocked one.
    plain : float
        Best time per unblocked step, in s.
    best : float or None
        Best time per step of the fastest candidate, in s; None without one.

    Returns
    -------
    dict
        ``rows``, ``threads``, ``unblocked_ms``, ``blocked_ms``, ``timestamp``.
    """
    return {
        "rows": chosen.rows if chosen else None,
        "threads": chosen.threads if chosen else None,
        "unblocked_ms": plain * MS_PER_S,
        "blocked_ms": None if best is None else best * MS_PER_S,
        "timestamp": datetime.now().isoformat(),
    }


def _warn_unwritable(reason: object) -> None:
    """Warn that measurements cannot be kept, so every process measures again.

    Parameters
    ----------
    reason : object
        What the cache directory raised, or why it cannot be written.
    """
    warnings.warn(
        f"The CPU step's tuning cannot be cached ({reason}), so every process "
        "measures it again. Set NLSE_CACHE_DIR to a writable directory.",
        stacklevel=2,
    )


def _cache_path() -> Path | None:
    """Return the cache file's path, or None where its directory cannot be made.

    Returns
    -------
    Path or None
        ``get_blocking_cache_path()``; None after a warning.
    """
    try:
        return get_blocking_cache_path()
    except OSError as e:
        _warn_unwritable(e)
        return None


def _read_entries(path: Path) -> dict:
    """Return the cached entries, or none where the file is missing or unreadable.

    Parameters
    ----------
    path : Path
        The cache file.

    Returns
    -------
    dict
        Entries by cache key.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            cache = json.load(f)
    # OSError: unreadable. ValueError: not JSON. RecursionError: nested too deep.
    except (OSError, ValueError, RecursionError) as e:
        say(f"Warning: Invalid CPU blocking cache, ignoring it: {e}")
        return {}
    if (
        not isinstance(cache, dict)
        or cache.get("version") != CACHE_VERSION
        or not isinstance(cache.get("entries"), dict)
    ):
        say("CPU blocking cache is not a record this version writes, ignoring it.")
        return {}
    return cache["entries"]


def _usable(entry: Any, shape: tuple, threads: int) -> bool:
    """Whether a cached entry is one ``_measure`` could have written, and fresh.

    Parameters
    ----------
    entry : Any
        What the cache holds under a key.
    shape : tuple
        ``(NY, NX)`` of the grid it is read for.
    threads : int
        From ``available_threads``.

    Returns
    -------
    bool
        True if its step can be built from it and run on this grid.
    """
    if not isinstance(entry, dict):
        return False
    rows, count = entry.get("rows"), entry.get("threads")
    unblocked = rows is None and count is None
    blocked = (
        type(rows) is int
        and type(count) is int
        and rows > 0
        and 0 < count <= threads
        and rows * count <= min(shape)
    )
    try:
        age = datetime.now() - datetime.fromisoformat(entry.get("timestamp", ""))
    # TypeError: not a string, or a time zone. ValueError: not a date.
    except (TypeError, ValueError):
        return False
    return (unblocked or blocked) and timedelta(0) <= age <= CACHE_LIFETIME


def _store(path: Path, key: str, entry: dict) -> None:
    """Add an entry to the cache file, keeping the others.

    Parameters
    ----------
    path : Path
        The cache file.
    key : str
        From ``_cache_key``.
    entry : dict
        From ``_measure``.
    """
    entries = _read_entries(path)
    entries[key] = entry
    try:
        write_json_atomically(path, {"version": CACHE_VERSION, "entries": entries})
        say(f"CPU blocking cached to {path}")
    except OSError as e:
        _warn_unwritable(e)


def _lock_token() -> str:
    """Return what a lock made by this thread holds: host, process and thread."""
    return f"{socket.gethostname()} {os.getpid()} {threading.get_ident()}"


def _create_lock(lock: Path) -> bool:
    """Make the lock file, holding ``_lock_token``, unless it exists.

    Parameters
    ----------
    lock : Path
        The lock file.

    Returns
    -------
    bool
        True if this call made it.
    """
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as f:
        f.write(_lock_token())
    return True


def _release_lock(lock: Path) -> None:
    """Remove the lock if it is still the one this thread made.

    A tuning that outlived ``LOCK_STALE_AFTER`` has had its lock taken over;
    removing that one would let a third process measure beside the second.

    Parameters
    ----------
    lock : Path
        The lock file.
    """
    try:
        held = lock.read_text()
    except FileNotFoundError:
        return  # removed by the process that took it over, then released
    except (OSError, ValueError):
        held = None  # unreadable, so not provably ours: called from a finally
    if held == _lock_token():
        # shortcut: a takeover between the read and the unlink is lost; it
        # needs this tuning past LOCK_STALE_AFTER, then a microsecond race.
        lock.unlink(missing_ok=True)
    else:
        say("The CPU tuning lock was taken over meanwhile: leaving it to its owner.")


def _is_stale(lock: Path) -> bool:
    """Whether a lock file is older than ``LOCK_STALE_AFTER``, or gone.

    Parameters
    ----------
    lock : Path
        The lock file.

    Returns
    -------
    bool
        True if it may be replaced.
    """
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return True  # released since it was found
    return age > LOCK_STALE_AFTER


def _may_measure(lock: Path) -> bool:
    """Take the lock, replacing a stale one; False if another process holds it.

    Parameters
    ----------
    lock : Path
        The lock file.

    Returns
    -------
    bool
        True if this process may measure: it holds the lock, or no lock can
        be made in the directory (warned), so there is no one to wait for.
    """
    try:
        if _create_lock(lock):
            return True
        if not _is_stale(lock):
            return False
        # shortcut: two processes replacing one stale lock at once can both
        # measure; it takes a tuning process killed, then a microsecond race.
        lock.unlink(missing_ok=True)
        return _create_lock(lock)
    except OSError as e:
        _warn_unwritable(e)
        return True


def _announce(shape: tuple, dtype: np.dtype, entry: dict) -> None:
    """Say what the measurement chose.

    Parameters
    ----------
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    entry : dict
        From ``_measure``.
    """
    grid = f"{shape[0]}x{shape[1]} {dtype.name}"
    if entry["rows"] is None:
        say(f"CPU step on {grid}: the unblocked step is fastest here.")
    else:
        gain = entry["unblocked_ms"] / entry["blocked_ms"]
        say(
            f"CPU step on {grid}: blocks of {entry['rows']} rows on "
            f"{entry['threads']} threads, {gain:.2f}x."
        )


def _measured(backend: Any, shape: tuple, dtype: np.dtype, threads: int) -> dict:
    """Measure a grid, saying so before and after.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``.

    Returns
    -------
    dict
        From ``_measure``.
    """
    say(
        f"Tuning the CPU step for {shape[0]}x{shape[1]} {dtype.name}, "
        "once for this machine..."
    )
    entry = _measure(backend, shape, dtype, threads)
    _announce(shape, dtype, entry)
    return entry


def _measure_and_store(
    backend: Any, shape: tuple, dtype: np.dtype, threads: int, path: Path | None
) -> dict | None:
    """Measure a grid under the lock and cache the entry.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``.
    path : Path or None
        The cache file; None where its directory cannot be made. There, and
        where the directory cannot be written, the entry is kept for this
        process only.

    Returns
    -------
    dict or None
        The entry; None, measuring nothing, while another process measures.
    """
    if path is None:
        return _measured(backend, shape, dtype, threads)
    if not os.access(path.parent, os.W_OK):
        _warn_unwritable(f"{path.parent} is not writable")
        return _measured(backend, shape, dtype, threads)
    lock = path.with_name(LOCK_NAME)
    if not _may_measure(lock):
        return None
    try:
        entry = _measured(backend, shape, dtype, threads)
        _store(path, _cache_key(shape, dtype, threads), entry)
    finally:
        _release_lock(lock)
    return entry


def _step_for(entry: dict | None, shape: tuple, dtype: np.dtype) -> BlockedStep | None:
    """Return the step an entry names, or None for the unblocked one.

    Parameters
    ----------
    entry : dict or None
        A usable entry, or None where there is none.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    BlockedStep or None
        The step.
    """
    if entry is None or entry["rows"] is None:
        return None
    return BlockedStep(shape, dtype, entry["rows"], entry["threads"])


def _defer(settled: tuple) -> None:
    """Leave a grid another process is tuning unsettled, to ask again later.

    Parameters
    ----------
    settled : tuple
        From ``_settled_key``.
    """
    if settled not in _BUSY_UNTIL:
        say(
            "Another process is tuning the CPU step: this one takes the "
            "unblocked step until it is done."
        )
    _BUSY_UNTIL[settled] = time.monotonic() + BUSY_RETRY


def _settled_key(shape: tuple, dtype: np.dtype, threads: int) -> tuple:
    """Return what ``_SETTLED`` holds a grid's step under, in this process."""
    # The dtype itself: its name is looked up anew, 1.2 us a call (M3 Max).
    return (os.environ.get("NLSE_CACHE_DIR"), tuple(shape), dtype, threads)


def _settle(
    backend: Any,
    shape: tuple,
    dtype: np.dtype,
    threads: int,
    force: bool,
    measure: bool,
) -> dict | None:
    """Return the grid's entry, measuring it where asked to and needed.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    threads : int
        From ``available_threads``.
    force : bool
        Measure even where the cache holds an entry.
    measure : bool
        Measure where it holds none; otherwise return None.

    Returns
    -------
    dict or None
        The entry, whose step is also remembered for the rest of the process;
        None for the unblocked step without an entry. While another process
        measures nothing is remembered, and the grid is asked for again.
    """
    settled = _settled_key(shape, dtype, threads)
    path = _cache_path()
    cached = (
        None
        if force or path is None
        else _read_entries(path).get(_cache_key(shape, dtype, threads))
    )
    if _usable(cached, shape, threads):
        entry = cached
    elif measure:
        entry = _measure_and_store(backend, shape, dtype, threads, path)
        if entry is None:
            _defer(settled)
            return None
    else:
        entry = None
    _SETTLED[settled] = _step_for(entry, shape, dtype)
    return entry


def tune(backend: Any, shape: tuple, dtype: np.dtype, force: bool) -> dict | None:
    """Measure a grid, unless the cache holds a fresh entry for it.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.
    force : bool
        Measure even where the cache holds an entry.

    Returns
    -------
    dict or None
        The entry, as ``_measure`` documents it; None while another process
        measures, and the grid runs unblocked in this one.
    """
    return _settle(backend, shape, dtype, available_threads(), force, measure=True)


def blocked_step(backend: Any, shape: tuple, dtype: np.dtype) -> BlockedStep | None:
    """Return the blocked step for a grid, or None to take the unblocked one.

    Settled once per process per grid, from the cache or by measuring
    (``TUNE_ON_FIRST_RUN``), and looked up after that. A grid another process
    is tuning runs unblocked, and is asked for again every ``BUSY_RETRY``.

    Parameters
    ----------
    backend : Backend
        The CPU backend.
    shape : tuple
        ``(NY, NX)``.
    dtype : np.dtype
        Complex dtype of the field.

    Returns
    -------
    BlockedStep or None
        The step to take.
    """
    if os.environ.get("NLSE_CPU_BLOCKING", "1") == "0":
        return None
    threads = available_threads()
    settled = _settled_key(shape, dtype, threads)
    if settled not in _SETTLED:
        if settled in _BUSY_UNTIL and time.monotonic() < _BUSY_UNTIL[settled]:
            return None
        force = os.environ.get("NLSE_FORCE_TUNING", "0") == "1"
        _settle(backend, shape, dtype, threads, force, TUNE_ON_FIRST_RUN or force)
    return _SETTLED.get(settled)
