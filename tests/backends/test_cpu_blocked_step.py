"""The CPU's blocked split step, the gate to it, and the cache of its tuning."""

import json
import mmap
import multiprocessing
import operator
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial

import numba
import numpy as np
import pytest
import scipy
from helpers import WINDOW, make
from NLSE import GPE, NLSE
from NLSE.backends import cpu_tuning, tune_cpu_step
from NLSE.backends.cpu import CPUBackend
from NLSE.kernels import cpu as kernels, cpu_blocked
from NLSE.kernels.cpu_blocked import BlockedStep, _shares, _wait
from NLSE.utils import get_blocking_cache_path
from scipy.constants import c, epsilon_0

NY, NX = 48, 40  # a grid no tested block divides, rows and columns told apart
# (rows, threads): one row on one thread, blocks dividing neither axis over an
# uneven share, blocks of a power of two, and one block of the whole field.
SPLITS = [(1, 1), (7, 3), (16, 4), (NY, 1)]
DZ, G = 0.3, 0.5  # step, and the Kerr coefficient: ~0.6 rad where |A| = 2
LOSSY = 0.05  # u = 2 alpha dz = 0.03, the solved branch
FROZEN = 1.0  # u = 0.6, past _LOSS_SOLVED_LIMIT
ISAT = 2.0
# Largest difference from the complex128 step, in eps of the field's width
# times max|A|: measured 4.2 (complex64) and 5.4 (complex128) over SPLITS and
# NONLINEAR; a skipped block or a factor missed is ~1e6 times more.
EPS_BOUND = 30
ROWS, THREADS = 5, 3  # the step fixed_entry names: no axis below is a multiple
# Threads the cache tests run as having, whatever the machine has: candidates
# at 256^2 and 512^2 complex64, and THREADS fits.
TEST_THREADS = 4
N = 32  # solver grid of the end-to-end tests
SOLVER_STEPS = 3
# Each solver with its step, inside its step limits: NLSE in m; GPE in s, with
# tests/solvers/test_gpe.py's atom number, interaction and step.
SOLVERS = {
    "NLSE": (NLSE, {}, 1e-4),
    "GPE": (GPE, {"gamma": 0, "N": 1e6, "g": 1e3 / (1e6 / 1e-3**2)}, 1e-8),
}
# Differences between the blocked and unblocked solver runs, in eps of the
# width times max|A|: measured 6.5 (complex64) and 6.6 (complex128).
SOLVER_EPS_BOUND = 50
STALE = timedelta(days=31)  # older than cpu_tuning.CACHE_LIFETIME
AHEAD = timedelta(days=1)  # a timestamp in the future, by more than clock skew
THREADED_STEPS = 200  # steps each of two threads takes: long enough to overlap
FORK_TIMEOUT = 60  # s a forked child gets for SOLVER_STEPS steps: ~1 s needed
OTHER_OWNER = "another-host 1 1"  # a lock another process holds, as _lock_token
UNREADABLE_LOCK = b"\xff\xfe"  # not UTF-8: no token this library wrote
SLOW_SHARE = 0.05  # s a share still runs after another has failed
PARENT_MARK, CHILD_MARK = 1, 2  # what each side of a fork fills its scratch with
# (fastest cores the OS reports, counts measured on ALL threads): an M3 Max's
# 12, none, all of them, more than there are, and an M2 Pro's 8 = ALL // 2.
ALL = 16
REPORTED = [
    (12, [16, 8, 12]),
    (None, [16, 8]),
    (16, [16, 8]),
    (20, [16, 8]),
    (8, [16, 8]),
]
SYSCTL_TIMEOUT = 10  # s for `sysctl -n`, which answers in milliseconds
# s a step of each candidate takes: the fastest neither first nor last.
CANDIDATE_SECONDS = {"slow": 3.0, "fast": 1.0, "middling": 2.0}
# s a step of each takes, round by round: the best of "early" is not its last.
ROUND_SECONDS = {"steady": (3.0, 3.0, 2.0), "early": (1.0, 5.0, 5.0)}


def unit_phases(rng, n):
    """Return ``n`` complex numbers of modulus 1 and random phase."""
    return np.exp(2j * np.pi * rng.random(n))


def problem(dtype, potential):
    """Return a random field, factors carrying 1/N, and a potential.

    Parameters
    ----------
    dtype : np.dtype
        Complex width.
    potential : str
        "none", "real" or "complex".

    Returns
    -------
    tuple
        ``(field, rows, columns, V)``.
    """
    rng = np.random.default_rng(0)
    field = (rng.normal(size=(NY, NX)) + 1j * rng.normal(size=(NY, NX))).astype(dtype)
    rows = (unit_phases(rng, NY) / (NY * NX)).astype(dtype)
    columns = unit_phases(rng, NX).astype(dtype)
    real = np.finfo(dtype).dtype
    V = {
        "none": None,
        "real": rng.normal(size=(NY, NX)).astype(real),
        "complex": (rng.normal(size=(NY, NX)) + 0.1j).astype(dtype),
    }[potential]
    return field, rows, columns, V


def unblocked_step(field, rows, columns, V, nonlinear):
    """Return the step in complex128, as the solver takes it unblocked.

    Parameters
    ----------
    field, rows, columns, V : np.ndarray
        From ``problem``.
    nonlinear : tuple
        ``(dz, alpha, g, Isat)``.

    Returns
    -------
    np.ndarray
        The field after the step.
    """
    wide = np.fft.fft2(field.astype(np.complex128))
    wide *= rows.astype(np.complex128)[:, None] * columns.astype(np.complex128)
    wide = np.ascontiguousarray(np.fft.ifft2(wide, norm="forward"))
    if V is None:
        return kernels.square_mod_nl_prop(wide, *nonlinear)
    wide_V = V.astype(np.complex128 if np.iscomplexobj(V) else np.float64)
    return kernels.square_mod_nl_prop_v(wide, wide_V, *nonlinear)


NONLINEAR = {
    "Kerr": ("none", (DZ, 0.0, G, np.inf)),
    "saturable, lossy": ("none", (DZ, LOSSY, G, ISAT)),
    "too lossy to solve": ("none", (DZ, FROZEN, G, ISAT)),
    "real potential": ("real", (DZ, LOSSY, G, ISAT)),
    "complex potential": ("complex", (DZ, 0.0, G, np.inf)),
}


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("split", SPLITS, ids=[f"{r}x{t}" for r, t in SPLITS])
@pytest.mark.parametrize("case", sorted(NONLINEAR))
def test_the_blocked_step_is_the_unblocked_step(dtype, split, case):
    """Blocked and unblocked steps agree to rounding at every block and share."""
    potential, nonlinear = NONLINEAR[case]
    field, rows, columns, V = problem(dtype, potential)
    expected = unblocked_step(field, rows, columns, V, nonlinear)
    BlockedStep(field.shape, np.dtype(dtype), *split)(
        field, rows, columns, V, nonlinear
    )
    error = np.abs(field - expected).max() / np.abs(expected).max()
    assert error <= EPS_BOUND * np.finfo(dtype).eps, f"{error:.2e} from complex128"


def test_a_plane_wave_turns_by_its_factors_and_its_intensity():
    """A Fourier mode comes out the same mode, at the closed form's amplitude."""
    rng = np.random.default_rng(1)
    p, q = 5, 3
    y, x = np.mgrid[:NY, :NX]
    mode = np.exp(2j * np.pi * (p * y / NY + q * x / NX))
    rows = unit_phases(rng, NY) * (1 + rng.random(NY)) / (NY * NX)
    columns = unit_phases(rng, NX) * (1 + rng.random(NX))
    field = 0.5 * mode
    amplitude = 0.5 * rows[p] * NY * NX * columns[q]
    expected = amplitude * mode * np.exp(1j * DZ * G * abs(amplitude) ** 2)
    nonlinear = (DZ, 0.0, G, np.inf)
    BlockedStep(field.shape, field.dtype, 7, 3)(field, rows, columns, None, nonlinear)
    assert np.abs(field - expected).max() <= 1e-13, "not the mode's closed form"


@pytest.mark.parametrize(
    "n,parts,multiple", [(48, 3, 7), (40, 4, 16), (5, 8, 1), (7, 2, 10), (64, 16, 4)]
)
def test_shares_cover_the_range_on_block_boundaries(n, parts, multiple):
    """In order, without gap or overlap, cut on multiples of the block.

    Catches a share dropped or repeated, and a cut inside a block.
    """
    shares = _shares(n, parts, multiple)
    covered = [i for start, stop in shares for i in range(start, stop)]
    assert covered == list(range(n)), "not every index exactly once"
    assert len(shares) <= parts, "more shares than threads"
    assert all(start % multiple == 0 for start, _ in shares), "cut inside a block"


def test_the_parallel_kernel_is_its_serial_body():
    """To the bit, on a field that is no whole number of spans.

    Catches a span skipped or cut short in the prange driver.
    """
    rng = np.random.default_rng(2)
    shape = (3, kernels._SPAN + 905)
    field = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    serial = field.copy().ravel()
    kernels._square_mod_nl_prop_span(serial, 0, serial.size, DZ, LOSSY, G, ISAT)
    kernels._square_mod_nl_prop(field, DZ, LOSSY, G, ISAT)
    assert np.array_equal(field.ravel(), serial), "an element stepped otherwise"


class AlwaysBlocked:
    """A plan whose grid is blocked, ROWS by ROWS on THREADS threads."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.step = BlockedStep(shape, np.dtype(dtype), ROWS, THREADS)

    def blocked_step(self):
        """Return the step."""
        return self.step


class NeverBlocked:
    """A plan whose grid is not blocked."""

    def __init__(self, shape):
        self.shape = shape

    def blocked_step(self):
        """Return None."""
        return None


def declined_arguments(case):
    """Return hook arguments the blocked step cannot take, as ``case`` says.

    Parameters
    ----------
    case : str
        Which argument is wrong.

    Returns
    -------
    tuple
        ``(A, plan, V, alpha, factors)``.
    """
    field, rows, columns, V = problem(np.complex64, "real")
    plan = AlwaysBlocked(field.shape, field.dtype)
    alpha, factors = 0.0, (rows, columns)
    if case == "batch":
        # Its first axis fits the row factor, and no potential to decline it.
        field, V = np.stack([field] * NY), None
    elif case == "Fortran order":
        field = np.asfortranarray(field)
    elif case == "read-only":
        field.flags.writeable = False
    elif case == "potential of another shape":
        V = np.ascontiguousarray(V[:, :1])
    elif case == "a loss per member":
        alpha = np.array([0.0, 0.1])
    elif case == "no factors":
        factors = None
    elif case == "factors swapped":
        factors = (columns, rows)
    elif case == "plan not blocked":
        plan = NeverBlocked(field.shape)
    elif case == "a field of another grid":
        # Wider than the plan's, all else fitting it: a step taken is wrong.
        field, V = (np.concatenate([a, a], axis=1) for a in (field, V))
        factors = (rows, np.concatenate([columns, columns]))
    else:
        raise ValueError(case)
    return field, plan, V, alpha, factors


@pytest.mark.parametrize(
    "case",
    [
        "batch",
        "Fortran order",
        "read-only",
        "potential of another shape",
        "a loss per member",
        "no factors",
        "factors swapped",
        "plan not blocked",
        "a field of another grid",
    ],
)
def test_the_hook_declines_and_leaves_the_field(case):
    """The hook declines, leaving A as it was, when the blocked step cannot run."""
    A, plan, V, alpha, factors = declined_arguments(case)
    before = A.copy()
    taken = kernels.linear_nonlinear_step(
        A, None, plan, V, DZ, alpha, G, ISAT, factors=lambda: factors
    )
    assert taken is False, "took a step it cannot compute"
    assert np.array_equal(A, before), "touched the field it declined"


def test_the_hook_takes_one_field_with_its_factors():
    """True, and the field stepped as the blocked step steps it."""
    A, rows, columns, V = problem(np.complex64, "real")
    expected = A.copy()
    plan = AlwaysBlocked(A.shape, A.dtype)
    plan.step(expected, rows, columns, V, (DZ, 0.0, G, ISAT))
    taken = kernels.linear_nonlinear_step(
        A, None, plan, V, DZ, 0.0, G, ISAT, factors=lambda: (rows, columns)
    )
    assert taken is True, "declined a step it can take"
    assert np.array_equal(A, expected), "not the blocked step"


def test_a_step_the_plan_declines_does_not_ask_for_the_factors():
    """Building them costs a pass over the dispersion: not for a step not taken.

    Catches every unblocked grid paying for the factors at its first step.
    """
    A, _, _, V = problem(np.complex64, "real")
    asked = []
    taken = kernels.linear_nonlinear_step(
        A, None, NeverBlocked(A.shape), V, DZ, 0.0, G, ISAT, factors=asked.append
    )
    assert taken is False and not asked


def test_a_failed_share_is_raised_once_every_share_is_done():
    """Catches a step returning while a share of it still writes the field."""
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(operator.truediv, 1, 0),
            pool.submit(time.sleep, SLOW_SHARE),
        ]
        with pytest.raises(ZeroDivisionError):
            _wait(futures)
        assert all(future.done() for future in futures), "a share still runs"


def scratch_owner(block):
    """Return the object whose memory ``block`` is a view of, None for numpy's own."""
    while isinstance(block, np.ndarray):
        block = block.base
    return getattr(block, "obj", None)


def test_the_scratch_is_one_mapping_of_its_own():
    """Catches scratch from malloc: on recycled pages a cache-sized block misses."""
    blocks = BlockedStep((NY, NX), np.dtype(np.complex64), ROWS, THREADS)._blocks
    starts = [block.ctypes.data for block in blocks]
    assert isinstance(scratch_owner(blocks[0]), mmap.mmap)
    assert np.diff(starts).tolist() == [blocks[0].nbytes] * (len(blocks) - 1)
    assert all(block.shape == (ROWS, NY) and block.flags.writeable for block in blocks)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
@pytest.mark.filterwarnings("ignore:This process")  # fork of a threaded process
def test_a_forked_child_writes_scratch_of_its_own():
    """Catches a shared mapping, mmap's default: parent and child on one scratch."""
    block = BlockedStep((NY, NX), np.dtype(np.complex64), ROWS, THREADS)._blocks[0]
    block[...] = PARENT_MARK
    context = multiprocessing.get_context("fork")
    child = context.Process(target=block.fill, args=(CHILD_MARK,))
    child.start()
    child.join(FORK_TIMEOUT)
    assert child.exitcode == 0, "the forked child failed"
    assert (block == PARENT_MARK).all()


# ── The tuning and its cache ─────────────────────────────────────────────────

MEASURED: list = []  # (shape, dtype) of every measurement fixed_entry stood in for


class CountingStep(BlockedStep):
    """A BlockedStep that counts the steps taken through it."""

    taken: list = []

    def __call__(self, *args):
        """Count, then step."""
        CountingStep.taken.append(self.rows)
        return super().__call__(*args)


def entry_with(**changes):
    """Return a usable entry for ROWS on THREADS, with some fields changed."""
    entry = {
        "rows": ROWS,
        "threads": THREADS,
        "unblocked_ms": 2.0,
        "blocked_ms": 1.0,
        "timestamp": datetime.now().isoformat(),
    }
    return {**entry, **changes}


def fixed_entry(backend, shape, dtype, threads):
    """Stand in for ``cpu_tuning._measure``: always ROWS on THREADS."""
    MEASURED.append((tuple(shape), np.dtype(dtype).name))
    return entry_with()


@pytest.fixture
def blocking(monkeypatch):
    """Blocking on, every measurement ``fixed_entry``, every step counted."""
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "1")
    monkeypatch.setenv("NLSE_QUIET", "1")
    monkeypatch.setattr(cpu_tuning, "available_threads", lambda: TEST_THREADS)
    monkeypatch.setattr(cpu_tuning, "_measure", fixed_entry)
    monkeypatch.setattr(cpu_tuning, "BlockedStep", CountingStep)
    MEASURED.clear()
    CountingStep.taken.clear()


def settle(shape=(NY, NX), dtype=np.complex64):
    """Return what the CPU plan of a grid answers, as a solver would ask it."""
    plan = CPUBackend().build_fft(shape, (-2, -1), np.dtype(dtype))[0]
    return plan.blocked_step()


def key_of(shape=(NY, NX), dtype=np.complex64, threads=TEST_THREADS, host=None):
    """Return the cache key of a grid, on another host if ``host`` says so.

    ``host`` is ``(position, value)`` in what ``cpu_tuning._host`` returns.
    """
    parts = list(cpu_tuning._host())
    if host is not None:
        parts[host[0]] = host[1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cpu_tuning, "_host", lambda: tuple(parts))
        return cpu_tuning._cache_key(shape, np.dtype(dtype), threads)


def lock_file():
    """Return the lock a tuning process holds."""
    return get_blocking_cache_path().with_name(cpu_tuning.LOCK_NAME)


def usable(entry):
    """Whether the cache tests' grid may use an entry."""
    return cpu_tuning._usable(entry, (NY, NX), TEST_THREADS)


def write_cache(content):
    """Write the cache file as given: a str verbatim, anything else as JSON."""
    text = content if isinstance(content, str) else json.dumps(content)
    get_blocking_cache_path().write_text(text)


def read_cache():
    """Return the decoded cache file."""
    return json.loads(get_blocking_cache_path().read_text())


def cached(entry, key=None):
    """Return a cache file holding one entry."""
    return {"version": cpu_tuning.CACHE_VERSION, "entries": {key or key_of(): entry}}


def test_a_grid_is_measured_once_then_read_back(blocking):
    """Measured on first use and written; read back, not measured, after.

    Catches a cache that is written and never read, or never written.
    """
    step = settle()
    assert isinstance(step, BlockedStep) and (step.rows, step.threads) == (
        ROWS,
        THREADS,
    )
    assert MEASURED == [((NY, NX), "complex64")]
    assert read_cache()["entries"][key_of()]["rows"] == ROWS
    assert not lock_file().exists(), "the lock was left behind"
    cpu_tuning._SETTLED.clear()  # as a new process
    assert settle().rows == ROWS
    assert len(MEASURED) == 1, "measured again with a fresh entry cached"


OTHER_KEYS = {
    "another grid": {"shape": (NX, NY)},
    "another dtype": {"dtype": np.complex128},
    "another thread count": {"threads": TEST_THREADS + 1},
    "another processor": {"host": (2, "another-cpu")},
    "another scipy": {"host": (3, "0.0")},
    "another numba": {"host": (4, "0.0")},
}


@pytest.mark.parametrize("other", sorted(OTHER_KEYS))
def test_an_entry_for_another_key_is_not_read(blocking, other):
    """An entry under another key is kept, and the grid is measured anew."""
    others = key_of(**OTHER_KEYS[other])
    assert others != key_of()
    write_cache(cached(entry_with(rows=None, threads=None), others))
    assert settle().rows == ROWS, "used the entry of another key"
    assert len(MEASURED) == 1
    assert set(read_cache()["entries"]) == {others, key_of()}


@pytest.mark.parametrize("cpus,numba_threads", [(16, 4), (3, 16)])
def test_the_threads_are_the_fewer_of_the_cpus_and_numba_s(
    monkeypatch, cpus, numba_threads
):
    """``NUMBA_NUM_THREADS`` caps the step's threads, as the CPUs do.

    Catches a key and candidates for threads numba was told not to use.
    """
    monkeypatch.setattr(cpu_tuning, "_usable_cpus", lambda: cpus)
    monkeypatch.setattr(numba, "get_num_threads", lambda: numba_threads)
    assert cpu_tuning.available_threads() == min(cpus, numba_threads)


CORRUPT = {
    "not JSON": "{",
    "a list": [],
    "another version": "version",
    "a version 1 cache, before the fastest cores' count": "version 1",
    "entries not a mapping": {"version": cpu_tuning.CACHE_VERSION, "entries": []},
    "rows a string": "rows",
    "rows zero": "zero",
    "threads missing": "threads",
    "no timestamp": "timestamp",
    "stale": "stale",
    "time zone": "zone",
    "in the future": "future",
    "rows past the grid": "past",
    "more threads than usable": "threads above",
}


def corrupt(name):
    """Return the cache content ``CORRUPT[name]`` stands for."""
    content = CORRUPT[name]
    entries = {
        "rows": entry_with(rows="8"),
        "zero": entry_with(rows=0),
        "threads": entry_with(threads=None),
        "timestamp": entry_with(timestamp=None),
        "stale": entry_with(timestamp=(datetime.now() - STALE).isoformat()),
        "zone": entry_with(timestamp=datetime.now(timezone.utc).isoformat()),
        "future": entry_with(timestamp=(datetime.now() + AHEAD).isoformat()),
        # Each fine on its own; together more rows than the grid's threads hold.
        "past": entry_with(rows=NX // THREADS + 1),
        "threads above": entry_with(rows=1, threads=TEST_THREADS + 1),
    }
    if content == "version":
        return {**cached(entry_with()), "version": "0"}
    if content == "version 1":
        return {**cached(entry_with()), "version": "1"}
    if isinstance(content, str) and content in entries:
        return cached(entries[content])
    return content


@pytest.mark.parametrize("name", sorted(CORRUPT))
def test_a_corrupt_cache_is_measured_over(blocking, name):
    """Measured again, and a usable file written in its place.

    Catches a crash on a bad file, and trust in a bad or stale entry.
    """
    write_cache(corrupt(name))
    MEASURED.clear()
    assert settle().rows == ROWS
    assert len(MEASURED) == 1, "trusted the cache"
    assert usable(read_cache()["entries"][key_of()])


def test_while_another_process_tunes_this_one_runs_unblocked(blocking):
    """A fresh lock: no measurement, no file, the lock left to its owner.

    Catches two processes timing each other's load.
    """
    lock_file().touch()
    assert settle() is None
    assert tune_cpu_step((NX, NY)) is None
    assert not MEASURED
    assert not get_blocking_cache_path().exists()
    assert lock_file().exists(), "took another process's lock"


def test_a_grid_another_process_was_tuning_is_asked_for_again(blocking):
    """A grid tuned elsewhere runs unblocked until BUSY_RETRY, then settles."""
    lock_file().touch()
    assert settle() is None
    lock_file().unlink()
    assert settle() is None and not MEASURED, "asked again before BUSY_RETRY"
    for key in cpu_tuning._BUSY_UNTIL:  # as BUSY_RETRY later
        cpu_tuning._BUSY_UNTIL[key] -= cpu_tuning.BUSY_RETRY
    assert settle().rows == ROWS
    assert len(MEASURED) == 1


def test_a_busy_grid_is_announced_once(blocking, monkeypatch, capsys):
    """Catches the notice repeated at every retry while the other process tunes."""
    monkeypatch.delenv("NLSE_QUIET")
    lock_file().touch()
    assert settle() is None
    for key in cpu_tuning._BUSY_UNTIL:  # as BUSY_RETRY later, the lock still held
        cpu_tuning._BUSY_UNTIL[key] -= cpu_tuning.BUSY_RETRY
    assert settle() is None
    assert capsys.readouterr().out.count("Another process is tuning") == 1


def test_a_plan_of_a_real_array_is_not_blocked(blocking):
    """Catches the blocked step, which transforms complex rows, offered one."""
    assert settle(dtype=np.float64) is None
    assert not MEASURED


def test_a_first_run_says_it_is_tuning(blocking, monkeypatch, capsys):
    """Catches a first step that stalls for seconds without a word."""
    monkeypatch.delenv("NLSE_QUIET")
    settle()
    assert "Tuning the CPU step for" in capsys.readouterr().out


def test_a_patch_release_keeps_its_entries():
    """The key names scipy and numba by major.minor: no tuning per bug fix."""
    assert cpu_tuning._release("1.16.2") == "1.16"
    *_, scipy_release, numba_release = cpu_tuning._host()
    assert scipy_release == cpu_tuning._release(scipy.__version__)
    assert numba_release == cpu_tuning._release(numba.__version__)


def test_a_stale_lock_is_taken_over(blocking):
    """A lock older than LOCK_STALE_AFTER was left by a dead process.

    Catches a crashed tuning that keeps every later process unblocked.
    """
    lock_file().touch()
    then = time.time() - cpu_tuning.LOCK_STALE_AFTER - 1
    os.utime(lock_file(), (then, then))
    assert settle().rows == ROWS
    assert len(MEASURED) == 1
    assert not lock_file().exists(), "the lock was left behind"


def measured_while_taken_over(backend, shape, dtype, threads):
    """Measure as a tuning that outlived LOCK_STALE_AFTER and lost its lock."""
    lock_file().unlink()
    lock_file().write_text(OTHER_OWNER)
    return fixed_entry(backend, shape, dtype, threads)


def test_a_lock_taken_over_is_left_to_its_new_owner(blocking, monkeypatch):
    """Catches a late tuning that removes the lock another process now holds."""
    monkeypatch.setattr(cpu_tuning, "_measure", measured_while_taken_over)
    assert settle().rows == ROWS
    assert lock_file().exists(), "removed the lock of the process that took it over"
    assert lock_file().read_text() == OTHER_OWNER


def test_an_unreadable_lock_is_left_alone(blocking):
    """Catches a release that raises out of the tuning's ``finally``."""
    lock_file().write_bytes(UNREADABLE_LOCK)
    cpu_tuning._release_lock(lock_file())
    assert lock_file().read_bytes() == UNREADABLE_LOCK


def exit_by_busy_grids():
    """In a forked child: exit 0 if it waits for no grid, 1 if it inherited one."""
    os._exit(1 if cpu_tuning._BUSY_UNTIL else 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
@pytest.mark.filterwarnings("ignore:This process")  # fork of a threaded process
def test_a_forked_child_does_not_inherit_a_wait(monkeypatch):
    """Catches a child that waits on the parent's deadline for a grid."""
    monkeypatch.setitem(cpu_tuning._BUSY_UNTIL, ("a grid",), time.monotonic() + 1)
    child = multiprocessing.get_context("fork").Process(target=exit_by_busy_grids)
    child.start()
    child.join(FORK_TIMEOUT)
    assert child.exitcode == 0, "the child inherited the parent's deadlines"


def test_an_unwritable_cache_warns_and_measures(
    blocking, monkeypatch, tmp_path, capsys
):
    """An unwritable cache warns, naming NLSE_CACHE_DIR; the grid is still tuned."""
    blocker = tmp_path / "a file"
    blocker.write_text("")
    monkeypatch.setenv("NLSE_CACHE_DIR", str(blocker / "cache"))
    monkeypatch.delenv("NLSE_QUIET")
    with pytest.warns(UserWarning, match="NLSE_CACHE_DIR"):
        assert settle().rows == ROWS
    assert len(MEASURED) == 1
    assert "Invalid" not in capsys.readouterr().out


def test_the_switch_off_neither_measures_nor_blocks(blocking, monkeypatch):
    """NLSE_CPU_BLOCKING=0: no step, no measurement, no file."""
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "0")
    assert settle() is None
    assert not MEASURED
    assert not get_blocking_cache_path().exists()


def test_forcing_measures_over_a_fresh_entry_once(blocking, monkeypatch):
    """NLSE_FORCE_TUNING=1: measured once in the process, cache or not."""
    write_cache(cached(entry_with(rows=None, threads=None)))
    monkeypatch.setenv("NLSE_FORCE_TUNING", "1")
    assert settle().rows == ROWS
    assert settle().rows == ROWS
    assert len(MEASURED) == 1


def test_without_tuning_on_first_run_the_utility_does_it(blocking, monkeypatch):
    """Unmeasured grids run unblocked until tune_cpu_step measures them.

    Catches tune_cpu_step leaving the process with the answer it had.
    """
    monkeypatch.setattr(cpu_tuning, "TUNE_ON_FIRST_RUN", False)
    assert settle() is None
    assert not MEASURED
    assert tune_cpu_step((NY, NX))["rows"] == ROWS
    assert settle().rows == ROWS


def scripted_time(blocked, run, arguments):
    """Stand in for ``_time_per_step``: 1 s unblocked, ``blocked`` s blocked."""
    return 1.0 if run is cpu_tuning._unblocked_steps else blocked


@pytest.mark.parametrize(
    "blocked,chosen", [(0.85, True), (0.9, True), (0.92, False), (1.1, False)]
)
def test_the_blocked_step_is_chosen_only_by_the_margin(monkeypatch, blocked, chosen):
    """Faster by more than MARGIN, or not chosen: 0.92 wins by 8.7%.

    Catches the comparison turned round, and the margin dropped or halved.
    """
    monkeypatch.setattr(cpu_tuning, "_time_per_step", partial(scripted_time, blocked))
    shape, dtype = (512, 512), np.dtype(np.complex64)
    assert cpu_tuning._candidates(shape, dtype, TEST_THREADS)
    entry = cpu_tuning._measure(CPUBackend(), shape, dtype, TEST_THREADS)
    assert (entry["rows"] is not None) is chosen
    assert entry["blocked_ms"] == pytest.approx(blocked * 1e3)


def named_time(run, arguments):
    """Stand in for ``_time_per_step``: what CANDIDATE_SECONDS says of ``run``."""
    return CANDIDATE_SECONDS[run]


def test_the_candidate_ranked_first_is_the_fastest(monkeypatch):
    """Catches the slowest block size, or simply the first, going to the duel."""
    monkeypatch.setattr(cpu_tuning, "_time_per_step", named_time)
    first = cpu_tuning._fastest([(name, ()) for name in CANDIDATE_SECONDS])
    assert list(CANDIDATE_SECONDS)[first] == "fast"


def round_time(taken, run, arguments):
    """Stand in for ``_time_per_step``: the next of ROUND_SECONDS for ``run``."""
    taken.append(run)
    return ROUND_SECONDS[run][taken.count(run) - 1]


def test_a_candidate_is_ranked_by_its_best_round(monkeypatch):
    """Catches a ranking by the last round, when the machine was busy then."""
    monkeypatch.setattr(cpu_tuning, "_time_per_step", partial(round_time, []))
    monkeypatch.setattr(cpu_tuning, "ROUNDS", len(ROUND_SECONDS["early"]))
    first = cpu_tuning._fastest([(name, ()) for name in ROUND_SECONDS])
    assert list(ROUND_SECONDS)[first] == "early"


def recorded_time(order, run, arguments):
    """Stand in for ``_time_per_step``: note which run was timed, 1 s each."""
    order.append(run)
    return 1.0


def test_the_duel_times_both_runs_as_often_and_in_turn(monkeypatch):
    """The duel times each side DUEL_ROUNDS times, alternating who goes first."""
    order = []
    monkeypatch.setattr(cpu_tuning, "_time_per_step", partial(recorded_time, order))
    cpu_tuning._duel(("plain", ()), ("blocked", ()))
    assert order.count("plain") == order.count("blocked") == cpu_tuning.DUEL_ROUNDS
    first = [("plain", "blocked")[i % 2] for i in range(cpu_tuning.DUEL_ROUNDS)]
    assert order[::2] == first


def test_a_real_measurement_is_a_usable_entry(monkeypatch):
    """Timed for real on a small grid, candidates and all: an entry that agrees.

    Catches a measurement that fails on a real step.
    """
    monkeypatch.setenv("NLSE_QUIET", "1")
    monkeypatch.setattr(cpu_tuning, "available_threads", lambda: TEST_THREADS)
    shape, dtype = (256, 256), np.dtype(np.complex64)
    assert cpu_tuning._candidates(shape, dtype, TEST_THREADS), "nothing blocked"
    entry = tune_cpu_step(shape, dtype)
    assert cpu_tuning._usable(entry, shape, TEST_THREADS)
    if entry["rows"] is not None:
        assert entry["blocked_ms"] * (1 + cpu_tuning.MARGIN) < entry["unblocked_ms"]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_candidates_are_the_block_sizes_every_thread_gets_one_of(dtype):
    """Each a BLOCK_BYTES block, no larger than a share of the grid."""
    shape, threads = (512, 256), 8
    width = np.dtype(dtype)
    candidates = cpu_tuning._candidates(shape, width, threads)
    sizes = {max(1, b // (shape[0] * width.itemsize)) for b in cpu_tuning.BLOCK_BYTES}
    assert candidates, "nothing to measure"
    assert all(rows in sizes for rows, _ in candidates)
    assert all(rows * count <= min(shape) for rows, count in candidates)
    assert {count for _, count in candidates} <= set(cpu_tuning._thread_counts(threads))


@pytest.mark.parametrize("reported,counts", REPORTED)
def test_the_counts_are_the_divisors_then_the_fastest_cores(
    monkeypatch, reported, counts
):
    """All threads first, so a tie keeps them, as before the fastest cores.

    Catches the fastest cores' count left out, or past the threads usable.
    """
    monkeypatch.setattr(cpu_tuning, "_performance_threads", lambda: reported)
    assert cpu_tuning._thread_counts(ALL) == counts


def test_one_thread_is_measured_on_one():
    """Catches a count of 0 from the divisors."""
    assert cpu_tuning._thread_counts(1) == [1]


def sysctl_count():
    """Return what ``sysctl -n`` says of the fastest cores, or None if it cannot."""
    tool = shutil.which("sysctl")
    if tool is None:
        return None
    name = cpu_tuning.PERFORMANCE_CPUS_SYSCTL.decode()
    done = subprocess.run(
        [tool, "-n", name], capture_output=True, text=True, timeout=SYSCTL_TIMEOUT
    )
    return int(done.stdout) if done.returncode == 0 else None


def test_the_fastest_cores_are_what_sysctl_says():
    """On macOS the count ``sysctl -n`` prints, where it prints one; else None.

    Catches a read that always fails, and a status or value read wrongly.
    """
    if sys.platform != "darwin":
        assert cpu_tuning._performance_threads() is None
        return
    expected = sysctl_count()
    if expected is None:
        pytest.skip("sysctl knows no performance levels here")
    assert cpu_tuning._performance_threads() == expected


def test_a_name_the_os_lacks_reads_as_none(monkeypatch):
    """MacOS 11, or a Mac of one kind of core: no count, not a crash."""
    monkeypatch.setattr(cpu_tuning, "PERFORMANCE_CPUS_SYSCTL", b"hw.no.such.name")
    cpu_tuning._performance_threads.cache_clear()
    try:
        assert cpu_tuning._performance_threads() is None
    finally:
        cpu_tuning._performance_threads.cache_clear()


def record_pool(asked, threads):
    """Stand in for ``cpu_blocked._thread_pool``: note the count, return the pool."""
    asked.append(threads)
    return REAL_POOL(threads)


REAL_POOL = cpu_blocked._thread_pool


def test_the_fastest_cores_count_is_measured_and_run_on(blocking, monkeypatch):
    """A candidate on that count; an entry of it runs a pool of it, numba untouched.

    Catches the count left out of the candidates, and a step that runs on
    another count than the one cached (or leaves numba's count changed).
    """
    monkeypatch.setattr(cpu_tuning, "_performance_threads", lambda: THREADS)
    width = np.dtype(np.complex64)
    candidates = cpu_tuning._candidates((512, 512), width, TEST_THREADS)
    assert THREADS in {count for _, count in candidates}, "not measured"
    write_cache(cached(entry_with()))
    step = settle()
    asked: list = []
    monkeypatch.setattr(cpu_blocked, "_thread_pool", partial(record_pool, asked))
    before = numba.get_num_threads()
    field, rows, columns, _ = problem(np.complex64, "none")
    step(field, rows, columns, None, (DZ, 0.0, G, np.inf))
    assert asked == [THREADS], "ran on another count"
    assert len(step._row_shares) == len(step._column_shares) == THREADS
    assert numba.get_num_threads() == before, "numba's count changed"


@pytest.mark.parametrize(
    "shape,dtype", [((4, 4, 4), np.complex64), ((0, 8), np.complex64), ((8, 8), float)]
)
def test_the_utility_refuses_what_it_cannot_tune(shape, dtype):
    """A 3D grid, an empty one, a real dtype."""
    with pytest.raises(ValueError):
        tune_cpu_step(shape, dtype)


# ── Through the solver ───────────────────────────────────────────────────────


def beam(simu, dtype, batch=None):
    """Return a Gaussian with a tilt on the solver's grid."""
    field = np.exp(-(simu.XX**2 + simu.YY**2) / (WINDOW / 4) ** 2)
    field = (field * np.exp(1j * simu.XX / WINDOW * 20)).astype(dtype)
    return field if batch is None else np.broadcast_to(field, (batch, *field.shape))


def run(solver, dtype, splitting, batch=None, steps=SOLVER_STEPS):
    """Propagate ``steps`` steps of a SOLVERS entry and return the field."""
    cls, physics, delta_z = SOLVERS[solver]
    simu = make(cls, "CPU", n=N, **physics)
    return np.asarray(
        simu.out_field(
            beam(simu, dtype, batch).copy(),
            steps * delta_z,
            delta_z=delta_z,
            splitting=splitting,
            verbose=False,
            plot=False,
        )
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("splitting", ["lie", "strang", "yoshida"])
@pytest.mark.parametrize("solver", sorted(SOLVERS))
@pytest.mark.filterwarnings("ignore:splitting=")
def test_the_solver_steps_blocked_to_the_unblocked_result(
    blocking, monkeypatch, solver, splitting, dtype
):
    """Each splitting takes the blocked step, and lands where it did unblocked.

    Catches the gate left shut for CPU, and a step handed the wrong half.
    """
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "0")
    expected = run(solver, dtype, splitting)
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "1")
    got = run(solver, dtype, splitting)
    assert len(CountingStep.taken) >= SOLVER_STEPS, "the blocked step was not taken"
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error <= SOLVER_EPS_BOUND * np.finfo(dtype).eps, f"{error:.2e}"


def test_a_batch_runs_unblocked_and_unchanged(blocking, monkeypatch):
    """A batch is declined: to the bit what it was, no blocked step taken."""
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "0")
    expected = run("NLSE", np.complex64, "lie", batch=3)
    monkeypatch.setenv("NLSE_CPU_BLOCKING", "1")
    got = run("NLSE", np.complex64, "lie", batch=3)
    assert not CountingStep.taken, "took the blocked step on a batch"
    assert np.array_equal(got, expected)


def openmp_vendor():
    """Return the vendor of numba's OpenMP runtime, None on another layer.

    Numba chooses its layer on the first parallel call: ask after one.
    """
    if numba.threading_layer() != "omp":
        return None
    from numba.np.ufunc import omppool  # imports only where OpenMP is found

    return omppool.openmp_vendor


def run_into(results, index, barrier):
    """Run THREADED_STEPS NLSE steps after ``barrier``, into ``results[index]``."""
    barrier.wait()
    results[index] = run("NLSE", np.complex64, "lie", steps=THREADED_STEPS)


def test_two_threads_step_as_they_would_one_after_the_other(blocking):
    """Two runs at once, on one settled step, give the runs one at a time.

    Catches the step's scratch shared between two calls at once.
    """
    expected = run("NLSE", np.complex64, "lie", steps=THREADED_STEPS)
    if numba.threading_layer() == "workqueue":
        pytest.skip("numba's workqueue layer aborts when two threads enter it")
    results = [None, None]
    barrier = threading.Barrier(len(results))
    threads = [
        threading.Thread(target=run_into, args=(results, i, barrier))
        for i in range(len(results))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert CountingStep.taken, "the blocked step was not taken"
    for got in results:
        assert np.array_equal(got, expected), "a run was changed by the other"


def child_steps(expected):
    """In a forked child: the blocked step taken, to the parent's result."""
    CountingStep.taken.clear()
    got = run("NLSE", np.complex64, "lie")
    assert CountingStep.taken, "the blocked step was not taken in the child"
    assert np.array_equal(got, expected)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
@pytest.mark.filterwarnings("ignore:This process")  # fork of a threaded process
def test_a_child_forked_mid_step_steps_blocked(blocking):
    """A child forked while a thread is mid-step still steps blocked."""
    expected = run("NLSE", np.complex64, "lie")
    if openmp_vendor() == "GNU":
        # https://numba.readthedocs.io/en/stable/user/threading-layer.html
        pytest.skip("numba ends a forked child that enters GNU OpenMP")
    step = settle((N, N))
    context = multiprocessing.get_context("fork")
    with step._lock:  # as a thread of the parent stepping when it forks
        child = context.Process(target=child_steps, args=(expected,))
        child.start()
    child.join(FORK_TIMEOUT)
    hung = child.is_alive()
    if hung:
        child.kill()
        child.join()
    else:
        pass  # finished: its exit code says how
    assert not hung, "the forked child hung"
    assert child.exitcode == 0, "the forked child failed"


PHYSICAL_N = 256  # grid of the closed-form tests, as tests/physics/
PHYSICAL_WINDOW = 4e-3  # m
PHYSICAL_WAIST = 200e-6  # m


def linear_solver(**overrides):
    """Return a lossless 2D NLSE with the closed-form tests' grid."""
    base = {
        "alpha": 0,
        "power": 1,
        "window": PHYSICAL_WINDOW,
        "n2": 0,
        "V": None,
        "L": 1e-2,
        "NX": PHYSICAL_N,
        "NY": PHYSICAL_N,
        "Isat": 1e30,
        "backend": "CPU",
    }
    return NLSE(**{**base, **overrides})


@pytest.mark.filterwarnings("ignore:delta_z=")  # the default step, capped
def test_a_blocked_gaussian_beam_diffracts_as_theory_says(blocking):
    """A blocked Gaussian beam has ``w = w0 sqrt(2)`` at one Rayleigh length."""
    simu = linear_solver()
    field = np.exp(-(simu.XX**2 + simu.YY**2) / PHYSICAL_WAIST**2)
    z_rayleigh = simu.k * PHYSICAL_WAIST**2 / 2
    out = np.asarray(
        simu.out_field(
            field.astype(np.complex64),
            z_rayleigh,
            verbose=False,
            plot=False,
            normalize=False,
        )
    )
    intensity = np.abs(out) ** 2
    r_sq = simu.XX**2 + simu.YY**2
    waist = float(np.sqrt(2 * np.sum(intensity * r_sq) / np.sum(intensity)))
    assert CountingStep.taken, "the blocked step was not taken"
    assert waist == pytest.approx(PHYSICAL_WAIST * np.sqrt(2), rel=2e-3)


def test_a_blocked_plane_wave_turns_by_its_saturated_kerr_phase(blocking):
    """``phi = k n2 I z / (1 + I/Isat)``, as in tests/physics/, at Isat = I."""
    n2, z, step = -1e-9, 5e-3, 1e-4
    probe = linear_solver(NX=64, NY=64)
    intensity = 1.0 / (probe.k * abs(n2) * z)  # about one radian of phase
    amplitude = float(np.sqrt(2 * intensity / (c * epsilon_0)))
    simu = linear_solver(NX=64, NY=64, n2=n2, Isat=intensity)
    out = np.asarray(
        simu.out_field(
            (amplitude * np.ones((64, 64))).astype(np.complex64),
            z,
            delta_z=step,
            verbose=False,
            plot=False,
            normalize=False,
        )
    )
    expected = simu.k * n2 * intensity * z / 2
    assert CountingStep.taken, "the blocked step was not taken"
    assert float(np.angle(out[32, 32])) == pytest.approx(expected, rel=1e-3)
