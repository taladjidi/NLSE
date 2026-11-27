"""The CPU split step in three passes over the field, on threads of its own.

``scipy.fft.fft2`` reads and writes the whole field once per axis, and the
propagator and the nonlinear step are a pass each on top. Done by blocks small
enough to stay in cache, the step is three passes: the forward transform of
each row; then, per band of columns, the column transforms with the propagator
applied between them; then the inverse transform of each row with the
nonlinear step on its way out. 1.17x a step at 4096x4096 complex64 and 1.25x
at complex128 on an idle M3 Max, and slower at 1024x1024, which is why the
block and the threads are measured per machine
(``NLSE/backends/cpu_tuning.py``, ``docs/optimization-log.md``).

Every piece releases the GIL: scipy's 1D transforms with ``workers=1`` and
serial ``nogil`` numba kernels, run on a pool of Python threads that each take
a static share of rows or columns. One step runs at a time per ``BlockedStep``:
its scratch is its own, and it already takes every thread it was measured on.
"""

import math
import mmap
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from itertools import pairwise

import numba
import numpy as np
from scipy import fft as scipy_fft

from .cpu import _square_mod_nl_prop_span, _square_mod_nl_prop_v_span

# B, the huge page of x86-64 and of arm64 on 4 KiB pages, where the scratch starts:
# https://docs.kernel.org/admin-guide/mm/transhuge.html
HUGE_PAGE = 2**21
ANONYMOUS = -1  # the fileno mmap takes for memory backed by no file


@numba.njit(nogil=True, cache=True, boundscheck=False)
def _gather_columns(field: np.ndarray, start: int, block: np.ndarray) -> None:
    """Copy columns ``start`` on of ``field`` into the rows of ``block``.

    Parameters
    ----------
    field : np.ndarray
        The field, ``(NY, NX)``.
    start : int
        First column to copy.
    block : np.ndarray
        ``(columns, NY)``, overwritten: ``block[j, i] = field[i, start + j]``.
    """
    for i in range(field.shape[0]):
        for j in range(block.shape[0]):
            block[j, i] = field[i, start + j]


@numba.njit(
    nogil=True, error_model="numpy", fastmath=True, cache=True, boundscheck=False
)
def _multiply_rows(block: np.ndarray, factor: np.ndarray) -> None:
    """Multiply each row of ``block`` by ``factor``, element by element.

    Parameters
    ----------
    block : np.ndarray
        ``(columns, NY)``, modified in place.
    factor : np.ndarray
        ``(NY,)``: the propagator's factor along y, which carries 1/N.
    """
    for j in range(block.shape[0]):
        for i in range(block.shape[1]):
            block[j, i] *= factor[i]


@numba.njit(
    nogil=True, error_model="numpy", fastmath=True, cache=True, boundscheck=False
)
def _scatter_columns(
    block: np.ndarray, field: np.ndarray, start: int, factor: np.ndarray
) -> None:
    """Write the rows of ``block`` back as columns ``start`` on of ``field``.

    Parameters
    ----------
    block : np.ndarray
        ``(columns, NY)``, as ``_gather_columns`` filled it.
    field : np.ndarray
        The field, ``(NY, NX)``, overwritten on those columns.
    start : int
        First column to write.
    factor : np.ndarray
        ``(NX,)``: the propagator's factor along x, applied on the way.
    """
    for i in range(field.shape[0]):
        for j in range(block.shape[0]):
            field[i, start + j] = block[j, i] * factor[start + j]


def _transform_rows(part: np.ndarray, forward: bool) -> None:
    """Transform each row of ``part`` in place, on the calling thread.

    The inverse is unnormalized (``norm="forward"``): the propagator's factor
    along y carries the 1/N.

    Parameters
    ----------
    part : np.ndarray
        C-contiguous rows, overwritten with their transform.
    forward : bool
        True for the forward transform, False for the inverse.
    """
    if forward:
        result = scipy_fft.fft(part, axis=-1, overwrite_x=True, workers=1)
    else:
        result = scipy_fft.ifft(
            part, axis=-1, overwrite_x=True, workers=1, norm="forward"
        )
    # overwrite_x allows scipy to write in place, without promising it.
    if not np.may_share_memory(result, part):
        part[...] = result


def _shares(n: int, parts: int, multiple: int) -> list[tuple[int, int]]:
    """Cut ``range(n)`` into at most ``parts`` runs, each a whole number of blocks.

    Parameters
    ----------
    n : int
        Length to cut.
    parts : int
        Runs wanted, one per thread.
    multiple : int
        Block length every cut but the last falls on a multiple of.

    Returns
    -------
    list of tuple of int
        ``(start, stop)`` of each non-empty run, in order, covering ``range(n)``.
    """
    cuts = [
        min(n, -(-(n * k // parts) // multiple) * multiple) for k in range(parts + 1)
    ]
    return [(start, stop) for start, stop in pairwise(cuts) if stop > start]


@cache
def _thread_pool(threads: int) -> ThreadPoolExecutor:
    """Return the process's pool of ``threads`` threads, made on first use.

    Parameters
    ----------
    threads : int
        Threads in the pool.

    Returns
    -------
    ThreadPoolExecutor
        The pool, shared by every blocked step of that many threads.
    """
    return ThreadPoolExecutor(threads, thread_name_prefix="NLSE-blocked")


# A forked child inherits the pools without their threads: work sent to one
# would wait forever, so the child starts pools of its own.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_thread_pool.cache_clear)
else:
    pass  # no fork on this platform (Windows), nothing to forget


def _private_mapping(size: int) -> mmap.mmap:
    """Return ``size`` B of fresh anonymous memory, in huge pages where granted.

    Private wherever a process forks: mmap's default is shared, and a child
    would then write the scratch its parent is stepping on.

    Parameters
    ----------
    size : int
        Bytes to map.

    Returns
    -------
    mmap.mmap
        The mapping, unmapped once nothing views it.
    """
    if sys.platform == "win32":
        return mmap.mmap(ANONYMOUS, size)  # shared, with no fork to share it over
    mapped = mmap.mmap(ANONYMOUS, size, access=mmap.ACCESS_COPY)
    if hasattr(mmap, "MADV_HUGEPAGE"):
        try:
            mapped.madvise(mmap.MADV_HUGEPAGE)
        except OSError:
            pass  # a kernel built without transparent huge pages: plain ones
    else:
        pass  # macOS: no huge page to ask for
    return mapped


def _scratch(count: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Return ``count`` blocks of ``shape`` in a mapping of their own.

    Not ``np.empty``: malloc hands back pages in the order earlier arrays left
    them, and a block the size of a cache then misses in it, 27 to 41 ms a step
    from one process to the next (``docs/optimization-log.md``).

    Parameters
    ----------
    count : int
        Blocks, one per thread.
    shape : tuple of int
        Shape of each.
    dtype : np.dtype
        Their dtype.

    Returns
    -------
    np.ndarray
        ``(count, *shape)`` of zeros, starting on a ``HUGE_PAGE`` boundary.
    """
    size = count * math.prod(shape) * np.dtype(dtype).itemsize
    # Whole huge pages, and one more to start on a boundary wherever it lands.
    mapped = _private_mapping(-(-size // HUGE_PAGE) * HUGE_PAGE + HUGE_PAGE)
    whole = np.frombuffer(mapped, np.uint8)
    start = -whole.ctypes.data % HUGE_PAGE
    return whole[start : start + size].view(dtype).reshape(count, *shape)


def _forward_rows(field: np.ndarray, start: int, stop: int, rows: int) -> None:
    """First pass: the forward transform of rows ``start`` to ``stop``.

    Parameters
    ----------
    field : np.ndarray
        The field, ``(NY, NX)``, modified in place.
    start : int
        First row.
    stop : int
        One past the last row.
    rows : int
        Rows transformed per call.
    """
    for first in range(start, stop, rows):
        _transform_rows(field[first : min(first + rows, stop)], True)


def _column_blocks(
    field: np.ndarray,
    start: int,
    stop: int,
    block: np.ndarray,
    row_factor: np.ndarray,
    column_factor: np.ndarray,
) -> None:
    """Second pass: columns ``start`` to ``stop`` transformed and propagated.

    Each band of ``len(block)`` columns is copied into ``block`` as rows,
    transformed forward, multiplied by the factor along y, transformed back,
    and written back multiplied by the factor along x.

    Parameters
    ----------
    field : np.ndarray
        The field, rows already transformed, modified in place.
    start : int
        First column.
    stop : int
        One past the last column.
    block : np.ndarray
        This thread's scratch, ``(columns per band, NY)``.
    row_factor : np.ndarray
        ``(NY,)``, the propagator's factor along y, carrying 1/N.
    column_factor : np.ndarray
        ``(NX,)``, the propagator's factor along x.
    """
    for first in range(start, stop, block.shape[0]):
        band = block[: min(block.shape[0], stop - first)]
        _gather_columns(field, first, band)
        _transform_rows(band, True)
        _multiply_rows(band, row_factor)
        _transform_rows(band, False)
        _scatter_columns(band, field, first, column_factor)


def _inverse_rows_nonlinear(
    field: np.ndarray,
    V: np.ndarray | None,
    start: int,
    stop: int,
    rows: int,
    nonlinear: tuple,
) -> None:
    """Third pass: inverse transform of rows ``start`` to ``stop``, then the nonlinear step.

    Parameters
    ----------
    field : np.ndarray
        The field, propagated in Fourier space, modified in place.
    V : np.ndarray or None
        Pre-scaled potential of the field's shape, or None.
    start : int
        First row.
    stop : int
        One past the last row.
    rows : int
        Rows per call.
    nonlinear : tuple
        ``(dz, alpha, g, Isat)`` as floats, for the span kernels.
    """
    width = field.shape[1]
    field_flat = field.ravel()
    for first in range(start, stop, rows):
        last = min(first + rows, stop)
        _transform_rows(field[first:last], False)
        if V is None:
            _square_mod_nl_prop_span(
                field_flat, first * width, last * width, *nonlinear
            )
        else:
            _square_mod_nl_prop_v_span(
                field_flat, V.ravel(), first * width, last * width, *nonlinear
            )


def _wait(futures: list[Future]) -> None:
    """Wait for every share of a pass, then raise the first error one raised.

    Every share is waited for, error or not: one still writing the field
    would overlap the next step, the caller's lock released under it.

    Parameters
    ----------
    futures : list of Future
        One per share.
    """
    errors = [future.exception() for future in futures]
    first = next((error for error in errors if error is not None), None)
    if first is not None:
        raise first


class BlockedStep:
    """A linear-then-nonlinear split step on one grid, in three blocked passes.

    Parameters
    ----------
    shape : tuple of int
        ``(NY, NX)`` of the field.
    dtype : np.dtype
        Complex dtype of the field.
    rows : int
        Rows per block in the row passes, and columns per band in the column
        pass.
    threads : int
        Python threads the passes are shared over.
    """

    def __init__(self, shape: tuple, dtype: np.dtype, rows: int, threads: int) -> None:
        ny, nx = shape
        self.rows = rows
        self.threads = threads
        self._row_shares = _shares(ny, threads, rows)
        self._column_shares = _shares(nx, threads, rows)
        self._blocks = list(_scratch(len(self._column_shares), (rows, ny), dtype))
        self._lock = threading.Lock()

    def __call__(
        self,
        field: np.ndarray,
        row_factor: np.ndarray,
        column_factor: np.ndarray,
        V: np.ndarray | None,
        nonlinear: tuple,
    ) -> np.ndarray:
        """Take the step in place, after any other thread's step on this one.

        Parameters
        ----------
        field : np.ndarray
            C-contiguous ``(NY, NX)`` field, modified in place.
        row_factor : np.ndarray
            ``(NY,)``, the propagator's factor along y, carrying 1/N.
        column_factor : np.ndarray
            ``(NX,)``, the propagator's factor along x.
        V : np.ndarray or None
            Pre-scaled potential of the field's shape, or None.
        nonlinear : tuple
            ``(dz, alpha, g, Isat)`` as floats.

        Returns
        -------
        np.ndarray
            The field.
        """
        # Asked here, not kept: after a fork the cached pool is a new one.
        submit = _thread_pool(self.threads).submit
        with self._lock:
            _wait(
                [
                    submit(_forward_rows, field, a, b, self.rows)
                    for a, b in self._row_shares
                ]
            )
            _wait(
                [
                    submit(
                        _column_blocks, field, a, b, block, row_factor, column_factor
                    )
                    for (a, b), block in zip(self._column_shares, self._blocks)
                ]
            )
            _wait(
                [
                    submit(
                        _inverse_rows_nonlinear, field, V, a, b, self.rows, nonlinear
                    )
                    for a, b in self._row_shares
                ]
            )
        return field
