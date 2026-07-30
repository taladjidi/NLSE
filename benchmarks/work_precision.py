#!/usr/bin/env python3
"""What each method costs to reach a given accuracy.

    python benchmarks/work_precision.py
    python benchmarks/work_precision.py --backends CPU CUPY --size 256

profile_backends.py asks how fast a step is. This asks the question that
decides which method to use: how much wall clock it takes to reach a given
error. A method that is twice the cost per step but converges an order faster
wins as soon as the accuracy asked for is tight enough, and the crossing point
is the only thing worth knowing about it.

Every run solves the same problem over the same distance and is scored against
one converged reference, so the numbers on a row are comparable. The step is
set from the phase the medium imprints per step rather than in metres, because
that is the quantity every limit in the solver is written against and the only
one that means the same thing across problems.

The reference is complex128 with the phase per step 200x smaller than the
finest measured point, and it is checked against a still finer one: a
work-precision table drawn against an unconverged reference measures the
reference.
"""

import argparse
import time

import numpy as np
from NLSE import NLSE

# The medium. A self-focusing beam strong enough that the nonlinearity, not
# the diffraction, sets the step.
WAIST = 2.23e-3
WINDOW = 4 * WAIST
PHYSICS = {
    "alpha": 0,
    "power": 4.0,
    "window": WINDOW,
    "n2": -1.6e-9,
    "V": None,
    "L": 5e-3,
    "Isat": 1e6,
}

# Phase per step to sweep. The solver's own default is 0.1 rad and its
# ceiling is pi.
PHASES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5)

METHODS = (
    ("split_step", "single"),
    ("split_step", "double"),
    ("RK4", "single"),
)


def field(n, dtype=np.complex64):
    """Return a Gaussian beam on an n x n grid."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, n)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / WAIST**2).astype(dtype)


_SOLVERS = {}


def solver(backend, n, dtype):
    """Return the solver for this backend and grid, built once.

    Reused across the sweep because a fresh one plans its transforms on the
    first propagation, and that plan costs more than the short runs at the
    coarse end of the sweep: timing a new solver each point measured the
    planning and reported the coarsest steps as the slowest.
    """
    key = (backend, n)
    if key not in _SOLVERS:
        _SOLVERS[key] = NLSE(NX=n, NY=n, backend=backend, **PHYSICS)
    return _SOLVERS[key]


def rates_of(simu, A, precision):
    """Return the phase rates, for the field the solver will actually run.

    ``out_field`` normalizes the input to the requested power before it
    propagates, and the rates are a property of that field, not of the array
    handed in. Reading them off the raw input understates them by whatever
    the normalization multiplies by -- here a factor of 5e6, which put every
    requested step past the stability limit and made every run take one step.
    """
    prepared, _ = simu._prepare_output_array(A.copy(), normalize=True)
    simu._precompute_step_constants(simu.V, precision)
    return simu._energy_rates(prepared)


def run(backend, n, dtype, method, precision, phase, repeats=1):
    """Propagate at this phase per step; return the field and the best time."""
    simu = solver(backend, n, dtype)
    A = field(n, dtype)
    rates = rates_of(simu, A, precision)
    # The rate each method's own limit is written against: every term for
    # RK4, which approximates the whole right-hand side; the real-space terms
    # alone for split step, which applies the linear part exactly.
    rate = (
        sum(rates.values())
        if method == "RK4"
        else (rates["potential"] + rates["interaction"])
    )
    delta_z = phase / rate
    # Plan the transforms and warm the caches before the clock starts.
    simu.out_field(
        A.copy(),
        delta_z * 2,
        delta_z=delta_z,
        verbose=False,
        plot=False,
        precision=precision,
        method=method,
    )
    simu._backend.synchronize()
    best = np.inf
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = simu.out_field(
            A.copy(),
            PHYSICS["L"],
            delta_z=delta_z,
            verbose=False,
            plot=False,
            precision=precision,
            method=method,
        )
        simu._backend.synchronize()
        best = min(best, time.perf_counter() - start)
        out = np.asarray(simu._backend.to_numpy(out))
    return out, best, int(np.ceil(PHYSICS["L"] / delta_z))


def error(got, reference):
    """Relative L2 difference between two fields."""
    return float(
        np.linalg.norm(got.astype(np.complex128) - reference)
        / np.linalg.norm(reference)
    )


def reference_field(backend, n):
    """Return a converged solution, checked against a finer one."""
    coarse, _, steps = run(backend, n, np.complex128, "split_step", "double", 2.5e-3)
    finer, _, _ = run(backend, n, np.complex128, "split_step", "double", 1.25e-3)
    drift = error(coarse, finer)
    print(f"  reference: {steps} steps, self-consistent to {drift:.2e}")
    if drift > 1e-9:
        print("  WARNING: the reference is not converged; read nothing below")
    return finer


def main(argv=None):
    """Measure and print the work-precision table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="*", default=["CPU"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    for backend in args.backends:
        print(f"\n=== {backend}, {args.size}x{args.size}, complex64 ===")
        reference = reference_field(backend, args.size)
        print(
            f"  {'method':<20} {'rad/step':>9} {'steps':>7} {'time':>9} {'rel err':>10}"
        )
        for method, precision in METHODS:
            for phase in PHASES:
                try:
                    got, seconds, steps = run(
                        backend,
                        args.size,
                        np.complex64,
                        method,
                        precision,
                        phase,
                        args.repeats,
                    )
                except Exception as exc:  # a step past a stability limit
                    print(
                        f"  {method + '/' + precision:<20} {phase:9.3f} "
                        f"{'-':>7} {'-':>9}   {type(exc).__name__}"
                    )
                    continue
                name = f"{method}/{precision}"
                print(
                    f"  {name:<20} {phase:9.3f} {steps:7d} "
                    f"{seconds * 1e3:8.1f}ms {error(got, reference):10.2e}"
                )


if __name__ == "__main__":
    main()
