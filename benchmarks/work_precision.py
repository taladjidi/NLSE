#!/usr/bin/env python3
"""What each method costs to reach a given accuracy.

    python benchmarks/work_precision.py
    python benchmarks/work_precision.py --backends CPU CUPY --size 256
    python benchmarks/work_precision.py --problem turbulence --backends CUPY

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

The reference is complex128 at a quarter of the finest measured step, and it is
checked against the half: a work-precision table drawn against an unconverged
reference measures the reference. The check is printed, and nothing in the table
means anything below it.

**The reference shares the grid.** What is measured is the error of the step,
not of the spatial discretization, so a problem whose fine structure the grid
cannot hold is still measured self-consistently -- and its table says nothing
about whether that grid is enough.

Two problems, because they ask different questions. ``beam`` is smooth and
lossless, where the ranking is decided by order alone. ``turbulence`` is
examples/fig2_turbulence.py: lossy, carrying a potential, and starting with
spectral content of its own. That is the case where a fourth-order splitting
with a backward sub-step and a Runge-Kutta method with a stability limit are
each asked something the smooth problem never asks them.
"""

import argparse
import time
import warnings
from collections import namedtuple

import numpy as np
from NLSE import NLSE

# A problem: what to build the solver with, the field to launch, the potential
# to hang on the solver afterwards (as the example does, since the constructor
# takes V but the grid it is written on comes from the solver), and which method
# to draw the reference with.
#
# The reference method is per problem because no method is the most converged
# one everywhere. On a lossless problem the splittings are the ones to trust:
# their real-space step is exact there, since a pure phase rotation leaves
# |A|^2 alone, and Strang reaches 1e-9 where RK4 stalls near 1e-7. Add loss and
# that reverses -- the sub-step freezes |A|^2 while the amplitude decays inside
# it, which costs *every* splitting its order and leaves them first order,
# while RK4 integrates the whole right-hand side and keeps its fourth. Drawn
# with a first-order reference, a table of first-order methods reports the
# reference's error and calls it theirs.
Problem = namedtuple("Problem", "physics field potential reference")

# The medium. A self-focusing beam strong enough that the nonlinearity, not
# the diffraction, sets the step.
WAIST = 2.23e-3
WINDOW = 4 * WAIST
BEAM_PHYSICS = {
    "alpha": 0,
    "power": 4.0,
    "window": WINDOW,
    "n2": -1.6e-9,
    "V": None,
    "L": 5e-3,
    "Isat": 1e6,
}

# examples/fig2_turbulence.py, at whatever grid is asked for. The loss is the
# point of it here: 20 per metre over 20 cm, which takes the amplitude down by
# 7.4x, and which a backward sub-step multiplies back up.
TURBULENCE_PHYSICS = {
    "alpha": 20,
    "power": 1.05,
    "window": 8e-3,
    "n2": -1.6e-9,
    "V": None,
    "L": 20e-2,
    "Isat": 10e4,
}
TURBULENCE_WAIST = 2e-3
TURBULENCE_DEFECT = 1e-3
TURBULENCE_KP = 2 * np.pi * 5e3

# Phase per step to sweep. The solver's own default is 0.1 rad and its
# ceiling is pi.
PHASES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5)

METHODS = (
    ("split_step", "lie"),
    ("split_step", "strang"),
    ("split_step", "yoshida"),
    ("RK4", "lie"),
)


def beam_field(simu, dtype=np.complex64):
    """Return a Gaussian beam on this solver's grid."""
    x = np.linspace(-WINDOW / 2, WINDOW / 2, simu.NX)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / WAIST**2).astype(dtype)


def turbulence_field(simu, dtype=np.complex64):
    """Return the example's field: a Gaussian, its halves tilted apart.

    The two counter-tilted halves are what makes this problem turbulent, and
    they are also spectral content the step criterion does not count -- it
    weighs the potential and the interaction, not the kinetic term the tilt
    loads.
    """
    E = np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / TURBULENCE_WAIST**2).astype(dtype)
    half = simu.NY // 2
    E[:half, :] *= np.exp(1j * TURBULENCE_KP * simu.XX[:half, :])
    E[half:, :] *= np.exp(-1j * TURBULENCE_KP * simu.XX[half:, :])
    return E


def turbulence_potential(simu):
    """Return the example's defect, on this solver's grid."""
    return 1e-4 * np.exp(-(np.hypot(simu.XX, simu.YY) ** 2) / TURBULENCE_DEFECT**2)


PROBLEMS = {
    "beam": Problem(BEAM_PHYSICS, beam_field, None, ("split_step", "strang")),
    "turbulence": Problem(
        TURBULENCE_PHYSICS, turbulence_field, turbulence_potential, ("RK4", "lie")
    ),
}

# Chosen once from the command line and read everywhere below, because it is a
# property of the whole table rather than of any row in it.
PROBLEM = PROBLEMS["beam"]

_SOLVERS = {}


def solver(backend, n, dtype):
    """Return the solver for this backend and grid, built once.

    Reused across the sweep because a fresh one plans its transforms on the
    first propagation, and that plan costs more than the short runs at the
    coarse end of the sweep: timing a new solver each point measured the
    planning and reported the coarsest steps as the slowest.
    """
    key = (backend, n, id(PROBLEM))
    if key not in _SOLVERS:
        simu = NLSE(NX=n, NY=n, backend=backend, **PROBLEM.physics)
        if PROBLEM.potential is not None:
            simu.V = PROBLEM.potential(simu)
        _SOLVERS[key] = simu
    return _SOLVERS[key]


def rates_of(simu, A, splitting):
    """Return the phase rates, for the field the solver will actually run.

    ``out_field`` normalizes the input to the requested power before it
    propagates, and the rates are a property of that field, not of the array
    handed in. Reading them off the raw input understates them by whatever
    the normalization multiplies by -- here a factor of 5e6, which put every
    requested step past the stability limit and made every run take one step.
    """
    prepared, _ = simu._prepare_output_array(A.copy(), normalize=True)
    simu._precompute_step_constants(simu.V, simu._field_dtype(A))
    return simu._energy_rates(prepared)


def run(backend, n, dtype, method, splitting, phase, repeats=1):
    """Propagate at this phase per step; return the field and the best time."""
    warnings.simplefilter("ignore")
    simu = solver(backend, n, dtype)
    A = PROBLEM.field(simu, dtype)
    rates = rates_of(simu, A, splitting)
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
        splitting=splitting,
        method=method,
    )
    simu._backend.synchronize()
    best = np.inf
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = simu.out_field(
            A.copy(),
            PROBLEM.physics["L"],
            delta_z=delta_z,
            verbose=False,
            plot=False,
            splitting=splitting,
            method=method,
        )
        simu._backend.synchronize()
        best = min(best, time.perf_counter() - start)
        out = np.asarray(simu._backend.to_numpy(out))
    return out, best, int(np.ceil(PROBLEM.physics["L"] / delta_z))


def error(got, reference):
    """Relative L2 difference between two fields."""
    return float(
        np.linalg.norm(got.astype(np.complex128) - reference)
        / np.linalg.norm(reference)
    )


def reference_field(backend, n):
    """Return a converged solution, checked against a finer one."""
    method, splitting = PROBLEM.reference
    coarse, _, steps = run(backend, n, np.complex128, method, splitting, 2.5e-3)
    finer, _, _ = run(backend, n, np.complex128, method, splitting, 1.25e-3)
    drift = error(coarse, finer)
    print(
        f"  reference: {method}/{splitting}, {steps} steps, "
        f"self-consistent to {drift:.2e}"
    )
    if drift > 1e-9:
        print("  WARNING: the reference is not converged; read nothing below")
    return finer


def main(argv=None):
    """Measure and print the work-splitting table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="*", default=["CPU"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--field",
        default="complex64",
        choices=["complex64", "complex128"],
        help="float width of the run, which is not the splitting",
    )
    parser.add_argument(
        "--problem",
        default="beam",
        choices=sorted(PROBLEMS),
        help="which medium and field to draw the table for",
    )
    args = parser.parse_args(argv)

    global PROBLEM
    PROBLEM = PROBLEMS[args.problem]

    for backend in args.backends:
        field_dtype = getattr(np, args.field)
        print(
            f"\n=== {args.problem}, {backend}, {args.size}x{args.size}, "
            f"{args.field} ==="
        )
        reference = reference_field(backend, args.size)
        print(
            f"  {'method':<20} {'rad/step':>9} {'steps':>7} {'time':>9} {'rel err':>10}"
        )
        for method, splitting in METHODS:
            for phase in PHASES:
                try:
                    got, seconds, steps = run(
                        backend,
                        args.size,
                        field_dtype,
                        method,
                        splitting,
                        phase,
                        args.repeats,
                    )
                except Exception as exc:  # a step past a stability limit
                    print(
                        f"  {method + '/' + splitting:<20} {phase:9.3f} "
                        f"{'-':>7} {'-':>9}   {type(exc).__name__}"
                    )
                    continue
                name = f"{method}/{splitting}"
                print(
                    f"  {name:<20} {phase:9.3f} {steps:7d} "
                    f"{seconds * 1e3:8.1f}ms {error(got, reference):10.2e}"
                )


if __name__ == "__main__":
    main()
