"""Time NLSE_3d's Strang step with its half steps merged and unmerged.

One process, the two sides interleaved, the time per step taken as the slope
between two run lengths so that the bracket and the setup drop out.

    python benchmarks/strang_merge_3d.py --backend CPU --nx 128 --nt 128
"""

import argparse
import statistics
import time

import numpy as np
from NLSE import NLSE_3d
from profile_backends import check_step_kept
from scipy.constants import c

WINDOW = 5e-3  # m, transverse
WINDOW_T = 1e-12  # s
WAIST = 1e-3  # m
PULSE_FRACTION = 6  # the pulse's 1/e half-width is the time window over this
LENGTH = 1e-2  # m
ALPHA = 10.0  # 1/m, so that the solved lossy step is in the run
ENERGY = 1e-6  # J
N2 = 1e-20  # m^2/W
ISAT = 1e10  # W/m^2
D0 = 1e-26  # s^2/m
STEPS_IN_LENGTH = 2000  # the step asked is LENGTH over this, under both limits
STEPS_LOW = 20
STEPS_HIGH = 60
ROUNDS = 5


def build(backend, nx, nt, merged):
    """Return a lossy, saturable NLSE_3d that merges its half steps or not."""
    simu = NLSE_3d(
        alpha=ALPHA,
        energy=ENERGY,
        window=[WINDOW, WINDOW, WINDOW_T],
        n2=N2,
        D0=D0,
        vg=c,
        V=None,
        L=LENGTH,
        NX=nx,
        NY=nx,
        NZ=nt,
        Isat=ISAT,
        backend=backend,
    )
    simu._lie_step_is_strang_body = merged
    return simu


def pulse(simu):
    """Return a Gaussian pulse on the solver's grid, in complex64."""
    X, Y, T = np.meshgrid(simu.X, simu.Y, simu.T)
    envelope = np.exp(
        -(X**2 + Y**2) / WAIST**2 - T**2 / (WINDOW_T / PULSE_FRACTION) ** 2
    )
    return envelope.astype(np.complex64)


def run(simu, field, steps):
    """Return the seconds of a Strang run of ``steps`` steps, and its result."""
    delta_z = LENGTH / STEPS_IN_LENGTH
    start = time.perf_counter()
    out = simu.out_field(
        field, steps * delta_z, delta_z=delta_z, splitting="strang", verbose=False
    )
    out = np.asarray(simu._backend.to_numpy(out))  # forces a lazy or queued result
    elapsed = time.perf_counter() - start
    check_step_kept(simu, delta_z)
    return elapsed, out


def per_step_ms(simu, field):
    """Return the milliseconds a step takes, from the slope between two runs."""
    low, _ = run(simu, field, STEPS_LOW)
    high, _ = run(simu, field, STEPS_HIGH)
    return (high - low) / (STEPS_HIGH - STEPS_LOW) * 1e3


def main(argv=None):
    """Print the step time of both sides and how far their results are apart."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--nx", type=int, default=128, help="points along x and y")
    parser.add_argument("--nt", type=int, default=128, help="points along time")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args(argv)

    sides = {
        name: build(args.backend, args.nx, args.nt, merged)
        for name, merged in (("unmerged", False), ("merged", True))
    }
    for name, simu in sides.items():
        if simu._backend.name != args.backend:
            raise RuntimeError(f"asked {args.backend}, got {simu._backend.name}")
        if simu._merges_strang_halves("strang") != (name == "merged"):
            raise RuntimeError(f"the {name} side does not run as its name says")
    field = pulse(sides["merged"])
    results = {name: run(simu, field, STEPS_LOW)[1] for name, simu in sides.items()}
    apart = np.max(np.abs(results["merged"] - results["unmerged"])) / np.max(
        np.abs(results["unmerged"])
    )

    times = {name: [] for name in sides}
    for round_index in range(args.rounds):
        order = list(sides) if round_index % 2 == 0 else list(sides)[::-1]
        for name in order:
            times[name].append(per_step_ms(sides[name], field))
    median = {name: statistics.median(values) for name, values in times.items()}
    spread = {name: max(values) - min(values) for name, values in times.items()}
    print(
        f"{args.backend} {args.nx}x{args.nx}x{args.nt} complex64 strang, slope "
        f"{STEPS_LOW}->{STEPS_HIGH} steps, {args.rounds} rounds: "
        f"unmerged {median['unmerged']:.3f} ms (spread {spread['unmerged']:.3f}), "
        f"merged {median['merged']:.3f} ms (spread {spread['merged']:.3f}), "
        f"ratio {median['merged'] / median['unmerged']:.3f}; "
        f"results {apart:.2e} of the peak apart after {STEPS_LOW} steps"
    )


if __name__ == "__main__":
    main()
