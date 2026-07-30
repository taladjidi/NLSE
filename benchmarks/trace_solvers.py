#!/usr/bin/env python3
"""Break a real propagation into phases and say where the time goes.

    python benchmarks/trace_solvers.py
    python benchmarks/trace_solvers.py --solvers NLSE CNLSE --methods RK4
    python benchmarks/trace_solvers.py --backends CUPY --nvtx --sizes 1024

profile_kernels.py times kernels in isolation. This runs the solver as a user
would and attributes the time inside it: how much goes to transforms, to the
linear step, to the nonlinear step, to the RK4 stage arithmetic, and how much
is not in a kernel at all.

Every wrapped call is synchronized so it can be attributed, which serialises
what the device would otherwise overlap. The traced total is therefore larger
than the real one, and both are reported: the gap is the cost of watching.
Read the shares, not the absolute times.

With --nvtx each phase is also pushed as an NVTX range, so `nsys profile
python benchmarks/trace_solvers.py --nvtx` lines the timeline up with these
names.
"""

import argparse
import collections
import contextlib
import time
from unittest import mock

import numpy as np

WAIST = 2.23e-3
DELTA_Z = 1e-4
# Enough steps that the once-per-run setup -- planning, the propagator build,
# the host-to-device transfer -- does not land in the per-step shares. It is
# not in a kernel, so it would otherwise inflate exactly the row that is
# hardest to interpret.
STEPS = 200

# Kernel or backend call -> the phase it belongs to. Anything unlisted is
# reported under its own name rather than silently folded away.
PHASES = {
    "fft": "transform",
    "ifft": "transform",
    "fft_oop": "transform",
    "apply_propagator": "linear",
    "linear_step": "linear (fused)",
    "square_mod": "nonlinear",
    "nl_prop": "nonlinear",
    "nl_prop_without_V": "nonlinear",
    "nl_prop_c": "nonlinear",
    "nl_prop_without_V_c": "nonlinear",
    "square_mod_nl_prop": "nonlinear",
    "square_mod_nl_prop_v": "nonlinear",
    "rabi_coupling": "nonlinear",
    "rk4_axpy": "RK4 stage",
    "rk4_accumulate": "RK4 stage",
    "rk4_set_and_axpy": "RK4 stage",
    "rk4_acc_and_axpy": "RK4 stage",
    "rk4_nl_rhs": "RK4 rhs",
    "rk4_nl_rhs_v": "RK4 rhs",
    "square_mod_rk4_nl_rhs": "RK4 rhs",
    "square_mod_rk4_nl_rhs_v": "RK4 rhs",
    "rk4_nl_rhs_c": "RK4 rhs",
    "rk4_nl_rhs_c_v": "RK4 rhs",
    "split_step_fused": "whole step (fused)",
    "split_step_coupled_fused": "whole step (fused)",
    "split_step_rk4_fused": "whole step (fused)",
    "rk4_rhs_fused": "RK4 rhs (fused)",
    "rk4_rhs_coupled_fused": "RK4 rhs (fused)",
}

PHASE_ORDER = [
    "transform",
    "linear",
    "linear (fused)",
    "nonlinear",
    "RK4 rhs",
    "RK4 rhs (fused)",
    "RK4 stage",
    "whole step (fused)",
]


class Recorder:
    """Accumulates time and call counts per kernel name."""

    def __init__(self, backend, nvtx=False):
        self.backend = backend
        self.seconds = collections.Counter()
        self.calls = collections.Counter()
        self._nvtx = _Nvtx.find() if nvtx else None

    def measure(self, name, fn, *args, **kwargs):
        """Call fn, forcing and timing its result so the time is its own.

        The result has to be handed to synchronize, not just the backend:
        MLX's graph is lazy and synchronize(None) forces nothing, which
        attributed every kernel 0% and the whole step to "not in a kernel".
        """
        if self._nvtx is not None:
            self._nvtx.range_push(name)
        start = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
            self.backend.synchronize(_forceable(out))
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1
            if self._nvtx is not None:
                self._nvtx.range_pop()
        return out


def _forceable(out):
    """Return something a backend can be asked to finish computing.

    Kernels return an array, a tuple of them, or None for a pure in-place
    update. Only the first case can be forced directly.
    """
    if isinstance(out, (tuple, list)):
        return out[0] if out else None
    return out


class _Nvtx:
    """Uniform push/pop over the several NVTX bindings that exist."""

    def __init__(self, push, pop):
        self.range_push = push
        self.range_pop = pop

    @staticmethod
    def find():
        """Return an _Nvtx, or None if nothing usable is installed."""
        try:  # the nvtx package on PyPI
            import nvtx

            return _Nvtx(nvtx.push_range, nvtx.pop_range)
        except (ImportError, AttributeError):
            pass
        try:  # CuPy ships its own, capitalised differently
            from cupy.cuda import nvtx as cupy_nvtx

            return _Nvtx(cupy_nvtx.RangePush, cupy_nvtx.RangePop)
        except (ImportError, AttributeError):
            pass
        print("  (nvtx requested but no binding found; ranges disabled)")
        return None


class TracingKernels:
    """Wraps a kernels object so every call through it is timed."""

    def __init__(self, kernels, recorder):
        self._kernels = kernels
        self._recorder = recorder

    def __getattr__(self, name):
        """Return the named kernel, wrapped so its time is recorded."""
        attr = getattr(self._kernels, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            return self._recorder.measure(name, attr, *args, **kwargs)

        return wrapped


# Recorded like a kernel, then reported separately: the whole step, so the
# time inside it but outside any kernel can be told apart from the time spent
# in the loop around it.
STEP_KEY = "__step__"


@contextlib.contextmanager
def tracing(simu, nvtx=False):
    """Route a solver's kernels, transforms and steps through a recorder."""
    backend = simu._backend
    recorder = Recorder(backend, nvtx=nvtx)
    cls = type(backend)
    solver_cls = type(simu)
    traced = TracingKernels(backend.kernels, recorder)

    real_fft, real_ifft = cls.fft, cls.ifft

    def plain_loop(self, step_fn, n_iters):
        """Run the steps from Python so each one's kernels are visible.

        CUPY captures one iteration into a CUDA graph and replays it, so the
        wrappers below would see the warmup and the capture and nothing else:
        the first run of this reported 0.01 calls per step and put the whole
        step in "not in a kernel". Bypassing the graph makes the phases
        visible, at the cost of measuring the unreplayed path -- which is why
        the untraced total is taken separately, with the graph in use. The gap
        between the two is what graph replay is worth.
        """
        for _ in range(n_iters):
            step_fn()

    def timed_fft(self, array, plan):
        return recorder.measure("fft", real_fft, self, array, plan)

    def timed_ifft(self, array, plan):
        return recorder.measure("ifft", real_ifft, self, array, plan)

    real_step = solver_cls.split_step
    real_rk4 = solver_cls.split_step_RK4

    def timed_step(self, *args, **kwargs):
        return recorder.measure(STEP_KEY, real_step, self, *args, **kwargs)

    def timed_rk4(self, *args, **kwargs):
        return recorder.measure(STEP_KEY, real_rk4, self, *args, **kwargs)

    with (
        mock.patch.object(cls, "kernels", property(lambda self: traced)),
        mock.patch.object(cls, "fft", timed_fft),
        mock.patch.object(cls, "ifft", timed_ifft),
        mock.patch.object(cls, "execute_loop", plain_loop),
        mock.patch.object(solver_cls, "split_step", timed_step),
        mock.patch.object(solver_cls, "split_step_RK4", timed_rk4),
    ):
        yield recorder


def build(solver_name, backend, n):
    """Return a solver of the requested kind on the requested backend."""
    from NLSE import solvers

    cls = getattr(solvers, solver_name)
    kwargs = {
        "alpha": 0.0,
        "power": 1.05,
        "window": 4 * WAIST,
        "n2": -1.6e-9,
        "V": None,
        "L": 1e-2,
        "NX": n,
        "Isat": 1e5,
        "backend": backend,
    }
    if not solver_name.endswith("_1d"):
        kwargs["NY"] = n
    if solver_name.startswith("CNLSE"):
        kwargs["n12"] = -1e-10
    return cls(**kwargs)


def input_field(simu, solver_name):
    """Return a Gaussian of the shape this solver expects."""
    if solver_name.endswith("_1d"):
        profile = np.exp(-(simu.X**2) / WAIST**2)
    else:
        profile = np.exp(-(simu.XX**2 + simu.YY**2) / WAIST**2)
    field = profile.astype(np.complex64)
    if solver_name.startswith("CNLSE"):
        field = np.array([field, 0.5 * field])
    return field


def run(simu, field, method):
    """Propagate STEPS steps and return the wall time."""
    start = time.perf_counter()
    out = simu.out_field(
        field.copy(),
        STEPS * DELTA_Z,
        verbose=False,
        plot=False,
        delta_z=DELTA_Z,
        method=method,
    )
    simu._backend.synchronize(out)
    return time.perf_counter() - start


def trace(solver_name, backend_name, n, method, nvtx):
    """Return (untraced seconds, traced seconds, recorder) for one case."""
    simu = build(solver_name, backend_name, n)
    field = input_field(simu, solver_name)

    run(simu, field, method)  # warm plans, JIT, autotuning
    untraced = min(run(simu, field, method) for _ in range(3))

    with tracing(simu, nvtx=nvtx) as recorder:
        traced = run(simu, field, method)
    return untraced, traced, recorder


def report(solver_name, backend_name, n, method, untraced, traced, recorder):
    """Print the phase breakdown for one case."""
    by_phase = collections.Counter()
    calls_by_phase = collections.Counter()
    for name, seconds in recorder.seconds.items():
        if name == STEP_KEY:
            continue
        phase = PHASES.get(name, name)
        by_phase[phase] += seconds
        calls_by_phase[phase] += recorder.calls[name]

    in_kernels = sum(by_phase.values())
    in_steps = recorder.seconds.get(STEP_KEY, 0.0)
    # The step calls the kernels, so its time contains theirs.
    step_overhead = max(in_steps - in_kernels, 0.0)
    around_steps = max(traced - in_steps, 0.0)

    print(f"\n=== {solver_name} / {backend_name} / {method} / {n}x{n} ===")
    print(
        f"  {STEPS} steps: {untraced * 1e3:.2f} ms untraced, "
        f"{traced * 1e3:.2f} ms traced "
        f"({traced / untraced:.2f}x: synchronizing each call, and on CUPY "
        f"also stepping from Python instead of replaying a CUDA graph)"
    )
    print(f"  {'phase':<22} {'per step':>10} {'share':>7} {'calls/step':>11}")
    ordered = [p for p in PHASE_ORDER if p in by_phase]
    ordered += sorted(p for p in by_phase if p not in PHASE_ORDER)
    for phase in ordered:
        seconds = by_phase[phase]
        print(
            f"  {phase:<22} {seconds / STEPS * 1e3:8.3f}ms {100 * seconds / traced:6.0f} %"
            f" {calls_by_phase[phase] / STEPS:10.1f}"
        )
    print(
        f"  {'step, outside kernels':<22} {step_overhead / STEPS * 1e3:8.3f}ms "
        f"{100 * step_overhead / traced:6.0f} %"
    )
    print(
        f"  {'outside the step':<22} {around_steps / STEPS * 1e3:8.3f}ms "
        f"{100 * around_steps / traced:6.0f} %"
        f"   (loop, callbacks, transfers, setup)"
    )


def main(argv=None):
    """Parse arguments and trace every requested case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", nargs="*", default=["NLSE", "CNLSE"])
    parser.add_argument("--methods", nargs="*", default=["split_step", "RK4"])
    parser.add_argument("--backends", nargs="*")
    parser.add_argument("--sizes", nargs="*", type=int, default=[1024])
    parser.add_argument("--nvtx", action="store_true", help="emit NVTX ranges for nsys")
    args = parser.parse_args(argv)

    from NLSE.backends import list_available_backends

    for backend_name in args.backends or list_available_backends():
        for solver_name in args.solvers:
            for n in args.sizes:
                for method in args.methods:
                    try:
                        result = trace(solver_name, backend_name, n, method, args.nvtx)
                    except Exception as exc:
                        print(
                            f"\n=== {solver_name} / {backend_name} / {method} "
                            f"/ {n}x{n} ===\n  {type(exc).__name__}: {exc}"
                        )
                        continue
                    report(solver_name, backend_name, n, method, *result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
