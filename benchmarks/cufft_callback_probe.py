#!/usr/bin/env python3
"""Ask whether this machine can fold the propagator multiply into the transform.

    python benchmarks/cufft_callback_probe.py
    python benchmarks/cufft_callback_probe.py --sizes 1024 2048

A split step reads and writes the field once for the transform pair, once more
for the propagator multiply and once more for the nonlinear step: 104 bytes per
grid point, of which the propagator pass is 24. cuFFT can run a store callback
as it writes each element, which would absorb that pass entirely -- 104 bytes
becoming 88, on a backend already at the bandwidth bound.

This does not implement that. It answers the four questions that decide whether
it can be implemented at all, because every one of them is a property of the
installed CUDA rather than of this repository, and none can be checked on a
machine without an NVIDIA GPU:

1. Is the callback machinery present? CuPy's is experimental, compiles the
   callback with nvcc and links cufft statically, so it needs a full toolkit
   and not just a driver.
2. Does a store callback compute the right thing?
3. **Does the callback see new values written into the aux array without the
   plan being rebuilt?** This is the one that decides the design. Callbacks
   bind at plan creation, and the propagator changes whenever an adaptive step
   does. If the array can be overwritten in place the plan is built once; if
   not, every step change costs a plan and the idea is dead.
4. Is it actually faster than the separate multiply it replaces?

Run it on the NVIDIA box and paste the output. Failures are the point: each one
says what is missing rather than that something went wrong.
"""

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np

# CuPy wants the device function pointer under this name; the function it
# points at may be called anything.
STORE_CALLBACK = r"""
__device__ void CB_MulAndStore(void *dataOut, size_t offset, cufftComplex element,
                               void *callerInfo, void *sharedPtr) {
    cufftComplex p = ((cufftComplex *)callerInfo)[offset];
    cufftComplex out;
    out.x = element.x * p.x - element.y * p.y;
    out.y = element.x * p.y + element.y * p.x;
    ((cufftComplex *)dataOut)[offset] = out;
}
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndStore;
"""


def report_environment():
    """Say what is installed, before anything tries to use it."""
    print("=== environment ===")
    try:
        import cupy as cp
    except ImportError as exc:
        print(f"  cupy: not importable ({exc})")
        return None
    print(f"  cupy            {cp.__version__}")
    try:
        print(f"  CUDA runtime    {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"  device          {cp.cuda.Device().attributes['Name']}")
    except Exception as exc:  # pragma: no cover - no device
        print(f"  CUDA runtime    unavailable ({type(exc).__name__}: {exc})")
        return None

    nvcc = shutil.which("nvcc")
    print(f"  nvcc            {nvcc or 'NOT FOUND -- callbacks need a full toolkit'}")
    if nvcc:
        version = (
            subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, check=False
            )
            .stdout.strip()
            .splitlines()
        )
        print(f"                  {version[-1] if version else '?'}")

    has = hasattr(cp.fft.config, "set_cufft_callbacks")
    print(f"  set_cufft_callbacks  {'present' if has else 'ABSENT in this cupy'}")
    return cp if has else None


def build_arrays(cp, n):
    """Return a field, a propagator, and the numpy answer to check against."""
    rng = np.random.default_rng(0)
    host = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex64)
    phase = rng.uniform(0, 2 * np.pi, size=(n, n))
    prop_host = np.exp(1j * phase).astype(np.complex64)
    want = np.fft.fft2(host) * prop_host
    return cp.asarray(host), cp.asarray(prop_host), want


def with_callback(cp, field, prop):
    """Transform with the multiply folded into the store, or raise saying why."""
    from cupyx.scipy.fftpack import get_fft_plan

    with cp.fft.config.set_cufft_callbacks(
        cb_store=STORE_CALLBACK, cb_store_aux_arr=prop
    ):
        # Built inside the context: the callback is compiled into the plan, so
        # a plan made outside it will not carry one.
        plan = get_fft_plan(field, axes=(-2, -1), value_type="C2C")
        out = cp.empty_like(field)
        with plan:
            plan.fft(field, out, cp.cuda.cufft.CUFFT_FORWARD)
        return out, plan


def check_correctness(cp, n):
    """Check the fused transform against the unfused answer (question 2)."""
    field, prop, want = build_arrays(cp, n)
    got, _ = with_callback(cp, field, prop)
    got = cp.asnumpy(got)
    err = np.max(np.abs(got - want)) / np.max(np.abs(want))
    print(f"  fused vs separate     max rel error {err:.3e}")
    return err < 1e-5


def check_aux_is_live(cp, n):
    """Check that overwriting the aux array reaches the callback (question 3).

    The one that decides the design. If it does, the plan is built once for
    the run; if not, every adaptive step change costs a new plan.
    """
    field, prop, _ = build_arrays(cp, n)
    first, plan = with_callback(cp, field, prop)

    # Same array object, new contents. A plan that captured the values rather
    # than the pointer will not notice.
    prop[...] = prop * cp.asarray(np.complex64(2.0))
    out = cp.empty_like(field)
    with plan:
        plan.fft(field, out, cp.cuda.cufft.CUFFT_FORWARD)

    ratio = cp.asnumpy(out) / cp.asnumpy(first)
    live = np.allclose(ratio, 2.0, rtol=1e-4, atol=1e-4)
    print(
        f"  aux array overwritten in place: "
        f"{'SEEN by the callback' if live else 'NOT seen -- plan must be rebuilt'}"
    )
    return live


def check_speed(cp, n, repeats=20):
    """Time the fused transform against transform + multiply (question 4)."""
    from cupyx.scipy.fftpack import get_fft_plan

    field, prop, _ = build_arrays(cp, n)
    sync = cp.cuda.runtime.deviceSynchronize

    plain = get_fft_plan(field, axes=(-2, -1), value_type="C2C")
    out = cp.empty_like(field)

    def unfused():
        with plain:
            plain.fft(field, out, cp.cuda.cufft.CUFFT_FORWARD)
        cp.multiply(out, prop, out=out)

    fused_out, fused_plan = with_callback(cp, field, prop)

    def fused():
        with fused_plan:
            fused_plan.fft(field, fused_out, cp.cuda.cufft.CUFFT_FORWARD)

    def best(fn):
        fn()
        sync()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - start)
        return min(times)

    # Interleaved, because between-process drift is larger than the effect.
    rounds = {"unfused": [], "fused": []}
    for i in range(4):
        order = ("unfused", "fused") if i % 2 == 0 else ("fused", "unfused")
        for name in order:
            rounds[name].append(best(unfused if name == "unfused" else fused))

    u, f = min(rounds["unfused"]), min(rounds["fused"])
    noise = max(max(v) / min(v) for v in rounds.values()) - 1
    pts = n * n
    print(
        f"  transform + multiply  {u * 1e3:7.3f} ms   "
        f"fused {f * 1e3:7.3f} ms   {u / f:5.2f}x   noise {noise:5.1%}"
    )
    print(
        f"  compulsory traffic    {(4 + 3) * 8 * pts / u / 1e9:6.1f} GB/s -> "
        f"{(4 + 1) * 8 * pts / f / 1e9:6.1f} GB/s"
    )


def main(argv=None):
    """Answer the four questions, in order, stopping at the first no."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[1024, 2048])
    args = parser.parse_args(argv)

    cp = report_environment()
    if cp is None:
        print("\ncuFFT callbacks are not reachable here; nothing further to ask.")
        return 1

    for n in args.sizes:
        print(f"\n=== {n}x{n} ===")
        try:
            if not check_correctness(cp, n):
                print("  the fused transform does not agree; stopping")
                return 1
            check_aux_is_live(cp, n)
            check_speed(cp, n)
        except Exception as exc:
            print(f"  {type(exc).__name__}: {exc}")
            print(
                "\n  If this is a link or compile error, the usual cause is a "
                "driver-only CUDA install: callbacks need nvcc and the static "
                "cufft library. See the CUDA toolkit install guide."
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
