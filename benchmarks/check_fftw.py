#!/usr/bin/env python3
"""Say whether this pyfftw can vectorize, and if not, which one you have.

    python benchmarks/check_fftw.py

The CPU backend warns when FFTW plans without vector codelets, because that
build is about four times slower and the transform is most of a CPU step. The
warning says what to do about it. This says *why* it is happening, which
matters because the two causes want different fixes and look identical from
the outside:

- the conda package was asked for but the PyPI wheel is still installed, which
  is easy to miss because ``conda install`` reports success. Conda sees a
  package named pyfftw at a satisfying version, marks the requirement met and
  changes nothing; ``conda list`` gives it away by showing the channel as
  ``pypi``.
- the build that is installed genuinely has no vector codelets for this
  machine, in which case reinstalling it again will not help.

The classification comes from ``NLSE.backends.cpu`` rather than being restated
here, so the tool and the warning cannot disagree about what a vector codelet
looks like.
"""

import ctypes.util
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyfftw
from NLSE.backends.cpu import _SIMD_CODELET

CODELET = re.compile(rb"fftwf?_codelet_[a-z0-9_]+")

# Sizes and alignments to plan. Small ones are included deliberately: a build
# that can vectorize does so even at 8x8, so a scalar plan there is evidence
# about the library rather than about the size.
CASES = ((64, 64), (256, 256), (1024,))


def codelets(wisdom):
    """Return the codelet names FFTW recorded, sorted."""
    return sorted({name.decode() for name in CODELET.findall(wisdom)})


def which_package():
    """Report which pyfftw was imported, and whether it brought its own FFTW."""
    print(f"pyfftw   {pyfftw.__version__}")
    print(f"         {pyfftw.__file__}")
    site = Path(pyfftw.__file__).resolve().parent.parent
    bundled = sorted(site.glob(".dylibs/*fftw*")) + sorted(
        site.glob("pyfftw.libs/*fftw*")
    )
    if bundled:
        print("\n[1] it bundles its own FFTW, so this is the PyPI wheel:")
        for lib in bundled:
            print(f"      {lib.name}  {lib.stat().st_size / 1e6:.1f} MB")
        print("    the conda package links libfftw3f from the environment instead.")
    else:
        print("\n[1] no bundled FFTW: it links one from the environment")
    return bool(bundled)


def which_installers():
    """Report what conda and pip each think is installed."""
    for tool, label in (
        (["conda", "list", "pyfftw"], "conda list"),
        ([sys.executable, "-m", "pip", "list"], "pip list"),
    ):
        try:
            out = subprocess.run(
                tool, capture_output=True, text=True, timeout=120
            ).stdout
        except Exception:
            print(f"\n[2] {label}: could not run")
            continue
        hits = [line.strip() for line in out.splitlines() if "fftw" in line.lower()]
        print(f"\n[2] {label}: " + ("; ".join(hits) or "nothing named fftw"))
    print("    a channel of 'pypi' in conda list means the wheel is what is")
    print("    installed, whatever conda install reported.")


def which_library():
    """Report the FFTW shared libraries visible to this environment."""
    print(f"\n[3] libfftw3f findable as: {ctypes.util.find_library('fftw3f')}")
    for lib in sorted(Path(sys.prefix, "lib").glob("libfftw3f*")):
        print(f"    in the environment: {lib.name}  {lib.stat().st_size / 1e6:.1f} MB")


def what_it_plans():
    """Plan several transforms and report the codelets FFTW chose."""
    print("\n[4] what it plans (single precision, FFTW_MEASURE):")
    vectorized = False
    for shape in CASES:
        for aligned in (True, False):
            pyfftw.forget_wisdom()
            if aligned:
                array = pyfftw.empty_aligned(shape, dtype="complex64")
            else:
                # A deliberately offset buffer: FFTW picks unaligned codelets
                # for one, and a build with no vector codelets at all shows
                # the same names either way.
                raw = np.empty(int(np.prod(shape)) + 1, dtype=np.complex64)
                array = raw[1:].reshape(shape)
            array[...] = 1 + 0j
            axes = tuple(range(-len(shape), 0))
            pyfftw.FFTW(
                array,
                array,
                direction="FFTW_FORWARD",
                axes=axes,
                flags=("FFTW_MEASURE",),
            )
            wisdom = b"".join(pyfftw.export_wisdom())
            simd = bool(_SIMD_CODELET.search(wisdom))
            vectorized = vectorized or simd
            label = "aligned  " if aligned else "unaligned"
            print(f"    {shape!s:10s} {label} vector={simd!s:5s} {codelets(wisdom)}")
    return vectorized


def main():
    """Print the report and a verdict."""
    print(
        f"python   {sys.version.split()[0]}  {platform.machine()}  {platform.system()}"
    )
    print(f"prefix   {sys.prefix}")
    bundled = which_package()
    which_installers()
    which_library()
    vectorized = what_it_plans()
    print(
        f"\n[5] pyfftw.simd_alignment = {pyfftw.simd_alignment}, which does not "
        f"answer the question:\n    it reads the same for a NEON build and a "
        f"scalar one. Ignore it."
    )

    print("\nverdict:")
    if vectorized:
        print("    this FFTW vectorizes. Nothing to do.")
    elif bundled:
        print("    the PyPI wheel is installed and it has no vector codelets.")
        print("    pip uninstall pyfftw   (until it says not installed)")
        print("    conda install -c conda-forge pyfftw")
        print("    then check conda list shows the channel as conda-forge.")
    else:
        print("    the build installed from the environment has no vector")
        print("    codelets for this machine. Reinstalling it will not help;")
        print("    the fftw package it links is the thing to change.")


if __name__ == "__main__":
    main()
