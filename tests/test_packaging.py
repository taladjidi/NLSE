"""Tests that the built distribution contains what the package needs at runtime.

These build a real wheel rather than inspecting the source tree, because an
editable install (``pip install -e .``, which is what CI uses) resolves every
path against the repository and therefore cannot catch a missing
``package-data`` entry or a stray directory swept up by ``packages.find``.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kernel templates read at runtime via Path(__file__).parent / ... .read_text().
# Every one of these must survive packaging or the matching backend dies with
# FileNotFoundError on a non-editable install.
REQUIRED_DATA_FILES = [
    "NLSE/kernels/cl_source/kernels.cl",
    "NLSE/kernels/cuda_source/kernels.cu",
]

# Modules from the pre-2.3 flat layout. They still exist under build/lib/ and
# must not be re-collected into the wheel, where they would shadow the real
# implementations in NLSE/solvers/ and NLSE/kernels/.
FORBIDDEN_STALE_MODULES = [
    "NLSE/nlse.py",
    "NLSE/nlse_1d.py",
    "NLSE/nlse_3d.py",
    "NLSE/cnlse.py",
    "NLSE/cnlse_1d.py",
    "NLSE/ddgpe.py",
    "NLSE/gpe.py",
    "NLSE/kernels_cpu.py",
    "NLSE/kernels_gpu.py",
    "NLSE/kernels_cl.py",
]


def _build_wheel_command(outdir: Path) -> list[str]:
    """Return the command that builds a wheel straight from the source tree.

    Prefers ``uv build --wheel`` and falls back to ``pip wheel``. Both stage
    through ``build/lib/``, which is what makes them able to detect files left
    behind there by an earlier build. Plain ``uv build`` is deliberately not
    used: it builds the wheel from a freshly generated sdist and would mask
    exactly that failure mode.

    ``--no-build-isolation`` keeps the build offline and deterministic; the
    dev extra pins the setuptools/wheel it needs.
    """
    if shutil.which("uv"):
        return [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--out-dir",
            str(outdir),
            str(PROJECT_ROOT),
        ]
    return [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(outdir),
        str(PROJECT_ROOT),
    ]


@pytest.fixture(scope="module")
def wheel_contents(tmp_path_factory) -> list[str]:
    """Build a wheel from the project root and return its member names."""
    pytest.importorskip("setuptools", reason="setuptools needed to build a wheel")

    outdir = tmp_path_factory.mktemp("wheel")
    command = _build_wheel_command(outdir)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"wheel build failed ({' '.join(command)}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        return zf.namelist()


@pytest.mark.parametrize("data_file", REQUIRED_DATA_FILES)
def test_kernel_templates_are_packaged(wheel_contents, data_file):
    """Kernel source templates must ship in the wheel."""
    assert data_file in wheel_contents, (
        f"{data_file} is missing from the wheel. The backend that loads it will "
        f"raise FileNotFoundError on a non-editable install. Add its directory "
        f"to [tool.setuptools.package-data] in pyproject.toml."
    )


@pytest.mark.parametrize("stale_module", FORBIDDEN_STALE_MODULES)
def test_stale_flat_layout_modules_are_not_packaged(wheel_contents, stale_module):
    """Pre-2.3 flat-layout modules must not be collected into the wheel."""
    assert stale_module not in wheel_contents, (
        f"{stale_module} was collected into the wheel, most likely from "
        f"build/lib/NLSE/. It shadows the current implementation. Exclude "
        f"build* in [tool.setuptools.packages.find]."
    )


def test_wheel_top_level_modules_match_source(wheel_contents):
    """Only the real top-level NLSE modules should be present."""
    top_level = {
        name
        for name in wheel_contents
        if name.startswith("NLSE/") and name.count("/") == 1 and name.endswith(".py")
    }
    assert top_level == {
        "NLSE/__init__.py",
        "NLSE/utils.py",
        "NLSE/callbacks.py",
    }, f"unexpected top-level modules in wheel: {sorted(top_level)}"
