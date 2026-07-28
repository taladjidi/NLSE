"""Tests that the built distribution contains what the package needs at runtime.

These build a real wheel rather than inspecting the source tree, because an
editable install (which is what CI uses) resolves every path against the
repository and therefore cannot catch a file that fails to ship.
"""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# setuptools staged builds through build/lib/ and copied sources over whatever
# an earlier build left there, so a checkout that once held a different layout
# kept shipping the old modules. hatchling builds straight from the source tree
# and has no staging directory, so this cannot recur; the assertions below stay
# as a backstop in case the build backend is ever changed back.
STALE_BUILD_DIR = PROJECT_ROOT / "build" / "lib" / "NLSE"


def _cause_hint() -> str:
    """Return a hint about where unexpected wheel contents came from."""
    if STALE_BUILD_DIR.exists():
        leftovers = sorted(p.name for p in STALE_BUILD_DIR.glob("*.py"))
        return (
            f"\n\n{STALE_BUILD_DIR} exists and contains {leftovers}. "
            f"That directory is left over from setuptools and should be "
            f"inert now, so if it is the source of these files the build "
            f"backend has regressed. Delete it and rebuild:"
            f"\n    rm -rf {PROJECT_ROOT / 'build'}"
        )
    return "\n\nNo stale build directory found; check the packaging config."


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

    Prefers ``uv build --wheel`` and falls back to ``pip wheel``. Both build
    the wheel directly from the source tree, which is the path an ordinary
    ``pip install .`` takes; the sdist route is covered separately.

    ``--no-build-isolation`` keeps the build offline and deterministic; the
    dev extra pins the hatchling it needs.
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
    pytest.importorskip("hatchling", reason="hatchling needed to build a wheel")

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


@pytest.fixture(scope="module")
def sdist_build(tmp_path_factory):
    """Build an sdist and then a wheel from it; return (sdist names, wheel names).

    Exercises the path a source install takes. It is separate from the
    in-tree wheel build because the failure modes differ: the sdist has to
    be self-contained, and nothing in the source tree can stand in for a
    file it forgot to include.
    """
    pytest.importorskip("hatchling", reason="hatchling needed to build")
    if not shutil.which("uv"):
        pytest.skip("uv is needed to build an sdist and a wheel from it")

    outdir = tmp_path_factory.mktemp("sdist")
    result = subprocess.run(
        ["uv", "build", "--out-dir", str(outdir), str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "building an sdist and then a wheel from it failed; the sdist is "
            "probably missing something the build needs.\n"
            f"{result.stdout}\n{result.stderr}"
        )

    sdists = list(outdir.glob("*.tar.gz"))
    wheels = list(outdir.glob("*.whl"))
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    with tarfile.open(sdists[0]) as tf:
        sdist_names = tf.getnames()
    with zipfile.ZipFile(wheels[0]) as zf:
        wheel_names = zf.namelist()
    return sdist_names, wheel_names


@pytest.mark.parametrize("data_file", REQUIRED_DATA_FILES)
def test_sdist_carries_the_kernel_templates(sdist_build, data_file):
    """Kernel templates must survive the sdist too, not just the wheel."""
    sdist_names, _ = sdist_build
    assert any(n.endswith(data_file) for n in sdist_names), (
        f"{data_file} is missing from the sdist, so a source install would "
        f"produce a package whose backend cannot load its kernels."
    )


def test_wheel_built_from_sdist_is_clean(sdist_build):
    """A wheel built via the sdist must have the same contents as a direct one."""
    _, wheel_names = sdist_build
    top_level = {
        n
        for n in wheel_names
        if n.startswith("NLSE/") and n.count("/") == 1 and n.endswith(".py")
    }
    assert top_level == {
        "NLSE/__init__.py",
        "NLSE/utils.py",
        "NLSE/callbacks.py",
    }, f"unexpected top-level modules via the sdist path: {sorted(top_level)}"


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
        f"{stale_module} was collected into the wheel, where it shadows the "
        f"current implementation in NLSE/solvers/ or NLSE/kernels/." + _cause_hint()
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
    }, f"unexpected top-level modules in wheel: {sorted(top_level)}" + _cause_hint()


def test_cache_dir_is_outside_the_installed_package(tmp_path, monkeypatch):
    """The runtime cache must never be written inside the package directory.

    Writing there fails on read-only installs, and on uninstall pip leaves
    the runtime-created files behind. What remains is a directory named
    after the package with no __init__.py, which Python imports as a
    namespace package, so `import NLSE` resolves to an empty module.
    """
    import NLSE
    from NLSE.utils import get_cache_dir

    package_dir = Path(NLSE.__file__).resolve().parent

    monkeypatch.delenv("NLSE_CACHE_DIR", raising=False)
    cache_dir = get_cache_dir().resolve()
    assert package_dir not in cache_dir.parents and cache_dir != package_dir, (
        f"cache directory {cache_dir} is inside the installed package {package_dir}"
    )

    monkeypatch.setenv("NLSE_CACHE_DIR", str(tmp_path / "custom"))
    assert get_cache_dir().resolve() == (tmp_path / "custom").resolve()
