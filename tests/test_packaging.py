"""Tests that the built distribution contains what the package needs at runtime."""

import email
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kernel templates read at runtime via Path(__file__).parent / ... .read_text(encoding="utf-8").
# Every one of these must survive packaging or the matching backend dies with
# FileNotFoundError on a non-editable install.
REQUIRED_DATA_FILES = [
    "NLSE/kernels/cl_source/kernels.cl",
    "NLSE/kernels/cuda_source/fft_callbacks.cu",
    "NLSE/kernels/cuda_source/kernels.cu",
    "NLSE/kernels/cuda_source/nonlinear_step.cuh",
]

# The extras a user can install. The dev and docs tools are dependency groups
# (PEP 735), which stay out of the published metadata.
PUBLISHED_EXTRAS = {"gpu", "mlx", "opencl"}


def _build_wheel_command(outdir: Path) -> list[str]:
    """Return the command that builds a wheel straight from the source tree."""
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


def _require_hatchling() -> None:
    """Fail, rather than skip, when the build backend is missing."""
    try:
        import hatchling  # noqa: F401
    except ImportError:
        pytest.fail(
            "hatchling is missing, so the packaging tests cannot build "
            "anything to inspect. It is in the dev dependency group: run "
            "`uv sync`."
        )


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory) -> Path:
    """Build a wheel from the project root and return its path."""
    _require_hatchling()

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
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_contents(wheel_path) -> list[str]:
    """Return the member names of the wheel built from the project root."""
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


def _run(command: list[str], what: str) -> None:
    """Run a build command from the project root, failing loudly."""
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        pytest.fail(
            f"{what} failed ({' '.join(str(c) for c in command)}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _build_sdist_and_wheel(outdir: Path) -> None:
    """Build an sdist into outdir, then a wheel from that sdist."""
    if shutil.which("uv"):
        _run(
            ["uv", "build", "--out-dir", str(outdir), str(PROJECT_ROOT)],
            "building an sdist and then a wheel from it",
        )
        return

    _run(
        [sys.executable, "-m", "hatchling", "build", "-t", "sdist", "-d", str(outdir)],
        "building an sdist",
    )
    sdists = list(outdir.glob("*.tar.gz"))
    assert len(sdists) == 1, f"expected one sdist, got {sdists}"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(outdir),
            str(sdists[0]),
        ],
        "building a wheel from the sdist",
    )


@pytest.fixture(scope="module")
def sdist_build(tmp_path_factory):
    """Build an sdist and then a wheel from it; return (sdist names, wheel names)."""
    _require_hatchling()

    outdir = tmp_path_factory.mktemp("sdist")
    _build_sdist_and_wheel(outdir)

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
        f"raise FileNotFoundError on a non-editable install. Check "
        f"[tool.hatch.build.targets.wheel] in pyproject.toml."
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


def test_the_wheel_holds_the_package_and_its_metadata_only(wheel_contents):
    """Tests, docs or examples in the wheel would install as top-level names."""
    roots = {name.split("/")[0] for name in wheel_contents}
    stray = sorted(root for root in roots - {"NLSE"} if not root.endswith(".dist-info"))
    assert not stray, f"the wheel carries {stray} next to NLSE/"


def test_the_wheel_publishes_only_the_backend_extras(wheel_path):
    """The dev and docs tools must not come back as extras of the package."""
    with zipfile.ZipFile(wheel_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(names) == 1, f"expected one METADATA file, got {names}"
        metadata = email.message_from_bytes(zf.read(names[0]))
    extras = set(metadata.get_all("Provides-Extra", []))
    assert extras == PUBLISHED_EXTRAS, (
        f"the wheel declares the extras {sorted(extras)}; the development "
        f"tools belong in [dependency-groups] of pyproject.toml"
    )


def ignored_package_dirs() -> tuple[str, ...]:
    """Return the .gitignore entries under NLSE/, which hatchling leaves out.

    Returns
    -------
    tuple of str
        Entries such as ``NLSE/kernels/metal/``, as written in .gitignore.
    """
    # shortcut: literal NLSE/ prefixes only; use pathspec if globs appear there.
    lines = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    return tuple(line.strip() for line in lines if line.startswith("NLSE/"))


def test_every_source_module_is_packaged(wheel_contents):
    """Every source module is in the wheel, not only in the editable install."""
    source = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "NLSE").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    source = {name for name in source if not name.startswith(ignored_package_dirs())}
    missing = sorted(source - set(wheel_contents))
    assert not missing, f"left out of the wheel: {missing}"


def test_cache_dir_is_outside_the_installed_package(tmp_path, monkeypatch):
    """The runtime cache must never be written inside the package directory."""
    import NLSE
    from NLSE.utils import get_cache_dir

    package_dir = Path(NLSE.__file__).resolve().parent

    # The platform default, under a home of this test's own: the real one
    # holds the user's benchmark cache.
    home = tmp_path / "home"
    monkeypatch.delenv("NLSE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    cache_dir = get_cache_dir().resolve()
    assert cache_dir.is_relative_to(home.resolve())
    assert package_dir not in cache_dir.parents and cache_dir != package_dir, (
        f"cache directory {cache_dir} is inside the installed package {package_dir}"
    )

    monkeypatch.setenv("NLSE_CACHE_DIR", str(tmp_path / "custom"))
    assert get_cache_dir().resolve() == (tmp_path / "custom").resolve()


def test_the_package_version_matches_the_project_metadata():
    """``__version__`` and the build's version have to be the same string."""
    from NLSE import __version__

    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # A regex, not tomllib: tomllib is 3.11+ and the package supports 3.10.
    found = re.findall(r'^version *= *"([^"]+)"', text, re.MULTILINE)
    assert len(found) == 1, (
        f"expected exactly one top-level version in pyproject.toml, found "
        f"{found}; this test can no longer tell which one ships"
    )
    declared = found[0]

    assert __version__ == declared, (
        f"NLSE.__version__ is {__version__} and pyproject.toml declares "
        f"{declared}; an install built from this tree would report the latter "
        f"while the package says the former"
    )
