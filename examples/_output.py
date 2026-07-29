"""Where the example scripts put what they produce.

They used to write to the current working directory, so running one from the
repository root dropped figures and timing arrays among the sources. The
.gitignore grew a pattern per file to hide them, including a repo-wide
``*.npy`` that would have hidden real data too.

They write here instead, next to the scripts and independent of where they
are run from.
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def output_path(name: str) -> Path:
    """Return the path to write ``name`` to, creating the directory.

    Parameters
    ----------
    name : str
        File name, without a directory.

    Returns
    -------
    Path
        Absolute path inside ``examples/output``.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name
