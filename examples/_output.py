"""Where the example scripts put what they produce.

Resolved from this file's location, so it does not depend on the working
directory the script is run from.
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
