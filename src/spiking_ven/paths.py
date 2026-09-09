"""Cache and data locations.

Nothing in this package hardcodes a path relative to the process working directory.
Callers pass explicit paths; these helpers only supply the defaults used by the CLI
entry points, and they can be redirected with two environment variables:

``SVEN_DATA_DIR``    raw, downloaded song data           (default ``./data``)
``SVEN_OUTPUT_DIR``  generated caches, models, figures   (default ``./outputs``)

Both directories are gitignored. Nothing here is created until something writes to it.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "data_dir",
    "output_dir",
    "motifs_npz",
    "of_encoder_npz",
    "ven_model_npz",
    "ensure_parent",
]


def data_dir() -> Path:
    """Directory holding raw downloaded song data."""
    return Path(os.environ.get("SVEN_DATA_DIR", "data"))


def output_dir() -> Path:
    """Directory holding generated caches, trained models and figures."""
    return Path(os.environ.get("SVEN_OUTPUT_DIR", "outputs"))


def motifs_npz() -> Path:
    """DTW-aligned song motifs produced by ``sven-prep-data``."""
    return output_dir() / "motifs.npz"


def of_encoder_npz() -> Path:
    """Trained Olshausen-Field sparse encoder produced by ``sven-train-encoder``."""
    return output_dir() / "of_encoder.npz"


def ven_model_npz(tag: str = "k4max") -> Path:
    """Trained vocal error network produced by ``sven-train-ven``."""
    return output_dir() / f"of_ven_model_{tag}.npz"


def ensure_parent(path: str | os.PathLike) -> Path:
    """Create the parent directory of ``path`` if needed; return ``path`` as a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
