"""Cache simulation outputs next to a trained model, so figures can be redrawn cheaply.

Running a trained network (hundreds of DAF trials, every rendition of the training pool)
is the slow part of most diagnostics; the model itself does not change once trained. This
stores the arrays a diagnostic simulated in ``<model>.<name>.npz`` beside the model,
together with a key: the model file's sha256 plus whatever else the result depends on
(stimulus cache, seeds, trial counts). A later call with the same key loads the arrays
instead of simulating; any change -- including retraining the model in place -- misses
and recomputes.

Public API
----------
cached_arrays(model_path, name, key, compute) -> dict[str, np.ndarray]
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import sha256_file

__all__ = ["cached_arrays"]


def cached_arrays(model_path: str | Path, name: str, key: dict,
                  compute: Callable[[], dict[str, np.ndarray]],
                  *, verbose: bool = True) -> dict[str, np.ndarray]:
    """``compute()``'s arrays for this model and ``key``, from the cache when it matches.

    ``key`` must be JSON-serialisable and name everything besides the model that the
    arrays depend on. Spike arrays are stored as ``uint8`` (and returned as such) when
    they hold only 0 and 1, so the files stay small.
    """
    model_path = Path(model_path)
    full_key = json.dumps({"model_sha256": sha256_file(model_path), **key}, sort_keys=True)
    path = model_path.with_name(f"{model_path.stem}.{name}.npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            if "_key" in d and str(d["_key"]) == full_key:
                if verbose:
                    print(f"[sim_cache] loaded {path}")
                return {k: d[k] for k in d.files if k != "_key"}
    out = compute()
    stored: dict[str, np.ndarray] = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.dtype != np.uint8 and a.size and np.isin(a, (0, 1)).all():
            a = a.astype(np.uint8)
        stored[k] = a
    to_save: dict[str, Any] = {"_key": np.array(full_key), **stored}
    np.savez_compressed(path, **to_save)
    if verbose:
        print(f"[sim_cache] saved {path}")
    return stored
