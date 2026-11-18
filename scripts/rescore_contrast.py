"""Score saved models on the song-driven contrast (``metrics.song_contrast``).

    python scripts/rescore_contrast.py --out contrast.json MODEL.npz [MODEL.npz ...]

Self-contained (only ``fit``/``transform`` and the aud -> E weights), so it also scores
numpy-engine models when that version of spiking-ven is on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.stimuli import build_stimuli
from spiking_ven.vocal_error_net import VocalErrorNet


def contrast(ven, st, n_trials: int = 10, seed: int = 7) -> dict:
    """Same definition as ``spiking_ven.metrics.song_contrast``."""
    rng = np.random.default_rng(seed)
    A = getattr(ven, "A", None)
    A = ven.B if A is None else A                  # numpy engine: aud -> E is ``B``
    cells = np.flatnonzero(np.asarray(A).any(axis=1))
    T, pre = int(st["T_song"]), st.get("hvc_prelude")

    def rate(aud):
        return float(np.mean([
            ven.transform(st["hvc_on"], aud, rng=rng, hvc_prelude=pre)[cells, :T].mean()
            for _ in range(n_trials)]) * 1000)

    h = rate(np.zeros_like(st["aud_correct"]))
    t, r = rate(st["aud_correct"]), rate(st["aud_reversed"])
    trained_all = float(np.mean([ven.transform(st["hvc_on"], st["aud_correct"], rng=rng,
                                               hvc_prelude=pre).mean()
                                 for _ in range(n_trials)]) * 1000)
    reversed_all = float(np.mean([ven.transform(st["hvc_on"], st["aud_reversed"], rng=rng,
                                                hvc_prelude=pre).mean()
                                  for _ in range(n_trials)]) * 1000)
    return {"n_cells": int(cells.size), "hvc_only_hz": h, "trained_hz": t, "reversed_hz": r,
            "excess_diff_hz": (r - h) - (t - h),
            "excess_ratio": (r - h) / (t - h) if t > h else float("nan"),
            "whole_rendition_ratio": reversed_all / trained_all}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
    st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600, n_ista=50,
                       verbose=False)
    out = {}
    for m in a.models:
        c = contrast(VocalErrorNet.load(m, seed=42), st)
        out[m] = c
        print(f"{m}\n   whole-rendition rev/tr {c['whole_rendition_ratio']:.3f} | song-driven "
              f"({c['n_cells']} cells): HVC-only {c['hvc_only_hz']:.2f}  trained "
              f"{c['trained_hz']:.2f}  reversed {c['reversed_hz']:.2f} Hz  excess ratio "
              f"{c['excess_ratio']:.2f}  diff {c['excess_diff_hz']:+.2f} Hz", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
