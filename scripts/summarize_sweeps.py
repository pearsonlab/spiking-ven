"""Group every sweep result by configuration, across training seeds.

    python scripts/summarize_sweeps.py [--min-seeds 2] [--match tau_gaba]

Reads ``outputs/sweeps/*/*.json`` (``scripts/sweep.py``), groups by the override dict
(and the number of renditions and the encoder, when not the defaults), and prints each
configuration's per-seed and mean metrics. ``diff`` is the song-driven contrast
(``metrics.song_contrast``), reversed minus trained excess rate (Hz).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [("score", lambda r: r["score"]["total"]),
        ("xs", lambda r: r["daf"]["extra_spikes_responders"]),
        ("resp", lambda r: r["daf"]["responder_fraction"]),
        ("lat", lambda r: r["daf"]["latency_ms"]),
        ("dur", lambda r: r["daf"]["duration_ms"]),
        ("pk", lambda r: r["daf"]["peak_latency_ms"]),
        ("rev/tr", lambda r: r["rates"]["reversed_over_trained"]),
        ("diff", lambda r: r.get("contrast", {}).get("excess_diff_hz", float("nan")))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-seeds", type=int, default=1)
    p.add_argument("--match", default="")
    a = p.parse_args()
    groups: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted(Path("outputs/sweeps").glob("*/*.json")):
        r = json.loads(f.read_text())
        ov = {k: v for k, v in r["overrides"].items() if k != "dt_ms" or v != 0.1}
        key = json.dumps(ov, sort_keys=True)
        if r.get("n_rend", 600) != 600:
            key += f" n_rend={r['n_rend']}"
        if "enc_" in f.parent.name:
            key += f" [{f.parent.name}]"
        if a.match not in key:
            continue
        groups[key][r["seed"]] = r            # later rounds overwrite identical reruns
    head = " ".join(f"{c:>7s}" for c, _ in COLS)
    for key, by_seed in sorted(groups.items()):
        if len(by_seed) < a.min_seeds:
            continue
        print(f"\n{key or '{} (shipped)'}\n   seed {head}")
        rows = []
        for seed, r in sorted(by_seed.items()):
            vals = [fn(r) for _, fn in COLS]
            rows.append(vals)
            print(f"   {seed:4d} " + " ".join(f"{v:7.3f}" for v in vals))
        if len(rows) > 1:
            m = np.nanmean(np.array(rows, dtype=float), axis=0)
            print("   mean " + " ".join(f"{v:7.3f}" for v in m))


if __name__ == "__main__":
    main()
