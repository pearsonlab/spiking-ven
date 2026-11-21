"""Re-score saved models' DAF response with fixed and with randomised noise onsets.

    SWEEP_AUD_NORM=1.0 python scripts/rescore_daf.py --out res.json M1.npz [M2.npz ...]

Uses the sweep's stimulus cache (``SWEEP_AUD_NORM`` selects the normalised one). For each
model: the DAF metrics and the responders' PSTH with the noise at the fixed window and at
per-trial onsets inside syllables (``metrics.daf_onsets``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # noqa: E402

from spiking_ven.metrics import daf_inputs, daf_response  # noqa: E402
from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

KEYS = ("extra_spikes_responders", "responder_fraction", "latency_ms", "duration_ms",
        "peak_latency_ms", "peak_change_hz", "late_over_peak", "duration_censored")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    st = sweep._stimuli()
    from spiking_ven.causal_filters import CausalFilterEncoder
    enc = CausalFilterEncoder.load(sweep.ENCODER)
    # Rebuilt rather than read from the cache, so a cache made before a fix to the
    # stimulus pipeline cannot leak stale DAF trials into the score.
    fixed_in = daf_inputs(enc, st, seed=42)
    rand_in = daf_inputs(enc, st, seed=42, random_onsets=True)
    out = {}
    for m in a.models:
        v = VocalErrorNet.load(m, seed=42)
        res = {}
        runs: tuple[tuple[str, dict[str, Any]], ...] = (
            ("fixed", {"inputs": fixed_in}),
            ("random", {"inputs": rand_in, "random_onsets": True}))
        for name, kw in runs:
            r = daf_response(v, enc, st, seed=42, **kw)
            res[name] = {k: r[k] for k in KEYS} | {"psth": r["psth_responders"]}
            print(f"{m} [{name:6s}] " + " ".join(f"{k.split('_')[0]}={r[k]:.2f}"
                                                  for k in KEYS), flush=True)
        out[m] = res
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
