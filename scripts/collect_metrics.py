"""Score one trained model and save what the comparison figures need.

    python scripts/collect_metrics.py --model outputs/ven_model.npz --label "Brian2" \
        --out outputs/compare/brian2.npz

Writes the rendition rates, the DAF response metrics and the responders' mean DAF PSTH
(1 ms bins, 150 ms either side of noise onset). Uses only API shared by the numpy engine
(spiking-ven <= 0.6) and the Brian2 engine, so the same script scores both: put the
version that can load the model on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.metrics import daf_responders, daf_response, daf_trials, rendition_rates
from spiking_ven.stimuli import build_stimuli, daf_window_ms
from spiking_ven.vocal_error_net import VocalErrorNet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
    st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=a.seed, n_hvc=600,
                       n_ista=50, verbose=False)
    ven = VocalErrorNet.load(a.model, seed=a.seed)
    t0 = time.time()
    m = rendition_rates(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                        aud_trained=st["aud_correct"], aud_daf=st["aud_daf"],
                        aud_reversed=st["aud_reversed"], hvc_prelude=st["hvc_prelude"])
    r = daf_response(ven, enc, st, seed=a.seed)
    # The same trials again (deterministic inputs; the network's noise stream moves on,
    # so this is an independent draw of the network noise) for the PSTH.
    trials = daf_trials(ven, enc, st, seed=a.seed)
    onset, _ = daf_window_ms(st["daf_window_s"])
    resp, _, _ = daf_responders(trials, onset_ms=onset, window_ms=150)
    W = 150
    psth = trials[:, resp, onset - W:onset + W].mean(axis=(0, 1)) * 1000.0
    psth_all = trials[:, :, onset - W:onset + W].mean(axis=(0, 1)) * 1000.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, label=a.label, metrics=json.dumps({**m, **r}, default=float),
             psth_responders=psth, psth_all=psth_all, t_ms=np.arange(-W, W),
             eval_s=time.time() - t0)
    print(json.dumps({**m, **r}, default=float, indent=1))


if __name__ == "__main__":
    main()
