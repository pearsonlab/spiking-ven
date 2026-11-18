"""DAF metrics of a saved model at one evaluation seed (the noise and DAF draws).

    python scripts/eval_seed.py --model outputs/ven_model.npz --seed 43 --out e43.json

The stimuli are built at seed 42 as in ``tests/test_reproduction_causal.py``; ``--seed``
sets the network's noise stream and the DAF noise draws. Run over seeds 42-51 to set the
reproduction tolerances (~3 SD).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.metrics import daf_response
from spiking_ven.stimuli import build_stimuli
from spiking_ven.vocal_error_net import VocalErrorNet

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", default="outputs/ven_model.npz")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600, n_ista=50,
                   verbose=False)
r = daf_response(VocalErrorNet.load(a.model, seed=a.seed), enc, st, seed=a.seed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(r, default=float, indent=1))
