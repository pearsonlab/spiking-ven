"""Wall-clock time of the whole training protocol, per engine.

Builds the stimuli, trains for ``--n-rend`` renditions (measuring the trained and reversed
song at the 15 ``LOG_AT`` renditions, as ``sven-train-ven`` does), then scores the model
(rendition rates and the 100-trial DAF response). Uses only API shared by the numpy engine
(spiking-ven <= 0.6) and the Brian2 engine; put the version to time on ``PYTHONPATH``.

    python scripts/benchmark_training.py --label "Brian2, 0.1 ms" --dt-ms 0.1 --out run.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

LOG_AT = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--dt-ms", type=float, default=None, help="Brian2 engine only")
    p.add_argument("--n-rend", type=int, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from spiking_ven.causal_filters import CausalFilterEncoder
    from spiking_ven.metrics import daf_response, rendition_rates
    from spiking_ven.stimuli import build_stimuli
    from spiking_ven.vocal_error_net import VocalErrorNet

    t0 = time.perf_counter()
    enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
    st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600, n_ista=50,
                       verbose=False)
    t_stim = time.perf_counter() - t0
    kw = {} if a.dt_ms is None else {"dt_ms": a.dt_ms}
    ven = VocalErrorNet(n_aud=enc.n_channels, seed=42, **kw)
    rng = np.random.default_rng(1)
    pre = st["hvc_prelude"]
    t0 = time.perf_counter()
    for r in range(a.n_rend):
        if r == 0:
            ven.fit(st["hvc_burn"], st["aud_burn"])
        else:
            ven.fit(st["hvc_on"], st["correct_pool"][r % len(st["correct_pool"])],
                    hvc_prelude=pre)
        if r + 1 in LOG_AT:
            ven.transform(st["hvc_on"], st["aud_correct"], rng=rng, hvc_prelude=pre)
            ven.transform(st["hvc_on"], st["aud_reversed"], rng=rng, hvc_prelude=pre)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    rendition_rates(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                    aud_trained=st["aud_correct"], aud_daf=st["aud_daf"],
                    aud_reversed=st["aud_reversed"], hvc_prelude=pre)
    daf_response(ven, enc, st, seed=42)
    t_eval = time.perf_counter() - t0
    res = {"label": a.label, "stimuli_s": t_stim, "train_s": t_train, "evaluate_s": t_eval,
           "minutes": (t_stim + t_train + t_eval) / 60.0, "n_rend": a.n_rend}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
