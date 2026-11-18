"""Train and score many network configurations in parallel.

    python scripts/sweep.py --name dt01_a --grid '{"dt_ms": [0.1], "hvc_mean_ns": [6.36, 8]}'

Each configuration is a set of ``VocalErrorNet`` keyword overrides on the shipped
defaults. It is trained for ``--n-rend`` renditions (``spiking_ven.training.train``) and
scored with the rendition rates and the 100-trial DAF response, exactly as
``sven-train-ven`` does; the result, with the overrides, goes to
``outputs/sweeps/<name>/<id>.json``. Stimuli and DAF inputs are encoded once and cached.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: The encoder the stimuli are built with; ``--encoder`` sets it (workers inherit it).
ENCODER = os.environ.get("SWEEP_ENCODER", "outputs/causal_encoder.npz")
CACHE = Path("outputs/sweeps/" + ("stimuli_cache.pkl" if ENCODER == "outputs/causal_encoder.npz"
                                  else f"stimuli_cache_{Path(ENCODER).stem}.pkl"))


def _stimuli():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    from spiking_ven.causal_filters import CausalFilterEncoder
    from spiking_ven.metrics import daf_inputs
    from spiking_ven.stimuli import build_stimuli
    enc = CausalFilterEncoder.load(ENCODER)
    st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600,
                       n_ista=50, verbose=False)
    st["daf_inputs"] = daf_inputs(enc, st, seed=42)
    st["n_channels"] = enc.n_channels
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(st))
    return st


def score(r: dict, m: dict) -> dict:
    """Each target's miss, in units of its tolerance (0 = on target)."""
    def outside(x, lo, hi):
        if not np.isfinite(x):
            return 3.0
        return max(0.0, lo - x, x - hi) / (hi - lo)
    s = {
        "rate": outside(r["baseline_hz_all"], 1.0, 10.0),
        "responders": outside(r["responder_fraction"], 0.30, 0.55),
        "latency": outside(r["latency_ms"], 11.0, 35.0),
        "duration": outside(r["duration_ms"], 47.0, 133.0),
        "peak_latency": outside(r["peak_latency_ms"], 35.0, 75.0),
        "late": outside(r["late_over_peak"], -1.0, 0.15),
        "extra_spikes": outside(r["extra_spikes_responders"], 0.8, 1.0),
        "specificity": outside(m["reversed_over_trained"], 1.0, 10.0),
    }
    s["total"] = float(sum(s.values()))
    return s


def run_one(args) -> dict:
    overrides, out_dir, n_rend, seed = args
    from spiking_ven.metrics import daf_response, rendition_rates, song_contrast
    from spiking_ven.training import train
    from spiking_ven.vocal_error_net import VocalErrorNet
    key = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:10]
    path = Path(out_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    st = _stimuli()
    t0 = time.time()
    ven = VocalErrorNet(n_aud=st["n_channels"], seed=seed, **overrides)
    traces = train(ven, st, n_rend, seed=seed, verbose=False)
    t_train = time.time() - t0
    m = rendition_rates(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                        aud_trained=st["aud_correct"], aud_daf=st["aud_daf"],
                        aud_reversed=st["aud_reversed"], hvc_prelude=st["hvc_prelude"])
    r = daf_response(ven, None, st, seed=seed, inputs=st["daf_inputs"])
    c = song_contrast(ven, st)
    res = {"id": key, "overrides": overrides, "n_rend": n_rend, "seed": seed,
           "rates": m, "daf": r, "contrast": c, "score": score(r, m),
           "trace": {k: v.tolist() for k, v in traces.items()},
           "j_ie_max": float(np.max(ven.J_IE)), "t_train_s": t_train,
           "t_total_s": time.time() - t0}
    ven.save(str(path.with_suffix(".npz")))
    path.write_text(json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--grid", default=None, help="JSON {param: [values]} (Cartesian product)")
    p.add_argument("--configs", default=None, help="JSON list of override dicts")
    p.add_argument("--n-rend", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", default=None,
                   help="encoder .npz (default outputs/causal_encoder.npz); its own cache")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    if a.encoder:
        os.environ["SWEEP_ENCODER"] = a.encoder
        global ENCODER, CACHE
        ENCODER = a.encoder
        CACHE = Path(f"outputs/sweeps/stimuli_cache_{Path(a.encoder).stem}.pkl")
    if a.grid:
        g = json.loads(a.grid)
        configs = [dict(zip(g, vals)) for vals in itertools.product(*g.values())]
    else:
        configs = json.loads(a.configs)
    out_dir = Path("outputs/sweeps") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    st = _stimuli()                    # build the cache once, before the workers start
    # Compile Brian2's code once too: workers compiling the same code at the same time can
    # fall back to the (much slower) numpy code generation.
    from spiking_ven.vocal_error_net import VocalErrorNet
    warm = VocalErrorNet(n_e=4, n_i=2, n_hvc=st["hvc_on"].shape[0], n_aud=st["n_channels"],
                         seed=0, dt_ms=configs[0].get("dt_ms", 0.1))
    warm.fit(st["hvc_on"][:, :5], st["aud_correct"][:, :5])
    jobs = [(c, str(out_dir), a.n_rend, a.seed) for c in configs]
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(run_one, jobs):
            r, s = res["daf"], res["score"]
            print(f"{res['id']} {json.dumps(res['overrides'])}\n"
                  f"    score {s['total']:.2f} | rate {r['baseline_hz_all']:.2f} "
                  f"resp {r['responder_fraction']:.2f} lat {r['latency_ms']:.1f} "
                  f"dur {r['duration_ms']:.1f} pk {r['peak_latency_ms']} "
                  f"late {r['late_over_peak']:.2f} xs {r['extra_spikes_responders']:.2f} "
                  f"rev/tr {res['rates']['reversed_over_trained']:.3f} "
                  f"song-excess rev/tr {res['contrast']['excess_ratio']:.2f} "
                  f"(diff {res['contrast']['excess_diff_hz']:+.2f} Hz) "
                  f"({res['t_total_s']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
