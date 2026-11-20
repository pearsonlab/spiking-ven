"""Short, monitored training run for plastic HVC -> I_HVC synapses.

    python scripts/monitor_hvc_plastic.py --out model.npz [--set kappa_h=0.01 ...]

Starts from the experiment baseline (``configs/baseline.json``, ``--config`` for another):
its model parameters, overridden by ``--set``, and its stimulus settings (encoder,
auditory normalisation, per-rendition rms) unless the ``SWEEP_*`` environment variables
already set them.

Trains as ``sweep.py`` does (per-cell rate targets, burn-in, then the training pool) and
every ``--every`` renditions prints: how concentrated each I_HVC cell's HVC input has
become (effective number of inputs, the participation ratio ``(sum w)^2 / sum w^2``; 540
uniform -> 1 for a single input) and its largest weight; the I_HVC rate and firing windows
per active cell; the mean I_HVC -> E weight; the E rate singing the trained song, the
reversed song and silent. One noise draw each, so the numbers are a trend, not a score.

Also writes ``<out>_trace.npz`` at every checkpoint (for ``plot_hvc_concentration.py``):
per-cell effective inputs, top-input weight fraction and I_HVC rate, and the full HVC
weight rows of ``--n-example`` cells, with each HVC neuron's burst time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

#: The experiment baseline (model, stimulus and training settings).
BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"


def windows(ih):
    """Firing windows per active cell: runs of spikes separated by > 10 ms."""
    out = []
    for row in ih:
        t = np.flatnonzero(row)
        if t.size:
            out.append(1 + int(np.sum(np.diff(t) > 10)))
    return np.array(out) if out else np.array([0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(BASELINE))
    p.add_argument("--n-rend", type=int, default=None,
                   help="renditions (default: the config's training.n_rend)")
    p.add_argument("--every", type=int, default=15)
    p.add_argument("--set", nargs="*", default=[], help="overrides, key=value")
    p.add_argument("--out", required=True)
    p.add_argument("--n-example", type=int, default=8)
    p.add_argument("--init", default=None,
                   help="continue training this saved model (skips targets and burn-in)")
    p.add_argument("--save-j", action="store_true",
                   help="also store the full I_HVC -> E weights and the E population PSTH "
                        "(singing the trained song) at each checkpoint")
    a = p.parse_args()
    base = json.loads(Path(a.config).read_text())
    cfg: dict[str, Any] = dict(base["model"])
    for kv in a.set:
        k, v = kv.split("=")
        cfg[k] = float(v)
    if a.n_rend is None:
        a.n_rend = int(base.get("training", {}).get("n_rend", 150))
    if not os.environ.get("SWEEP_ENCODER"):
        stim = base.get("stimulus", {})
        sweep.configure(stim.get("encoder"), stim.get("aud_norm"),
                        stim.get("aud_norm_power"), bool(stim.get("rms_per_rendition")))
    st = sweep._stimuli()
    T = int(st["T_song"])
    pre = st["hvc_prelude"]
    if a.init:
        ven = VocalErrorNet.load(a.init, seed=42)
    else:
        ven = VocalErrorNet(n_aud=st["n_channels"], seed=42, **cfg)
        ven.set_rate_targets(ven.n_hvc, ven.n_aud)
    rng = np.random.default_rng(7)
    M = np.asarray(ven.M_HH, dtype=bool)
    pool = st["correct_pool"]
    from scipy.ndimage import gaussian_filter1d
    burst_t = np.argmax(gaussian_filter1d(np.asarray(st["hvc_on"], dtype=float)[:, :T], 2,
                                          axis=1), axis=1)
    ex = np.random.default_rng(0).choice(ven.n_ih, a.n_example, replace=False)
    trace: dict[str, list] = {"rend": [], "eff_inputs": [], "top_frac": [], "ih_rate": [],
                              "example_rows": [], "e_sing": [], "e_rev": [], "e_quiet": [],
                              "i_h": [], "j_he": [], "e_psth": []}
    trace_path = a.out.replace(".npz", "_trace.npz")
    print(f"config: {cfg}", flush=True)
    print(" rend   eff.inputs(med/min)  maxw   IHVC Hz  silent  win/cell  J_HE   "
          "E sing  E rev  E silent   s/rend", flush=True)
    t0 = time.time()
    for r in range(a.n_rend + 1):
        if r % a.every == 0:
            B = np.where(M, ven.B_H, 0.0).astype(np.float64)
            pr = B.sum(1) ** 2 / np.maximum((B ** 2).sum(1), 1e-12)
            e_sing = ven.transform(st["hvc_on"], st["aud_correct"], rng=rng,
                                   hvc_prelude=pre)[:, :T]
            sing = e_sing.mean() * 1000
            ih = np.asarray(ven.last_ih)[:, :T]
            rate = ih.sum(1) / (T / 1000)
            rev = ven.transform(st["hvc_on"], st["aud_reversed"], rng=rng,
                                hvc_prelude=pre)[:, :T].mean() * 1000
            quiet = ven.transform(st["hvc_off"], np.zeros_like(st["aud_correct"]),
                                  rng=rng)[:, 20:T].mean() * 1000
            ih_quiet = float(np.asarray(ven.last_ih)[:, 20:T].mean() * 1000)
            i_h = np.asarray(ven.I_H, dtype=float)
            w = windows(ih)
            trace["rend"].append(r)
            trace["eff_inputs"].append(pr)
            trace["top_frac"].append(B.max(1) / np.maximum(B.sum(1), 1e-12))
            trace["ih_rate"].append(rate)
            trace["example_rows"].append(B[ex])
            trace["e_sing"].append(sing)
            trace["e_rev"].append(rev)
            trace["e_quiet"].append(quiet)
            if a.save_j:
                trace["j_he"].append(np.asarray(ven.J_HE, dtype=np.float16))
                trace["e_psth"].append(e_sing.mean(axis=0) * 1000)
            trace["i_h"].append(i_h)
            arrays: dict[str, Any] = {"burst_t": burst_t, "example": ex, "M_HH": M,
                                      **{k: np.asarray(v) for k, v in trace.items()}}
            np.savez(trace_path, **arrays)
            jhe = float(ven.J_HE[ven.M_HE].mean())
            dt = (time.time() - t0) / max(r, 1)
            print(f"{r:5d}   {np.median(pr):7.1f} / {pr.min():6.1f}   {B.max():5.1f}  "
                  f"{rate.mean():6.2f}  {np.mean(rate == 0):5.0%}  {np.median(w):6.0f}"
                  f"    {jhe:5.2f}  {sing:6.2f}  {rev:5.2f}  {quiet:6.2f}     {dt:5.1f}"
                  f"   IHVC silent {ih_quiet:.2f} Hz, I_h {i_h.mean():+.1f} pA "
                  f"[{i_h.min():+.0f}, {i_h.max():+.0f}], S_H median "
                  f"{np.median(ven.S_H):.0f} [{np.min(ven.S_H):.0f}, {np.max(ven.S_H):.0f}] nS",
                  flush=True)
            if not np.isfinite(B).all() or rate.mean() > 100:
                print("diverging; stopping", flush=True)
                break
        if r == a.n_rend:
            break
        if r == 0 and not a.init:
            ven.fit(st["hvc_burn"], st["aud_burn"])
        else:
            ven.fit(st["hvc_on"], pool[r % len(pool)], hvc_prelude=pre)
    ven.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
