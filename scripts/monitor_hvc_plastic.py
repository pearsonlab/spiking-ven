"""Short, monitored training run for plastic HVC -> I_HVC synapses.

    SWEEP_ENCODER=... SWEEP_AUD_NORM=0.3 SWEEP_AUD_NORM_POW=2 SWEEP_RMS_SELF=1 \\
    python scripts/monitor_hvc_plastic.py --n-rend 150 --every 15 --out model.npz \\
        --set kappa_h=0.02 tau_elig_ms=10

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
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # type: ignore  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

BASE = dict(c_ie=0.5, ie_init_mean_ns=6.0, ie_init_sd_ns=3.0, i_drive_i_pa=0.0,
            i_drive_e_pa=2500.0, c_ei=0.05, ei_mean_ns=11.0, ei_sd_ns=5.5, c_hvc=0.0,
            aud_mean_ns=400.0, aud_sd_ns=200.0, c_aud=0.05, g_nmda_mean_ns=0.0,
            n_ih=1200, rho_ns=0.0, c_hvc_h=0.9, hvc_h_mean_ns=0.9, hvc_h_sd_ns=0.0,
            vogels_h=1.0, eta_h_ns=3.0, c_he=10 / 600, hvc_plastic=1.0, kappa_h=0.02,
            tau_elig_ms=10.0, h_relax=0.02)


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
    p.add_argument("--n-rend", type=int, default=150)
    p.add_argument("--every", type=int, default=15)
    p.add_argument("--set", nargs="*", default=[], help="overrides, key=value")
    p.add_argument("--out", required=True)
    p.add_argument("--n-example", type=int, default=8)
    a = p.parse_args()
    cfg = dict(BASE)
    for kv in a.set:
        k, v = kv.split("=")
        cfg[k] = float(v)
    st = sweep._stimuli()
    T = int(st["T_song"])
    pre = st["hvc_prelude"]
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
                              "example_rows": [], "e_sing": [], "e_rev": [], "e_quiet": []}
    trace_path = a.out.replace(".npz", "_trace.npz")
    print(f"config: {cfg}", flush=True)
    print(" rend   eff.inputs(med/min)  maxw   IHVC Hz  silent  win/cell  J_HE   "
          "E sing  E rev  E silent   s/rend", flush=True)
    t0 = time.time()
    for r in range(a.n_rend + 1):
        if r % a.every == 0:
            B = np.where(M, ven.B_H, 0.0).astype(np.float64)
            pr = B.sum(1) ** 2 / np.maximum((B ** 2).sum(1), 1e-12)
            sing = ven.transform(st["hvc_on"], st["aud_correct"], rng=rng,
                                 hvc_prelude=pre)[:, :T].mean() * 1000
            ih = np.asarray(ven.last_ih)[:, :T]
            rate = ih.sum(1) / (T / 1000)
            rev = ven.transform(st["hvc_on"], st["aud_reversed"], rng=rng,
                                hvc_prelude=pre)[:, :T].mean() * 1000
            quiet = ven.transform(st["hvc_off"], np.zeros_like(st["aud_correct"]),
                                  rng=rng)[:, 20:T].mean() * 1000
            w = windows(ih)
            trace["rend"].append(r)
            trace["eff_inputs"].append(pr)
            trace["top_frac"].append(B.max(1) / np.maximum(B.sum(1), 1e-12))
            trace["ih_rate"].append(rate)
            trace["example_rows"].append(B[ex])
            trace["e_sing"].append(sing)
            trace["e_rev"].append(rev)
            trace["e_quiet"].append(quiet)
            np.savez(trace_path, burst_t=burst_t, example=ex, M_HH=M,
                     **{k: np.asarray(v) for k, v in trace.items()})
            jhe = float(ven.J_HE[ven.M_HE].mean())
            dt = (time.time() - t0) / max(r, 1)
            print(f"{r:5d}   {np.median(pr):7.1f} / {pr.min():6.1f}   {B.max():5.1f}  "
                  f"{rate.mean():6.2f}  {np.mean(rate == 0):5.0%}  {np.median(w):6.0f}"
                  f"    {jhe:5.2f}  {sing:6.2f}  {rev:5.2f}  {quiet:6.2f}     {dt:5.1f}",
                  flush=True)
            if not np.isfinite(B).all() or rate.mean() > 100:
                print("diverging; stopping", flush=True)
                break
        if r == a.n_rend:
            break
        if r == 0:
            ven.fit(st["hvc_burn"], st["aud_burn"])
        else:
            ven.fit(st["hvc_on"], pool[r % len(pool)], hvc_prelude=pre)
    ven.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
