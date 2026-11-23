"""Time course of the DAF response across all trials: latency, peak, post-DAF dip, rebound.

    python scripts/daf_shape.py --model M.npz --cache stimuli.pkl --encoder ENC.npz \\
        --out fig.png [--label "..."]

The same trials and responder test ``metrics.daf_response`` scores (``daf_trials`` with
the cache's DAF inputs; ``daf_responders``), over longer windows than the metric keeps,
for fixed noise onsets (the cache's ``daf_window_s``) and for randomised onsets inside
syllables (``daf_onsets``, each trial aligned to its own onset). Per column: responder
PSTH as change from each cell's pre-onset rate (mean +- s.e.m. over responders),
non-responders and all E cells; and a heatmap of the responder-averaged response on every
trial. Reference marks: the paper's peak latency range (35-75 ms) and ~20 Hz peak
(a reading of Mandelblat-Cerf et al. 2014 Fig 8E, not data).

The simulated trials are cached beside the model (``spiking_ven.sim_cache``), so redrawing
the figure for the same model and stimuli takes seconds.
"""

from __future__ import annotations

import argparse
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from spiking_ven.causal_filters import CausalFilterEncoder  # noqa: E402
from spiking_ven.metrics import (daf_inputs, daf_onsets, daf_responders, daf_trials,  # noqa: E402
                                 first_peak)
from spiking_ven.sim_cache import cached_arrays  # noqa: E402
from spiking_ven.stimuli import daf_window_ms  # noqa: E402
from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

PRE = 150          # ms before onset (also the responder test window)
SMOOTH = 4.0       # ms, Gaussian, for the traces
C_R, C_N, C_A = "#c0392b", "#5f5e58", "#2a78d6"


def analyse(trials, onset, post, pre_plot=PRE):
    """Align, find responders, return (t, resp_change (n_resp, t), nonresp, all, per-trial).

    The responder test and each cell's baseline use the PRE ms before onset; the returned
    traces start ``pre_plot`` ms before it (>= PRE)."""
    resp, pre, _ = daf_responders(trials, onset_ms=onset, window_ms=PRE, alpha=0.02)
    seg = trials[:, :, onset - pre_plot:onset + post] * 1000.0     # (trials, n_e, t) Hz
    base = seg[:, :, pre_plot - PRE:pre_plot].mean(axis=(0, 2))    # each cell's pre rate
    cell = seg.mean(axis=0) - base[:, None]                        # (n_e, t) change
    t = np.arange(-pre_plot, post)
    per_trial = (seg[:, resp, :] - base[resp][None, :, None]).mean(axis=1) if resp.any() \
        else np.zeros((seg.shape[0], seg.shape[2]))
    return t, cell[resp], cell[~resp], cell, per_trial, resp


def panel(ax, ax_h, t, r, nr, allc, per_trial, resp, burst_ms, title, xlim=None):
    z = int(np.searchsorted(t, 0))                                 # index of onset
    sm = lambda x: gaussian_filter1d(x, SMOOTH, axis=-1)  # noqa: E731
    ax.axvspan(0, burst_ms, color="#ecebe7", lw=0, zorder=0)
    ax.axvspan(35, 75, ymin=0.97, ymax=1.0, color="#e0c06a", lw=0, zorder=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(20, color="#e0c06a", lw=0.8, ls=":")
    if len(r):
        m = sm(r.mean(0))
        se = sm(r.std(0) / np.sqrt(len(r)))
        ax.fill_between(t, m - se, m + se, color=C_R, alpha=0.2, lw=0)
        ax.plot(t, m, color=C_R, lw=1.6, label=f"responders (n={len(r)})")
        # Latency from the first prominent peak; magnitude from the largest. When they
        # differ the response has more than one hump, and both are marked.
        k = first_peak(m[z:], smooth=0) + z
        g = int(np.argmax(m[z:])) + z
        ax.annotate(f"first peak +{m[k]:.0f} Hz at {t[k]} ms", (t[k], m[k]),
                    xytext=(t[k] + 25, m[k]), fontsize=7, color=C_R,
                    arrowprops=dict(arrowstyle="-", color=C_R, lw=0.6))
        if abs(t[g] - t[k]) > 5:
            ax.annotate(f"max +{m[g]:.0f} Hz at {t[g]} ms", (t[g], m[g]),
                        xytext=(t[g] + 25, m[g] + 1.5), fontsize=7, color=C_R,
                        arrowprops=dict(arrowstyle="-", color=C_R, lw=0.6))
        post_m = m[z + burst_ms:]
        j = int(np.argmin(post_m)) + z + burst_ms
        ax.annotate(f"dip {m[j]:+.1f} Hz at {t[j]} ms", (t[j], m[j]),
                    xytext=(t[j] + 20, m[j] - 4), fontsize=7, color=C_R,
                    arrowprops=dict(arrowstyle="-", color=C_R, lw=0.6))
    ax.plot(t, sm(nr.mean(0)), color=C_N, lw=1.1, label="non-responders")
    ax.plot(t, sm(allc.mean(0)), color=C_A, lw=1.1, ls="--", label="all E cells")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_ylabel("rate change from\npre-onset (Hz)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    v = float(np.percentile(np.abs(sm(per_trial)), 99)) or 1.0
    ax_h.imshow(sm(per_trial), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                extent=(t[0], t[-1] + 1, per_trial.shape[0], 0))
    ax_h.axvline(0, color="k", lw=0.6)
    ax_h.axvline(burst_ms, color="k", lw=0.6, ls=":")
    ax_h.set_ylabel("trial")
    ax_h.set_xlabel("time from noise onset (ms)")
    if xlim is not None:
        ax.set_xlim(*xlim)
        ax_h.set_xlim(*xlim)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--post-fixed", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="x-axis range in ms (e.g. -500 1500 for Mandelblat-Cerf's scale); "
                        "the fixed-onset trace then uses all the rendition has before and "
                        "after onset, and the rest of the axis is left blank")
    a = p.parse_args()
    with open(a.cache, "rb") as fh:
        st = pickle.load(fh)
    enc = CausalFilterEncoder.load(a.encoder)
    ven = VocalErrorNet.load(a.model, seed=a.seed)
    onset, burst = daf_window_ms(st["daf_window_s"])
    burst_ms = int(round(burst))
    T = int(st["T_song"])

    def simulate():
        fixed = daf_trials(ven, enc, st, seed=a.seed, inputs=st["daf_inputs"])
        r_in = daf_inputs(enc, st, seed=a.seed, random_onsets=True)
        rnd = daf_trials(ven, enc, st, seed=a.seed, inputs=r_in)
        return {"fixed": fixed, "random": rnd}

    sims = cached_arrays(a.model, "daf_trials",
                         {"stimuli": str(a.cache), "encoder": str(a.encoder), "seed": a.seed},
                         simulate)
    fixed = sims["fixed"].astype(np.float32)
    rnd = sims["random"].astype(np.float32)
    if a.xlim is not None:
        post_f = fixed.shape[2] - onset
        pre_f = max(PRE, min(onset, int(-a.xlim[0])))
    else:
        post_f, pre_f = min(a.post_fixed, T - onset), PRE
    res_f = analyse(fixed, onset, post_f, pre_f)
    ons = daf_onsets(st, n_trials=rnd.shape[0], seed=a.seed)        # pre 150, post 300
    post_r = 300
    aligned = np.stack([rnd[k, :, o - PRE:o + post_r] for k, o in enumerate(ons)])
    res_r = analyse(aligned, PRE, post_r)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    panel(axes[0, 0], axes[1, 0], *res_f[:5], res_f[5], burst_ms,
          f"Fixed onset ({onset} ms into the motif), {fixed.shape[0]} trials", a.xlim)
    panel(axes[0, 1], axes[1, 1], *res_r[:5], res_r[5], burst_ms,
          f"Random onsets inside syllables, {rnd.shape[0]} trials (each aligned)", a.xlim)
    fig.suptitle(f"DAF response time course{': ' + a.label if a.label else ''}  "
                 f"(grey: noise; gold: paper's peak-latency range and ~20 Hz peak, "
                 f"read off Fig 8E)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    summary = {"label": a.label, "model": a.model, "encoder": a.encoder,
               "fixed": summarize(res_f, burst_ms), "random": summarize(res_r, burst_ms)}
    out_json = a.out.rsplit(".", 1)[0] + ".json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1)
    for k in ("fixed", "random"):
        s = summary[k]
        print(f"{k}: responders {s['n_responders']}, first peak {s['first_peak_hz']:+.1f} Hz at "
              f"{s['first_peak_ms']} ms, max {s['max_hz']:+.1f} Hz at {s['max_ms']} ms, dip "
              f"{s['dip_hz']:+.1f} Hz at {s['dip_ms']} ms, tail 50-150 ms after offset "
              f"{s['tail_frac']:.2f} of max")


def summarize(res, burst_ms) -> dict:
    """Responder-mean time-course numbers (same smoothing and peak rules as the figure)."""
    t, r = res[0], res[1]
    z = int(np.searchsorted(t, 0))
    if not len(r):
        return {"n_responders": 0, "first_peak_hz": 0.0, "first_peak_ms": 0, "max_hz": 0.0,
                "max_ms": 0, "dip_hz": 0.0, "dip_ms": 0, "tail_frac": 0.0}
    m = gaussian_filter1d(r.mean(0), SMOOTH)
    k = first_peak(m[z:], smooth=0) + z
    g = int(np.argmax(m[z:])) + z
    j = int(np.argmin(m[z + burst_ms:])) + z + burst_ms
    tail = m[(t >= burst_ms + 50) & (t < burst_ms + 150)]
    return {"n_responders": int(len(r)), "first_peak_hz": float(m[k]), "first_peak_ms": int(t[k]),
            "max_hz": float(m[g]), "max_ms": int(t[g]), "dip_hz": float(m[j]),
            "dip_ms": int(t[j]),
            "tail_frac": float(tail.mean() / m[g]) if tail.size and m[g] > 0 else 0.0}


if __name__ == "__main__":
    main()
