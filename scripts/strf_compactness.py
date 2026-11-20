"""Compare S&W encoders on STRF compactness and how long DAF noise lingers in their output.

    python scripts/strf_compactness.py --encoders A.npz B.npz --labels a b --out fig.png

Per encoder: share of STRF energy in the most recent 20 ms of the 64 ms window and the
energy-weighted mean lag (Schneider & Woolley 2013, Fig S6D: primary-AC STRFs are drawn over
0-60 ms with their energy in the first ~15 ms); reconstruction of the template motif; and,
for the DAF stimulus, the raw (unnormalized) higher-AC population activity around the noise
burst -- its peak time and how long after noise offset it takes to fall to 10% of the peak.
The figure shows that time course and eight example STRFs per encoder.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from spiking_ven.corpus import load_corpus  # noqa: E402
from spiking_ven.stimuli import daf_waveform  # noqa: E402
from spiking_ven.sw_encoder import SWEncoder  # noqa: E402

ON, NOISE = 400, 50       # DAF burst at 400-450 ms


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    corpus = load_corpus(a.motifs)
    sig, _ = corpus.template()
    sr = int(corpus.sr)
    daf = daf_waveform(sig, sr, seed=42)
    n = len(a.encoders)
    fig = plt.figure(figsize=(13, 2.2 + 1.3 * n))
    gs = fig.add_gridspec(n + 1, 9, height_ratios=[2.2] + [1] * n, hspace=0.5, wspace=0.15)
    ax_t = fig.add_subplot(gs[0, :])
    t = np.arange(-100, 250)
    for r, (path, lab) in enumerate(zip(a.encoders, a.labels)):
        enc = SWEncoder.load(path)
        C, L = int(enc.coch_params["n_channels"]), enc.L
        V = enc.V                                            # (n, C, L), oldest lag first
        ms = 1000 // int(enc.coch_params["frame_rate"])
        e_lag = (V ** 2).sum(axis=1)                         # (n, L)
        e_lag = e_lag / np.maximum(e_lag.sum(axis=1, keepdims=True), 1e-12)
        lag_ms = (L - 1 - np.arange(L)) * ms + ms / 2        # newest column = 1 ms
        recent = float(np.median(e_lag[:, lag_ms <= 20].sum(axis=1)))
        mean_lag = float(np.median((e_lag * lag_ms).sum(axis=1)))
        X = enc.cochleagram(sig, sr=sr)
        R = enc.reconstruct(X)
        Xp = np.concatenate([np.zeros((C, 1)), X[:, :-1]], axis=1)
        ve = 1 - ((R - Xp)[:, L:] ** 2).sum() / ((Xp - Xp.mean())[:, L:] ** 2).sum()
        acts = enc.encode_signal(daf, sr=sr, rms_from=sig, T_out_ms=ON + 300, seed=42)
        pop = gaussian_filter1d(acts.sum(axis=0).astype(float), 2)
        seg = pop[ON - 100:ON + 250]
        base = float(pop[ON - 100:ON].mean())
        pk = int(np.argmax(seg[100:])) + 100
        above = np.flatnonzero(seg[pk:] - base > 0.1 * (seg[pk] - base))
        end = pk + (above[-1] if above.size and above[-1] < len(seg) - pk - 1 else 0)
        tail = (t[end] - NOISE) if end > pk else np.nan
        print(f"{lab:14s} STRF energy <=20 ms {recent:.0%}, mean lag {mean_lag:.0f} ms; "
              f"recon {ve:.0%}; DAF raw peak at {t[pk]} ms, falls to 10% "
              f"{tail:.0f} ms after noise offset", flush=True)
        ax_t.plot(t, (seg - base) / max(seg[pk] - base, 1e-9), lw=1.3,
                  label=f"{lab} (peak {t[pk]} ms, tail {tail:.0f} ms)")
        pref = np.argmax(np.abs(V).sum(axis=2), axis=1)
        pick = np.argsort(pref)[np.linspace(0, len(pref) - 1, 8).astype(int)]
        axl = fig.add_subplot(gs[r + 1, 0])
        axl.axis("off")
        axl.text(0, 0.5, f"{lab}\n<=20 ms: {recent:.0%}\nmean lag {mean_lag:.0f} ms\n"
                 f"recon {ve:.0%}", fontsize=7, va="center")
        for j, k in enumerate(pick):
            ax = fig.add_subplot(gs[r + 1, j + 1])
            v = float(np.abs(V[k]).max()) or 1.0
            ax.imshow(V[k], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=(-L * ms, 0, 0, C))
            ax.set_xticks([-60, -30, 0] if r == n - 1 else [])
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
    ax_t.axvspan(0, NOISE, color="#ecebe7", lw=0)
    ax_t.axhline(0, color="0.6", lw=0.8)
    ax_t.set_xlabel("time from noise onset (ms)", fontsize=8)
    ax_t.set_ylabel("raw higher-AC\npopulation (peak = 1)", fontsize=8)
    ax_t.legend(fontsize=7, frameon=False)
    ax_t.set_title("DAF in the encoder output (no normalization), and example STRFs "
                   "(lag -64..0 ms)", loc="left", fontsize=9)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
