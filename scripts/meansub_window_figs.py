"""Figures comparing running-mean subtractions (existing filters and network).

    python scripts/meansub_window_figs.py --model M.npz --norm-sigma 1 --out-dir DIR

For windows of 16 / 8 / 4 cochleagram frames: (1) auditory input and the responders' DAF
PSTH, both aligned to randomised noise onsets; (2) the cancellation figure per window.
The encoder's filters and the network were trained with the 16-frame window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiking_ven.causal_filters import CausalFilterEncoder  # noqa: E402
from spiking_ven.figures.cancellation import make_figure  # noqa: E402
from spiking_ven.metrics import daf_inputs, daf_onsets, daf_response  # noqa: E402
from spiking_ven.stimuli import build_stimuli  # noqa: E402
from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

#: (label, boxcar frames, exponential tau frames or None)
VARIANTS = (("boxcar 16 frames (32 ms)", 16, None), ("exp tau 8 frames (16 ms)", 16, 8.0),
            ("exp tau 4 frames (8 ms)", 16, 4.0), ("exp tau 2 frames (4 ms)", 16, 2.0))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
INK2, GRID, BAND = "#5f5e58", "#e4e3df", "#ecebe7"


def binned(x, b):
    n = len(x) // b
    return x[:n * b].reshape(n, b).mean(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--norm-sigma", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", default=None,
                   help='JSON list of [label, boxcar frames, tau frames or null]')
    p.add_argument("--tag", default="meansub_exp_daf", help="comparison figure name")
    a = p.parse_args()
    variants = VARIANTS if a.variants is None else [tuple(v) for v in json.loads(a.variants)]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for k, (lab, w, tau) in enumerate(variants):
        enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
        enc.meansub_frames = w
        enc.meansub_tau = tau
        st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600, n_ista=50,
                           norm_sigma=a.norm_sigma, verbose=False)
        rin = daf_inputs(enc, st, seed=42, random_onsets=True)
        on = daf_onsets(st, seed=42)
        aud = np.stack([rin[t, :, o - 150:o + 150] for t, o in enumerate(on)]).sum(1).mean(0)
        v = VocalErrorNet.load(a.model, seed=42)
        r = daf_response(v, enc, st, seed=42, inputs=rin, random_onsets=True)
        res[k] = (aud * 1000 / rin.shape[1], np.array(r["psth_responders"]), r)
        tag = f"_box{w:g}" if tau is None else f"_exp{tau:g}"
        make_figure(VocalErrorNet.load(a.model, seed=42), st, tag, out_dir=out)
        print(f"{lab}: fwd/rev corr {st['fwd_rev_corr']:.3f} dur {r['duration_ms']:.1f} "
              f"xs {r['extra_spikes_responders']:.2f} resp {r['responder_fraction']:.2f} "
              f"lat {r['latency_ms']:.1f} pk {r['peak_latency_ms']} "
              f"late {r['late_over_peak']:.2f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    t = np.arange(-150, 150)
    b = 5
    tb = binned(t.astype(float), b)
    for k, (lab, w, tau) in enumerate(variants):
        aud, psth, r = res[k]
        axes[0].plot(tb, binned(aud, b), color=COLORS[k], lw=1.6, label=lab)
        pb = binned(psth, b)
        pb = pb - pb[tb < 0].mean()
        axes[1].plot(tb, pb, color=COLORS[k], lw=1.6,
                     label=f"{lab}: duration {r['duration_ms']:.0f} ms, "
                           f"{r['extra_spikes_responders']:.2f} extra spikes")
    for ax, ylab, title in ((axes[0], "auditory rate per channel (Hz)",
                             "Auditory input around the noise burst"),
                            (axes[1], "rate change, responders (Hz)",
                             "Responder DAF PSTH (5 ms bins)")):
        ax.axvspan(0, 50, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("time from noise onset (ms)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Running-mean subtraction, boxcar vs exponential: noise onsets randomised; "
                 "filters and network trained with the 16-frame boxcar", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = out / f"{a.tag}.png"
    fig.savefig(path, dpi=170)
    print("wrote", path)


if __name__ == "__main__":
    main()
