"""Rasters for a network with an I_HVC population: auditory, I_bal, I_HVC, E, E rate.

    python scripts/raster_figure.py --model MODEL.npz --out fig.png [--title "..."]

Three columns as the cancellation figure (training song, time-reversed motif, DAF), all
while singing (HVC on, after the intro notes). I_HVC rows are sorted by each cell's first
spike, so the HVC-driven basis reads as a diagonal. The E rate is over all E cells,
Gaussian-smoothed (6 ms).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # type: ignore  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

CLR = {"aud": "#1baf7a", "I": "#eb6834", "IH": "#e87ba4", "E": "#2a78d6"}
N_SHOW = 64


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = p.parse_args()
    st = sweep._stimuli()
    T = st["T_song"]
    v = VocalErrorNet.load(a.model, seed=7)
    rng = np.random.default_rng(0)
    pick = {k: np.sort(rng.choice(n, min(N_SHOW, n), replace=False))
            for k, n in (("I", v.n_i), ("E", v.n_e), ("IH", max(v.n_ih, 1)))}
    cols = [("Training song", st["aud_correct"]), ("Time-reversed motif", st["aud_reversed"]),
            ("Distorted auditory feedback", st["aud_daf"])]
    runs = []
    for _, aud in cols:
        se, si = v.transform_all(st["hvc_on"], aud, hvc_prelude=st["hvc_prelude"])
        runs.append((aud[:, :T], si[:, :T], None if v.last_ih is None else v.last_ih[:, :T],
                     se[:, :T]))
    order = None
    if runs[0][2] is not None:
        first = np.array([np.flatnonzero(r)[0] if r.any() else T for r in runs[0][2]])
        order = np.argsort(first)[:: max(1, v.n_ih // N_SHOW)][:N_SHOW]
    rows = ["aud", "I", "IH", "E", "rate"] if order is not None else ["aud", "I", "E", "rate"]
    labels = {"aud": "Auditory", "I": "I_bal", "IH": "I_HVC (sorted)", "E": "E", "rate": "E rate (Hz)"}
    ymax = 1.15 * max(gaussian_filter1d(r[3].mean(0) * 1000, 6).max() for r in runs)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 1.6 * len(rows) + 0.6), sharex=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    for c, ((title, _), (aud, si, ih, se)) in enumerate(zip(cols, runs)):
        data = {"aud": aud, "I": si[pick["I"]], "E": se[pick["E"]]}
        if order is not None:
            data["IH"] = ih[order]
        for r, key in enumerate(rows):
            ax = axes[r, c]
            if key == "rate":
                ax.plot(gaussian_filter1d(se.mean(0) * 1000, 6), color=CLR["E"], lw=1.2)
                ax.set_ylim(0, ymax)
            else:
                m = data[key]
                yy, tt = np.nonzero(m)
                ax.scatter(tt, yy, s=1.2, color=CLR[key], marker="|", linewidths=0.8)
                ax.set_ylim(-1, m.shape[0])
                ax.set_yticks([])
                full = {"aud": aud, "I": si, "IH": ih, "E": se}[key]
                ax.text(0.99, 0.92, f"{full.mean() * 1000:.0f} Hz", transform=ax.transAxes,
                        ha="right", fontsize=7, color="#5f5e58")
            if c == 0:
                ax.set_ylabel(labels[key], fontsize=9)
            if c > 0 and key == "rate":
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[-1, c].set_xlabel("time in motif (ms)")
    if a.title:
        fig.suptitle(a.title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(a.out, dpi=170, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
