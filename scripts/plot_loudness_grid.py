"""Small multiples: each refined colony corpus against song, one column per corpus.

    python scripts/plot_loudness_grid.py --song outputs/song_corpus_koch.npz --out fig.png \\
        --corpus "colony, all=outputs/colony_corpus_cr.npz" \\
        --corpus "colony >=-35 dB, RMS-normalised=outputs/colony_corpus_cr_active35.npz:rms"

Levels as in ``plot_loudness.py`` (10 ms frames, 200-8000 Hz, dB re full scale, as the
trainer sees them: song bouts RMS-normalised, prescaled corpora as stored). A ``:rms`` suffix
on a corpus path normalises each of its segments to RMS ``CAUSAL_RMS_REF`` the way song bouts
are, to show what that treatment would do. Top row: frame-level histogram; bottom row: share
of total power by level. Song (grey-free orange) is repeated in every column for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_loudness import frame_db  # noqa: E402

from spiking_ven.constants import CAUSAL_HI_HZ, CAUSAL_LO_HZ, CAUSAL_RMS_REF  # noqa: E402

COLONY, SONG = "#2a78d6", "#eb6834"   # categorical slots 1, 2


def levels(path: str, renorm: bool) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    sr = int(d["sr"])
    pre = "prescaled" in d.files and bool(d["prescaled"])
    sos = butter(4, [CAUSAL_LO_HZ, min(CAUSAL_HI_HZ, 0.95 * sr / 2)], "bandpass", fs=sr,
                 output="sos")
    out = []
    for s0, n0 in zip(d["starts"], d["lengths"]):
        x = d["audio"][s0:s0 + n0].astype(np.float64)
        if renorm or not pre:
            x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12) * CAUSAL_RMS_REF
        out.append(frame_db(x, sr, sos))
    return np.concatenate(out)


def summary(v: np.ndarray) -> str:
    pw = 10 ** (v / 10)
    top = np.sort(pw)[::-1][: max(1, v.size // 100)].sum() / pw.sum()
    return f"median {np.median(v):.0f} dB, IQR {np.subtract(*np.percentile(v, [75, 25])):.0f} dB\n" \
           f"top 1% of frames: {top:.0%} of power"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--song", required=True)
    p.add_argument("--corpus", action="append", required=True, metavar="LABEL=PATH[:rms]")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    song = levels(a.song, False)
    cols = []
    for spec in a.corpus:
        label, path = spec.rsplit("=", 1)
        renorm = path.endswith(":rms")
        cols.append((label, levels(path[:-4] if renorm else path, renorm)))
    allv = np.concatenate([song] + [v for _, v in cols])
    bins = np.arange(np.floor(np.percentile(allv, 0.1)), np.ceil(allv.max()) + 1, 1.0)
    ref = 20 * np.log10(CAUSAL_RMS_REF)

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.2), sharex=True, sharey="row",
                             squeeze=False, constrained_layout=True)
    hs, _ = np.histogram(song, bins, density=True)
    ps, _ = np.histogram(song, bins, weights=10 ** (song / 10) / (10 ** (song / 10)).sum())
    for j, (label, v) in enumerate(cols):
        h, _ = np.histogram(v, bins, density=True)
        pw = 10 ** (v / 10)
        pp, _ = np.histogram(v, bins, weights=pw / pw.sum())
        for ax, cv, sv in ((axes[0, j], h, hs), (axes[1, j], pp, ps)):
            ax.step(bins[:-1], sv, where="post", color=SONG, lw=1.8, label="song")
            ax.step(bins[:-1], cv, where="post", color=COLONY, lw=1.8, label="colony")
            ax.axvline(ref, color="0.45", ls="--", lw=1)
            ax.grid(color="0.92", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, j].set_title(label, fontsize=9)
        axes[0, j].text(0.02, 0.97, f"colony: {summary(v)}\nsong: {summary(song)}",
                        transform=axes[0, j].transAxes, fontsize=6.5, va="top")
        axes[1, j].set_xlabel("10 ms frame level (dB re FS)", fontsize=8)
    axes[0, 0].set_ylabel("density of frames")
    axes[1, 0].set_ylabel("share of total power")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("refined colony corpora (blue) vs song (orange); dashed = training reference "
                 "(RMS 0.1)", fontsize=10)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    print(f"{'song':34s} {summary(song).replace(chr(10), '; ')}")
    for label, v in cols:
        print(f"{label:34s} {summary(v).replace(chr(10), '; ')}")


if __name__ == "__main__":
    main()
