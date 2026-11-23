"""Loudness distributions of training corpora, as the encoder trainer sees them.

    python scripts/plot_loudness.py --colony outputs/colony_corpus_cr.npz \\
        --song outputs/song_corpus_koch.npz --out fig.png [--events events.png --n-events 12]

Level = mean square of 10 ms frames (hop 5 ms) in the cochleagram band (200-8000 Hz), in dB
re full scale, after the trainer's scaling: song bouts are each normalised to RMS
``CAUSAL_RMS_REF`` (as ``train_sw_encoder.py`` does for a corpus that is not prescaled);
a prescaled corpus (the colony) is used as stored. RMS ``CAUSAL_RMS_REF`` (-20 dB) is the
level the colony build maps to ``ref_spl_db`` (80 dB SPL), so the top axis gives nominal SPL.

Left: histogram of frame levels. Right: share of total power at each level (what dominates
the input variance). With ``--events``, the loudest colony frames (at least 1 s apart) are
drawn as spectrograms labelled with source file and time, to listen to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfilt, spectrogram  # noqa: E402

from spiking_ven.constants import CAUSAL_HI_HZ, CAUSAL_LO_HZ, CAUSAL_RMS_REF  # noqa: E402

COLORS = {"colony": "#2a78d6", "song": "#eb6834"}   # categorical slots 1, 2


def frame_db(x: np.ndarray, sr: int, sos) -> np.ndarray:
    """dB of the band-passed signal's mean square in 10 ms frames, hop 5 ms."""
    y = sosfilt(sos, x)
    n, hop = int(0.010 * sr), int(0.005 * sr)
    if len(y) < n:
        return np.empty(0)
    f = np.lib.stride_tricks.sliding_window_view(y, n)[::hop]
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-20)


def segments(path: str):
    d = np.load(path, allow_pickle=False)
    sr = int(d["sr"])
    pre = "prescaled" in d.files and bool(d["prescaled"])
    out = []
    for i, (s0, n0) in enumerate(zip(d["starts"], d["lengths"])):
        x = d["audio"][s0:s0 + n0].astype(np.float64)
        if not pre:
            x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12) * CAUSAL_RMS_REF
        out.append((i, x))
    return d, sr, pre, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colony", required=True)
    p.add_argument("--song", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--events", default=None, help="also draw the loudest colony events here")
    p.add_argument("--n-events", type=int, default=12)
    a = p.parse_args()

    levels, meta = {}, {}
    for name, path in (("colony", a.colony), ("song", a.song)):
        d, sr, pre, segs = segments(path)
        sos = butter(4, [CAUSAL_LO_HZ, min(CAUSAL_HI_HZ, 0.95 * sr / 2)], "bandpass", fs=sr,
                     output="sos")
        per = [frame_db(x, sr, sos) for _, x in segs]
        levels[name] = np.concatenate(per)
        meta[name] = (d, sr, sos, segs, per)

    ref = 20 * np.log10(CAUSAL_RMS_REF)
    spl0 = float(meta["colony"][0]["ref_spl_db"]) - ref if "ref_spl_db" in meta["colony"][0].files \
        else None
    lo = min(np.percentile(v, 0.1) for v in levels.values())
    hi = max(v.max() for v in levels.values())
    bins = np.arange(np.floor(lo), np.ceil(hi) + 1, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, v in levels.items():
        c = COLORS[name]
        h, _ = np.histogram(v, bins, density=True)
        ax1.step(bins[:-1], h, where="post", color=c, lw=2,
                 label=f"{name}: median {np.median(v):.1f}, 99.9% {np.percentile(v, 99.9):.1f}, "
                       f"max {v.max():.1f} dB")
        pw = 10 ** (v / 10)
        hp, _ = np.histogram(v, bins, weights=pw / pw.sum())
        top = np.sort(pw)[::-1]
        share1 = top[: max(1, len(top) // 100)].sum() / pw.sum()
        ax2.step(bins[:-1], hp, where="post", color=c, lw=2,
                 label=f"{name}: loudest 1% of frames carry {share1:.0%} of power")
    for ax in (ax1, ax2):
        ax.axvline(ref, color="0.45", ls="--", lw=1.2)
        ax.grid(color="0.92", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlabel("10 ms frame level (dB re full scale; dashed = training reference)")
        if spl0 is not None:
            sec = ax.secondary_xaxis("top", functions=(lambda x: x + spl0, lambda x: x - spl0))
            sec.set_xlabel("nominal dB SPL (colony calibration)", fontsize=8)
    ax1.set(ylabel="density of frames", title="how often each level occurs")
    ax2.set(ylabel="share of total power", title="what carries the input variance")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    for name, v in levels.items():
        print(f"{name:6s} frames {v.size}  pct 1/50/90/99/99.9/max: "
              + " ".join(f"{q:.1f}" for q in np.percentile(v, [1, 50, 90, 99, 99.9, 100])))

    if not a.events:
        return
    d, sr, _, segs, per = meta["colony"]
    src = d["source"] if "source" in d.files else np.array([f"seg{i}" for i in range(len(segs))])
    # time of each segment within its source file (files split into consecutive pieces)
    seg_t0, seen = [], {}
    for i, (s0, n0) in enumerate(zip(d["starts"], d["lengths"])):
        k = str(src[i])
        seg_t0.append(seen.get(k, 0.0))
        seen[k] = seen.get(k, 0.0) + n0 / sr
    hop = 0.005
    cand = sorted(((lv, i, j) for i, f in enumerate(per) for j, lv in enumerate(f)), reverse=True)
    picks: list[tuple[float, int, float]] = []
    for lv, i, j in cand:
        t = j * hop
        if all(not (i == pi and abs(t - pt) < 1.0) for _, pi, pt in picks):
            picks.append((lv, i, t))
        if len(picks) >= a.n_events:
            break
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.6), squeeze=False,
                             constrained_layout=True)
    for ax, (lv, i, t) in zip(axes.flat, picks):
        x = segs[i][1]
        c = int(t * sr)
        w = x[max(0, c - sr // 2): c + sr // 2]
        f, tt, S = spectrogram(w, fs=sr, nperseg=256, noverlap=192)
        ax.pcolormesh(tt - 0.5, f / 1000, 10 * np.log10(S + 1e-20), shading="auto",
                      cmap="magma", vmin=10 * np.log10(S.max()) - 70)
        ts = seg_t0[i] + t
        ax.set_title(f"{lv - ref:+.0f} dB re ref | {src[i]}\n@ {ts:.2f} s", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("s from peak", fontsize=6)
        ax.set_ylabel("kHz", fontsize=6)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle("loudest colony events (1 s windows, at least 1 s apart)", fontsize=9)
    fig.savefig(a.events, dpi=150)
    print("wrote", a.events)
    for lv, i, t in picks:
        print(f"  {lv - ref:+5.1f} dB re ref  {src[i]}  @ {seg_t0[i] + t:7.2f} s")


if __name__ == "__main__":
    main()
