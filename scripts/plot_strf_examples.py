"""Example STRFs from one encoder, large enough to read.

    python scripts/plot_strf_examples.py FILE.npz --out fig.png [--title colony] [--n 32]

Accepts what ``plot_strfs.py`` accepts. Picks ``--n`` live units spread evenly across
preferred frequency (as ``plot_strfs.py`` does) and draws them in a grid with real axes:
lag in ms (0 = newest frame, at the right) and channel centre frequency in kHz. Each panel
has its own symmetric colour scale (red excitatory, blue suppressive) and is titled with
the unit's energy-weighted mean lag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_strfs import load_strfs  # noqa: E402

from spiking_ven.constants import CAUSAL_HI_HZ, CAUSAL_LO_HZ  # noqa: E402
from spiking_ven.filterbank import centre_frequencies  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--sr", type=int, default=16000)
    a = p.parse_args()

    V, fm, note = load_strfs(a.file)
    K, C, L = V.shape
    lag = (L - 1 - np.arange(L)) * fm + fm / 2            # column j's lag, oldest first
    cf = centre_frequencies(a.sr, C, CAUSAL_LO_HZ, CAUSAL_HI_HZ)[::-1] / 1000  # ch 0 lowest
    e = (V ** 2).sum(axis=1)
    e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
    mlag = (e * lag).sum(axis=1)
    pref = np.argmax(np.abs(V).sum(axis=2), axis=1)
    live = np.flatnonzero(np.abs(V).reshape(K, -1).max(axis=1) > 0)
    order = live[np.argsort(pref[live], kind="stable")]
    pick = order[np.linspace(0, len(order) - 1, min(a.n, len(order))).astype(int)]

    rows = int(np.ceil(len(pick) / a.cols))
    fig, axes = plt.subplots(rows, a.cols, figsize=(a.cols * 1.6, rows * 1.75 + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    yt = [i for i in range(C) if i % 10 == 0] + [C - 1]
    for j, ax in enumerate(axes.flat):
        if j >= len(pick):
            ax.axis("off")
            continue
        k = pick[j]
        w = V[k]
        v = float(np.abs(w).max()) or 1.0
        ax.imshow(w, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=(-L * fm, 0, -0.5, C - 0.5), interpolation="nearest")
        ax.set_title(f"unit {k}: {mlag[k]:.0f} ms", fontsize=7, pad=2)
        ax.tick_params(labelsize=6, length=2)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{cf[i]:.1f}" for i in yt])
    for ax in axes[-1]:
        ax.set_xlabel("lag (ms)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz", fontsize=7)
    title = a.title or Path(a.file).stem
    med = float(np.median(mlag[live]))
    fig.suptitle(f"{title} ({note}): {len(pick)} of {len(live)} units, spread over "
                 f"preferred frequency; median mean lag {med:.1f} ms. "
                 "Red excitatory, blue suppressive; own scale per panel",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
