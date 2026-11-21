"""Contact sheet for an STRF-penalty sweep of the S&W encoder.

    python scripts/sw_penalty_sheet.py --encoders outputs/encoders/pen/*.npz --out FIG.png

One row per encoder: variance explained on the template motif, fraction of zero STRF entries,
units kept, and eight STRFs evenly spaced in preferred frequency.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiking_ven.corpus import load_corpus  # noqa: E402
from spiking_ven.sw_encoder import SWEncoder  # noqa: E402


def stats(enc: SWEncoder, sig, sr: int) -> tuple[float, float]:
    X = enc.cochleagram(sig, sr=sr)
    R = enc.reconstruct(X)
    Xp = np.concatenate([np.zeros((X.shape[0], 1)), X[:, :-1]], axis=1)
    ve = 1 - ((R - Xp)[:, enc.L:] ** 2).sum() / ((Xp - Xp.mean())[:, enc.L:] ** 2).sum()
    return float(ve), float(np.mean(enc.W[enc.keep] == 0))


def label(path: Path) -> str:
    m = re.search(r"l1_([^_]+)_sm_([^_]+)$", path.stem)
    return f"L1 {m.group(1)}\nsmooth {m.group(2)}" if m else path.stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--n-show", type=int, default=8)
    a = p.parse_args()
    corpus = load_corpus(a.motifs)
    sig, _ = corpus.template()
    sr = int(corpus.sr)
    paths = [Path(e) for e in a.encoders]
    n = a.n_show
    fig, axes = plt.subplots(len(paths), n + 1, figsize=((n + 1) * 1.3, len(paths) * 1.15),
                             gridspec_kw={"width_ratios": [1.6] + [1] * n})
    axes = np.atleast_2d(axes)
    for row, path in zip(axes, paths):
        enc = SWEncoder.load(path)
        C, L = int(enc.coch_params["n_channels"]), enc.L
        K = enc.W[enc.keep].reshape(-1, C, L)
        ve, z = stats(enc, sig, sr)
        for ax in row:
            ax.axis("off")
        row[0].text(0, 0.5, f"{label(path)}\nrecon {ve:.0%}\nzeros {z:.0%}\n"
                    f"units {len(enc.keep)}", va="center", fontsize=7)
        pref = np.argmax(np.abs(K).sum(axis=2), axis=1)
        order = np.argsort(pref)
        pick = order[np.linspace(0, len(order) - 1, n).astype(int)]
        for ax, k in zip(row[1:], pick):
            w = K[k]
            v = np.abs(w).max() or 1.0
            ax.imshow(w, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        print(f"{path.stem}: recon {ve:.3f} zeros {z:.3f} units {len(enc.keep)}")
    fig.suptitle("STRF penalty sweep (x: lag -64..0 ms, y: 40 ERB channels; recon = variance "
                 "explained on the template motif)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
