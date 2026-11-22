"""Covariance and correlation of primary-unit activity for S&W encoder checkpoints.

    python scripts/plot_unit_cov.py A.ckpt.npz [B.ckpt.npz ...] --labels a b --out fig.png
        [--corpus outputs/song_corpus_koch.npz] [--scale 0.05324] [--beta 9]

Activity r = softplus_beta(V . x + b) (noise-free; ReLU if --beta 0) on the held-out bouts of
the corpus, with x the centred strictly-past window divided by --scale (the training input
normalisation). Units are ordered by preferred frequency (channel of largest summed |V|), as
in plot_strfs.py. Top row: covariance (symmetric colour scale per panel); bottom row:
correlation (-1..1). Panel titles give the mean |off-diagonal correlation|.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiking_ven.causal_filters import _past_windows  # noqa: E402
from spiking_ven.cochleagram import cochleagram  # noqa: E402
from spiking_ven.constants import (  # noqa: E402
    CAUSAL_FRAME_RATE, CAUSAL_HI_HZ, CAUSAL_LO_HZ, CAUSAL_N_COCH_CH, CAUSAL_RMS_REF,
)


def heldout_cochleagrams(path: str) -> list[np.ndarray]:
    d = np.load(path, allow_pickle=False)
    out = []
    for s0, n0, held in zip(d["starts"], d["lengths"], d["val"]):
        if not held:
            continue
        x = d["audio"][s0:s0 + n0].astype(np.float64)
        x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12) * CAUSAL_RMS_REF
        out.append(cochleagram(x, int(d["sr"]), n_channels=CAUSAL_N_COCH_CH,
                               frame_rate=CAUSAL_FRAME_RATE, lo_hz=CAUSAL_LO_HZ,
                               hi_hz=CAUSAL_HI_HZ).astype(np.float64))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--corpus", default="outputs/song_corpus_koch.npz")
    p.add_argument("--scale", type=float, default=0.05324)
    p.add_argument("--beta", type=float, default=9.0)
    a = p.parse_args()
    labels = a.labels or a.files
    cochs = heldout_cochleagrams(a.corpus)
    n = len(a.files)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.2), squeeze=False)
    for j, (path, lab) in enumerate(zip(a.files, labels)):
        d = np.load(path, allow_pickle=False)
        V, b, mu = np.asarray(d["V"]), np.asarray(d["b"]), np.asarray(d["mu"])
        K, C, L = V.shape
        W = V.reshape(K, -1)
        R = []
        for X in cochs:
            if X.shape[1] <= L:
                continue
            Z = ((_past_windows(X, L) - mu) / a.scale) @ W.T + b
            R.append(np.logaddexp(0.0, a.beta * Z) / a.beta if a.beta > 0
                     else np.maximum(Z, 0.0))
        R = np.concatenate(R, axis=0)
        order = np.argsort(np.argmax(np.abs(V).sum(axis=2), axis=1), kind="stable")
        R = R[:, order]
        cov = np.cov(R.T)
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-24))
        cor = cov / np.outer(sd, sd)
        off = ~np.eye(K, dtype=bool)
        v = float(np.percentile(np.abs(cov), 99)) or 1.0
        epoch = f", epoch {int(d['epoch'])}" if "epoch" in d else ""
        lab = lab.replace("\\n", "\n")
        im0 = axes[0, j].imshow(cov, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[0, j].set_title(f"{lab}{epoch}\ncovariance", fontsize=8)
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.03)
        im1 = axes[1, j].imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, j].set_title(f"correlation (mean |off-diag| "
                             f"{float(np.abs(cor[off]).mean()):.2f})", fontsize=8)
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.03)
        for ax in axes[:, j]:
            ax.set_xlabel("unit (by preferred frequency)", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Primary-unit activity on held-out bouts (noise-free softplus output)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
