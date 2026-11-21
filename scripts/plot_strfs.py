"""Plot STRFs from a training checkpoint or a saved S&W encoder.

    python scripts/plot_strfs.py FILE.npz [FILE2.npz ...] --out fig.png [--n 32]
        [--labels "row 1 title" "row 2 title" ...]

Accepts ``<out>.ckpt.npz`` checkpoints written every epoch by ``train_sw_encoder.py``
(``V`` of shape (K, C, L)) or full encoder files (``SWEncoder``). One row of STRFs per file,
sorted by preferred frequency, labelled with the epoch (checkpoints), the share of energy at
lags <= 20 ms and the energy-weighted mean lag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_strfs(path: str) -> tuple[np.ndarray, float, str]:
    d = np.load(path, allow_pickle=False)
    if "kind" in d and str(d["kind"]) == "sw":
        from spiking_ven.sw_encoder import SWEncoder
        enc = SWEncoder.load(path)
        return enc.V, 1000.0 / float(enc.coch_params["frame_rate"]), "final"
    note = f"epoch {int(d['epoch'])}" if "epoch" in d else ""
    return np.asarray(d["V"]), float(d["frame_ms"]) if "frame_ms" in d else 2.0, note


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=16, help="STRFs per row")
    p.add_argument("--labels", nargs="+", default=None,
                   help="readable row titles (one per file; default: file names); use '\\n' "
                        "for a line break")
    p.add_argument("--minus-init", type=int, default=None, metavar="SEED",
                   help="plot the learned change V - V_init, regenerating learn_relu's random "
                        "initial STRFs from this seed (useful without penalties, where the "
                        "initial noise never decays)")
    a = p.parse_args()
    rows = len(a.files)
    if a.labels is not None and len(a.labels) != rows:
        p.error(f"--labels needs {rows} entries, got {len(a.labels)}")
    fig, axes = plt.subplots(rows, a.n + 1, figsize=((a.n + 2.2) * 1.05, rows * 1.25),
                             gridspec_kw={"width_ratios": [3.4] + [1] * a.n}, squeeze=False)
    for r, path in enumerate(a.files):
        V, fm, note = load_strfs(path)
        K, C, L = V.shape
        if a.minus_init is not None:
            V0 = np.random.default_rng(a.minus_init).standard_normal((K, C, L))
            V = V - V0 / np.linalg.norm(V0.reshape(K, -1), axis=1)[:, None, None]
            note += " (minus init)"
        lag = (L - 1 - np.arange(L)) * fm + fm / 2
        e = (V ** 2).sum(axis=1)
        e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        near = float(np.median(e[:, lag <= 20].sum(axis=1)))
        mlag = float(np.median((e * lag).sum(axis=1)))
        pref = np.argmax(np.abs(V).sum(axis=2), axis=1)
        live = np.flatnonzero(np.abs(V).reshape(K, -1).max(axis=1) > 0)
        order = live[np.argsort(pref[live])]
        pick = order[np.linspace(0, len(order) - 1, min(a.n, len(order))).astype(int)]
        ax0 = axes[r, 0]
        ax0.axis("off")
        title = a.labels[r].replace("\\n", "\n") if a.labels else Path(path).stem
        ax0.text(0, 0.5, f"{title}\n({note}; <=20 ms {near:.0%}, mean lag {mlag:.0f} ms)",
                 fontsize=6.5, va="center")
        for j in range(a.n):
            ax = axes[r, j + 1]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= len(pick):
                ax.axis("off")
                continue
            w = V[pick[j]]
            v = float(np.abs(w).max()) or 1.0
            ax.imshow(w, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=(-L * fm, 0, 0, C))
    fig.suptitle("STRFs (x: lag -64..0 ms; y: 40 ERB channels; red excitatory, blue "
                 "suppressive), sorted by preferred frequency", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
