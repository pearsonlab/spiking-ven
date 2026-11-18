"""Per-neuron E rate while singing vs not singing, as Mandelblat-Cerf et al. 2014 Fig 7D.

    python scripts/singing_scatter.py --models A.npz B.npz --labels "A" "B" --out fig.png

Singing: the trained song with HVC input (after the intro notes), over the motif. Not
singing: silence, no HVC, over the same window. Each rate is averaged over ``--trials``
noise draws. Log axes, like the paper; rates of zero are drawn at the axis floor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # type: ignore  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"
COLOR = "#2a78d6"
FLOOR = 0.1     # Hz: axis floor; neurons silent in a condition are drawn here


def rates(v, st, trials):
    T = st["T_song"]
    rng = np.random.default_rng(11)
    z = np.zeros_like(st["aud_correct"])
    sing = np.mean([v.transform(st["hvc_on"], st["aud_correct"], rng=rng,
                                hvc_prelude=st["hvc_prelude"])[:, :T] for _ in range(trials)],
                   axis=(0, 2)) * 1000
    quiet = np.mean([v.transform(st["hvc_off"], z, rng=rng)[:, :T] for _ in range(trials)],
                    axis=(0, 2)) * 1000
    return quiet, sing


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    st = sweep._stimuli()
    fig, axes = plt.subplots(1, len(a.models), figsize=(4.2 * len(a.models), 4.4), squeeze=False)
    for ax, path, label in zip(axes[0], a.models, a.labels):
        quiet, sing = rates(VocalErrorNet.load(path, seed=3), st, a.trials)
        x, y = np.maximum(quiet, FLOOR), np.maximum(sing, FLOOR)
        lo, hi = FLOOR * 0.8, max(x.max(), y.max(), 50.0) * 1.3
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, zorder=1)
        ax.scatter(x, y, s=9, color=COLOR, alpha=0.45, linewidths=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("firing rate, not singing (Hz)", color=INK)
        ax.set_ylabel("firing rate, singing (Hz)", color=INK)
        ratio = sing.mean() / max(quiet.mean(), 1e-9)
        above = float(np.mean(sing > quiet))
        ax.set_title(f"{label}\nmean {quiet.mean():.1f} -> {sing.mean():.1f} Hz "
                     f"({ratio:.2f}x); {above:.0%} of cells higher singing",
                     loc="left", fontsize=9, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"{label}: not singing {quiet.mean():.2f} Hz, singing {sing.mean():.2f} Hz, "
              f"ratio {ratio:.2f}, {above:.0%} higher while singing; median per-cell ratio "
              f"{np.median(np.maximum(sing, FLOOR) / np.maximum(quiet, FLOOR)):.2f}")
    fig.suptitle("E neurons: singing vs not singing (cf. Mandelblat-Cerf et al. 2014, Fig. 7D)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=180)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
