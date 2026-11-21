"""Plot how I_HVC cells' HVC inputs concentrate over training (``monitor_hvc_plastic.py``).

    python scripts/plot_hvc_concentration.py --trace run_trace.npz --out fig.png

Top left: effective number of HVC inputs per I_HVC cell (participation ratio; starts at
~540, 1 = a single input) over renditions, median and 10-90% band. Top right: the largest
input's share of the cell's total HVC weight. Middle: I_HVC rate and E rates. Bottom: the
HVC weight rows of example cells at each checkpoint, HVC neurons ordered by burst time in
the motif, so concentration onto one time shows as a single bright column.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C1, C2, C3, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#5f5e58"


def band(ax, x, Y, color, label):
    lo, med, hi = np.percentile(Y, [10, 50, 90], axis=1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, med, color=color, lw=1.6, label=label)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = p.parse_args()
    d = np.load(a.trace)
    r = d["rend"]
    rows = d["example_rows"]                       # (n_check, n_ex, n_hvc)
    order = np.argsort(d["burst_t"])
    n_ex = rows.shape[1]
    n_show = min(len(r), 6)
    checks = np.unique(np.linspace(0, len(r) - 1, n_show).astype(int))

    fig = plt.figure(figsize=(12, 4.2 + 1.1 * n_ex))
    gs = fig.add_gridspec(2 + n_ex, len(checks), height_ratios=[3, 3] + [1] * n_ex,
                          hspace=0.6, wspace=0.08)
    ax = fig.add_subplot(gs[0, : len(checks) // 2])
    band(ax, r, d["eff_inputs"], C1, "effective inputs")
    ax.set_yscale("log")
    ax.set_ylabel("effective HVC inputs\nper I_HVC cell")
    ax.set_xlabel("rendition")
    ax2 = fig.add_subplot(gs[0, len(checks) // 2:])
    band(ax2, r, d["top_frac"], C2, "top input share")
    ax2.set_ylabel("largest input's share\nof total HVC weight")
    ax2.set_xlabel("rendition")
    ax3 = fig.add_subplot(gs[1, : len(checks) // 2])
    band(ax3, r, d["ih_rate"], C3, "I_HVC")
    ax3.set_ylabel("I_HVC rate, singing (Hz)")
    ax3.set_xlabel("rendition")
    ax4 = fig.add_subplot(gs[1, len(checks) // 2:])
    ax4.plot(r, d["e_sing"], color=C1, lw=1.4, label="singing, trained")
    ax4.plot(r, d["e_rev"], color=C3, lw=1.4, label="singing, reversed")
    ax4.plot(r, d["e_quiet"], color=INK2, lw=1.2, ls="--", label="silent")
    ax4.set_ylabel("E rate (Hz)")
    ax4.set_xlabel("rendition")
    ax4.legend(fontsize=7, frameon=False)
    for x in (ax, ax2, ax3, ax4):
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    vmax = float(np.percentile(rows, 99.9)) or 1.0
    bt = np.sort(d["burst_t"])
    for j, c in enumerate(checks):
        for k in range(n_ex):
            axk = fig.add_subplot(gs[2 + k, j])
            axk.imshow(rows[c, k][order][None, :], aspect="auto", cmap="magma", vmin=0,
                       vmax=vmax, extent=(bt[0], bt[-1], 0, 1))
            axk.set_yticks([])
            if k == 0:
                axk.set_title(f"rendition {r[c]}", fontsize=8)
            if k == n_ex - 1:
                axk.set_xlabel("HVC burst time (ms)", fontsize=7)
            else:
                axk.set_xticks([])
            if j == 0:
                axk.set_ylabel(f"cell {d['example'][k]}", fontsize=6, rotation=0,
                               ha="right", va="center")
            axk.tick_params(labelsize=6)
    fig.suptitle(a.title or "HVC -> I_HVC concentration over training", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
