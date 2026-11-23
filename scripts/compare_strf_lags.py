"""Compare where in time STRFs put their energy, across encoders.

    python scripts/compare_strf_lags.py A.npz B.npz --labels colony song --out fig.png

Accepts what ``plot_strfs.py`` accepts (``.ckpt.npz`` checkpoints or saved S&W encoders).
For each unit, the STRF's energy is summed over frequency and normalised over lag. Left:
median (line) and interquartile range (band) of that profile across units, lag 0 = the
newest frame. Right: distribution (ECDF) of each unit's energy-weighted mean lag. A
corpus whose STRFs favour shorter-range information sits left in both.
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

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order


def lag_profiles(V: np.ndarray) -> np.ndarray:
    """(K, L) energy per lag, newest lag first, each row summing to 1 (dead units dropped)."""
    E = (V ** 2).sum(axis=1)[:, ::-1]
    tot = E.sum(axis=1)
    E = E[tot > 1e-12 * tot.max()]
    return E / E.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--basis", default=None, metavar="LO,HI,N",
                   help="the --tm-basis used; draws a random STRF's profile as reference")
    a = p.parse_args()
    if len(a.labels) != len(a.files):
        raise SystemExit("one label per file")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, (f, lab) in enumerate(zip(a.files, a.labels)):
        V, frame_ms, note = load_strfs(f)
        E = lag_profiles(V)
        L = E.shape[1]
        lag = (np.arange(L) + 0.5) * frame_ms
        q25, q50, q75 = np.percentile(E, [25, 50, 75], axis=0)
        mean_lag = E @ lag
        early = E[:, lag <= 20].sum(axis=1)
        c = COLORS[i]
        name = f"{lab} ({note}, {E.shape[0]} units)"
        ax1.fill_between(lag, q25, q75, color=c, alpha=0.18, lw=0)
        ax1.plot(lag, q50, color=c, lw=2, label=name)
        x = np.sort(mean_lag)
        ax2.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=c, lw=2,
                 label=f"{lab}: median {np.median(mean_lag):.1f} ms, "
                       f"<=20 ms {np.median(early):.0%}")
    if a.basis:
        # A random STRF in the basis span: expected energy per lag is the projector's
        # diagonal. This is what the basis alone favours, before any learning.
        from spiking_ven.sw_encoder import tm_basis

        lo, hi, nb = a.basis.split(",")
        B = tm_basis(np.geomspace(float(lo), float(hi), int(nb)), L, frame_ms)
        d = np.diag(B.T @ np.linalg.solve(B @ B.T, B))[::-1]
        d = d / d.sum()
        ax1.plot(lag, d, color="0.45", lw=1.5, ls="--", label="random STRF in basis")
        ax2.axvline(d @ lag, color="0.45", lw=1.5, ls="--",
                    label=f"random STRF in basis: {d @ lag:.1f} ms")
    ax1.set(xlabel="lag (ms, 0 = newest frame)", ylabel="share of STRF energy per frame",
            title="temporal energy profile (median, IQR)")
    ax2.set(xlabel="energy-weighted mean lag (ms)", ylabel="fraction of units",
            title="per-unit mean lag (ECDF)")
    for ax in (ax1, ax2):
        ax.grid(color="0.92", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    if a.title:
        fig.suptitle(a.title, fontsize=11)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
