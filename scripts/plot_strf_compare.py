"""Side-by-side STRFs from several encoders: one row per encoder, one shared colour scale.

    python scripts/plot_strf_compare.py A.npz B.npz C.npz --labels A B C --out fig.png \\
        [--n 12] [--basis 2,64,16] [--title ...]

Accepts what ``plot_strfs.py`` accepts. Each row shows ``--n`` live units spread evenly over
preferred frequency, with real axes (lag ms, 0 = newest frame at right; channel centre kHz)
and every panel on ONE symmetric colour scale (largest |weight| among the plotted units).
A last column shows each encoder's distribution (ECDF) of per-unit energy-weighted mean lag,
with a random STRF in ``--basis`` as reference.
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

COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100"]   # fixed categorical order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--basis", default=None, metavar="LO,HI,N")
    p.add_argument("--title", default="")
    p.add_argument("--sr", type=int, default=16000)
    a = p.parse_args()
    if len(a.labels) != len(a.files):
        raise SystemExit("one label per file")

    runs = []
    for f, lab in zip(a.files, a.labels):
        V, fm, note = load_strfs(f)
        K, C, L = V.shape
        lag = (L - 1 - np.arange(L)) * fm + fm / 2               # column j's lag (oldest first)
        e = (V ** 2).sum(axis=1)
        live = np.flatnonzero(e.sum(axis=1) > 0)
        e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)
        mlag = (e * lag).sum(axis=1)
        pref = np.argmax(np.abs(V).sum(axis=2), axis=1)
        order = live[np.argsort(pref[live], kind="stable")]
        pick = order[np.linspace(0, len(order) - 1, min(a.n, len(order))).astype(int)] \
            if len(order) else order
        runs.append(dict(V=V, fm=fm, note=note, lab=lab, lag=lag, mlag=mlag[live], pick=pick,
                         mlag_all=mlag, near=np.median(e[live][:, lag <= 20].sum(axis=1))
                         if len(live) else np.nan))
    C, L, fm = runs[0]["V"].shape[1], runs[0]["V"].shape[2], runs[0]["fm"]
    cf = centre_frequencies(a.sr, C, CAUSAL_LO_HZ, CAUSAL_HI_HZ)[::-1] / 1000
    vmax = max((float(np.abs(r["V"][r["pick"]]).max()) for r in runs if len(r["pick"])),
               default=1.0) or 1.0

    nr, nc = len(runs), a.n
    fig = plt.figure(figsize=(nc * 1.25 + 3.6, nr * 1.45 + 0.9))
    gs = fig.add_gridspec(nr, nc + 3, width_ratios=[1] * nc + [0.12, 0.9, 2.6], wspace=0.08,
                          hspace=0.45, left=0.05, right=0.98, top=0.86, bottom=0.1)
    im = None
    yt = [0, C // 2, C - 1]
    for i, r in enumerate(runs):
        for j in range(nc):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([] if i < nr - 1 else [-60, -30, 0])
            ax.set_yticks(yt if j == 0 else [])
            if j == 0:
                ax.set_yticklabels([f"{cf[k]:.1f}" for k in yt], fontsize=6)
                ax.set_ylabel(f"{r['lab']}\nkHz", fontsize=7)
            ax.tick_params(labelsize=6, length=2)
            if j >= len(r["pick"]):
                ax.axis("off")
                continue
            k = r["pick"][j]
            im = ax.imshow(r["V"][k], aspect="auto", origin="lower", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, extent=(-L * fm, 0, -0.5, C - 0.5),
                           interpolation="nearest")
            ax.set_title(f"{r['mlag_all'][k]:.0f} ms", fontsize=6, pad=1.5)
            if i == nr - 1:
                ax.set_xlabel("lag ms", fontsize=6)
    if im is not None:
        cax = fig.add_subplot(gs[:, nc])
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[:, nc + 2])
    for i, r in enumerate(runs):
        x = np.sort(r["mlag"])
        if x.size:
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=COLORS[i], lw=2,
                    label=f"{r['lab']} ({r['note']}): median {np.median(x):.1f} ms, "
                          f"<=20 ms {r['near']:.0%}")
    if a.basis:
        from spiking_ven.sw_encoder import tm_basis
        lo, hi, nb = a.basis.split(",")
        B = tm_basis(np.geomspace(float(lo), float(hi), int(nb)), L, fm)
        dgl = np.diag(B.T @ np.linalg.solve(B @ B.T, B))[::-1]
        ref = float((dgl / dgl.sum()) @ ((np.arange(L) + 0.5) * fm))
        ax.axvline(ref, color="0.45", ls="--", lw=1.2, label=f"random STRF in basis: {ref:.1f} ms")
    ax.set(xlabel="energy-weighted mean lag (ms)", ylabel="fraction of units",
           title="per-unit mean lag")
    ax.grid(color="0.92", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle((a.title + ": " if a.title else "") + f"{nc} STRFs per row spread over preferred"
                 f" frequency; one colour scale |w| <= {vmax:.3g} (red excitatory, blue "
                 "suppressive); panel titles = mean lag", fontsize=9, x=0.01, ha="left")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)
    for r in runs:
        x = r["mlag"]
        print(f"{r['lab']:12s} {r['note']}: median mean lag {np.median(x):.1f} ms, "
              f"10-90% {np.percentile(x, 10):.1f}-{np.percentile(x, 90):.1f}, "
              f"<23 ms {100 * (x < 23).mean():.0f}%, <=20 ms energy {r['near']:.0%}, "
              f"peak |w| {np.abs(r['V']).max():.3f}, live {x.size}")


if __name__ == "__main__":
    main()
