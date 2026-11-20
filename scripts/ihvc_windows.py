"""When does each I_HVC cell fire during the motif, and how well do they tile it?

    python scripts/ihvc_windows.py --model M.npz --cache stimuli.pkl --out fig.png

Trial-averaged I_HVC PSTHs while singing the trained song (``--trials`` noise draws),
against each cell's not-singing rate. A cell's firing windows are the stretches where its
smoothed PSTH exceeds its not-singing rate by more than half its peak excess (FWHM-style).
Reports per cell: rate singing / not singing, number of windows, their widths and total
time; for the population: how much of the motif at least one window covers, and how many
cells cover each moment. Also counts each cell's HVC inputs and their bursts per motif.
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

SMOOTH_MS = 3.0


def segments(mask):
    """(start, stop) of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    with open(a.cache, "rb") as fh:
        st = pickle.load(fh)
    ven = VocalErrorNet.load(a.model, seed=42)
    rng = np.random.default_rng(7)
    T = int(st["T_song"])
    pre = st.get("hvc_prelude")

    def run(hvc, aud, prelude):
        acc = None
        for _ in range(a.trials):
            ven.transform(hvc, aud, rng=rng, hvc_prelude=prelude)
            s = np.asarray(ven.last_ih, dtype=np.float64)[:, :T]
            acc = s if acc is None else acc + s
        return acc / a.trials * 1000.0                      # (n_ih, T) Hz

    sing = run(st["hvc_on"], st["aud_correct"], pre)
    quiet = run(st["hvc_off"], np.zeros_like(st["aud_correct"]), None)
    base = quiet[:, 20:].mean(axis=1)                        # not-singing rate per cell
    sm = gaussian_filter1d(sing, SMOOTH_MS, axis=1)
    exc = sm - base[:, None]
    peak = exc.max(axis=1)

    n_ih = sing.shape[0]
    widths, n_win, active = [], np.zeros(n_ih, int), np.zeros((n_ih, T), bool)
    for i in range(n_ih):
        if peak[i] <= 0:
            continue
        m = exc[i] > 0.5 * peak[i]
        active[i] = m
        segs = segments(m)
        n_win[i] = len(segs)
        widths += [b - s for s, b in segs]
    widths = np.asarray(widths)
    cover = active.sum(axis=0)

    # HVC inputs per I_HVC cell and their bursts per motif.
    hvc = np.asarray(st["hvc_on"])[:, :T]
    hvc_bursts = np.array([len(segments(gaussian_filter1d(h.astype(float), 2) > 0.05))
                           for h in hvc])
    B = np.asarray(ven.B_H) > 0
    n_in = B.sum(axis=1)
    bursts_in = np.array([hvc_bursts[B[i]].sum() for i in range(n_ih)])

    rs, rq = sing.mean(axis=1), quiet[:, 20:].mean(axis=1)
    lines = [
        f"{a.label}",
        f"I_HVC rate: singing {rs.mean():.1f} Hz, not singing {rq.mean():.1f} Hz "
        f"({np.mean(rq > 0):.0%} of cells fire at all when not singing)",
        f"HVC inputs per cell: median {np.median(n_in):.0f} (range {n_in.min()}-{n_in.max()}); "
        f"HVC bursts per HVC neuron per motif: median {np.median(hvc_bursts):.0f}; "
        f"HVC bursts reaching each I_HVC cell: median {np.median(bursts_in):.0f}",
        f"windows per cell: median {np.median(n_win):.0f} (10-90% "
        f"{np.percentile(n_win, 10):.0f}-{np.percentile(n_win, 90):.0f}); "
        f"window width: median {np.median(widths):.0f} ms (10-90% "
        f"{np.percentile(widths, 10):.0f}-{np.percentile(widths, 90):.0f})",
        f"time in windows per cell: median {np.median(active.sum(axis=1)):.0f} ms of {T} "
        f"({np.median(active.sum(axis=1)) / T:.0%})",
        f"motif covered by >=1 window: {np.mean(cover > 0):.0%}; cells covering each ms: "
        f"median {np.median(cover):.0f} (10-90% {np.percentile(cover, 10):.0f}-"
        f"{np.percentile(cover, 90):.0f})",
        f"peak excess over not-singing: median {np.median(peak):.0f} Hz; modulation depth "
        f"(peak excess / mean singing rate) median {np.median(peak / np.maximum(rs, 1e-9)):.1f}",
    ]
    print("\n".join(lines))

    order = np.argsort(np.argmax(exc, axis=1))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    v = float(np.percentile(sm, 99))
    axes[0].imshow(sm[order], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=v,
                   extent=(0, T, 0, n_ih))
    axes[0].set_ylabel("I_HVC cell (sorted by peak time)")
    axes[0].set_title(f"{a.label}: trial-averaged I_HVC PSTH while singing "
                      f"({a.trials} trials; colour to {v:.0f} Hz)", loc="left", fontsize=9)
    axes[1].plot(cover, color="#2a78d6", lw=1.2)
    axes[1].set_ylabel("cells in a\nwindow")
    axes[2].plot(gaussian_filter1d(sing.mean(axis=0), SMOOTH_MS), color="#eb6834", lw=1.2,
                 label="singing")
    axes[2].axhline(rq.mean(), color="#5f5e58", lw=1, ls="--", label="not singing")
    axes[2].set_ylabel("I_HVC pop.\nrate (Hz)")
    axes[2].set_xlabel("time in motif (ms)")
    axes[2].legend(fontsize=8, frameon=False, loc="upper right")
    for ax in axes[1:]:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "\n".join(lines[1:]), fontsize=7, va="bottom", family="monospace")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
