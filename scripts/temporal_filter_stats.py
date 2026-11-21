"""Nagel & Doupe (2006)-style temporal-filter statistics of S&W encoder STRFs.

    python scripts/temporal_filter_stats.py A.npz B.npz ... [--labels a b ...] [--out fig.png]

Nagel & Doupe characterise Field L neurons by their filter for the (log) amplitude envelope
of a broadband or best-frequency narrowband noise carrier. Our analogue for an STRF V (C x L)
is its temporal profile summed over channels (broadband) or over the best-frequency band
(+-2 channels around the channel of largest |V|). Per unit, with their definitions:

- Pos/Neg: sum of the positive parts / |sum of the negative parts| (1 = balanced
  "differentiator"; >> 1 a positive "integrator"). Nagel & Doupe: mean 1.41 at 30 dB,
  0.97 at 63 dB.
- BMF: frequency of the power-spectrum peak of the filter segment from 25 ms before |filter|
  first reaches 25% of its maximum up to lag 0 (zero-padded FFT; 0 Hz = low-pass).
  Nagel & Doupe: 10-150 Hz, roughly log-uniform.
- 50% width: duration over which |filter| >= half its maximum.

Units with an all-zero profile are skipped. Prints medians / quartiles per encoder and, with
--out, histograms.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiking_ven.sw_encoder import SWEncoder  # noqa: E402


def filter_stats(f: np.ndarray, frame_ms: float, nfft: int = 1024) -> tuple[float, float, float]:
    """(Pos/Neg, BMF Hz, 50% width ms) of one temporal filter, ordered oldest lag first."""
    a = np.abs(f)
    m = float(a.max())
    pos, neg = float(f[f > 0].sum()), float(-f[f < 0].sum())
    pn = pos / neg if neg > 0 else np.inf
    width = float((a >= 0.5 * m).sum()) * frame_ms
    first = int(np.flatnonzero(a >= 0.25 * m)[0])            # earliest (oldest) 25% crossing
    start = max(first - int(round(25.0 / frame_ms)), 0)
    seg = f[start:]
    p = np.abs(np.fft.rfft(seg - 0.0, n=nfft)) ** 2
    bmf = float(np.fft.rfftfreq(nfft, d=frame_ms / 1000.0)[int(np.argmax(p))])
    return pn, bmf, width


def profiles(V: np.ndarray, band: int | None) -> np.ndarray:
    """(K, L) temporal profiles: sum over all channels, or +-band around the best channel."""
    if band is None:
        return V.sum(axis=1)
    best = np.argmax(np.abs(V).max(axis=2), axis=1)
    C = V.shape[1]
    return np.stack([V[k, max(c - band, 0):min(c + band + 1, C)].sum(axis=0)
                     for k, c in enumerate(best)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("encoders", nargs="+")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    labels = a.labels or [Path(e).stem for e in a.encoders]
    rows = []
    print(f"{'encoder':16s} {'carrier':10s} {'Pos/Neg med [IQR]':>22s} {'balanced':>9s} "
          f"{'BMF med [IQR] Hz':>20s} {'low-pass':>9s} {'BMF 10-150':>10s} {'50% width ms':>14s}")
    for path, lab in zip(a.encoders, labels):
        enc = SWEncoder.load(path)
        fm = 1000.0 / float(enc.coch_params["frame_rate"])
        V = enc.V
        for carrier, band in (("broadband", None), ("best-band", 2)):
            P = profiles(V, band)
            st = np.array([filter_stats(f, fm) for f in P if np.abs(f).max() > 0])
            pn, bmf, w = st[:, 0], st[:, 1], st[:, 2]
            q = lambda x: np.percentile(x[np.isfinite(x)], [25, 50, 75])  # noqa: E731
            pq, bq, wq = q(pn), q(bmf), q(w)
            bal = float(np.mean((pn >= 0.5) & (pn <= 2.0)))
            print(f"{lab:16s} {carrier:10s} {pq[1]:6.2f} [{pq[0]:5.2f}-{pq[2]:6.2f}] "
                  f"{bal:9.0%} {bq[1]:6.0f} [{bq[0]:4.0f}-{bq[2]:4.0f}] "
                  f"{np.mean(bmf == 0):9.0%} {np.mean((bmf >= 10) & (bmf <= 150)):10.0%} "
                  f"{wq[1]:6.0f} [{wq[0]:3.0f}-{wq[2]:3.0f}]")
            rows.append((lab, carrier, pn, bmf, w))
    if a.out:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for lab, carrier, pn, bmf, w in rows:
            if carrier != "broadband":
                continue
            lpn = np.log10(np.clip(pn, 1e-2, 1e2))
            axes[0].hist(lpn, bins=np.linspace(-2, 2, 33), histtype="step", lw=1.3, label=lab)
            axes[1].hist(bmf, bins=np.linspace(0, 250, 26), histtype="step", lw=1.3)
            axes[2].hist(w, bins=np.linspace(0, 64, 33), histtype="step", lw=1.3)
        axes[0].axvline(np.log10(0.97), color="0.3", ls="--", lw=0.8)
        axes[0].axvline(np.log10(1.41), color="0.3", ls=":", lw=0.8)
        axes[0].set_xlabel("log10 Pos/Neg (dashed 0.97 loud, dotted 1.41 soft; N&D 2006)")
        axes[1].axvspan(10, 150, color="#ecebe7", lw=0)
        axes[1].set_xlabel("BMF (Hz; shaded: N&D range 10-150 Hz)")
        axes[2].set_xlabel("50% width (ms)")
        for ax in axes:
            ax.set_ylabel("units")
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle("Temporal filters of encoder STRFs (summed over channels = broadband "
                     "carrier)", x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(a.out, dpi=150)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
