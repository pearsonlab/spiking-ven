"""STRF and population statistics for saved S&W encoders (one row per encoder).

    python scripts/encoder_stats.py NAME=ENC.npz [NAME2=ENC2.npz ...] [--out table.md]

Primary-unit rates (``SWEncoder.primary_rate``, with each encoder's own normalisation and
nonlinearity) on the held-out bouts of the corpus. Per encoder: share of STRF energy at
lags <= 20 ms and energy-weighted mean lag (medians over units); dead units (activity std
< 2% of the median unit's); units with a near-duplicate (activity correlation > 0.9 with
another unit); mean |off-diagonal correlation|.
"""

from __future__ import annotations

import argparse

import numpy as np

from spiking_ven.sw_encoder import SWEncoder

from plot_unit_cov import heldout_cochleagrams  # noqa: E402  (scripts/ on the path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("encoders", nargs="+", help="NAME=path.npz")
    p.add_argument("--corpus", default="outputs/song_corpus_koch.npz")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cochs = heldout_cochleagrams(a.corpus)
    rows = ["| encoder | STRF <=20 ms | mean lag | dead units | units with r>0.9 partner | "
            "mean abs corr |", "|---|---|---|---|---|---|"]
    for item in a.encoders:
        name, path = item.split("=", 1)
        enc = SWEncoder.load(path)
        V = enc.V
        n, C, L = V.shape
        fm = 1000.0 / float(enc.coch_params["frame_rate"])
        lag = (L - 1 - np.arange(L)) * fm + fm / 2
        e = (V ** 2).sum(axis=1)
        e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        near = float(np.median(e[:, lag <= 20].sum(axis=1)))
        mlag = float(np.median((e * lag).sum(axis=1)))
        R = np.concatenate([enc.primary_rate(X) for X in cochs if X.shape[1] > L], axis=1)
        sd = R.std(axis=1)
        dead = int((sd < 0.02 * np.median(sd)).sum())
        live = sd > 0
        cor = np.corrcoef(R[live])
        off = ~np.eye(cor.shape[0], dtype=bool)
        dup = int(((cor > 0.9) & off).any(axis=1).sum())
        rows.append(f"| {name} | {near:.0%} | {mlag:.0f} ms | {dead}/{n} | {dup}/{n} | "
                    f"{float(np.abs(cor[off]).mean()):.2f} |")
        print(rows[-1], flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
