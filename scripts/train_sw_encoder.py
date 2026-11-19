"""Train the Schneider & Woolley two-stage auditory front end (``spiking_ven.sw_encoder``).

    python scripts/train_sw_encoder.py --out outputs/encoders/sw_encoder.npz

Stage 1: a rectified autoencoder on the raw cochleagrams of the corpus motifs (same
cochleagram settings as the causal encoder). Units that never activate are dropped; the
primary gain is set so the song's mean primary rate is 32 Hz. Stage 2: the
difference-of-Gaussians threshold is set at a percentile of the DoG output on the song, so
that higher-AC activity is sparse; its spike rate is calibrated later, by the stimulus
pipeline.
"""

from __future__ import annotations

import argparse

import numpy as np

from spiking_ven.cochleagram import cochleagram
from spiking_ven.constants import (
    CAUSAL_FRAME_RATE, CAUSAL_HI_HZ, CAUSAL_LO_HZ, CAUSAL_N_COCH_CH, CAUSAL_RMS_REF,
)
from spiking_ven.corpus import load_corpus
from spiking_ven.sw_encoder import PRIMARY_RATE_HZ, SWEncoder, train_autoencoder, unfold


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=128, help="primary AC units")
    p.add_argument("--l-frames", type=int, default=32, help="STRF length (2 ms frames)")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--threshold-pct", type=float, default=90.0,
                   help="higher-AC threshold: this percentile of the DoG output on song")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    corpus = load_corpus(a.motifs)
    sr = int(corpus.sr)
    cp = {"n_channels": CAUSAL_N_COCH_CH, "K_frames": a.l_frames,
          "frame_rate": CAUSAL_FRAME_RATE, "lo_hz": CAUSAL_LO_HZ, "hi_hz": CAUSAL_HI_HZ,
          "rms_ref": CAUSAL_RMS_REF}
    cochs = []
    for sig in corpus.signals():
        sig = sig / max(float(np.sqrt(np.mean(sig ** 2))), 1e-12) * CAUSAL_RMS_REF
        cochs.append(cochleagram(sig, sr, n_channels=cp["n_channels"],
                                 frame_rate=cp["frame_rate"], lo_hz=cp["lo_hz"],
                                 hi_hz=cp["hi_hz"]).astype(np.float64))
    print(f"{len(cochs)} motifs, {sum(c.shape[1] for c in cochs)} frames; "
          f"training {a.k} STRFs x {cp['n_channels']} ch x {a.l_frames} frames")
    W, b, D, d0, mu, scale = train_autoencoder(cochs, K=a.k, L=a.l_frames,
                                               n_epochs=a.epochs, lr=a.lr, l2=a.l2,
                                               seed=a.seed)

    # Units that never activate on the corpus are dropped (silent channels).
    R = np.concatenate([np.maximum(W @ ((unfold(X, a.l_frames) - mu) / scale) + b, 0.0)
                        for X in cochs], axis=1)
    keep = np.flatnonzero((R > 0).any(axis=1))
    gain = PRIMARY_RATE_HZ / float(R[keep].mean())
    enc = SWEncoder(W, b, scale, cp, primary_gain=gain, keep=keep, D=D, d0=d0, mu=mu)
    print(f"kept {keep.size}/{a.k} units; primary gain {gain:.2f} Hz/unit; "
          f"active fraction {(R[keep] > 0).mean():.1%}")

    # Higher-AC threshold from the DoG output on the song (primary spikes included).
    drive = []
    for i, sig in enumerate(corpus.signals()):
        X = enc.cochleagram(sig, sr=sr)
        drive.append(enc.higher_drive(enc.primary_spikes(enc.primary_rate(X),
                                                         seed=a.seed + i)))
    drive = np.concatenate(drive, axis=1)
    enc.threshold = float(np.percentile(drive, a.threshold_pct))
    act = np.maximum(drive - enc.threshold, 0.0)
    print(f"higher-AC threshold {enc.threshold:.4f} ({a.threshold_pct:g}th pct); "
          f"active fraction {(act > 0).mean():.1%}; units ever active "
          f"{(act > 0).any(axis=1).sum()}/{keep.size}")
    enc.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
