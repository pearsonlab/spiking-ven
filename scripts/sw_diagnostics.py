"""Diagnostic figures for the Schneider & Woolley front end: STRFs, reconstructions, rasters.

    python scripts/sw_diagnostics.py --encoder outputs/encoders/sw_encoder.npz --out-dir DIR

Writes ``sw_strfs.png`` (the primary-AC STRFs, sorted by preferred frequency),
``sw_reconstruction.png`` (template motif cochleagram, the autoencoder's reconstruction and
their difference) and ``sw_rasters.png`` (primary AC and higher-AC/BS rasters and population
rates for the training song, the time-reversed motif and the DAF stimulus).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from spiking_ven.common import acts_to_spikes  # noqa: E402
from spiking_ven.constants import AUD_MEAN_RATE_HZ  # noqa: E402
from spiking_ven.corpus import load_corpus  # noqa: E402
from spiking_ven.stimuli import daf_waveform  # noqa: E402
from spiking_ven.sw_encoder import SWEncoder  # noqa: E402

INK, INK2 = "#1f1f1e", "#5f5e58"
C_AC, C_BS = "#eb6834", "#2a78d6"


def strf_figure(enc: SWEncoder, out: Path, n_show: int = 64) -> None:
    C, L = int(enc.coch_params["n_channels"]), enc.L
    K = enc.W[enc.keep].reshape(-1, C, L)
    pref = np.argmax(np.abs(K).sum(axis=2), axis=1)          # preferred channel
    order = np.argsort(pref)[:: max(1, len(pref) // n_show)][:n_show]
    n = len(order)
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.25))
    for ax in axes.flat:
        ax.axis("off")
    ms = 1000 // int(enc.coch_params["frame_rate"])
    for ax, k in zip(axes.flat, order):
        w = K[k]
        v = np.abs(w).max() or 1.0
        ax.imshow(w, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=(-L * ms, 0, 0, C))
        ax.set_title(f"unit {enc.keep[k]}", fontsize=6, color=INK2)
    fig.suptitle(f"Primary AC STRFs ({n} of {len(enc.keep)} units, sorted by preferred "
                 f"frequency; x: lag {-L * ms}..0 ms, y: 40 ERB channels low->high; "
                 f"red excitatory, blue suppressive)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "sw_strfs.png", dpi=150)
    plt.close(fig)


def recon_figure(enc: SWEncoder, sig, sr, out: Path) -> None:
    X = enc.cochleagram(sig, sr=sr)
    R = enc.reconstruct(X)
    Xp = np.concatenate([np.zeros((X.shape[0], 1)), X[:, :-1]], axis=1)   # frame t-1 at t
    ve = 1 - ((R - Xp)[:, enc.L:] ** 2).sum() / ((Xp - Xp.mean())[:, enc.L:] ** 2).sum()
    ms = 1000 // int(enc.coch_params["frame_rate"])
    ext = (0, X.shape[1] * ms, 0, X.shape[0])
    vmax = float(np.percentile(Xp, 99.5))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    for ax, M, title, kw in ((axes[0], Xp, "cochleagram (frame t-1)", {}),
                             (axes[1], R, "reconstruction from the code at t", {}),
                             (axes[2], R - Xp, "reconstruction - cochleagram", {"cm": True})):
        if kw.get("cm"):
            v = float(np.percentile(np.abs(M), 99.5))
            ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=ext)
        else:
            ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                      extent=ext)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_ylabel("channel")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"Autoencoder reconstruction of the template motif: {ve:.1%} of variance "
                 f"explained", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sw_reconstruction.png", dpi=150)
    plt.close(fig)


def raster_figure(enc: SWEncoder, corpus, out: Path, seed: int = 42) -> None:
    sig, _ = corpus.template()
    sr = int(corpus.sr)
    stims = [("Training song", sig), ("Time-reversed motif", sig[::-1].copy()),
             ("Distorted auditory feedback", daf_waveform(sig, sr, seed=seed))]
    T = int(corpus.T_song)
    acts_song = enc.encode_signal(sig, sr=sr, rms_from=sig, T_out_ms=T, seed=seed)
    fig, axes = plt.subplots(5, 3, figsize=(14, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2.2, 2.2, 1, 1]})
    for c, (title, s) in enumerate(stims):
        X = enc.cochleagram(s, sr=sr, rms_from=sig)
        ac = enc.primary_spikes(enc.primary_rate(X), seed=seed + c)[:, :T]
        acts = enc.encode_signal(s, sr=sr, rms_from=sig, T_out_ms=T, seed=seed + c)
        bs = acts_to_spikes(acts, mean_rate_hz=AUD_MEAN_RATE_HZ, frame_rate=1000,
                            seed=seed + 10 + c, calibrate_on=acts_song)
        ms = 1000 // int(enc.coch_params["frame_rate"])
        axes[0, c].imshow(X, aspect="auto", origin="lower", cmap="magma",
                          extent=(0, X.shape[1] * ms, 0, X.shape[0]))
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")
        for r, (spk, col, lab) in enumerate(((ac, C_AC, "primary AC"),
                                             (bs, C_BS, "higher AC (BS)")), start=1):
            yy, tt = np.nonzero(spk)
            axes[r, c].scatter(tt, yy, s=0.6, color=col, marker="|", linewidths=0.5)
            axes[r, c].set_ylim(-1, spk.shape[0])
            axes[r, c].text(0.99, 0.97, f"{spk.mean() * 1000:.1f} Hz",
                            transform=axes[r, c].transAxes, ha="right", va="top",
                            fontsize=8, color=INK2)
            if c == 0:
                axes[r, c].set_ylabel(f"{lab}\nunit")
            axes[r + 2, c].plot(gaussian_filter1d(spk.mean(axis=0) * 1000, 5), color=col,
                                lw=1.1)
            if c == 0:
                axes[r + 2, c].set_ylabel(f"{lab}\nrate (Hz)")
        if c == 2:
            for r in range(1, 5):
                axes[r, c].axvspan(400, 450, color="#ecebe7", lw=0, zorder=0)
        axes[4, c].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("cochleagram\nchannel")
    fig.suptitle("Schneider & Woolley front end: primary AC (dense STRFs, ~32 Hz) -> higher "
                 "AC (difference of Gaussians on each primary spike train, ~2.5 Hz)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "sw_rasters.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", required=True)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    enc = SWEncoder.load(a.encoder)
    corpus = load_corpus(a.motifs)
    strf_figure(enc, out)
    sig, _ = corpus.template()
    recon_figure(enc, sig, int(corpus.sr), out)
    raster_figure(enc, corpus, out)
    print("wrote", *(out / f for f in ("sw_strfs.png", "sw_reconstruction.png",
                                         "sw_rasters.png")))


if __name__ == "__main__":
    main()
