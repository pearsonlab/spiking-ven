"""plots/plot_encoding_comparison_rates.py — population-rate view of the audio
encoding comparison.

Alternative to plot_encoding_comparison.py (the raster view). Instead of one
raster row per population, each column shows a single shared axes with one
population firing-rate line per population (encoder / inhibitory / excitatory)
under its spectrogram. The waveform and white-noise columns are dropped, and
the figure is half the width of the raster view.

This is a *supplementary* view; it deliberately reuses
``compute_encoding_columns`` from plot_encoding_comparison so the network
inference is never duplicated, and it does not replace the raster figure.

Columns
-------
1. Training song
2. Time-reversed motif
3. Distorted auditory feedback

Rows (per column)
-----------------
1. Spectrogram
2. Population firing rates (encoder, inhibitory, excitatory) on shared axes

Requires:
  outputs/of_encoder.npz
  outputs/of_ven_model_k4max.npz  (default; or another via --model)
  outputs/motifs.npz

Saves
-----
  outputs/encoding_comparison_rates{out_tag}.png
  outputs/encoding_comparison_rates{out_tag}.pdf
"""

import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from ..constants import KERNEL_WIDTH_MS, PEAK_RATE_HZ
from .encoding_comparison import compute_encoding_columns, specgram

__all__ = ["make_rate_figure"]

# Columns kept for this view (index into compute_encoding_columns output):
# 0 training, 1 time-reversed, 3 DAF — white noise (2) is dropped.
KEEP_COLS = [0, 1, 3]

CLR = {"aud": "#2ca02c", "I": "#c0392b", "E": "#1f77b4"}


def _pop_rate(mat, T_song, *, sigma_ms=6.0):
    """Population mean firing rate (Hz) over 0..T_song, Gaussian-smoothed.

    ``mat`` is an (N, T) binary spike matrix at 1 ms resolution; the mean over
    neurons of each 1 ms column is a per-ms spike probability, ×1000 → Hz.
    """
    r = mat[:, :T_song].mean(axis=0) * 1000.0
    return gaussian_filter1d(r, sigma_ms)


def make_rate_figure(
    ven,
    encoder,
    sig_train,
    T_song,
    out_tag="",
    *,
    out_dir="outputs",
    T_post: int = 200,
    sr: int = 16000,
    seed: int = 42,
    kernel_width: float = KERNEL_WIDTH_MS,
    peak_rate: float = PEAK_RATE_HZ,
    sigma_ms: float = 6.0,
    of_mean_rate_hz: float = 15.0,
    of_n_ista: int = 50,
):
    """Population-rate comparison figure with the current VEN state."""
    os.makedirs(str(out_dir), exist_ok=True)
    data      = compute_encoding_columns(
        ven, encoder, sig_train, T_song,
        T_post=T_post, sr=sr, seed=seed,
        kernel_width=kernel_width, peak_rate=peak_rate,
        of_mean_rate_hz=of_mean_rate_hz, of_n_ista=of_n_ista,
    )
    all_cols = data["columns"]
    cols     = [all_cols[i] for i in KEEP_COLS]
    n_col    = len(cols)

    _enc_label = "Auditory"

    n_wav = T_song * sr // 1000

    def _pad_sig(sig, n):
        return sig[:n] if len(sig) >= n else np.concatenate([sig, np.zeros(n - len(sig), dtype=sig.dtype)])

    _, _, _logS_ref = specgram(_pad_sig(sig_train, n_wav), n_wav, sr)
    VMAX = _logS_ref.max()
    VMIN = VMAX - 60.0

    # One row per population (Auditory, Error); each takes half the vertical
    # space the single rate panel used, so the 1.35 rate unit splits into two.
    RATE_SPECS = [("aud", _enc_label, CLR["aud"]), ("sE", "Error", CLR["E"])]

    # Half the width of the raster view (20 in → 10 in), scaled to column count,
    # then 1.6× so the doubled axis type has room.
    fig_w = 16.0 * n_col / 4.0
    fig, axes = plt.subplots(
        1 + len(RATE_SPECS), n_col,
        figsize=(fig_w, 6.2),
        gridspec_kw={"height_ratios": [1.0, 0.675, 0.675],
                     "hspace": 0.30, "wspace": 0.10},
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    t_ms = np.arange(T_song)

    for ci, cd in enumerate(cols):
        ax_spec = axes[0, ci]
        is_left = ci == 0
        sig_disp = _pad_sig(cd["sig"], n_wav)

        # Row 0: spectrogram
        fk, ts, logS = specgram(sig_disp, n_wav, sr)
        ax_spec.pcolormesh(ts, fk, logS, cmap="inferno",
                           vmin=VMIN, vmax=VMAX, shading="auto", rasterized=True)
        ax_spec.set_facecolor("#F8F8F8")
        # Wrapped: at this size the long labels are wider than one column.
        ax_spec.set_title(textwrap.fill(cd["label"], 18),
                          fontsize=24.0, pad=6, fontweight="bold")
        ax_spec.set_ylim(0, 8)
        ax_spec.tick_params(axis="x", length=0, labelbottom=False)
        if is_left:
            ax_spec.set_yticks([0, 4, 8])
            ax_spec.set_yticklabels(["0", "4", "8"], fontsize=21)
            ax_spec.set_ylabel("Frequency\n(kHz)", fontsize=24, labelpad=14, linespacing=1.5)
        else:
            ax_spec.set_yticks([])
            ax_spec.spines["left"].set_visible(False)
        for sp in ("top", "right", "bottom"):
            ax_spec.spines[sp].set_visible(False)

        # Rows 1..N: one population firing-rate line per axes
        for ri, (key, lab, clr) in enumerate(RATE_SPECS):
            axr     = axes[1 + ri, ci]
            is_last = ri == len(RATE_SPECS) - 1
            axr.plot(t_ms, _pop_rate(cd[key], T_song, sigma_ms=sigma_ms),
                     color=clr, lw=2.6)
            axr.set_xlim(0, T_song)
            axr.set_ylim(0, 90)
            axr.set_yticks([0, 30, 60, 90])
            axr.margins(x=0)
            for sp in ("top", "right"):
                axr.spines[sp].set_visible(False)
            if is_last:
                axr.set_xlabel("Time (ms)", fontsize=24)
                axr.tick_params(axis="x", labelsize=21, length=3)
            else:
                axr.tick_params(axis="x", length=0, labelbottom=False)
            if is_left:
                axr.set_ylabel(f"{lab}\nrate (Hz)", fontsize=24, labelpad=14, color=clr,
                               linespacing=1.5)
                axr.tick_params(axis="y", labelsize=21)
            else:
                axr.spines["left"].set_visible(False)
                axr.tick_params(axis="y", length=0, labelleft=False)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.13)

    for fmt in ("png", "pdf"):
        out = os.path.join(str(out_dir), f"encoding_comparison_rates{out_tag}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight",
                    **({"dpi": 150} if fmt == "png" else {}))
        print(f"Saved {out}")
    plt.close(fig)
