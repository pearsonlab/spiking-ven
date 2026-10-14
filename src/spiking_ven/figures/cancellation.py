"""The auditory cancellation figure.

Three stimulus columns -- training song, time-reversed motif, distorted auditory feedback
(the song with a noise burst) -- by six rows: waveform, spectrogram, auditory neurons,
inhibitory interneurons, excitatory projection neurons, and the E population rate.

The stimuli are the arrays :func:`spiking_ven.evaluate.build_stimuli` returns, so the
figure draws exactly the spikes the metrics score.

Public API
----------
make_figure(ven, st, out_tag="", *, out_dir="outputs")

Saves ``cancellation{out_tag}.png`` and ``.pdf`` in ``out_dir``.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.signal import spectrogram as scipy_spectrogram  # noqa: E402

__all__ = ["make_figure", "specgram"]

#: (column title, waveform key, auditory-spike key) in ``build_stimuli``'s output.
COLUMNS = [
    ("Training song", "sig_train", "aud_correct"),
    ("Time-reversed motif", "sig_reversed", "aud_reversed"),
    ("Distorted auditory feedback", "sig_daf", "aud_daf"),
]

#: Neurons drawn per raster row.
N_RASTER = 64


def specgram(sig, n_samples, sr):
    """STFT spectrogram of ``sig[:n_samples]`` → (freq_kHz, time_ms, logS_dB)."""
    n = min(len(sig), n_samples)
    f, t, Sxx = scipy_spectrogram(sig[:n], fs=sr, nperseg=256, noverlap=224,
                                   nfft=512, scaling="spectrum")
    return f / 1000.0, t * 1000.0, 10.0 * np.log10(Sxx + 1e-10)


def make_figure(ven, st, out_tag="", *, out_dir="outputs"):
    """Run the network on each column's stimulus and save the figure.

    ``ven`` is a VocalErrorNetV2; ``st`` is ``evaluate.build_stimuli``'s output. The
    conduction delays are properties of the trained model and are applied inside its
    own inference pass.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    T_song, sr = st["T_song"], st["sr"]
    cols = []
    for title, sig_key, aud_key in COLUMNS:
        sE, sI = ven.transform_all(st["hvc_on"], st[aud_key])
        cols.append(dict(title=title, sig=st[sig_key], aud=st[aud_key], sE=sE, sI=sI))

    # ── Neuron subsets ────────────────────────────────────────────────────────
    rng_sub = np.random.default_rng(0)
    n_aud   = st["aud_correct"].shape[0]
    ids_a   = np.sort(rng_sub.choice(n_aud,   size=min(N_RASTER, n_aud),   replace=False))
    ids_i   = np.sort(rng_sub.choice(ven.n_i, size=min(N_RASTER, ven.n_i), replace=False))
    ids_e   = np.sort(rng_sub.choice(ven.n_e, size=min(N_RASTER, ven.n_e), replace=False))

    def _to_eventplot(mat: np.ndarray, ids_: np.ndarray) -> list[np.ndarray]:
        return [np.where(mat[i, :T_song] > 0.5)[0].astype(float) for i in ids_]

    # ── Figure layout ─────────────────────────────────────────────────────────
    ROW_LABELS = ["", "", "Auditory\nneurons", "Inhibitory\ninterneurons",
                  "Excitatory\nprojection\nneurons", "E population\nrate (Hz)"]
    CLR = {"aud": "#2ca02c", "I": "#c0392b", "E": "#1f77b4"}
    BG  = {"wav": "#FFFFFF",  "spec": "#F8F8F8",
           "aud": "#F0FAF0",  "I": "#FDF0F0", "E": "#EFF5FF"}

    _u = 0.015   # inches per raster row
    HR = [
        2.0 / _u,
        2.0 / _u,
        len(ids_a),
        len(ids_i),
        len(ids_e),
        # The E population rate, on a shared y-scale across columns. The reversed-motif
        # response is spread over the motif rather than concentrated like the DAF burst,
        # so in a raster of sampled neurons it is nearly invisible; averaged over all of
        # them it is plain. A thin trace: it is a reference strip under the rasters, not
        # a panel in its own right.
        0.53 / _u,
    ]
    fig, axes = plt.subplots(
        6, len(cols),
        figsize=(5.0 * len(cols), sum(HR) * _u + 1.2),
        gridspec_kw={"height_ratios": HR, "hspace": 0.0, "wspace": 0.06},
        sharex="col",
    )

    n_wav = T_song * sr // 1000

    def _pad_sig(sig: np.ndarray, n: int) -> np.ndarray:
        return sig[:n] if len(sig) >= n else np.concatenate([sig, np.zeros(n - len(sig))])

    _, _, _logS_ref = specgram(_pad_sig(st["sig_train"], n_wav), n_wav, sr)
    VMAX = _logS_ref.max()
    VMIN = VMAX - 60.0

    t_wav_ms  = np.arange(n_wav) / sr * 1000.0
    _wav_ylim = max(np.abs(_pad_sig(c["sig"], n_wav)).max() for c in cols) * 1.05

    # One y-limit for every column, or the subtlety the row exists to show would be
    # rescaled away column by column.
    rate_ymax = 1.15 * max(
        float(gaussian_filter1d(c["sE"][:, :T_song].mean(axis=0) * 1000.0, 6.0).max())
        for c in cols)

    for ci, cd in enumerate(cols):
        ax_wav, ax_spec, ax_aud, ax_i, ax_e, ax_r = axes[:, ci]
        is_left  = ci == 0
        sig_disp = _pad_sig(cd["sig"], n_wav)

        # Row 0: waveform
        ax_wav.plot(t_wav_ms, sig_disp, lw=0.3, color="0.2")
        ax_wav.set_ylim(-_wav_ylim, _wav_ylim)
        ax_wav.set_facecolor(BG["wav"])
        ax_wav.set_yticks([])
        ax_wav.set_title(cd["title"], fontsize=11.0, pad=4, fontweight="bold")
        for sp in ("top", "right", "left", "bottom"):
            ax_wav.spines[sp].set_visible(False)
        ax_wav.tick_params(axis="x", length=0, labelbottom=False)

        # Row 1: spectrogram
        fk, ts, logS = specgram(sig_disp, n_wav, sr)
        ax_spec.pcolormesh(ts, fk, logS, cmap="inferno",
                            vmin=VMIN, vmax=VMAX, shading="auto", rasterized=True)
        ax_spec.set_facecolor(BG["spec"])
        ax_spec.tick_params(axis="x", length=0, labelbottom=False)
        ax_spec.set_ylim(0, 8)
        if is_left:
            ax_spec.set_yticks([0, 4, 8])
            ax_spec.set_yticklabels(["0", "4", "8"], fontsize=7)
            ax_spec.set_ylabel("Frequency (kHz)", fontsize=8, labelpad=2)
        else:
            ax_spec.set_yticks([])
            ax_spec.spines["left"].set_visible(False)
        for sp in ("top", "right", "bottom"):
            ax_spec.spines[sp].set_visible(False)

        def _style_raster(ax, clr, ev, ids_, row_label, rate_mat):
            n = len(ids_)
            ax.eventplot(ev, colors=clr, linewidths=1.2, linelengths=1.125,
                         lineoffsets=np.arange(n))
            ax.set_ylim(-0.5, n - 0.5)
            ax.set_yticks([])
            ax.axhline(n - 0.5, color="#BBBBBB", lw=0.3)   # separator at panel top
            for sp in ("top", "right", "left", "bottom"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            if is_left:
                ax.set_ylabel(row_label, fontsize=8.25, rotation=0,
                              ha="right", va="center", labelpad=4)
            rate = float(rate_mat[:, :T_song].mean()) * 1000
            ax.text(0.995, 0.92, f"{rate:.0f} Hz", transform=ax.transAxes,
                    fontsize=5.5, ha="right", va="top", color=clr,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))

        # Rows 2-4: auditory, I and E rasters
        for ax, key, clr, ids_, label in (
            (ax_aud, "aud", "aud", ids_a, ROW_LABELS[2]),
            (ax_i,   "sI",  "I",   ids_i, ROW_LABELS[3]),
            (ax_e,   "sE",  "E",   ids_e, ROW_LABELS[4]),
        ):
            ax.set_facecolor(BG[clr])
            ax.set_xlim(0, T_song)
            _style_raster(ax, CLR[clr], _to_eventplot(cd[key], ids_), ids_, label, cd[key])

        # Row 5: E population rate, every unit, Gaussian-smoothed.
        rate = gaussian_filter1d(cd["sE"][:, :T_song].mean(axis=0) * 1000.0, 6.0)
        ax_r.set_facecolor(BG["E"])
        ax_r.plot(np.arange(T_song), rate, color=CLR["E"], lw=1.0)
        ax_r.set_xlim(0, T_song)
        ax_r.set_ylim(0, rate_ymax)
        ax_r.set_xlabel("Time (ms)", fontsize=9)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)
        ax_r.tick_params(labelsize=8)
        if is_left:
            ax_r.set_ylabel(ROW_LABELS[5], fontsize=8.25, rotation=0,
                            ha="right", va="center", labelpad=4)
        else:
            ax_r.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.04)

    for fmt in ("png", "pdf"):
        out = os.path.join(str(out_dir), f"cancellation{out_tag}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight",
                    **({"dpi": 150} if fmt == "png" else {}))
        print(f"Saved {out}")
    plt.close(fig)
