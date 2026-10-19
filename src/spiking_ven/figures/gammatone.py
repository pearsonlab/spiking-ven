"""The encoder's front end on one motif, from waveform to spikes.

One row, four panels, following the signal through the encoder:

1. Spectrogram of the template rendition's waveform.
2. Magnitude response of the five lowest-frequency gammatone filters.
3. Cochleagram of the same rendition -- the gammatone bank's output, computed with the
   encoder's own parameters and RMS normalisation.
4. The encoder's spike output on that rendition, one row per unit, ordered by the
   kernel's peak channel.

The rendition is the template (:attr:`spiking_ven.corpus.Corpus.template_index`), the one
the network trains on, and the spikes go through :func:`spiking_ven.evaluate.encode_stimulus`
-- the same path ``sven-train-ven`` uses -- so the raster is a sample of what the network
actually hears, not a re-derivation of it.

Public API
----------
make_gammatone_figure(encoder, corpus, out_dir, *, seed=42, mean_rate_hz=AUD_MEAN_RATE_HZ)
    Render it.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import signal as sgn  # noqa: E402

from ..cochleagram import cochleagram  # noqa: E402
from ..constants import AUD_MEAN_RATE_HZ, N_ISTA  # noqa: E402
from ..evaluate import encode_acts, encode_stimulus  # noqa: E402
from ..filterbank import centre_frequencies, erb_filterbank, make_erb_filters  # noqa: E402

__all__ = ["make_gammatone_figure"]

_INK, _INK2 = "#1f1f1e", "#5f5e58"
_BLUES5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light -> dark
# Sized like seaborn's "talk" context, so it reads on a slide.
_STYLE = {
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 17, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "axes.linewidth": 1.9, "xtick.major.width": 1.9,
    "ytick.major.width": 1.9, "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.width": 1.4, "xtick.minor.size": 4,
    "axes.edgecolor": _INK, "axes.labelcolor": _INK, "xtick.color": _INK,
    "ytick.color": _INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": _INK, "pdf.fonttype": 42,
}


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return paths


def _spectrogram(ax, sig, sr, T_ms):
    nper = 256                                         # 16 ms at 16 kHz
    f, t, S = sgn.spectrogram(sig, fs=sr, window="hann", nperseg=nper,
                              noverlap=nper - 32)
    S_db = 10 * np.log10(S + 1e-20)
    # 50 dB of range: any wider and the noise floor washes the song out.
    ax.pcolormesh(t * 1000, f / 1000, S_db, cmap="Blues", shading="auto",
                  vmin=S_db.max() - 50, vmax=S_db.max(), rasterized=True)
    ax.set_xlim(0, T_ms)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title("Spectrogram")


def _filter_responses(ax, cf_hz, sr, n=5):
    imp = np.zeros(2 ** 15)
    imp[0] = 1.0
    h = erb_filterbank(imp, make_erb_filters(sr, cf_hz[:n]))
    fr = np.fft.rfftfreq(len(imp), 1.0 / sr)
    H = 20 * np.log10(np.abs(np.fft.rfft(h, axis=1)) + 1e-12)
    for k in range(n):
        ax.plot(fr, H[k], color=_BLUES5[k], lw=3)
    ax.set_xscale("log")
    ax.set_xlim(100, 800)
    ax.set_ylim(-50, 5)
    ax.set_xticks([100, 200, 300, 500, 800])
    ax.set_xticklabels(["100", "200", "300", "500", "800"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB)")
    ax.set_title("Five lowest gammatone filters")


def _cochleagram(ax, sig, sr, cp, cf_hz, T_ms):
    x = sig / max(float(np.sqrt(np.mean(sig ** 2))), 1e-12) * cp["rms_ref"]
    X = cochleagram(x, sr, n_channels=len(cf_hz), frame_rate=cp["frame_rate"],
                    lo_hz=cp["lo_hz"], hi_hz=cp["hi_hz"])
    dt_ms = 1000.0 / float(cp["frame_rate"])
    t_edges = np.arange(X.shape[1] + 1) * dt_ms
    ax.pcolormesh(t_edges, _band_edges(cf_hz), X, cmap="Blues", shading="flat",
                  rasterized=True)
    _frequency_axis(ax, cf_hz, T_ms)
    ax.set_title("Cochleagram")


def _band_edges(cf_hz):
    """Edges of each channel's band on a linear axis: midway between centre frequencies,
    with the outer two bands mirrored about their centres."""
    mid = 0.5 * (cf_hz[1:] + cf_hz[:-1])
    return np.concatenate([[2 * cf_hz[0] - mid[0]], mid, [2 * cf_hz[-1] - mid[-1]]])


def _frequency_axis(ax, cf_hz, T_ms):
    """Linear centre-frequency axis, shared by the cochleagram and the raster. Linear
    rather than per-channel: the ERB channels widen with frequency, so the high channels
    where the filters concentrate get the room and the quiet low ones are compressed."""
    edges = _band_edges(cf_hz)
    ax.set_ylim(edges[0], edges[-1])
    ax.set_yticks([1000, 3000, 5000, 7000])
    ax.set_ylabel("center frequency (Hz)")
    ax.set_xlim(0, T_ms)
    ax.set_xlabel("time (ms)")


def _spike_raster(ax, encoder, spikes, cf_hz, T_ms):
    # One equal row per unit. The units are spectrotemporal kernels, not frequency bands,
    # so the axis is ordinal; they are ordered by peak channel -- as the kernel figure
    # orders them -- which gives a loose tonotopy without claiming a frequency for each.
    K = spikes.shape[0]
    bf = (np.asarray(encoder.V) ** 2).sum(axis=2).argmax(axis=1)
    row = np.empty(K)
    row[np.argsort(bf, kind="stable")] = np.arange(K)
    units, t = np.nonzero(spikes)
    ax.vlines(t, row[units] - 0.4, row[units] + 0.4, color=_INK, linewidth=1.8)
    ax.set_xlim(0, T_ms)
    ax.set_ylim(-0.5, K - 0.5)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("encoder unit")
    ax.set_title("Spiking feature encoding")


def make_gammatone_figure(encoder, corpus, out_dir, *, seed=42,
                          mean_rate_hz=AUD_MEAN_RATE_HZ, n_ista=N_ISTA):
    """Render the front-end figure for a :class:`CausalFilterEncoder` on ``corpus``."""
    sig, _ = corpus.template()
    sr = corpus.sr
    cp = encoder.coch_params
    C = int(cp["n_channels"])
    cf_hz = np.sort(centre_frequencies(sr, C, float(cp["lo_hz"]), float(cp["hi_hz"])))
    T_ms = len(sig) / sr * 1000.0

    T_out = corpus.T_song
    acts_train = encode_acts(sig, encoder, sr=sr, n_ista=n_ista, T_out_ms=T_out,
                             rms_from=sig)
    spikes = encode_stimulus(sig, encoder, sr=sr, T_out_ms=T_out, acts_train=acts_train,
                             rms_from=sig, mean_rate_hz=mean_rate_hz, n_ista=n_ista,
                             seed=seed)

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(22, 5.2))
        # Narrow spacer columns, so the cochleagram's y-label clears the panel to its left.
        # The raster has only a y-label, no axis, so the pair sit in their own tighter grid.
        gs = fig.add_gridspec(1, 5, width_ratios=[1.1, 0.02, 1, 0.04, 2.35], wspace=0.2)
        pair = gs[0, 4].subgridspec(1, 2, wspace=0.17)
        ax_spec, ax_filt = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
        ax_coch, ax_spk = fig.add_subplot(pair[0, 0]), fig.add_subplot(pair[0, 1])
        _spectrogram(ax_spec, sig, sr, T_ms)
        _filter_responses(ax_filt, cf_hz, sr)
        _cochleagram(ax_coch, sig, sr, cp, cf_hz, T_ms)
        _spike_raster(ax_spk, encoder, spikes, cf_hz, T_ms)
        return _save(fig, out_dir, "gammatone_filterbank")
