"""Example auditory units: each one's kernel (its STRF) over its spikes to every motif.

Top row, the kernel as a spectrotemporal image -- frequency against lag, the present at
the right edge. Bottom row, one spike train per rendition in the corpus, motif onset at
0. Read down a column to see what the unit is tuned to and when the song delivers it:
a unit whose kernel matches a syllable fires at that syllable on every rendition.

The rasters are the auditory input the VEN trains on, not a re-encoding of it: every
rendition goes through :func:`~spiking_ven.stimuli.encode_stimulus` on the template's
reference, with the seeds :func:`~spiking_ven.stimuli.build_stimuli` gives its
training pool.

Public API
----------
pick_units(spikes, encoder, n=3, min_rate_hz=1.0)
    The most reliable unit in each of ``n`` peak-frequency bands.
make_unit_figure(encoder, motifs_path, out_dir, *, units=None, ...)
    Render it.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..constants import AUD_MEAN_RATE_HZ, N_ISTA, SR, T_POST  # noqa: E402
from ..corpus import load_corpus  # noqa: E402
from ..stimuli import encode_acts, encode_stimulus  # noqa: E402
from ..filterbank import centre_frequencies  # noqa: E402

__all__ = ["pick_units", "make_unit_figure"]

#: Gaussian width (ms) for the smoothed PSTHs the reliability score compares.
SMOOTH_MS = 5.0


def _centre_hz(encoder, sr):
    cp = encoder.coch_params
    cf = centre_frequencies(sr, encoder.V.shape[1], float(cp["lo_hz"]), float(cp["hi_hz"]))
    return np.sort(np.asarray(cf, dtype=float))


def _peak_hz(encoder, sr):
    """(K,) frequency of each kernel's highest-energy channel."""
    return _centre_hz(encoder, sr)[(encoder.V ** 2).sum(axis=2).argmax(axis=1)]


def motif_spikes(encoder, motifs_path, *, sr=SR, seed=42, t_post=T_POST,
                 mean_rate_hz=AUD_MEAN_RATE_HZ, n_ista=N_ISTA):
    """(n_motifs, K, T) auditory spikes for every rendition, in corpus order.

    Seeds follow ``build_stimuli``: ``seed + 10`` for the template, ``seed + 100 + i``
    for rendition ``i``, so these are the spike trains in its training pool.
    """
    corpus = load_corpus(motifs_path)
    sig_train, train_idx = corpus.template()
    T = corpus.T_song + t_post
    acts_train = encode_acts(sig_train, encoder, sr=sr, n_ista=n_ista, T_out_ms=T,
                             rms_from=sig_train)
    out = []
    for i in range(corpus.n_motifs):
        out.append(encode_stimulus(corpus.signal(i), encoder, sr=sr, T_out_ms=T,
                                   acts_train=acts_train, rms_from=sig_train,
                                   mean_rate_hz=mean_rate_hz, n_ista=n_ista,
                                   seed=seed + (10 if i == train_idx else 100 + i)))
    return np.stack(out), corpus.T_song, train_idx


def reliability(spikes, T_song):
    """(K,) mean pairwise correlation of each unit's smoothed response across motifs.

    Renditions with no spikes from a unit are left out of its average rather than
    counted as zero correlation, so a sparse unit is scored on the motifs it answers.
    """
    from scipy.ndimage import gaussian_filter1d

    sm = gaussian_filter1d(spikes[:, :, :T_song].astype(np.float64), SMOOTH_MS, axis=2)
    sm -= sm.mean(axis=2, keepdims=True)
    rel = np.zeros(spikes.shape[1])
    for k in range(spikes.shape[1]):
        x = sm[:, k]
        n = np.linalg.norm(x, axis=1)
        x = x[n > 0] / n[n > 0, None]
        m = len(x)
        if m > 1:
            rel[k] = ((x @ x.T).sum() - m) / (m * (m - 1))
    return rel


def pick_units(spikes, encoder, T_song, *, n=3, min_rate_hz=1.0, sr=SR):
    """The most reliable unit in each of ``n`` bands of peak frequency, low to high.

    Bands are quantiles of the eligible units' peak frequencies, not equal octaves: the
    bank's tuning is skewed high, so fixed octaves leave the bottom band empty.
    ``min_rate_hz`` keeps out units that answer once on a few motifs -- they correlate
    perfectly with themselves and say nothing about the song.
    """
    rate = spikes[:, :, :T_song].sum(axis=(0, 2)) / (spikes.shape[0] * T_song / 1000.0)
    rel = reliability(spikes, T_song)
    pk = _peak_hz(encoder, sr)
    ok = np.flatnonzero(rate >= min_rate_hz)
    if len(ok) < n:
        ok = np.argsort(rate)[::-1][:n]
    edges = np.quantile(np.log(pk[ok]), np.linspace(0, 1, n + 1))
    band = np.clip(np.searchsorted(edges, np.log(pk[ok]), side="right") - 1, 0, n - 1)
    chosen = []
    for b in range(n):
        cand = ok[band == b]
        cand = cand[~np.isin(cand, chosen)]
        if len(cand):
            chosen.append(int(cand[np.argmax(rel[cand])]))
    return chosen


def make_unit_figure(encoder, motifs_path, out_dir, *, units=None, sr=SR, seed=42,
                     t_post=T_POST, mean_rate_hz=AUD_MEAN_RATE_HZ, n_ista=N_ISTA, tag=""):
    """Kernel over multi-motif raster for each of ``units`` (default :func:`pick_units`)."""
    spikes, T_song, _ = motif_spikes(encoder, motifs_path, sr=sr, seed=seed,
                                             t_post=t_post, mean_rate_hz=mean_rate_hz,
                                             n_ista=n_ista)
    if units is None:
        units = pick_units(spikes, encoder, T_song, sr=sr)
    units = [int(u) for u in units]

    V = np.asarray(encoder.V, dtype=np.float64)
    K, C, L = V.shape
    dt_ms = 1000.0 / float(encoder.coch_params["frame_rate"])
    cf = _centre_hz(encoder, sr)
    n_motifs = spikes.shape[0]

    # Ticks over the filterbank's nominal range, not its centre frequencies: the top
    # centre sits below hi_hz, and filtering on it drops the 8k label.
    cp = encoder.coch_params
    tick_hz = [t for t in (200, 500, 1000, 2000, 4000, 8000)
               if float(cp["lo_hz"]) <= t <= float(cp["hi_hz"])]
    tick_pos = np.interp(np.log(tick_hz), np.log(cf), np.arange(C) + 0.5, left=0, right=C)
    tick_lab = [f"{t // 1000}k" if t >= 1000 else str(t) for t in tick_hz]

    fig, axes = plt.subplots(2, len(units), figsize=(3.9 * len(units), 7.2), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1, 1.55], hspace=0.22,
                                              wspace=0.38))
    for j, k in enumerate(units):
        ax = axes[0, j]
        m = float(np.abs(V[k]).max()) or 1.0
        im = ax.imshow(V[k], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-m, vmax=m,
                       extent=[-L * dt_ms, 0, 0, C])
        ax.set_yticks(tick_pos, tick_lab)
        ax.set_xlabel("Lag (ms)")
        ax.set_title(f"Unit {k}")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[-m, 0, m])
        cb.ax.set_yticklabels(["−", "0", "+"])
        cb.ax.tick_params(labelsize=8)

        ax = axes[1, j]
        for i in range(n_motifs):
            t = np.flatnonzero(spikes[i, k, :T_song])
            ax.vlines(t, i - 0.4, i + 0.4, color="0.3", lw=1.5)
        ax.set_xlim(0, T_song)
        ax.set_ylim(-1, n_motifs)
        ax.set_yticks(range(0, n_motifs, 9))
        ax.set_xlabel("Time (ms)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Motif")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"unit_strfs{tag}.png", out / f"unit_strfs{tag}.pdf"]
    for p in paths:
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return paths, units
