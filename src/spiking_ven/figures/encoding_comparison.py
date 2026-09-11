"""plots/plot_encoding_comparison.py — Four-column audio encoding comparison figure.

Columns
-------
1. Target training song
2. Time-reversed motif (K4)
3. Novel broadband WN  1× RMS
4. Training song + DAF-level WN  5.6× RMS  (K2)

Rows (per column)
-----------------
1. Raw audio waveform
2. Spectrogram
3. Encoder spikes — raster
4. I neurons — raster
5. E neurons — raster

Public API
----------
make_figure(ven, encoder, sig_train, T_song, out_tag, *, ...)
    Generate and save the figure with the current VEN state.
    encoder: OlshausenFieldEncoder.
    Call this from experiment scripts to snapshot learning progress.

compute_encoding_columns(ven, encoder, sig_train, T_song, *, ...)
    Build the four stimuli, encode them, and run VEN inference. Returns the
    per-column spike matrices. Shared with the population-rate view
    (plot_encoding_comparison_rates.py) so the network inference is defined once.

Requires:
  outputs/of_encoder.npz
  of_ven_model_k4max.npz  (default; or another via --model)
  outputs/motifs.npz

Saves
-----
  outputs/encoding_comparison{out_tag}.png
  outputs/encoding_comparison{out_tag}.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram

from ..constants import DAF_WINDOW_S, DAF_WN_AMPLITUDE, KERNEL_WIDTH_MS, PEAK_RATE_HZ
from ..olshausen_field import OlshausenFieldEncoder, coch_encode, of_to_spikes
from ..common import generate_hvc_spikes

__all__ = ["make_figure", "compute_encoding_columns", "specgram"]

# Column labels shared by every view of this comparison.
COLUMN_LABELS = [
    "Training song",
    "Time-reversed motif",
    "White noise",
    "Distorted auditory feedback",
]


def specgram(sig, n_samples, sr):
    """STFT spectrogram of ``sig[:n_samples]`` → (freq_kHz, time_ms, logS_dB)."""
    n = min(len(sig), n_samples)
    f, t, Sxx = scipy_spectrogram(sig[:n], fs=sr, nperseg=256, noverlap=224,
                                   nfft=512, scaling="spectrum")
    return f / 1000.0, t * 1000.0, 10.0 * np.log10(Sxx + 1e-10)


def compute_encoding_columns(
    ven,
    encoder,
    sig_train,
    T_song,
    *,
    T_post: int = 200,
    sr: int = 16000,
    seed: int = 42,
    kernel_width: float = KERNEL_WIDTH_MS,
    peak_rate: float = PEAK_RATE_HZ,
    daf_wn_amplitude: float = DAF_WN_AMPLITUDE,
    of_mean_rate_hz: float = 15.0,
    of_n_ista: int = 50,
):
    """Build the four stimuli, encode them, and run VEN inference.

    Returns
    -------
    dict with keys:
      T_rend  : int  — full rendition length in ms (T_song + T_post)
      columns : list of 4 dicts, one per stimulus in the order
                [training, time-reversed, white-noise, DAF]. Each dict has
                keys ``label`` (str), ``sig`` (audio), ``aud`` (encoder spikes),
                ``sE`` (excitatory spikes), ``sI`` (inhibitory spikes).

    This is the single source of truth for the network inference behind both
    the raster figure (make_figure) and the population-rate figure.
    """
    if not isinstance(encoder, OlshausenFieldEncoder):
        raise TypeError(
            f"this figure requires an OlshausenFieldEncoder, got {type(encoder).__name__}"
        )
    T_rend    = T_song + T_post
    rms_train = float(np.sqrt(np.mean(sig_train ** 2)))

    # ── Build the four stimuli ────────────────────────────────────────────────
    sig_rev    = sig_train[::-1].copy().astype(np.float64)
    # Separate streams: the white-noise column and the noise added to the DAF column are
    # different stimuli and must be independent draws. Seeding both at seed+1 made them
    # the same realisation, so columns 3 and 4 shared their noise.
    rng_novel  = np.random.default_rng(seed + 1)
    sig_novel  = rng_novel.standard_normal(len(sig_train)) * rms_train
    _rng_daf   = np.random.default_rng(seed + 2)
    daf_noise  = _rng_daf.standard_normal(len(sig_train)) * rms_train * daf_wn_amplitude
    sig_daf    = sig_train.copy()
    _s0, _s1   = int(DAF_WINDOW_S[0] * sr), int(DAF_WINDOW_S[1] * sr)
    if _s0 >= len(sig_train):
        raise ValueError(
            f"the DAF window {DAF_WINDOW_S} s starts past the end of a "
            f"{len(sig_train) / sr:.3f} s motif, so the DAF column would be identical to "
            "the training column. Adjust constants.DAF_WINDOW_S for this corpus."
        )
    sig_daf[_s0:_s1] += daf_noise[_s0:_s1]

    # ── Encoder → Poisson spikes ──────────────────────────────────────────────
    _sign_split = hasattr(ven, "n_aud") and ven.n_aud == 2 * encoder.n_channels

    def _sig_to_aud(sig: np.ndarray, seed_offset: int) -> np.ndarray:
        acts = coch_encode(sig.astype(np.float64), encoder, sr=sr,
                           n_ista=of_n_ista, upsample_to_ms=True, T_out_ms=T_rend)
        spk = of_to_spikes(acts, mean_rate_hz=of_mean_rate_hz, frame_rate=1000,
                           sign_split=_sign_split, seed=seed + seed_offset)
        return spk.astype(np.float32)

    aud_train = _sig_to_aud(sig_train, 10)
    aud_rev   = _sig_to_aud(sig_rev,   13)
    aud_novel = _sig_to_aud(sig_novel, 11)
    aud_daf   = _sig_to_aud(sig_daf,   12)

    # ── HVC inputs ────────────────────────────────────────────────────────────
    hvc_one = generate_hvc_spikes(
        n_hvc=ven.n_hvc, T=T_rend, n_renditions=1,
        T_song=T_song, T_burn=0, T_post=T_post,
        peak_rate=peak_rate, kernel_width=kernel_width, seed=seed,
    )

    # ── VEN inference ─────────────────────────────────────────────────────────
    def _infer_once(hvc, aud):
        # No global np.random.seed here: the network owns a seeded generator of its
        # own, so seeding the global RNG would imply a dependency that no longer
        # exists. See VocalErrorNetV2's _sim_rng.
        return ven.transform_all(hvc, aud)

    sE_train, sI_train = _infer_once(hvc_one, aud_train)
    sE_rev,   sI_rev   = _infer_once(hvc_one, aud_rev)
    sE_novel, sI_novel = _infer_once(hvc_one, aud_novel)
    sE_daf,   sI_daf   = _infer_once(hvc_one, aud_daf)

    sigs = [sig_train, sig_rev, sig_novel, sig_daf]
    auds = [aud_train, aud_rev, aud_novel, aud_daf]
    sEs  = [sE_train, sE_rev, sE_novel, sE_daf]
    sIs  = [sI_train, sI_rev, sI_novel, sI_daf]
    columns = [
        dict(label=lab, sig=s, aud=a, sE=e, sI=i)
        for lab, s, a, e, i in zip(COLUMN_LABELS, sigs, auds, sEs, sIs)
    ]
    return dict(T_rend=T_rend, columns=columns)


def make_figure(
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
    n_kernels: int = 64,
    kernel_width: float = KERNEL_WIDTH_MS,
    peak_rate: float = PEAK_RATE_HZ,
    daf_wn_amplitude: float = DAF_WN_AMPLITUDE,
    sub_l: int = 64,
    sub_i: int = 64,
    sub_e: int = 64,
    of_mean_rate_hz: float = 15.0,
    of_n_ista: int = 50,
):
    """Generate and save encoding comparison figure with the current VEN state.

    Parameters
    ----------
    ven         : VocalErrorNetV2 (trained or partially trained)
    encoder     : OlshausenFieldEncoder
    sig_train   : (N,) float64 training audio signal
    T_song      : int, song duration in ms
    out_tag     : str appended to output filenames, e.g. "_r0050"

    The conduction delays are not arguments: they are properties of the trained model
    (``ven.aud_delay_ms`` / ``ven.hvc_delay_ms``) and are applied inside its own
    inference pass. This function used to accept ``aud_delay_ms``/``hvc_delay_ms`` and
    ignore them, which let a caller believe it had changed something.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    data      = compute_encoding_columns(
        ven, encoder, sig_train, T_song,
        T_post=T_post, sr=sr, seed=seed,
        kernel_width=kernel_width, peak_rate=peak_rate,
        daf_wn_amplitude=daf_wn_amplitude, of_mean_rate_hz=of_mean_rate_hz,
        of_n_ista=of_n_ista,
    )
    COL_DATA  = data["columns"]
    sig_rev   = COL_DATA[1]["sig"]
    sig_novel = COL_DATA[2]["sig"]
    sig_daf   = COL_DATA[3]["sig"]

    # ── Neuron subsets ────────────────────────────────────────────────────────
    rng_sub = np.random.default_rng(0)
    ids_l   = np.sort(rng_sub.choice(n_kernels, size=min(sub_l, n_kernels), replace=False))
    ids_i   = np.sort(rng_sub.choice(ven.n_i,   size=min(sub_i, ven.n_i),   replace=False))
    ids_e   = np.sort(rng_sub.choice(ven.n_e,   size=min(sub_e, ven.n_e),   replace=False))

    def _to_eventplot(mat: np.ndarray, ids_: np.ndarray) -> list[np.ndarray]:
        return [np.where(mat[i, :T_song] > 0.5)[0].astype(float) for i in ids_]

    # ── Figure layout ─────────────────────────────────────────────────────────
    COL_LABELS = COLUMN_LABELS
    _enc_label = "Auditory\nneurons"
    ROW_LABELS = ["", "", _enc_label, "Inhibitory\ninterneurons", "Excitatory\nprojection\nneurons"]
    CLR = {"lew": "#2ca02c", "I": "#c0392b", "E": "#1f77b4"}
    BG  = {"wav": "#FFFFFF",  "spec": "#F8F8F8",
           "lew": "#F0FAF0",  "I": "#FDF0F0", "E": "#EFF5FF"}

    ROW_HEIGHT_IN = 0.015
    _u = ROW_HEIGHT_IN
    HR = [
        2.0 / _u,
        2.0 / _u,
        len(ids_l),
        len(ids_i),
        len(ids_e),
    ]
    fig_height = sum(HR) * _u + 1.2

    fig, axes = plt.subplots(
        5, 4,
        figsize=(20, fig_height),
        gridspec_kw={"height_ratios": HR, "hspace": 0.0, "wspace": 0.06},
        sharex="col",
    )

    n_wav = T_song * sr // 1000

    def _pad_sig(sig: np.ndarray, n: int) -> np.ndarray:
        return sig[:n] if len(sig) >= n else np.concatenate([sig, np.zeros(n - len(sig), dtype=sig.dtype)])

    _sig_train_disp = _pad_sig(sig_train, n_wav)
    _, _, _logS_ref = specgram(_sig_train_disp, n_wav, sr)
    VMAX = _logS_ref.max()
    VMIN = VMAX - 60.0

    t_wav_ms  = np.arange(n_wav) / sr * 1000.0
    _wav_sigs = [_pad_sig(s, n_wav) for s in (sig_train, sig_rev, sig_novel, sig_daf)]
    _wav_ylim = max(np.abs(s).max() for s in _wav_sigs) * 1.05

    for ci, cd in enumerate(COL_DATA):
        ax_wav, ax_spec, ax_lew, ax_i, ax_e = axes[:, ci]
        is_left  = ci == 0
        sig_disp = _pad_sig(cd["sig"], n_wav)

        # Row 0: waveform
        ax_wav.plot(t_wav_ms, sig_disp, lw=0.3, color="0.2")
        ax_wav.set_ylim(-_wav_ylim, _wav_ylim)
        ax_wav.set_facecolor(BG["wav"])
        ax_wav.set_yticks([])
        ax_wav.set_title(COL_LABELS[ci], fontsize=11.0, pad=4, fontweight="bold")
        for sp in ("top", "right", "left", "bottom"):
            ax_wav.spines[sp].set_visible(False)
        ax_wav.tick_params(axis="x", length=0, labelbottom=False)
        if is_left:
            ax_wav.set_ylabel(ROW_LABELS[0], fontsize=8.25, rotation=0,
                               ha="right", va="center", labelpad=4)

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

        def _style_raster(ax, clr, ev, ids_, row_label, rate_mat, *, is_last=False):
            n = len(ids_)
            ax.eventplot(ev, colors=clr, linewidths=1.2, linelengths=1.125,
                         lineoffsets=np.arange(n))
            ax.set_ylim(-0.5, n - 0.5)
            ax.set_yticks([])
            # separator line at top of panel (matches master_raster axhline style)
            ax.axhline(n - 0.5, color="#BBBBBB", lw=0.3)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_visible(is_last)
            if is_last:
                ax.tick_params(axis="x", labelsize=7, length=4, width=0.8)
                ax.set_xlabel("Time (ms)", fontsize=8)
            else:
                ax.tick_params(axis="x", length=0, labelbottom=False)
            if is_left:
                ax.set_ylabel(row_label, fontsize=8.25, rotation=0,
                              ha="right", va="center", labelpad=4)
            rate = float(rate_mat[:, :T_song].mean()) * 1000
            ax.text(0.995, 0.92, f"{rate:.0f} Hz", transform=ax.transAxes,
                    fontsize=5.5, ha="right", va="top", color=clr,
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))

        # Row 2: encoder spikes
        lew_ev = _to_eventplot(cd["aud"], ids_l)
        ax_lew.set_facecolor(BG["lew"])
        ax_lew.set_xlim(0, T_song)
        _style_raster(ax_lew, CLR["lew"], lew_ev, ids_l, ROW_LABELS[2], cd["aud"])

        # Row 3: I raster
        i_ev = _to_eventplot(cd["sI"], ids_i)
        ax_i.set_facecolor(BG["I"])
        _style_raster(ax_i, CLR["I"], i_ev, ids_i, ROW_LABELS[3], cd["sI"])

        # Row 4: E raster
        e_ev = _to_eventplot(cd["sE"], ids_e)
        ax_e.set_facecolor(BG["E"])
        ax_e.set_xlim(0, T_song)
        _style_raster(ax_e, CLR["E"], e_ev, ids_e, ROW_LABELS[4], cd["sE"], is_last=True)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.04)

    for fmt in ("png", "pdf"):
        out = os.path.join(str(out_dir), f"encoding_comparison{out_tag}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight",
                    **({"dpi": 150} if fmt == "png" else {}))
        print(f"Saved {out}")
    plt.close(fig)
