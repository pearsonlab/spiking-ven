"""Error-signal metrics for a trained vocal error network.

The K1-K4 family from Mandelblat-Cerf et al. 2014 (Fig. 7/8), which is how "did the
network learn to cancel the bird's own song?" gets turned into numbers:

===== ============================================ ==========================
K1     correct song + HVC, mean E rate              target mean 7.7 Hz, SD 8.7
K2     white noise at DAF amplitude + HVC           pop. avg ~16 Hz; responders ~28 Hz
K3     K2 / K1                                      pop. avg ~2.1x; responders ~3.6x
K4     (time-reversed motif + HVC) / K1             > 1x
===== ============================================ ==========================

The model is a **responder-only** population -- every excitatory unit receives identical
auditory drive, so there is no non-responder subpopulation to average in. The
responder-only targets are therefore the relevant ones.

This lives here, as a function returning a dict, rather than as a block of prints at the
bottom of a training script, so the same numbers can be asserted in tests and recomputed
for a saved model without retraining.
"""

from __future__ import annotations

__all__ = ["daf_metrics", "format_metrics", "build_stimuli", "BIOLOGICAL_TARGETS", "DAF_WN_AMPLITUDE"]

# Verbatim targets, kept next to the code that is judged against them.
BIOLOGICAL_TARGETS = {
    "k1_hz": (7.7, 8.7),        # mean, SD  (Fig. 7C)
    "k2_hz_pop": 16.0,          # population average (Fig. 7K)
    "k2_hz_responders": 28.0,   # responders only  (Fig. 8E)
    "k3_pop": 2.1,
    "k3_responders": 3.6,
    "k4_min": 1.0,              # direction only
}

# DAF white noise is mixed at this multiple of song RMS: ~95 dBSPL WN vs ~80 dBSPL song.
DAF_WN_AMPLITUDE = 5.6


def _rate_hz(ven, hvc, aud) -> float:
    """Mean E-population rate in Hz for one rendition of (hvc, aud) input."""
    return float(ven.transform(hvc, aud).mean() * 1000)


def daf_metrics(ven, *, hvc_on, hvc_off, aud_correct, aud_daf, aud_reversed) -> dict:
    """Compute K1-K4 for a trained network.

    Parameters
    ----------
    ven          : trained VocalErrorNetV2
    hvc_on       : (n_hvc, T) HVC premotor spikes -- the singing condition
    hvc_off      : (n_hvc, T) zeros -- the not-singing control
    aud_correct  : (n_aud, T) spikes for the trained song
    aud_daf      : (n_aud, T) spikes for white noise at DAF amplitude
    aud_reversed : (n_aud, T) spikes for the time-reversed motif

    Returns
    -------
    dict with k1, k2, k2_no_hvc, k4_rate (all Hz) and the k3, k4 ratios.
    """
    k1 = _rate_hz(ven, hvc_on, aud_correct)
    k2 = _rate_hz(ven, hvc_on, aud_daf)
    k2_no_hvc = _rate_hz(ven, hvc_off, aud_daf)
    k4_rate = _rate_hz(ven, hvc_on, aud_reversed)
    nan = float("nan")
    return {
        "k1": k1,
        "k2": k2,
        "k2_no_hvc": k2_no_hvc,
        "k4_rate": k4_rate,
        "k3": (k2 / k1) if k1 > 0 else nan,
        "k4": (k4_rate / k1) if k1 > 0 else nan,
    }


def format_metrics(m: dict, *, r_e_target: float | None = None) -> str:
    """Render :func:`daf_metrics` output next to the biological targets."""
    t = BIOLOGICAL_TARGETS
    k1_mean, k1_sd = t["k1_hz"]
    sanity = (f"  (sanity; homeostasis target {r_e_target:.0f} Hz)"
              if r_e_target is not None else "")
    return "\n".join([
        "DAF evaluation (Mandelblat-Cerf 2014 Fig. 7/8 targets):",
        f"  correct + HVC   (K1): {m['k1']:6.2f} Hz   target mean {k1_mean} Hz, SD {k1_sd}",
        f"  WN(DAF) + HVC   (K2): {m['k2']:6.2f} Hz   pop ~{t['k2_hz_pop']:.0f} Hz; "
        f"responders ~{t['k2_hz_responders']:.0f} Hz",
        f"  WN(DAF) + noHVC     : {m['k2_no_hvc']:6.2f} Hz{sanity}",
        f"  WN / correct    (K3): {m['k3']:6.2f}x     pop ~{t['k3_pop']}x; "
        f"responders ~{t['k3_responders']}x",
        f"  reversed + HVC      : {m['k4_rate']:6.2f} Hz",
        f"  reversed/correct(K4): {m['k4']:6.2f}x     target > {t['k4_min']:.0f}x",
    ])

# ---------------------------------------------------------------------------
# Stimulus construction
# ---------------------------------------------------------------------------
# Defined once, here, and used by BOTH the trainer and the reproduction tests. If the
# two ever built their stimuli separately they would drift, and the metrics would stop
# describing the model that was trained.

# HVC burst envelope: peak rate scales inversely with kernel width so integrated drive
# per burst is constant.
KERNEL_WIDTH = 10.0
PEAK_RATE = 150.0 * 20.0 / KERNEL_WIDTH


def build_stimuli(
    encoder,
    motifs_path,
    *,
    sr: int = 16000,
    seed: int = 42,
    t_post: int = 200,
    t_burn: int = 500,
    n_hvc: int = 60,
    mean_rate_hz: float = 15.0,
    n_ista: int = 50,
    verbose: bool = True,
) -> dict:
    """Build every stimulus the trainer and the metrics need.

    Returns a dict with the auditory spike arrays (``aud_correct``, ``aud_reversed``,
    ``aud_daf``, plus the full ``correct_pool``), the HVC inputs (``hvc_on``,
    ``hvc_off``, ``hvc_burn`` and its padded ``aud_burn``), and the bookkeeping the
    caller needs (``T_song``, ``T_rend``, ``train_idx``, ``fwd_rev_corr``, ``n_kernels``).
    """
    import numpy as np

    from .common import generate_hvc_spikes
    from .olshausen_field import coch_encode, of_to_spikes

    md = np.load(str(motifs_path))
    audio_m, lengths, song_Ts = md["audio"], md["lengths"], md["song_Ts"]
    n_motifs = audio_m.shape[0]
    T_song = int(np.ceil(song_Ts.max()))
    T_rend = T_song + t_post
    n_kernels = encoder.n_bases

    rms_all = np.array([float(np.sqrt(np.mean(audio_m[i, : lengths[i]] ** 2)))
                        for i in range(n_motifs)])
    train_idx = int(np.argmax(rms_all))          # highest-RMS rendition is the template
    sig_train = audio_m[train_idx, : lengths[train_idx]].astype(np.float64)
    rms_train = float(np.sqrt(np.mean(sig_train**2)))

    def sig_to_aud(sig, seed_offset: int):
        acts = coch_encode(sig.astype(np.float64), encoder, sr=sr, n_ista=n_ista,
                           upsample_to_ms=True, T_out_ms=T_rend)
        spk = of_to_spikes(acts, mean_rate_hz=mean_rate_hz, frame_rate=1000,
                           seed=seed + seed_offset)
        return spk.astype(np.float32)

    if verbose:
        print(f"T_song={T_song} ms  T_rend={T_rend} ms  train_motif={train_idx}  "
              f"rms={rms_train:.4f}")
        print(f"Building correct training pool ({n_motifs} motifs)...")

    aud_correct = sig_to_aud(sig_train, 10)
    correct_pool = [aud_correct]                 # index 0 is the template motif
    for ci in range(n_motifs):
        if ci == train_idx:
            continue
        correct_pool.append(sig_to_aud(audio_m[ci, : lengths[ci]].astype(np.float64),
                                       100 + ci))

    sig_rev = sig_train[::-1].astype(np.float64)
    aud_reversed = sig_to_aud(sig_rev, 200)

    # How separable forward and reversed song are in the encoder is the ceiling on K4.
    acts_fwd = coch_encode(sig_train, encoder, sr=sr, n_ista=n_ista,
                           upsample_to_ms=True, T_out_ms=T_rend)
    acts_rev = coch_encode(sig_rev, encoder, sr=sr, n_ista=n_ista,
                           upsample_to_ms=True, T_out_ms=T_rend)
    fv, rv = acts_fwd.ravel(), acts_rev.ravel()
    fwd_rev_corr = (float(np.corrcoef(fv, rv)[0, 1])
                    if fv.std() > 0 and rv.std() > 0 else float("nan"))
    del acts_fwd, acts_rev, fv, rv

    # coch_encode RMS-normalises internally, so DAF amplitude is washed out; only the
    # broadband spectrum distinguishes white noise from song here.
    rng_daf = np.random.default_rng(seed + 1)
    noise_daf = rng_daf.standard_normal(len(sig_train)) * rms_train * DAF_WN_AMPLITUDE
    aud_daf = sig_to_aud(noise_daf, 20)

    if verbose:
        for label, arr in (("training motif", aud_correct), ("reversed motif", aud_reversed),
                           (f"DAF WN ({DAF_WN_AMPLITUDE}x RMS)", aud_daf)):
            print(f"  {label}: mean rate={arr.mean() * 1000:.1f} Hz  "
                  f"active fraction={float((arr > 0).mean()) * 100:.1f}%")
        print(f"  forward vs reversed activation correlation: {fwd_rev_corr:.4f}  "
              f"(lower -> cleaner K4 ceiling)")
        print("Building HVC inputs...")

    T_total_burn = t_burn + T_rend
    hvc_burn = generate_hvc_spikes(n_hvc=n_hvc, T=T_total_burn, n_renditions=1,
                                   T_song=T_song, T_burn=t_burn, T_post=t_post,
                                   peak_rate=PEAK_RATE, kernel_width=KERNEL_WIDTH,
                                   seed=seed)
    aud_burn = np.zeros((n_kernels, T_total_burn), dtype=np.float32)
    aud_burn[:, t_burn: t_burn + T_rend] = aud_correct

    hvc_on = generate_hvc_spikes(n_hvc=n_hvc, T=T_rend, n_renditions=1, T_song=T_song,
                                 T_burn=0, T_post=t_post, peak_rate=PEAK_RATE,
                                 kernel_width=KERNEL_WIDTH, seed=seed)
    hvc_off = np.zeros((n_hvc, T_rend), dtype=np.float32)

    return {
        "aud_correct": aud_correct,
        "aud_reversed": aud_reversed,
        "aud_daf": aud_daf,
        "correct_pool": correct_pool,
        "hvc_on": hvc_on,
        "hvc_off": hvc_off,
        "hvc_burn": hvc_burn,
        "aud_burn": aud_burn,
        "T_song": T_song,
        "T_rend": T_rend,
        "train_idx": train_idx,
        "fwd_rev_corr": fwd_rev_corr,
        "n_kernels": n_kernels,
        "n_motifs": n_motifs,
    }
