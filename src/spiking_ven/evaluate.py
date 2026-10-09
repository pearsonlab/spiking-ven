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

What K2/K3 measure
------------------
The response to white noise presented at ``DAF_WN_AMPLITUDE`` times the song's RMS,
against the response to the song -- level difference included.

This was not always so. ``coch_encode`` used to scale every signal to the encoder's
reference RMS *individually*, which divided ``DAF_WN_AMPLITUDE`` straight back out and
left K2/K3 reporting a pure spectrotemporal mismatch at matched drive. The stimulus set
is now normalised against one reference (the song) instead, so relative level survives
into the cochleagram. ``of_to_spikes`` still rate-calibrates, but on the song window via
``calibrate_on``, so it no longer flattens the levels either.

This lives here, as a function returning a dict, rather than as a block of prints at the
bottom of a training script, so the same numbers can be asserted in tests and recomputed
for a saved model without retraining.
"""

from __future__ import annotations

import numpy as np

from .constants import DAF_WINDOW_S, DAF_WN_AMPLITUDE, KERNEL_WIDTH_MS, PEAK_RATE_HZ

__all__ = [
    "daf_waveform",
    "encode_acts",
    "encode_stimulus",
    "daf_response",
    "format_responders",
    "daf_metrics",
    "format_metrics",
    "build_stimuli",
    "BIOLOGICAL_TARGETS",
    "DAF_WN_AMPLITUDE",
]

# Verbatim targets, kept next to the code that is judged against them.
BIOLOGICAL_TARGETS = {
    # Quoted from Mandelblat-Cerf et al. 2014 (eLife 3:e02152), pp. 10-13.
    #: "AIV single-units discharged at low rates during singing (1-10 Hz, Figure 7C)".
    #: A range for the population of 37 single units, not a mean.
    "singing_rate_hz": (1.0, 10.0),
    #: "More than 40% of the VTA/SNc-projecting neurons exhibited a significant neuronal
    #: response to distorted auditory feedback (50 ms noise bursts) presented during
    #: singing ... p<0.02 for n = 7/17 neurons".
    "responder_fraction": 7 / 17,
    #: Fig 8E, black trace ("noise during singing", n=7). Note the panel's y-axis is
    #: CHANGE in firing rate, so this is an increase above baseline, and it is the peak
    #: of a trial-averaged PSTH -- not a mean over a window. Read off the axis.
    "responder_peak_change_hz": 20.0,
    #: "an average latency of 23 +/- 12 ms from noise onset and ... an average duration
    #: of 90 +/- 43 ms" (mean, SD).
    "response_latency_ms": (23.0, 12.0),
    "response_duration_ms": (90.0, 43.0),
    #: The burst length the responder statistics were measured with.
    "noise_burst_ms": 50.0,
    "reversed_min_ratio": 1.0,  # direction only

    # --- auditory input population ------------------------------------------------
    # Schneider & Woolley 2013, Neuron 79:141-152, DOI 10.1016/j.neuron.2013.04.038.
    # Zebra finch, awake restrained, song-driven rates by station (mean +/- SD), p. 142:
    #   midbrain MLd (n=100)                      39.8 +/- 25.4 Hz
    #   primary AC / Field L (n=99)               32.4 +/- 20.1 Hz
    #   higher-level AC, narrow-spiking (n=35)    19.0 +/- 11.7 Hz
    #   higher-level AC, broad-spiking (n=135)     2.4 +/-  2.7 Hz
    # The broad-spiking population is the sparse, selective, background-invariant one, so
    # it is what this encoder models -- not the dense upstream stations.
    "aud_song_rate_hz": (2.4, 2.7),
    #: "while more than 70% of neurons in upstream auditory areas fired during an average
    #: 63 ms window, only 5% of BS neurons were active during the same epoch" (Fig 3B).
    #: 63 ms is the average duration of a zebra finch song note.
    "aud_pop_active_frac": 0.05,
    "aud_pop_window_ms": 63,
    #: Selectivity: BS neurons were driven by 6.9 +/- 5.2 of 15 songs, against 14.4 +/- 2.5
    #: (midbrain), 14.7 +/- 1.9 (primary AC), 14.96 +/- 0.21 (higher-level AC NS).
    "aud_songs_driving_of_15": (6.9, 5.2),
    # NOTE: the paper reports no intra-burst or instantaneous rate for these neurons, so
    # there is no target for burst strength here. Fig 3A's neurogram tops out at 67
    # spikes/s, but that is a trial-averaged population display, not an instantaneous rate.
}

# Superseded, and wrong: a previous table listed 7.7 +/- 8.7 Hz for singing (not a number
# the paper states) and 28 Hz for responders citing Fig 8E (which plots a *change* in rate
# peaking near 20 Hz, not an absolute rate). The 2.1x and 3.6x "targets" were quotients of
# those two, computed here rather than reported by the paper. Do not reinstate them.


def daf_waveform(sig_train, sr: int, *, seed: int = 42,
                 amplitude: float = DAF_WN_AMPLITUDE,
                 window_s: tuple[float, float] = DAF_WINDOW_S):
    """The DAF stimulus: the song, with white noise mixed into one window.

    Defined once, here, and used by both the metrics and the figure. They used to
    build different stimuli under the same name -- the metrics scored *pure* white
    noise for the whole rendition, with no song in it at all, while the figure drew
    song plus a noise burst. Only the latter is the DAF paradigm the biological
    numbers come from: the bird sings, and noise is played back during singing.

    Returns the waveform; the caller encodes it.
    """
    import numpy as np

    sig = np.asarray(sig_train, dtype=np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    noise = np.random.default_rng(seed).standard_normal(len(sig)) * rms * amplitude
    s0, s1 = int(window_s[0] * sr), int(window_s[1] * sr)
    if s0 >= len(sig):
        raise ValueError(
            f"the DAF window {window_s} s starts past the end of a "
            f"{len(sig) / sr:.3f} s motif, so the DAF stimulus would be identical to "
            "the song. Adjust constants.DAF_WINDOW_S for this corpus."
        )
    out = sig.copy()
    out[s0:s1] += noise[s0:s1]
    return out


def encode_acts(sig, encoder, *, sr, n_ista, T_out_ms, rms_from, divisive_gain=True):
    """Waveform -> (n_channels, T) activations, dispatching on the encoder type.

    Every activation in this module goes through here. The dispatch used to live only in
    encode_stimulus, so build_stimuli's other three call sites bypassed it and a
    non-patch encoder failed on the first one.
    """
    from .olshausen_field import coch_encode

    sig = np.asarray(sig, dtype=np.float64)
    if hasattr(encoder, "encode_signal"):        # causal filter bank
        return encoder.encode_signal(sig, sr=sr, n_ista=n_ista, upsample_to_ms=True,
                                     T_out_ms=T_out_ms, rms_from=rms_from,
                                     divisive_gain=divisive_gain)
    return coch_encode(sig, encoder, sr=sr, n_ista=n_ista, upsample_to_ms=True,
                       T_out_ms=T_out_ms, rms_from=rms_from,
                       divisive_gain=divisive_gain)


def encode_stimulus(sig, encoder, *, sr, T_out_ms, acts_train, rms_from,
                    mean_rate_hz: float = 15.0, n_ista: int = 50, seed: int = 0):
    """Waveform to auditory spike train, on the shared reference.

    One definition of the encode path, so the metrics, the responder analysis and the
    figure cannot normalise differently. Both normalisation stages take their reference
    from the song: the waveform scaling from ``rms_from``, the spike rate from
    ``acts_train``.
    """
    from .olshausen_field import of_to_spikes

    acts = encode_acts(sig, encoder, sr=sr, n_ista=n_ista, T_out_ms=T_out_ms,
                       rms_from=rms_from)
    return of_to_spikes(acts, mean_rate_hz=mean_rate_hz, frame_rate=1000, seed=seed,
                        calibrate_on=acts_train).astype(np.float32)


def daf_response(ven, encoder, *, sig_train, hvc_on, acts_train, sr, T_out_ms,
                 n_trials: int = 20, window_ms: int = 150, alpha: float = 0.02,
                 seed: int = 42, mean_rate_hz: float = 15.0, n_ista: int = 50) -> dict:
    """The DAF response, measured the way the experiment measures it.

    Replaces averaging the excitatory rate over a whole rendition, which cannot see a
    50 ms burst: diluted ~20:1 in a 1048 ms rendition, that number is dominated by the
    cancelled song and lands on top of the reversed-song rate.

    Follows Mandelblat-Cerf et al. 2014: spike counts in a ``window_ms`` window before
    and after burst onset, compared per neuron with a paired t-test across trials, and
    a unit counts as a responder at ``p < alpha`` with an increase. Trials are
    renditions with independent noise draws; the song and the HVC drive are fixed, as
    in the experiment.
    """
    from scipy import stats

    onset = int(DAF_WINDOW_S[0] * 1000)
    W = int(window_ms)
    n_e = int(ven.n_e)
    pre = np.zeros((n_trials, n_e))
    post = np.zeros((n_trials, n_e))
    psth = np.zeros((n_trials, n_e, 2 * W), dtype=np.float32)
    for t in range(n_trials):
        aud = encode_stimulus(
            daf_waveform(sig_train, sr, seed=seed + 1000 + t), encoder,
            sr=sr, T_out_ms=T_out_ms, acts_train=acts_train, rms_from=sig_train,
            mean_rate_hz=mean_rate_hz, n_ista=n_ista, seed=seed + 2000 + t,
        )
        sE = ven.transform(hvc_on, aud)
        pre[t] = sE[:, onset - W:onset].sum(axis=1)
        post[t] = sE[:, onset:onset + W].sum(axis=1)
        psth[t] = sE[:, onset - W:onset + W] * 1000

    tval, pval = stats.ttest_rel(post, pre, axis=0)
    resp = (pval < alpha) & (tval > 0)
    to_hz = 1000.0 / W
    base = pre.mean(axis=0) * to_hz
    inc = (post - pre).mean(axis=0) * to_hz
    nan = float("nan")

    # The paper's ~20 Hz peak is the trial average over its SEVEN RESPONDERS (Fig 8E,
    # n=7), so the model's peak has to be averaged over responders too. It used to average
    # over every unit, which mixes the non-responders' flat PSTHs into a number compared
    # against a responders-only one -- and understates the gap, since responders are
    # roughly half the population here. The all-unit value is kept separately rather than
    # dropped, because it is the right denominator for anything about the population.
    def _peak(mask):
        if not np.any(mask):
            return nan, 0
        mp = psth[:, mask, :].mean(axis=(0, 1))
        return float(mp.max() - float(mp[:W].mean())), int(np.argmax(mp) - W)

    peak_resp, lat_resp = _peak(resp)
    peak_all, _ = _peak(np.ones(n_e, dtype=bool))

    # Response onset latency and duration, by the paper's own rule: 2 ms bins, a bin is
    # significant at p < 0.05 against the pre-onset distribution, latency is the first bin
    # followed by five consecutive significant ones, duration runs to the first run of ten
    # non-significant bins. This is an ONSET, which is what 23 +/- 12 ms refers to -- not
    # the PSTH argmax, which is later (49 ms here) and was being printed against it.
    BIN = 2
    nb = (2 * W) // BIN
    half = nb // 2
    rate = psth.reshape(n_trials, n_e, nb, BIN).mean(axis=3).mean(axis=0)
    lats, durs = [], []
    for u in np.flatnonzero(resp):
        rr = rate[u]
        mu, sd = rr[:half].mean(), rr[:half].std()
        if sd == 0:
            continue
        sig = (1.0 - stats.norm.cdf((rr[half:] - mu) / sd)) < 0.05
        la = next((i for i in range(len(sig) - 5) if sig[i:i + 5].all()), None)
        if la is None:
            continue
        lats.append(la * BIN)
        en = next((j for j in range(la, len(sig) - 10) if not sig[j:j + 10].any()), None)
        if en is not None:
            durs.append((en - la) * BIN)
    return {
        "n_trials": int(n_trials),
        "n_units": n_e,
        "n_responders": int(resp.sum()),
        "responder_fraction": float(resp.mean()),
        "baseline_hz_all": float(base.mean()),
        "baseline_hz_responders": float(base[resp].mean()) if resp.any() else nan,
        "increase_hz_responders": float(inc[resp].mean()) if resp.any() else nan,
        #: Extra spikes per neuron per burst, over the analysis window. The directly
        #: interpretable form: the paper's responder gains ~1-2 spikes when it hears the
        #: distortion, on a baseline of well under one spike in the same window.
        "extra_spikes_responders": (float((post - pre)[:, resp].mean())
                                    if resp.any() else nan),
        "increase_hz_all": float(inc.mean()),
        "peak_change_hz": peak_resp,          # responders only, as in Fig 8E
        "peak_change_hz_all": peak_all,
        "peak_latency_ms": lat_resp,
        "latency_ms": float(np.mean(lats)) if lats else nan,
        "duration_ms": float(np.mean(durs)) if durs else nan,
        "n_with_latency": len(lats),
        "window_ms": W,
        "burst_ms": float((DAF_WINDOW_S[1] - DAF_WINDOW_S[0]) * 1000),
    }


def format_responders(r: dict) -> str:
    """Render :func:`daf_response` next to the paper's numbers."""
    t = BIOLOGICAL_TARGETS
    lo, hi = t["singing_rate_hz"]
    lat, lat_sd = t["response_latency_ms"]
    pk = t["responder_peak_change_hz"]
    dur_m, dur_sd = t["response_duration_ms"]
    dur = dur_m / 1000.0
    return "\n".join([
        f"DAF response, {r['burst_ms']:.0f} ms burst, {r['window_ms']} ms windows, "
        f"{r['n_trials']} trials (Mandelblat-Cerf 2014):",
        f"  responders                 {r['n_responders']:4d}/{r['n_units']} "
        f"= {r['responder_fraction']:5.0%}      paper {t['responder_fraction']:.0%} (7/17)",
        f"  baseline, all units       {r['baseline_hz_all']:7.2f} Hz      "
        f"paper {lo:.0f}-{hi:.0f} Hz during singing",
        f"  baseline, responders      {r['baseline_hz_responders']:7.2f} Hz",
        f"  peak change, responders   {r['peak_change_hz']:7.2f} Hz      "
        f"paper ~{t['responder_peak_change_hz']:.0f} Hz (Fig 8E, n=7)",
        f"  peak change, all units    {r['peak_change_hz_all']:7.2f} Hz      "
        f"(no paper equivalent; responders only are plotted there)",
        f"  latency, responders       {r['latency_ms']:7.1f} ms      "
        f"paper {lat:.0f} +/- {lat_sd:.0f} ms  (onset: first of 5 consecutive 2 ms bins)",
        f"  duration, responders      {r['duration_ms']:7.1f} ms      "
        f"paper {dur_m:.0f} +/- {dur_sd:.0f} ms",
        f"  peak latency, responders  {r['peak_latency_ms']:+7d} ms      "
        f"(PSTH argmax; later than onset, no paper equivalent)",
        # The window mean and the peak are different quantities, so the paper's peak is
        # converted here rather than printed beside a mean as if they matched. A 20 Hz
        # peak lasting 90 +/- 43 ms is 1.8 extra spikes if the response is rectangular,
        # 0.9 if triangular -- the honest range given the paper reports no shape.
        f"  extra spikes/neuron/burst {r['extra_spikes_responders']:7.2f}         "
        f"paper ~{0.5 * pk * dur:.1f}-{pk * dur:.1f} "
        f"({pk:.0f} Hz peak x {1000 * dur:.0f} ms, triangular..rectangular)",
    ])


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
    """Render the rendition-level rates from :func:`daf_metrics`.

    The DAF rate is deliberately NOT reported here. Averaged over a whole rendition it
    cannot see a 50 ms burst -- diluted ~20:1, it sits on top of the reversed-song rate
    and reports the cancelled song instead. See :func:`daf_response`.
    """
    t = BIOLOGICAL_TARGETS
    lo, hi = t["singing_rate_hz"]
    sanity = (f"  (sanity; homeostasis target {r_e_target:.0f} Hz)"
              if r_e_target is not None else "")
    return "\n".join([
        "Excitatory rate per rendition (Mandelblat-Cerf 2014):",
        f"  trained song, singing     {m['k1']:6.2f} Hz   paper {lo:.0f}-{hi:.0f} Hz "
        f"during singing",
        f"  reversed song, singing    {m['k4_rate']:6.2f} Hz",
        f"  reversed / trained        {m['k4']:6.2f}x     target > "
        f"{t['reversed_min_ratio']:.0f}x",
        f"  DAF stimulus, not singing {m['k2_no_hvc']:6.2f} Hz{sanity}",
    ])

# ---------------------------------------------------------------------------
# Stimulus construction
# ---------------------------------------------------------------------------
# Defined once, here, and used by BOTH the trainer and the reproduction tests. If the
# two ever built their stimuli separately they would drift, and the metrics would stop
# describing the model that was trained.



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
    from .corpus import load_corpus
    from .olshausen_field import of_to_spikes

    corpus = load_corpus(motifs_path)
    n_motifs = corpus.n_motifs
    T_song = corpus.T_song
    T_rend = T_song + t_post
    n_kernels = encoder.n_channels

    sig_train, train_idx = corpus.template()
    rms_train = float(np.sqrt(np.mean(sig_train**2)))

    # The encoder path normalises TWICE, and both have to share one reference or the
    # level difference does not reach the network:
    #   coch_encode   scales the waveform  -> rms_from=sig_train
    #   of_to_spikes  scales the spike rate -> calibrate_on=acts_train
    # Fixing only the first restores level in the activations and then flattens it again
    # at the spike stage: measured, DAF at 5.6x song RMS gives 2.81x the activation
    # energy but still 15.6 Hz/channel, against 43.4 Hz/channel when both are shared.
    acts_train = encode_acts(sig_train, encoder, sr=sr, n_ista=n_ista,
                             T_out_ms=T_rend, rms_from=sig_train)

    def sig_to_aud(sig, seed_offset: int, acts=None):
        if acts is None:
            acts = encode_acts(sig, encoder, sr=sr, n_ista=n_ista,
                               T_out_ms=T_rend, rms_from=sig_train)
        spk = of_to_spikes(acts, mean_rate_hz=mean_rate_hz, frame_rate=1000,
                           seed=seed + seed_offset, calibrate_on=acts_train)
        return spk.astype(np.float32)

    if verbose:
        print(f"T_song={T_song} ms  T_rend={T_rend} ms  train_motif={train_idx}  "
              f"rms={rms_train:.4f}")
        print(f"Building correct training pool ({n_motifs} motifs)...")

    aud_correct = sig_to_aud(sig_train, 10, acts=acts_train)
    correct_pool = [aud_correct]                 # index 0 is the template motif
    for ci in range(n_motifs):
        if ci == train_idx:
            continue
        correct_pool.append(sig_to_aud(corpus.signal(ci), 100 + ci))

    sig_rev = sig_train[::-1].astype(np.float64)
    aud_reversed = sig_to_aud(sig_rev, 200)

    # How separable forward and reversed song are in the encoder is the ceiling on K4.
    # acts_train is the forward encoding already, so only the reversed one is new.
    acts_fwd = acts_train
    acts_rev = encode_acts(sig_rev, encoder, sr=sr, n_ista=n_ista,
                           T_out_ms=T_rend, rms_from=sig_train)
    fv, rv = acts_fwd.ravel(), acts_rev.ravel()
    fwd_rev_corr = (float(np.corrcoef(fv, rv)[0, 1])
                    if fv.std() > 0 and rv.std() > 0 else float("nan"))
    del acts_rev, fv, rv

    # Song with noise mixed into one window -- the same construction the figure draws,
    # from the same function. seed + 2 matches the figure's stream, so the two describe
    # the identical stimulus rather than two draws of a similar one.
    aud_daf = sig_to_aud(daf_waveform(sig_train, sr, seed=seed + 2), 20)

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
                                   peak_rate=PEAK_RATE_HZ, kernel_width=KERNEL_WIDTH_MS,
                                   seed=seed)
    aud_burn = np.zeros((n_kernels, T_total_burn), dtype=np.float32)
    aud_burn[:, t_burn: t_burn + T_rend] = aud_correct

    hvc_on = generate_hvc_spikes(n_hvc=n_hvc, T=T_rend, n_renditions=1, T_song=T_song,
                                 T_burn=0, T_post=t_post, peak_rate=PEAK_RATE_HZ,
                                 kernel_width=KERNEL_WIDTH_MS, seed=seed)
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
        "acts_train": acts_train,
        "sig_train": sig_train,
        "sr": sr,
        "fwd_rev_corr": fwd_rev_corr,
        "n_kernels": n_kernels,
        "n_motifs": n_motifs,
    }
