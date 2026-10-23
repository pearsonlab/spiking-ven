"""Error-signal metrics for a trained vocal error network.

How "did the network learn to cancel the bird's own song?" is turned into numbers,
against Mandelblat-Cerf et al. 2014 (eLife 3:e02152, Figs. 7/8):

- :func:`rendition_rates` -- mean E rate per rendition: the trained song and the
  time-reversed motif while singing (their ratio should exceed 1), and the DAF stimulus
  heard while not singing.
- :func:`daf_response` -- the DAF response as the experiment measures it: responders by a
  per-neuron paired t-test, their peak, onset latency, duration and extra spikes.

The stimuli come from :func:`spiking_ven.stimuli.build_stimuli`; the numbers aimed at are
in :mod:`spiking_ven.targets`.
"""

from __future__ import annotations

import numpy as np

from .stimuli import daf_waveform, daf_window_ms, encode_stimulus
from .targets import BIOLOGICAL_TARGETS, PEAK_BIN_MS

__all__ = [
    "daf_trials",
    "daf_responders",
    "daf_response",
    "format_responders",
    "rendition_rates",
    "format_rendition_rates",
]


def daf_trials(ven, encoder, st: dict, *, n_trials: int = 20, seed: int = 42):
    """E spikes for ``n_trials`` DAF renditions: ``(n_trials, n_e, T_rend)`` float32.

    Trials are renditions with independent noise draws; the song and the HVC drive are
    fixed, as in the experiment. ``st`` is :func:`~spiking_ven.stimuli.build_stimuli`'s
    output, so the trials are encoded exactly like every other stimulus, with the noise
    in ``st["daf_window_s"]``.
    """
    out = np.zeros((n_trials, int(ven.n_e), st["T_rend"]), dtype=np.float32)
    for t in range(n_trials):
        aud = encode_stimulus(
            daf_waveform(st["sig_train"], st["sr"], seed=seed + 1000 + t,
                         window_s=st["daf_window_s"]), encoder,
            sr=st["sr"], T_out_ms=st["T_rend"], acts_train=st["acts_train"],
            rms_from=st["sig_train"], mean_rate_hz=st["mean_rate_hz"],
            n_ista=st["n_ista"], seed=seed + 2000 + t,
        )
        out[t] = ven.transform(st["hvc_on"], aud)
    return out


def daf_responders(trials, *, onset_ms: int, window_ms: int = 150, alpha: float = 0.02):
    """Per-neuron responders by the paper's criterion. Returns ``(resp, pre, post)``.

    Spike counts in a ``window_ms`` window before and after burst onset (``onset_ms``, an
    index into the 1 ms ``trials`` axis), compared per neuron with a paired t-test across
    trials; a unit responds at ``p < alpha`` with an increase. ``pre``/``post`` are the
    ``(n_trials, n_e)`` counts.
    """
    from scipy import stats

    onset = int(onset_ms)
    W = int(window_ms)
    if onset - W < 0 or onset + W > trials.shape[2]:
        raise ValueError(
            f"{W} ms windows either side of onset {onset} ms do not fit in a "
            f"{trials.shape[2]} ms rendition"
        )
    pre = trials[:, :, onset - W:onset].sum(axis=2).astype(np.float64)
    post = trials[:, :, onset:onset + W].sum(axis=2).astype(np.float64)
    tval, pval = stats.ttest_rel(post, pre, axis=0)
    return (pval < alpha) & (tval > 0), pre, post


def daf_response(ven, encoder, st: dict, *, n_trials: int = 20, window_ms: int = 150,
                 alpha: float = 0.02, seed: int = 42,
                 duration_window_ms: int = 300) -> dict:
    """The DAF response, measured the way the experiment measures it.

    Per neuron, in windows either side of noise onset (from ``st["daf_window_s"]``),
    because an E rate averaged over a whole rendition dilutes a brief burst (~20:1 for
    50 ms) and cannot see it.

    ``window_ms`` sets the responder test and the onset search. The end of a response is
    searched for over ``duration_window_ms`` after onset (capped at the end of the motif),
    so that a response outlasting ``window_ms`` is timed rather than lost.

    Follows Mandelblat-Cerf et al. 2014; see :func:`daf_trials` and
    :func:`daf_responders`.
    """
    from scipy import stats

    onset, burst_ms = daf_window_ms(st["daf_window_s"])
    W = int(window_ms)
    trials = daf_trials(ven, encoder, st, n_trials=n_trials, seed=seed)
    n_e = int(ven.n_e)
    resp, pre, post = daf_responders(trials, onset_ms=onset, window_ms=W, alpha=alpha)
    psth = trials[:, :, onset - W:onset + W] * 1000
    base = pre.mean(axis=0) * (1000.0 / W)
    nan = float("nan")

    # The paper's ~20 Hz peak averages over its seven responders (Fig 8E, n=7), so the
    # model's peak averages over responders too. Binned at PEAK_BIN_MS; see
    # BIOLOGICAL_TARGETS["responder_peak_change_hz"] for why it is not a hard target.
    def _peak(mask):
        if not np.any(mask):
            return nan, 0
        mp = psth[:, mask, :].mean(axis=(0, 1))
        nb = (2 * W) // PEAK_BIN_MS
        mb = mp[:nb * PEAK_BIN_MS].reshape(nb, PEAK_BIN_MS).mean(axis=1)
        base = float(mb[:nb // 2].mean())
        return float(mb.max() - base), int(np.argmax(mb) * PEAK_BIN_MS - W)

    peak_resp, lat_resp = _peak(resp)

    # Transient or step: the responder PSTH 100-150 ms after onset, relative to its peak.
    def _late_over_peak(mask):
        if not np.any(mask):
            return nan
        mp = psth[:, mask, :].mean(axis=(0, 1))
        d = mp - float(mp[:W].mean())
        pk = float(d[W:].max())
        return float(d[W + 100:W + 150].mean() / pk) if pk > 0 else nan

    late_over_peak = _late_over_peak(resp)

    # Response onset latency and duration, per responder: trial-averaged rate in 2 ms
    # bins, each significant at p < 0.05 against that neuron's pre-onset bins. Latency is
    # the first of five consecutive significant bins, searched within ``window_ms``; it is
    # an onset, which is what the paper's 23 +/- 12 ms is -- the PSTH peak
    # (peak_latency_ms) comes ~25 ms later. Duration runs from there to the start of the
    # first run of ten non-significant bins. A response with no such run before the end
    # of the duration window is counted at the length it reached ("censored").
    BIN = 2
    dur_W = min(int(duration_window_ms), st.get("T_song", trials.shape[2]) - onset,
                trials.shape[2] - onset)
    dur_W -= dur_W % BIN
    pre_b = (trials[:, :, onset - W:onset].mean(axis=0) * 1000
             ).reshape(n_e, W // BIN, BIN).mean(axis=2)
    post_b = (trials[:, :, onset:onset + dur_W].mean(axis=0) * 1000
              ).reshape(n_e, dur_W // BIN, BIN).mean(axis=2)
    n_onset_bins = W // BIN
    lats, durs, n_censored = [], [], 0
    for u in np.flatnonzero(resp):
        mu, sd = pre_b[u].mean(), pre_b[u].std()
        if sd == 0:
            continue
        sig = (1.0 - stats.norm.cdf((post_b[u] - mu) / sd)) < 0.05
        la = next((i for i in range(n_onset_bins - 4) if sig[i:i + 5].all()), None)
        if la is None:
            continue
        lats.append(la * BIN)
        en = next((j for j in range(la, len(sig) - 9) if not sig[j:j + 10].any()), None)
        if en is None:
            n_censored += 1
            en = len(sig)
        durs.append((en - la) * BIN)
    return {
        "n_trials": int(n_trials),
        "n_units": n_e,
        "n_responders": int(resp.sum()),
        "responder_fraction": float(resp.mean()),
        "baseline_hz_all": float(base.mean()),
        "baseline_hz_responders": float(base[resp].mean()) if resp.any() else nan,
        #: Extra spikes per neuron per burst, over the analysis window. Compare with
        #: BIOLOGICAL_TARGETS["extra_spikes_per_burst"].
        "extra_spikes_responders": (float((post - pre)[:, resp].mean())
                                    if resp.any() else nan),
        #: The spread across responders (0.3-9 extra spikes in the shipped model): a
        #: uniform gain change would move both ends, where the excess is in the tail.
        "extra_spikes_sd": (float((post - pre)[:, resp].mean(axis=0).std())
                            if resp.any() else nan),
        "extra_spikes_p90": (float(np.percentile((post - pre)[:, resp].mean(axis=0), 90))
                             if resp.any() else nan),
        "extra_spikes_max": (float((post - pre)[:, resp].mean(axis=0).max())
                             if resp.any() else nan),
        "peak_change_hz": peak_resp,          # responders only, as in Fig 8E
        "peak_latency_ms": lat_resp,
        "late_over_peak": late_over_peak,
        "latency_ms": float(np.mean(lats)) if lats else nan,
        "duration_ms": float(np.mean(durs)) if durs else nan,
        #: Responders with a detectable onset (the latency/duration sample), and how many
        #: of them were still responding at the end of the duration window.
        "n_timed": len(lats),
        "duration_censored": n_censored,
        "duration_window_ms": dur_W,
        "window_ms": W,
        "burst_ms": burst_ms,
    }


def format_responders(r: dict) -> str:
    """Render :func:`daf_response` next to the paper's numbers."""
    t = BIOLOGICAL_TARGETS
    lo, hi = t["singing_rate_hz"]
    lat, lat_sd = t["response_latency_ms"]
    pk = t["responder_peak_change_hz"]
    dur_m, dur_sd = t["response_duration_ms"]
    pl_lo, pl_hi = t["responder_peak_latency_ms"]
    xs_lo, xs_hi = t["extra_spikes_per_burst"]
    return "\n".join([
        f"DAF response, {r['burst_ms']:.0f} ms burst, {r['window_ms']} ms windows, "
        f"{r['n_trials']} trials (Mandelblat-Cerf 2014):",
        f"  responders                 {r['n_responders']:4d}/{r['n_units']} "
        f"= {r['responder_fraction']:5.0%}      paper {t['responder_fraction']:.0%} (7/17)",
        f"  baseline, all units       {r['baseline_hz_all']:7.2f} Hz      "
        f"paper {lo:.0f}-{hi:.0f} Hz during singing",
        f"  baseline, responders      {r['baseline_hz_responders']:7.2f} Hz      "
        f"(biased low by the responder selection; not a rate target)",
        f"  peak change, responders   {r['peak_change_hz']:7.2f} Hz      "
        f"paper ~{t['responder_peak_change_hz']:.0f} Hz (Fig 8E, n=7; both at "
        f"{PEAK_BIN_MS} ms bins on our side, UNKNOWN on theirs -- bin-width dependent, "
        f"not a hard target)",
        f"  latency, responders       {r['latency_ms']:7.1f} ms      "
        f"paper {lat:.0f} +/- {lat_sd:.0f} ms  (onset: first of 5 consecutive 2 ms bins)",
        f"  duration, responders      {r['duration_ms']:7.1f} ms      "
        f"paper {dur_m:.0f} +/- {dur_sd:.0f} ms  ({r['n_timed']} with a detectable onset, "
        f"{r['duration_censored']} still responding at +{r['duration_window_ms']} ms)",
        f"  peak latency, responders  {r['peak_latency_ms']:+7d} ms      "
        f"paper ~{pl_lo:.0f}-{pl_hi:.0f} ms (Fig 8E, read off the panel)",
        f"  late (100-150ms) / peak   {r['late_over_peak']:7.3f}         "
        f"paper < {t['responder_late_over_peak_max']:.2f} (Fig 8E: a transient, not a step)",
        f"  extra spikes  p90 {r['extra_spikes_p90']:6.2f}  max {r['extra_spikes_max']:6.2f}"
        f"   sd {r['extra_spikes_sd']:5.2f}",
        f"  extra spikes/neuron/burst {r['extra_spikes_responders']:7.2f}         "
        f"bird ~{xs_lo:.1f}-{xs_hi:.1f} ({pk:.0f} Hz peak over {dur_m:.0f} ms; "
        f"no shape reported, so a conversion)",
    ])


def _rate_hz(ven, hvc, aud) -> float:
    """Mean E-population rate in Hz for one rendition of (hvc, aud) input."""
    return float(ven.transform(hvc, aud).mean() * 1000)


def rendition_rates(ven, *, hvc_on, hvc_off, aud_trained, aud_daf, aud_reversed) -> dict:
    """Mean E rate over one rendition, per condition, for a trained network.

    Parameters
    ----------
    ven          : trained VocalErrorNet
    hvc_on       : (n_hvc, T) HVC premotor spikes -- the singing condition
    hvc_off      : (n_hvc, T) zeros -- the not-singing control
    aud_trained  : (n_aud, T) spikes for the trained song
    aud_daf      : (n_aud, T) spikes for the DAF stimulus (song + noise burst)
    aud_reversed : (n_aud, T) spikes for the time-reversed motif

    Returns
    -------
    dict with ``trained_hz`` and ``reversed_hz`` (both while singing),
    ``daf_not_singing_hz`` (all Hz), and ``reversed_over_trained``, the ratio of the
    first two -- above 1 when cancellation is specific to the trained song.
    """
    trained_hz = _rate_hz(ven, hvc_on, aud_trained)
    daf_not_singing_hz = _rate_hz(ven, hvc_off, aud_daf)
    reversed_hz = _rate_hz(ven, hvc_on, aud_reversed)
    nan = float("nan")
    return {
        "trained_hz": trained_hz,
        "reversed_hz": reversed_hz,
        "reversed_over_trained": (reversed_hz / trained_hz) if trained_hz > 0 else nan,
        "daf_not_singing_hz": daf_not_singing_hz,
    }


def format_rendition_rates(m: dict) -> str:
    """Render the rendition-level rates from :func:`rendition_rates`.

    The DAF rate while singing is deliberately NOT reported here. Averaged over a whole
    rendition it cannot see a brief burst -- diluted ~20:1 for 50 ms, it sits on top of
    the reversed-song rate and reports the cancelled song instead. See
    :func:`daf_response`.
    """
    t = BIOLOGICAL_TARGETS
    lo, hi = t["singing_rate_hz"]
    return "\n".join([
        "Excitatory rate per rendition (Mandelblat-Cerf 2014):",
        f"  trained song, singing     {m['trained_hz']:6.2f} Hz   paper {lo:.0f}-{hi:.0f} Hz "
        f"during singing",
        f"  reversed song, singing    {m['reversed_hz']:6.2f} Hz",
        f"  reversed / trained        {m['reversed_over_trained']:6.2f}x     target > "
        f"{t['reversed_min_ratio']:.0f}x",
        f"  DAF stimulus, not singing {m['daf_not_singing_hz']:6.2f} Hz",
    ])
