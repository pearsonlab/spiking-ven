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

__all__ = ["daf_metrics", "format_metrics", "BIOLOGICAL_TARGETS"]

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
