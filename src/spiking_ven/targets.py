"""Literature targets the model is scored against, with their citations.

Kept apart from the code that measures the model so the numbers being aimed at, and the
caveats on each, can be read in one place. The DAF response numbers are from
Mandelblat-Cerf et al. 2014 (eLife 3:e02152); the auditory input rate is from Schneider
& Woolley 2013 (Neuron 79:141-152).
"""

from __future__ import annotations

__all__ = ["BIOLOGICAL_TARGETS", "PEAK_BIN_MS"]

#: Bin width (ms) for the responder PSTH peak. 2 ms is the only binning Mandelblat-Cerf
#: state -- for their latency and duration z-tests -- and nothing in the paper says how
#: the plotted PSTHs were binned or whether they were smoothed.
PEAK_BIN_MS = 2

# Verbatim targets, quoted or read off the papers' figures.
BIOLOGICAL_TARGETS = {
    # Quoted from Mandelblat-Cerf et al. 2014 (eLife 3:e02152), pp. 10-13.
    #: "AIV single-units discharged at low rates during singing (1-10 Hz, Figure 7C)".
    #: A range for the population of 37 single units, not a mean.
    #:
    #: Compare ``baseline_hz_all`` against this, NEVER ``baseline_hz_responders``.
    #: Responders are selected by a paired t-test for post > pre, which preferentially
    #: picks neurons that happened to be quiet in the pre-window, so their baseline is
    #: biased low by the selection itself -- measured here ~0.4 Hz against the
    #: population's ~2.6 Hz. The paper applies the same criterion, so the
    #: bias is shared and the responder-vs-responder comparisons stay fair; it just means
    #: the responder baseline is not an estimate of anything's firing rate.
    "singing_rate_hz": (1.0, 10.0),
    #: "More than 40% of the VTA/SNc-projecting neurons exhibited a significant neuronal
    #: response to distorted auditory feedback (50 ms noise bursts) presented during
    #: singing ... p<0.02 for n = 7/17 neurons".
    "responder_fraction": 7 / 17,
    #: Fig 8E, black trace ("noise during singing", n=7), read off the panel's y-axis,
    #: which is the CHANGE in firing rate. A diagnostic, not a target: a peak depends on
    #: bin width, and the paper states none for this PSTH (2 ms bins appear only in the
    #: latency/duration tests). The model's peak falls from 49 to 35 Hz between 1 and
    #: 10 ms bins. Compare spike counts (``extra_spikes_per_burst``) instead, which do not
    #: depend on binning.
    "responder_peak_change_hz": 20.0,
    #: Extra spikes per responder per burst, derived from the two numbers above and below:
    #: a ~20 Hz peak over a 90 ms response is ~0.8-1.0 spikes for a triangular or
    #: rise-then-exponential shape (a rectangle would give 1.8). The paper reports no
    #: shape, so this is a conversion, not a measurement.
    "extra_spikes_per_burst": (0.8, 1.0),
    #: Fig 8E, black trace, read off the panel (the paper states no number): the peak
    #: falls ~40-60 ms after noise onset, widened to 35-75 ms for figure-reading error.
    #: Less bin-dependent than the amplitude: the model's peak time stays at 50-55 ms
    #: from 1 to 10 ms bins.
    "responder_peak_latency_ms": (35.0, 75.0),
    #: Also off Fig 8E: the black trace is back to ~0 by 120-150 ms and stays there, i.e.
    #: a transient rather than a step. The isolated-noise trace in the same panel does
    #: show a late sustained rise, so this discriminates between the two shapes. Note it
    #: only catches a response that ramps and stays up; a response SHORTER than the
    #: bird's also passes, and "response_duration_ms" is what covers that side.
    "responder_late_over_peak_max": 0.15,
    #: "an average latency of 23 +/- 12 ms from noise onset and ... an average duration
    #: of 90 +/- 43 ms" (mean, SD).
    "response_latency_ms": (23.0, 12.0),
    "response_duration_ms": (90.0, 43.0),
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
}
