"""Calibration constants shared by the trainer, the metrics and the figures.

These were previously repeated across those three places, which is a drift hazard rather
than a style problem: the stimulus the figure draws has to be the stimulus the network was
trained and scored on, or the figure describes a different experiment.

Kept deliberately small. Anything that is genuinely a per-run choice belongs on a CLI
flag, not here.
"""

from __future__ import annotations

__all__ = [
    "SR",
    "KERNEL_WIDTH_MS",
    "PEAK_RATE_HZ",
    "DAF_WN_AMPLITUDE",
    "DAF_WINDOW_S",
    "AUD_DELAY_MS",
    "AUD_MEAN_RATE_HZ",
    "HVC_DELAY_MS",
]

#: Corpus sample rate (Hz). The corpus is resampled to this during preprocessing.
SR = 16000

#: Width of the Gaussian HVC burst envelope (ms).
KERNEL_WIDTH_MS = 10.0

#: Peak HVC burst rate (spk/s). Scales inversely with the kernel width so the integrated
#: drive per burst is constant -- derived, never written as a literal.
PEAK_RATE_HZ = 150.0 * 20.0 / KERNEL_WIDTH_MS

#: DAF white noise is mixed at this multiple of song RMS: ~95 dBSPL WN vs ~80 dBSPL song.
#:
#: This scales the *figure's* DAF column, where the noise is added into a window of the
#: song and the relative level therefore matters. It does NOT affect the K2/K3 metrics:
#: ``evaluate.build_stimuli`` sends its white noise through the same RMS-normalising
#: encoder path as every other stimulus, so the amplitude divides straight back out.
#: See the note in :mod:`spiking_ven.evaluate`.
DAF_WN_AMPLITUDE = 5.6

#: Auditory population mean firing rate during song (Hz), the encoder's spike-rate
#: calibration target.
#:
#: Schneider & Woolley 2013, Neuron 79:141-152 (DOI 10.1016/j.neuron.2013.04.038) measure
#: 2.4 +/- 2.7 Hz for the broad-spiking neurons of the higher-level auditory cortex -- the
#: sparse, selective, background-invariant population this encoder models. 2.5 realises
#: 2.30 Hz after the refractory period. It was 15.0, an unsourced default repeated across
#: seven call sites, which made the auditory input ~4x too fast.
AUD_MEAN_RATE_HZ = 2.5

#: Window within the motif that the DAF stimulus perturbs (seconds).
#:
#: 50 ms, matching the noise bursts in Mandelblat-Cerf et al. 2014 -- the experiment the
#: responder statistics come from. It was 200 ms, which is long enough that the modelled
#: response keeps building through the burst instead of being the transient the paper
#: measures (peak 118 ms in at 200 ms, against 88 ms at 50 ms).
DAF_WINDOW_S = (0.400, 0.450)

#: Cochlea -> AIV-E conduction delay (ms).
#:
#: NOT a measured conduction delay. Mandelblat-Cerf et al. 2014 report 23 +/- 12 ms as the
#: AIV *response latency* -- noise onset to the first 2 ms bin followed by five
#: significant bins -- which is the end-to-end observable this model should PREDICT, not
#: consume. Using it as the conduction delay double-counts: the network then integrates
#: for a further ~43 ms, putting the modelled latency at ~66 ms, and makes the 23 ms
#: comparison circular.
#:
#: Now 5 ms, a plausible synaptic value, which is what every validated run uses. The
#: latency is therefore PREDICTED, not consumed: the model lands at 27-34 ms against the
#: bird's 23 +/- 12, with the residual being genuine network integration time. At 23 ms it
#: was double-counted and the comparison was circular.
AUD_DELAY_MS = 5

#: HVC -> AIV-I conduction delay (ms). Estimated -- no direct measurement exists. Zero in
#: the validated operating point: HVC drives I directly and any delay here shifts the
#: motor prediction relative to the auditory signal it has to cancel.
HVC_DELAY_MS = 0
