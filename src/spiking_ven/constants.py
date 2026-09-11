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

#: Window within the motif that the figure's DAF column perturbs (seconds).
DAF_WINDOW_S = (0.400, 0.600)

#: Cochlea -> AIV-E conduction delay (ms). Measured; Mandelblat-Cerf et al. 2014.
AUD_DELAY_MS = 23

#: HVC -> AIV-I conduction delay (ms). Estimated -- no direct measurement exists.
HVC_DELAY_MS = 5
