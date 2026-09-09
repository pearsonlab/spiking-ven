"""Cochleagram: gammatone spectrogram with sqrt compression.

Pipeline:
    raw waveform
        → gammatone filterbank (ERB-spaced)
        → per-band energy (non-negative)
        → sqrt compression (approximates ~40 dB dynamic range compression)
        → (n_channels, n_frames) at frame_rate Hz
"""

import numpy as np

from .filterbank import gammatone_spectrogram


def cochleagram(
    signal: np.ndarray,
    sr: int,
    n_channels: int = 64,
    frame_rate: int = 1000,
    lo_hz: float = 100.0,
    hi_hz: float = 8000.0,
) -> np.ndarray:
    """Compute cochleagram from a mono waveform.

    Parameters
    ----------
    signal : 1-D float array, mono waveform (any amplitude scale)
    sr : sample rate in Hz
    n_channels : number of ERB-spaced frequency channels
    frame_rate : output frame rate in Hz (cochleagram time resolution)
    lo_hz, hi_hz : frequency range of the filter bank

    Returns
    -------
    gram : (n_channels, n_frames) float32 array, sqrt-compressed filter
           energies.  Values are proportional to signal amplitude; no
           per-signal normalisation is applied so that amplitude information
           is preserved across calls (suitable for streaming / causal use).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D (mono)")

    gram = gammatone_spectrogram(signal, sr, n_channels, frame_rate, lo_hz, hi_hz)
    gram = np.sqrt(np.maximum(gram, 0.0))
    return gram.astype(np.float32)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV (or any libsndfile-supported) audio file.

    Returns (signal, sr) where signal is mono float64 normalised to [-1, 1].
    If the file is stereo, channels are averaged.
    """
    import soundfile as sf

    data, sr = sf.read(path, always_2d=True, dtype="float64")
    mono = data.mean(axis=1)
    return mono, sr
