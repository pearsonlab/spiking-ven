"""The corpus helper and the shared constants.

Template selection used to be re-derived in four places. These tests pin the single
implementation, including the invariant that actually matters: the template is the
loudest rendition, because that is the one the network trains on.
"""

import numpy as np
import pytest

from spiking_ven import constants
from spiking_ven.corpus import Corpus, highest_rms_index


def _corpus(n=4, T=1000, loud=2):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((n, T)) * 0.01).astype(np.float32)
    audio[loud] *= 10.0                       # make one rendition unambiguously loudest
    return Corpus(audio=audio, lengths=np.full(n, T, dtype=np.int32),
                  song_Ts=np.full(n, 500.0), sr=16000), loud


def test_template_is_the_loudest_rendition():
    c, loud = _corpus()
    assert c.template_index == loud
    sig, idx = c.template()
    assert idx == loud
    assert sig.shape == (c.audio.shape[1],)


def test_template_respects_valid_lengths():
    """A rendition that is loud only inside its zero-padding must not win."""
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal((3, 100)) * 0.01).astype(np.float32)
    audio[0, 50:] = 100.0                     # huge, but beyond this rendition's length
    lengths = np.array([50, 100, 100], dtype=np.int32)
    c = Corpus(audio=audio, lengths=lengths, song_Ts=np.full(3, 10.0), sr=16000)
    assert c.template_index != 0


def test_signal_is_trimmed_and_float64():
    c, _ = _corpus(T=800)
    c = Corpus(audio=c.audio, lengths=np.array([10, 20, 30, 40], dtype=np.int32),
               song_Ts=c.song_Ts, sr=c.sr)
    assert c.signal(2).shape == (30,)
    assert c.signal(2).dtype == np.float64


def test_T_song_rounds_up():
    c, _ = _corpus()
    c = Corpus(audio=c.audio, lengths=c.lengths,
               song_Ts=np.array([847.2, 846.0, 840.0, 830.0]), sr=c.sr)
    assert c.T_song == 848


def test_highest_rms_index_rejects_empty():
    with pytest.raises(ValueError):
        highest_rms_index([])


def test_peak_rate_is_derived_not_hardcoded():
    """Changing the kernel width must carry the peak rate with it."""
    assert constants.PEAK_RATE_HZ == pytest.approx(
        150.0 * 20.0 / constants.KERNEL_WIDTH_MS)


def test_encoder_classes_share_a_size_property():
    """Both encoders expose n_channels, so callers need no hasattr fallback."""
    from spiking_ven import OlshausenFieldEncoder, SmithLewickiDictionary

    of = OlshausenFieldEncoder(n_bases=7, patch_len=40, seed=0)
    assert of.n_channels == 7
    sl = SmithLewickiDictionary(n_kernels=5, kernel_ms=10.0)
    assert sl.n_channels == 5
