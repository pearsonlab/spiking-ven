"""The corpus helper and the shared constants.

These pin the single template-selection rule, including the invariant that matters: the
template is the loudest rendition, because that is the one the network trains on.
"""

import numpy as np
import pytest

from spiking_ven import constants
from spiking_ven.corpus import Corpus, highest_rms_index, load_corpus


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


def test_stored_template_index_wins_over_computed():
    c, loud = _corpus()
    other = (loud + 1) % c.n_motifs
    c = Corpus(audio=c.audio, lengths=c.lengths, song_Ts=c.song_Ts, sr=c.sr,
               stored_template_index=other)
    assert c.template_index == other
    assert c.template()[1] == other


def test_template_falls_back_to_highest_rms_without_stored_index():
    c, loud = _corpus()
    assert c.stored_template_index is None
    assert c.template_index == loud


@pytest.mark.parametrize("bad", [-1, 4, 10])
def test_out_of_range_stored_index_rejected(bad):
    c, _ = _corpus()
    with pytest.raises(ValueError, match="out of range"):
        Corpus(audio=c.audio, lengths=c.lengths, song_Ts=c.song_Ts, sr=c.sr,
               stored_template_index=bad)


@pytest.mark.parametrize("stored", [None, 0])
def test_load_corpus_round_trip(tmp_path, stored):
    c, loud = _corpus()
    extra = {} if stored is None else {"template_index": stored}
    path = tmp_path / "motifs.npz"
    np.savez_compressed(path, audio=c.audio, lengths=c.lengths, sr=c.sr,
                        song_Ts=c.song_Ts, **extra)
    loaded = load_corpus(path)
    assert loaded.stored_template_index == stored
    assert loaded.template_index == (loud if stored is None else stored)
    np.testing.assert_array_equal(loaded.audio, c.audio)
    assert loaded.sr == c.sr
