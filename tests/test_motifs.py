"""The motif preprocessing stage: DTW helpers and an end-to-end run on synthetic song."""

import numpy as np
import pytest
import scipy.io

from spiking_ven.data import motifs

STRETCH = 1.25


def _smooth_features(T, n=12):
    """Smooth, positive features whose frames are distinct under cosine distance."""
    t = np.arange(T, dtype=np.float64)
    k = np.arange(1, n + 1)[:, None]
    return 2.0 + np.sin(2 * np.pi * k * t / (3.0 * T) + k)


def _stretched_pair(T_ref=60):
    ref = _smooth_features(T_ref)
    T_q = int(round(T_ref * STRETCH))
    src = np.arange(T_q) / STRETCH          # query frame j shows reference frame j/STRETCH
    query = np.stack([np.interp(src, np.arange(T_ref), row) for row in ref])
    return ref, query


def test_dtw_warp_recovers_known_stretch():
    ref, query = _stretched_pair()
    warp = motifs.dtw_warp(ref, query)
    assert warp.shape == (ref.shape[1],)
    assert np.all(np.diff(warp) >= 0)
    expected = STRETCH * np.arange(ref.shape[1])
    assert np.max(np.abs(warp - expected)) < 2.0


def test_warp_time_ms_inverts_the_warp():
    ref, query = _stretched_pair()
    warp = motifs.dtw_warp(ref, query)
    hop = 10.0
    for t_ref in (50.0, 200.0, 400.0):
        assert motifs.warp_time_ms(t_ref * STRETCH, warp, hop) == pytest.approx(
            t_ref, abs=2 * hop)


def test_apply_audio_warp_returns_requested_length():
    sr, hop = 16000, 10
    warp = STRETCH * np.arange(50)
    q = np.random.default_rng(0).standard_normal(int(50 * STRETCH * sr * hop / 1000))
    out = motifs.apply_audio_warp(q, 8000, warp, sr, hop)
    assert out.shape == (8000,)
    assert out.dtype == np.float32


def _write_song(path, sf, amp, stretch, rng, fs=44100):
    """One WAV + .not.mat pair holding a single 'abcded' motif framed by intro notes."""
    freqs = {"i": 600.0, "a": 1500.0, "b": 2500.0, "c": 3500.0, "d": 4500.0, "e": 5500.0}
    labels = "iabcdedi"
    syl_ms, gap_ms = 40.0 * stretch, 15.0 * stretch
    onsets, offsets, t = [], [], 20.0
    for _ in labels:
        onsets.append(t)
        offsets.append(t + syl_ms)
        t += syl_ms + gap_ms
    audio = 1e-3 * rng.standard_normal(int((t + 20.0) * fs / 1000))
    for lab, on, off in zip(labels, onsets, offsets):
        i0, i1 = int(on * fs / 1000), int(off * fs / 1000)
        tt = np.arange(i1 - i0) / fs
        audio[i0:i1] += amp * np.sin(2 * np.pi * freqs[lab] * tt)
    sf.write(str(path), audio, fs)
    scipy.io.savemat(str(path) + ".not.mat", {
        "labels": labels,
        "onsets": np.array(onsets, dtype=np.float64),
        "offsets": np.array(offsets, dtype=np.float64),
        "Fs": float(fs),
    })


def test_main_end_to_end_records_template_index(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    rng = np.random.default_rng(0)
    # The loudest raw rendition is the second file, so the index cannot pass by default.
    specs = [(0.2, 1.0), (0.5, 1.1), (0.3, 0.92)]   # (amplitude, time stretch)
    for k, (amp, stretch) in enumerate(specs):
        _write_song(wav_dir / f"song{k}.wav", sf, amp, stretch, rng)

    out = tmp_path / "motifs.npz"
    motifs.main(["--wav-dir", str(wav_dir), "--out", str(out)])

    d = np.load(out)
    assert set(d.files) == {"audio", "lengths", "sr", "syl_on", "syl_off", "song_Ts",
                            "raw_song_Ts", "template_index"}
    assert d["audio"].shape[0] == len(specs)
    tpl = int(d["template_index"])
    assert tpl == 1
    assert np.all(d["lengths"] == d["audio"].shape[1])
    # Every rendition's syllables land near the template's after alignment.
    np.testing.assert_allclose(d["syl_on"], np.broadcast_to(d["syl_on"][tpl], d["syl_on"].shape),
                               atol=3 * motifs.HOP_MS)
