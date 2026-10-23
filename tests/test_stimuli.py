"""Stimulus construction, the DAF window's path into the analysis, and stimulus provenance.

All synthetic: a tiny random encoder, a few hundred ms of noise as the corpus, and
hand-built trial arrays, so none of this needs ``outputs/``.
"""

import json

import numpy as np
import pytest

from spiking_ven import constants as C
from spiking_ven import metrics
from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.manifest import sha256_file
from spiking_ven.metrics import daf_responders, daf_response, daf_trials
from spiking_ven.stimuli import (
    build_stimuli,
    daf_window_ms,
    resolve_stimulus_settings,
    stimulus_record,
)

SR = 16000


@pytest.fixture
def encoder():
    K, Ch, L = 6, 8, 5
    V = np.random.default_rng(0).standard_normal((K, Ch, L))
    cp = {"n_channels": Ch, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    return CausalFilterEncoder(V, np.zeros(K), 0.1, cp)


@pytest.fixture
def motifs(tmp_path):
    n = SR // 2                                           # three 500 ms renditions
    audio = np.random.default_rng(1).standard_normal((3, n)).astype(np.float32) * 0.1
    path = tmp_path / "motifs.npz"
    np.savez(path, audio=audio, lengths=np.full(3, n), song_Ts=np.full(3, 500.0), sr=SR)
    return path


# ------------------------------------------------------------------ sample rate


def test_sr_defaults_to_the_corpus_and_a_mismatch_is_an_error(encoder, motifs):
    st = build_stimuli(encoder, motifs, n_hvc=4, t_post=50, verbose=False)
    assert st["sr"] == SR
    with pytest.raises(ValueError, match="does not match the corpus"):
        build_stimuli(encoder, motifs, sr=22050, n_hvc=4, verbose=False)


def test_build_stimuli_records_its_settings(encoder, motifs):
    st = build_stimuli(encoder, motifs, n_hvc=4, verbose=False)
    assert (st["t_post"], st["t_burn"], st["n_ista"]) == (C.T_POST, C.T_BURN, C.N_ISTA)
    assert st["mean_rate_hz"] == C.AUD_MEAN_RATE_HZ
    assert st["daf_window_s"] == tuple(C.DAF_WINDOW_S)
    assert st["hvc_on"].shape[0] == 4


# ------------------------------------------------------------------- DAF window


def test_daf_window_reaches_the_waveform(encoder, motifs):
    """The noise lands in the window build_stimuli is given, and nowhere else."""
    window = (0.10, 0.13)
    st = build_stimuli(encoder, motifs, n_hvc=4, t_post=50, daf_window_s=window,
                       verbose=False)
    assert st["daf_window_s"] == window
    changed = np.flatnonzero(st["sig_daf"] != st["sig_train"])
    assert changed.min() >= int(window[0] * SR)
    assert changed.max() < int(window[1] * SR)


def _trials(onset_ms, burst_ms, *, n_trials=20, n_e=20, T=600, n_driven=5, seed=0):
    """Poisson background, plus a rate increase over the burst in the first units."""
    rng = np.random.default_rng(seed)
    tr = (rng.random((n_trials, n_e, T)) < 0.003).astype(np.float32)
    burst = rng.random((n_trials, n_driven, burst_ms)) < 0.3
    tr[:, :n_driven, onset_ms:onset_ms + burst_ms] = np.maximum(
        tr[:, :n_driven, onset_ms:onset_ms + burst_ms], burst)
    return tr


def test_responders_are_found_at_the_onset_they_are_given():
    tr = _trials(250, 30)
    resp, _, _ = daf_responders(tr, onset_ms=250, window_ms=100)
    assert resp[:5].all()
    assert resp[5:].sum() <= 1
    # Looking at the wrong onset, the same units do not respond.
    resp_wrong, _, _ = daf_responders(tr, onset_ms=450, window_ms=100)
    assert not resp_wrong[:5].any()


def test_responder_windows_must_fit_in_the_rendition():
    tr = _trials(250, 30, T=300)
    with pytest.raises(ValueError, match="do not fit"):
        daf_responders(tr, onset_ms=250, window_ms=100)
    with pytest.raises(ValueError, match="do not fit"):
        daf_responders(tr, onset_ms=50, window_ms=100)


class _FakeVen:
    n_e = 20

    def transform(self, hvc, aud):
        return np.zeros((self.n_e, aud.shape[1]), dtype=np.float32)


def test_daf_trials_use_the_stimulus_window(monkeypatch):
    seen = []

    def fake_waveform(sig, sr, *, seed, window_s):
        seen.append(window_s)
        return sig

    monkeypatch.setattr(metrics, "daf_waveform", fake_waveform)
    monkeypatch.setattr(metrics, "encode_stimulus",
                        lambda sig, enc, **kw: np.zeros((3, kw["T_out_ms"]), np.float32))
    st = {"sig_train": np.zeros(100), "sr": SR, "T_rend": 50, "acts_train": None,
          "mean_rate_hz": 2.5, "n_ista": 1, "hvc_on": None, "daf_window_s": (0.2, 0.25)}
    out = daf_trials(_FakeVen(), None, st, n_trials=3)
    assert out.shape == (3, 20, 50)
    assert seen == [(0.2, 0.25)] * 3


def test_daf_response_takes_onset_and_burst_from_the_stimulus(monkeypatch):
    onset, burst = 250, 30
    monkeypatch.setattr(metrics, "daf_trials",
                        lambda ven, enc, st, *, n_trials, seed: _trials(onset, burst))
    st = {"daf_window_s": (onset / 1000, (onset + burst) / 1000)}
    r = daf_response(_FakeVen(), None, st, n_trials=20)
    assert r["burst_ms"] == pytest.approx(burst)
    assert r["n_responders"] >= 5
    assert r["extra_spikes_responders"] > 0


def test_daf_window_ms():
    assert daf_window_ms((0.4, 0.45)) == (400, pytest.approx(50.0))
    assert daf_window_ms([0.3, 0.32])[0] == 300


# ------------------------------------------------------------------- provenance

_ALL_NONE = {"sr": None, "seed": None, "t_post": None, "t_burn": None,
             "mean_rate_hz": None, "n_ista": None, "daf_window_s": None}


def test_an_empty_record_falls_back_to_the_constants(capsys):
    s = resolve_stimulus_settings({}, dict(_ALL_NONE))
    assert s == {"sr": None, "seed": None, "t_post": C.T_POST, "t_burn": C.T_BURN,
                 "mean_rate_hz": C.AUD_MEAN_RATE_HZ, "n_ista": C.N_ISTA,
                 "daf_window_s": tuple(C.DAF_WINDOW_S)}
    assert capsys.readouterr().err == ""


def test_the_record_is_used_and_an_explicit_flag_overrides_it_with_a_warning(capsys):
    rec = {"t_post": 100, "mean_rate_hz": 3.0, "seed": 7, "daf_window_s": [0.2, 0.25]}
    s = resolve_stimulus_settings(rec, dict(_ALL_NONE))
    assert (s["t_post"], s["mean_rate_hz"], s["seed"]) == (100, 3.0, 7)
    assert s["daf_window_s"] == (0.2, 0.25)
    assert capsys.readouterr().err == ""

    s = resolve_stimulus_settings(rec, {**_ALL_NONE, "mean_rate_hz": 2.5, "t_post": 100,
                                        "daf_window_s": [0.2, 0.25]})
    assert s["mean_rate_hz"] == 2.5
    err = capsys.readouterr().err
    assert "mean_rate_hz=2.5" in err and "mean_rate_hz=3.0" in err
    assert "t_post" not in err and "daf_window_s" not in err, "agreeing flags are silent"


def test_a_checksum_mismatch_warns(tmp_path, capsys):
    enc, mot = tmp_path / "enc.npz", tmp_path / "motifs.npz"
    enc.write_bytes(b"encoder")
    mot.write_bytes(b"motifs")
    rec = {"encoder_sha256": sha256_file(enc), "motifs_sha256": "0" * 64}
    resolve_stimulus_settings(rec, {}, encoder_path=enc, motifs_path=mot)
    err = capsys.readouterr().err
    assert str(mot) in err and "motifs" in err
    assert str(enc) not in err


def test_a_record_round_trips_through_json(tmp_path, capsys):
    enc = tmp_path / "enc.npz"
    enc.write_bytes(b"encoder")
    st = {"sr": SR, "seed": 3, "t_post": 120, "t_burn": 400, "n_hvc": 6,
          "mean_rate_hz": 2.5, "n_ista": 20, "daf_window_s": (0.3, 0.35)}
    rec = json.loads(json.dumps(stimulus_record(st, encoder_path=enc)))
    assert rec["encoder_sha256"] == sha256_file(enc) and "motifs_sha256" not in rec
    s = resolve_stimulus_settings(rec, dict(_ALL_NONE), encoder_path=enc)
    assert s == {k: st[k] for k in _ALL_NONE}
    assert capsys.readouterr().err == ""
