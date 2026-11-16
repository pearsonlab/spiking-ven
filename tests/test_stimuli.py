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
    return CausalFilterEncoder(V, 0.1, cp)


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

    def transform(self, hvc, aud, hvc_prelude=None):
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
             "mean_rate_hz": None, "n_ista": None, "daf_window_s": None,
             "hvc_intro": None, "hvc_model": None}


def test_an_empty_record_falls_back_to_the_constants(capsys):
    s = resolve_stimulus_settings({}, dict(_ALL_NONE))
    assert s == {"sr": None, "seed": None, "t_post": C.T_POST, "t_burn": C.T_BURN,
                 "mean_rate_hz": C.AUD_MEAN_RATE_HZ, "n_ista": C.N_ISTA,
                 "daf_window_s": tuple(C.DAF_WINDOW_S), "hvc_intro": tuple(C.HVC_INTRO),
                 "hvc_model": C.HVC_MODEL}
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
          "mean_rate_hz": 2.5, "n_ista": 20, "daf_window_s": (0.3, 0.35),
          "hvc_intro": (2, 60, 30), "hvc_model": "burst"}
    rec = json.loads(json.dumps(stimulus_record(st, encoder_path=enc)))
    assert rec["encoder_sha256"] == sha256_file(enc) and "motifs_sha256" not in rec
    s = resolve_stimulus_settings(rec, dict(_ALL_NONE), encoder_path=enc)
    assert s == {k: st[k] for k in _ALL_NONE}
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------- HVC intro notes


def test_intro_notes_repeat_one_sparse_sequence_at_the_motif_density():
    from spiking_ven.common import generate_hvc_intro

    pre = generate_hvc_intro(60, 840, n_notes=2, note_ms=84, gap_ms=20, peak_rate=300.0,
                             kernel_width=10.0, seed=1)
    assert pre.shape == (60, 2 * 104)
    active = np.flatnonzero(pre.any(axis=1))
    assert len(active) == 6                       # 60 * 84 / 840 neurons, as in the motif
    np.testing.assert_array_equal(pre[:, :104], pre[:, 104:])      # the same note twice
    assert not pre[:, 84:104].any() and not pre[:, 188:].any()     # silent gaps
    assert generate_hvc_intro(60, 840, 0, 80, 20).shape == (60, 0)


def test_a_record_without_intro_notes_means_none_and_no_record_means_the_default():
    rec = {"t_post": 200}                         # recorded, but before intro notes existed
    assert resolve_stimulus_settings(rec, {"hvc_intro": None})["hvc_intro"] == (0, 0, 0)
    assert resolve_stimulus_settings({}, {"hvc_intro": None})["hvc_intro"] == tuple(
        C.HVC_INTRO)


def test_build_stimuli_puts_the_intro_notes_before_the_motif(encoder, motifs):
    st = build_stimuli(encoder, motifs, hvc_intro=(2, 40, 10), t_burn=300, verbose=False)
    pre = st["hvc_prelude"]
    assert pre.shape[1] == 100 and pre.any()
    np.testing.assert_array_equal(st["hvc_burn"][:, 200:300], pre)  # ends the burn-in
    assert not st["hvc_burn"][:, :200].any()
    none = build_stimuli(encoder, motifs, hvc_intro=(0, 0, 0), verbose=False)
    assert none["hvc_prelude"].shape[1] == 0


# ------------------------------------------------------------- HVC_RA burst generator


def test_bursts_have_finchsim_hvcra_shape_and_repeat_every_rendition():
    """One burst per neuron per motif: 4 spikes 2 ms apart (500 Hz, 6 ms long), onset
    in the neuron's own slot of the motif, identical across renditions."""
    from spiking_ven.common import generate_hvc_bursts

    n, T_song, T_post, T_burn, n_rend = 60, 840, 160, 100, 3
    R = T_song + T_post
    T = T_burn + n_rend * R
    s = generate_hvc_bursts(n, T, n_rend, T_song, T_burn, T_post=T_post, seed=0)
    assert s.shape == (n, T) and set(np.unique(s)) <= {0.0, 1.0}
    assert (s.sum(axis=1) == 4 * n_rend).all()
    assert not s[:, :T_burn].any()
    slot = T_song / n
    for i in range(n):
        t = np.flatnonzero(s[i, T_burn:T_burn + R])
        np.testing.assert_array_equal(np.diff(t), [2, 2, 2])
        assert np.floor(i * slot) <= t[0] < np.ceil((i + 1) * slot)
    one = s[:, T_burn:T_burn + R]
    for r in range(1, n_rend):
        np.testing.assert_array_equal(s[:, T_burn + r * R:T_burn + (r + 1) * R], one)
    np.testing.assert_array_equal(
        s, generate_hvc_bursts(n, T, n_rend, T_song, T_burn, T_post=T_post, seed=0))
    assert not np.array_equal(
        s, generate_hvc_bursts(n, T, n_rend, T_song, T_burn, T_post=T_post, seed=1))
    with pytest.raises(ValueError, match="too short"):
        generate_hvc_bursts(n, T - 1, n_rend, T_song, T_burn, T_post=T_post)


def test_hvc_drive_dispatches_and_rejects_unknown_models():
    from spiking_ven.common import generate_hvc_spikes, hvc_drive

    kw = dict(n_hvc=8, T=300, n_renditions=1, T_song=200, T_burn=0, T_post=100, seed=3,
              peak_rate=300.0, kernel_width=10.0, spikes_per_burst=3, intra_burst_hz=400.0)
    np.testing.assert_array_equal(
        hvc_drive("gaussian", **kw),
        generate_hvc_spikes(8, 300, 1, 200, 0, T_post=100, peak_rate=300.0,
                            kernel_width=10.0, seed=3))
    assert (hvc_drive("burst", **kw).sum(axis=1) == 3).all()
    with pytest.raises(ValueError, match="unknown HVC model"):
        hvc_drive("hvcx", **kw)


def test_intro_notes_use_the_chosen_generator():
    from spiking_ven.common import generate_hvc_intro

    pre = generate_hvc_intro(60, 840, n_notes=2, note_ms=84, gap_ms=20, seed=1,
                             model="burst", spikes_per_burst=4, intra_burst_hz=500.0)
    active = np.flatnonzero(pre.any(axis=1))
    assert len(active) == 6
    assert (pre[active, :104].sum(axis=1) == 4).all()
    np.testing.assert_array_equal(pre[:, :104], pre[:, 104:])


def test_a_record_without_a_generator_means_gaussian():
    assert resolve_stimulus_settings({"seed": 42},
                                     {"hvc_model": None})["hvc_model"] == "gaussian"
    assert resolve_stimulus_settings({}, {"hvc_model": None})["hvc_model"] == C.HVC_MODEL
    assert resolve_stimulus_settings({"hvc_model": "burst"},
                                     {"hvc_model": None})["hvc_model"] == "burst"


def test_build_stimuli_records_and_uses_the_burst_generator(encoder, motifs):
    st = build_stimuli(encoder, motifs, hvc_model="burst", verbose=False)
    assert st["hvc_model"] == "burst"
    assert (st["hvc_on"].sum(axis=1) == C.HVC_SPIKES_PER_BURST).all()
    assert stimulus_record(st)["hvc_model"] == "burst"
