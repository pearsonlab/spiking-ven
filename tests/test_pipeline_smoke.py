"""Hermetic end-to-end smoke test: corpus -> encoder -> network.

Uses a small synthetic corpus rather than the real song data, so this runs anywhere with
no network. That matters because the real corpus cannot be fetched in CI at all: the
dataset host sits behind CloudFront, which returns 403 to datacenter IP ranges regardless
of the User-Agent. The browser UA in ``spiking_ven.data.download`` is what makes the
download work from a workstation; nothing makes it work from a hosted runner.

So the coverage split is deliberate:
  * this test exercises the encoder and network CLIs on synthetic input, in CI
  * ``make test-repro`` exercises the real corpus and asserts the published numbers,
    on a machine that can actually download it
"""

import numpy as np
import pytest

from spiking_ven.cli import train_encoder, train_ven


def _synthetic_motifs(path, *, n_motifs=3, sr=16000, dur_ms=500, n_syl=6, seed=0):
    """Write a motifs.npz with the same keys and shapes as the real corpus.

    The audio is band-limited noise bursts rather than song -- enough structure for the
    sparse coder to find something, without pretending to be birdsong.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * dur_ms / 1000)
    audio = np.zeros((n_motifs, n), dtype=np.float32)
    t = np.arange(n) / sr
    for m in range(n_motifs):
        sig = np.zeros(n)
        for k in range(n_syl):
            f = 800 + 500 * k + 100 * m
            lo, hi = int(n * k / n_syl), int(n * (k + 0.7) / n_syl)
            env = np.hanning(hi - lo)
            sig[lo:hi] += env * np.sin(2 * np.pi * f * t[lo:hi])
        sig += 0.02 * rng.standard_normal(n)
        audio[m] = (sig / max(float(np.abs(sig).max()), 1e-9) * 0.1).astype(np.float32)

    edges = np.linspace(0, dur_ms, n_syl + 1)
    syl_on = np.tile(edges[:-1], (n_motifs, 1))
    syl_off = np.tile(edges[1:] * 0.9, (n_motifs, 1))
    np.savez_compressed(
        path,
        audio=audio,
        lengths=np.full(n_motifs, n, dtype=np.int32),
        sr=sr,
        syl_on=syl_on,
        syl_off=syl_off,
        song_Ts=np.full(n_motifs, float(dur_ms)),
        raw_song_Ts=np.full(n_motifs, float(dur_ms)),
    )
    return path


def test_encoder_then_network_end_to_end(tmp_path):
    motifs = _synthetic_motifs(tmp_path / "motifs.npz")
    encoder_out = tmp_path / "enc.npz"
    model_out = tmp_path / "ven.npz"

    train_encoder.main([
        "--motifs", str(motifs), "--out", str(encoder_out),
        "--n-bases", "8", "--n-epochs", "2", "--k-frames", "8",
        "--n-coch-ch", "20", "--seed", "42",
    ])
    assert encoder_out.exists(), "encoder artifact not written"

    train_ven.main([
        "--encoder", str(encoder_out), "--motifs", str(motifs),
        "--out", str(model_out), "--n-rend", "2",
        "--n-e", "30", "--n-i", "8", "--n-hvc", "6", "--seed", "42",
    ])
    assert model_out.exists(), "model artifact not written"
    assert (tmp_path / "ven_traces.npz").exists(), "traces not written alongside the model"

    tr = np.load(tmp_path / "ven_traces.npz")
    # Not asserting biology on synthetic audio -- only that the pipeline produced
    # finite, structurally sane numbers.
    for key in ("rc", "rn", "jie", "theta"):
        assert tr[key].shape == (2,), f"{key} should have one entry per rendition"
        assert np.isfinite(tr[key]).all(), f"{key} contains non-finite values"
    for key in ("k1", "rc_hvc"):
        if key in tr.files:
            assert np.isfinite(tr[key]).all()
    assert float(tr["jie"][-1]) > 0, "JIE should be positive after training"


def test_synthetic_corpus_matches_the_real_schema(tmp_path):
    """The fixture must keep the same keys as the real corpus, or it tests nothing."""
    motifs = _synthetic_motifs(tmp_path / "m.npz")
    d = np.load(motifs)
    assert set(d.files) == {
        "audio", "lengths", "sr", "syl_on", "syl_off", "song_Ts", "raw_song_Ts",
    }
    assert d["audio"].dtype == np.float32
    assert d["audio"].shape[0] == d["lengths"].shape[0] == d["song_Ts"].shape[0]
    assert np.isfinite(d["audio"]).all()
