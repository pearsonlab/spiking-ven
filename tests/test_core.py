"""Core-surface tests: import isolation, and the numeric path on numpy + scipy alone.

Deliberately free of downloaded data and trained artifacts, so these run in CI on a
bare checkout. The reproduction assertions that need real song live in the slow suite.
"""

import subprocess
import sys

import numpy as np
import pytest

import spiking_ven as sv


def test_core_import_does_not_pull_in_brian2():
    """`import spiking_ven` must never drag in Brian2 (it is an optional extra).

    Checked in a subprocess on purpose: an in-process ``sys.modules`` assertion is
    order-dependent, because any other test that imports Brian2 pollutes this one.
    """
    subprocess.run(
        [sys.executable, "-c", "import sys, spiking_ven; assert 'brian2' not in sys.modules"],
        check=True,
    )


def test_version_exposed():
    assert sv.__version__


@pytest.mark.parametrize("n_channels", [20, 40])
def test_cochleagram_shape_and_finite(n_channels):
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000) * 0.01
    gram = sv.cochleagram(sig, 16000, n_channels=n_channels, frame_rate=500)
    assert gram.shape[0] == n_channels
    assert np.isfinite(gram).all()
    assert (gram >= 0).all(), "cochleagram energy must be non-negative"


def test_centre_frequencies_are_ordered_and_in_band():
    freqs = sv.centre_frequencies(16000, 40, 200.0, 8000.0)
    assert freqs.shape == (40,)
    # ERB spacing returns descending centre frequencies; check monotonicity either way.
    assert np.all(np.diff(freqs) < 0) or np.all(np.diff(freqs) > 0)
    # Band edges are hit to within floating-point tolerance, not exactly.
    assert freqs.min() == pytest.approx(200.0, rel=1e-9)
    assert freqs.max() <= 8000.0 + 1e-6


def test_encoder_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    patches = rng.standard_normal((256, 40 * 4)).astype(np.float64)
    enc = sv.OlshausenFieldEncoder(n_bases=8, patch_len=patches.shape[1], seed=42)
    enc.fit(patches, n_epochs=2)
    path = tmp_path / "enc.npz"
    enc.save(str(path))
    enc2 = sv.OlshausenFieldEncoder.load(str(path))
    np.testing.assert_array_equal(enc.A, enc2.A)


def test_hvc_spikes_are_binary_and_seed_deterministic():
    # T must equal T_burn + n_renditions * (T_song + T_post).
    kw = {"n_hvc": 10, "n_renditions": 2, "T_song": 300, "T_burn": 100, "T_post": 50}
    T = kw["T_burn"] + kw["n_renditions"] * (kw["T_song"] + kw["T_post"])
    a = sv.generate_hvc_spikes(T=T, seed=42, **kw)
    b = sv.generate_hvc_spikes(T=T, seed=42, **kw)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (kw["n_hvc"], T)
    assert set(np.unique(a)).issubset({0, 1})
    assert a.sum() > 0, "HVC population should emit spikes"
