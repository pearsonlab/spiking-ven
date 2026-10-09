"""Regression pins for the causal minimum-phase encoder + VEN operating point.

These are the numbers ``sven-train-ven`` produces with pure defaults against
``outputs/causal_encoder.npz``. They exist so that a change to the dynamics, the encoder
or the metrics cannot silently move the result: several times during development a
"harmless" change (the refractory period, the auditory rate calibration, a metric
definition) moved these by 2x or more, and only a recorded reference makes that visible.

Marked ``slow`` -- each test re-runs VEN inference over several stimuli -- and skipped
when the artifacts are absent so a bare checkout still gets a green suite. Regenerate with

    sven-train-ven --causal-encoder outputs/causal_encoder.npz \
                   --out outputs/of_ven_model_default.npz

The model is deterministic (test_determinism.py), so these are tight: the tolerances
allow for platform differences in floating-point reductions, not run-to-run noise.
"""

from pathlib import Path

import pytest

from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.constants import AUD_MEAN_RATE_HZ
from spiking_ven.evaluate import BIOLOGICAL_TARGETS, build_stimuli, daf_response
from spiking_ven.vocal_error_net import VocalErrorNetV2

OUT = Path("outputs")
MOTIFS = OUT / "motifs.npz"
ENCODER = OUT / "causal_encoder.npz"
MODEL = OUT / "of_ven_model_default.npz"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MOTIFS.exists() and ENCODER.exists() and MODEL.exists()),
        reason="causal reproduction artifacts missing - run sven-train-ven first",
    ),
]

SEED = 42


@pytest.fixture(scope="module")
def setup():
    enc = CausalFilterEncoder.load(str(ENCODER))
    st = build_stimuli(enc, str(MOTIFS), sr=16000, seed=SEED, n_hvc=60, n_ista=50,
                       verbose=False)
    ven = VocalErrorNetV2.load(str(MODEL), seed=SEED)
    return enc, st, ven


@pytest.fixture(scope="module")
def daf(setup):
    enc, st, ven = setup
    return daf_response(ven, enc, sig_train=st["sig_train"], hvc_on=st["hvc_on"],
                        acts_train=st["acts_train"], sr=16000, T_out_ms=st["T_rend"],
                        n_ista=50, n_trials=20, seed=SEED)


def test_encoder_is_the_pruned_causal_bank(setup):
    enc, _, _ = setup
    assert enc.n_channels == 56          # 64 trained, silent ones pruned
    assert enc.L == 16
    assert enc.lam == pytest.approx(0.3)


def test_auditory_input_matches_schneider_woolley(setup):
    """The encoder's realised rate, against the broad-spiking population it models."""
    _, st, _ = setup
    rate = float(st["aud_correct"].mean() * 1000)
    assert rate == pytest.approx(2.15, abs=0.15)
    # 2.4 +/- 2.7 Hz; the SD is larger than the mean, so only the mean is worth asserting.
    assert abs(rate - 2.4) < 2.7


def test_model_carries_the_operating_point(setup):
    _, _, ven = setup
    assert ven.t_refr_ms == pytest.approx(2.0)
    assert ven.a_sfa == pytest.approx(0.3)
    assert ven.tau_sfa == pytest.approx(10.0)
    assert ven.aud_delay_ms == 5 and ven.hvc_delay_ms == 0
    assert AUD_MEAN_RATE_HZ == pytest.approx(2.5)


def test_daf_response_matches_the_recorded_run(daf):
    assert daf["responder_fraction"] == pytest.approx(0.537, abs=0.03)
    assert daf["baseline_hz_all"] == pytest.approx(3.20, rel=0.08)
    assert daf["peak_change_hz"] == pytest.approx(96.9, rel=0.10)
    assert daf["latency_ms"] == pytest.approx(29.3, abs=2.5)
    assert daf["duration_ms"] == pytest.approx(55.0, abs=4.0)
    assert daf["extra_spikes_responders"] == pytest.approx(3.34, rel=0.10)


def test_the_response_magnitude_distribution_is_pinned(daf):
    """The spread, not just the mean -- it is what distinguishes a gain change from a
    truncation, and the mean alone hid that distinction across two sweeps."""
    assert daf["extra_spikes_p90"] == pytest.approx(8.10, rel=0.12)
    assert daf["extra_spikes_max"] == pytest.approx(17.3, rel=0.15)
    assert daf["extra_spikes_sd"] == pytest.approx(3.11, rel=0.12)


def test_biological_targets_that_are_met(daf):
    """Only the ones that actually pass. Response magnitude is still ~2x over the bird's
    upper bound and deliberately has no assertion here."""
    lo, hi = BIOLOGICAL_TARGETS["singing_rate_hz"]
    assert lo <= daf["baseline_hz_all"] <= hi
    lat, lat_sd = BIOLOGICAL_TARGETS["response_latency_ms"]
    assert abs(daf["latency_ms"] - lat) <= lat_sd
    dur, dur_sd = BIOLOGICAL_TARGETS["response_duration_ms"]
    assert abs(daf["duration_ms"] - dur) <= dur_sd
    # 7/17 has a 95% CI of 21-64%, so the responder fraction only has to land inside that.
    assert 0.21 <= daf["responder_fraction"] <= 0.64


def test_song_is_cancelled_more_than_the_reversed_motif(setup):
    """Time-local, in 50 ms windows. A motif mean dilutes a signal that is local in time
    and understates the contrast -- it read 1.48 where the peak-window ratio was 1.98."""
    import numpy as np

    _, st, ven = setup
    T = st["T_song"]
    W = 50
    def windows(aud):
        sE = ven.transform(st["hvc_on"], aud)[:, :T]
        return np.array([sE[:, i:i + W].mean() * 1000 for i in range(0, T - W, W)])
    song, rev = windows(st["aud_correct"]), windows(st["aud_reversed"])
    assert song.max() < rev.max()
    assert np.percentile(rev, 90) / np.percentile(song, 90) > 1.3
