"""Regression pins for the causal minimum-phase encoder + VEN operating point.

These are the numbers ``sven-train-ven`` produces with pure defaults against
``outputs/causal_encoder.npz``. They exist so that a change to the dynamics, the encoder
or the metrics cannot silently move the result: small-looking changes (a refractory
period, a rate calibration, a metric definition) can move these by 2x or more.

Marked ``slow`` -- each test re-runs VEN inference over several stimuli -- and skipped
when the artifacts are absent so a bare checkout still gets a green suite. Regenerate with

    sven-train-ven --encoder outputs/causal_encoder.npz \
                   --out outputs/ven_model.npz

The run is deterministic (test_determinism.py), but each tolerance is ~3 SD of that
metric across 10 evaluation seeds (42-51) of the shipped model. A change that only
redraws the simulation noise -- reordering calls, say -- therefore still passes, and a
failure means the model or the metric moved by more than noise.
"""

from pathlib import Path

import pytest

from spiking_ven.causal_filters import CausalFilterEncoder
from spiking_ven.constants import AUD_DELAY_MS, AUD_MEAN_RATE_HZ, HVC_DELAY_MS, N_HVC
from spiking_ven.metrics import daf_response
from spiking_ven.stimuli import build_stimuli
from spiking_ven.targets import BIOLOGICAL_TARGETS, PEAK_BIN_MS
from spiking_ven.vocal_error_net import VocalErrorNet

OUT = Path("outputs")
MOTIFS = OUT / "motifs.npz"
ENCODER = OUT / "causal_encoder.npz"
MODEL = OUT / "ven_model.npz"

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
    st = build_stimuli(enc, str(MOTIFS), sr=16000, seed=SEED, n_hvc=N_HVC, n_ista=50,
                       verbose=False)
    ven = VocalErrorNet.load(str(MODEL), seed=SEED)
    return enc, st, ven


@pytest.fixture(scope="module")
def daf(setup):
    enc, st, ven = setup
    return daf_response(ven, enc, st, seed=SEED)


def test_encoder_is_the_pruned_causal_bank(setup):
    enc, _, _ = setup
    assert enc.n_channels == 54          # 64 trained, silent ones pruned
    assert enc.L == 16
    assert enc.lam == pytest.approx(0.3)


def test_auditory_input_matches_schneider_woolley(setup):
    """The encoder's realised rate, against the broad-spiking population it models."""
    _, st, _ = setup
    rate = float(st["aud_correct"].mean() * 1000)
    assert rate == pytest.approx(2.16, abs=0.15)
    # The SD is larger than the mean, so only the mean is worth asserting.
    mean, sd = BIOLOGICAL_TARGETS["aud_song_rate_hz"]
    assert abs(rate - mean) < sd


def test_model_carries_the_operating_point(setup):
    _, _, ven = setup
    assert ven.t_refr_ms == pytest.approx(2.0)
    assert ven.beta_adapt_pa == pytest.approx(7540.0)
    assert (ven.e_l_mv, ven.v_th_mv) == (-70.0, -50.0)
    assert (ven.e_ampa_mv, ven.e_gaba_mv) == (0.0, -70.0)   # conductance synapses
    assert ven.tau_adapt == pytest.approx(10.0)
    assert ven.tau_ampa == pytest.approx(3.0)        # AMPA: aud->E, HVC->I, E->I
    assert ven.r_i_hz == pytest.approx(46.0)         # just above the I rate during song
    assert ven.tau_gaba == pytest.approx(15.0)       # GABA_A I->E
    assert (ven.aud_delay_ms, ven.hvc_delay_ms) == (AUD_DELAY_MS, HVC_DELAY_MS)
    assert AUD_MEAN_RATE_HZ == pytest.approx(2.5)
    # The peak is bin-width dependent, so the bin width is part of the metric's
    # definition. 2 ms is the only binning Mandelblat-Cerf state. Pinned so that changing
    # it has to be deliberate; the measured bin-width series lives in
    # BIOLOGICAL_TARGETS["responder_peak_change_hz"], which is where its caveats are.
    assert PEAK_BIN_MS == 2


def test_daf_response_matches_the_recorded_run(daf):
    # value at seed 42, tolerance ~3 SD over seeds 42-51 (SD in the comment)
    assert daf["responder_fraction"] == pytest.approx(0.548, abs=0.03)       # 0.005
    assert daf["baseline_hz_all"] == pytest.approx(2.660, abs=0.04)          # 0.011 Hz
    assert daf["peak_change_hz"] == pytest.approx(21.5, abs=1.5)             # 0.49 Hz
    assert daf["latency_ms"] == pytest.approx(25.1, abs=4.0)                 # 0.96 ms
    assert daf["duration_ms"] == pytest.approx(50.3, abs=6.5)                # 2.18 ms
    assert daf["extra_spikes_responders"] == pytest.approx(0.931, abs=0.03)  # 0.007


def test_the_response_magnitude_distribution_is_pinned(daf):
    """The spread, not just the mean -- it is what distinguishes a gain change from a
    truncation of the strongest responses."""
    assert daf["extra_spikes_p90"] == pytest.approx(2.06, abs=0.08)          # 0.027
    assert daf["extra_spikes_max"] == pytest.approx(9.72, abs=0.45)          # 0.13
    assert daf["extra_spikes_sd"] == pytest.approx(1.25, abs=0.03)           # 0.008


def test_biological_targets_that_are_met(daf):
    """Every biological target, at the paper's ~100 trials per neuron (DAF_N_TRIALS)."""
    lo, hi = BIOLOGICAL_TARGETS["singing_rate_hz"]
    assert lo <= daf["baseline_hz_all"] <= hi
    lat, lat_sd = BIOLOGICAL_TARGETS["response_latency_ms"]
    # 25.1 ms here, 22.9 +/- 1.0 over evaluation seeds 42-51; training seeds 43 and 44
    # give 21.4 and 26.4 ms. Duration 50 ms here, 51.2 +/- 2.2 (48.2-54.7) over evaluation
    # seeds, 54.0 and 49.3 ms on training seeds 43 and 44 -- the tightest margin, against 47.
    assert abs(daf["latency_ms"] - lat) <= lat_sd
    dur, dur_sd = BIOLOGICAL_TARGETS["response_duration_ms"]
    assert abs(daf["duration_ms"] - dur) <= dur_sd
    # 7/17 has a 95% CI of 21-64%, so the responder fraction only has to land inside that.
    assert 0.21 <= daf["responder_fraction"] <= 0.64
    lo, hi = BIOLOGICAL_TARGETS["extra_spikes_per_burst"]
    assert lo <= daf["extra_spikes_responders"] <= hi


def test_response_shape_matches_figure_8e(daf):
    """Fig. 8E, black trace (noise during singing, n = 7), read off the panel.

    Shape rather than magnitude. The peak's AMPLITUDE depends on a bin width the paper
    never states -- from 1 to 10 ms bins the model's peak falls 23 -> 18 Hz -- so it is
    deliberately not asserted anywhere. When the peak occurs depends less on it: 38, 38, 35
    and 30 ms at 1, 2, 5 and 10 ms bins, and 38 ms at 2 ms bins on every evaluation seed.
    Training seed 44's response sags mid-burst, which puts its 2 ms peak at 22 ms.

    Both bounds come from reading a published panel, not from a stated number, so they are
    wider than the figure appears to justify.
    """
    lo, hi = BIOLOGICAL_TARGETS["responder_peak_latency_ms"]
    assert lo <= daf["peak_latency_ms"] <= hi
    assert daf["late_over_peak"] < BIOLOGICAL_TARGETS["responder_late_over_peak_max"]
