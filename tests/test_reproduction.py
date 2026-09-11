"""Reproduction assertions for the published cancellation result.

These run against the artifacts in ``outputs/`` and are marked ``slow`` -- they re-run
VEN inference over several stimuli. Skipped when the artifacts are absent, so a bare
checkout still gets a green suite.

Populate the artifacts with ``make figure`` (about 10 minutes), then::

    make test-repro

Reference values are from the seeded 600-rendition run recorded in the README. The model
is deterministic (see test_determinism.py), so re-deriving metrics from a pinned model
file should land on the same numbers; the tolerances allow for platform differences in
floating-point reductions, not for run-to-run noise.
"""

from pathlib import Path

import pytest

from spiking_ven.corpus import load_corpus
from spiking_ven.evaluate import BIOLOGICAL_TARGETS, build_stimuli, daf_metrics
from spiking_ven.olshausen_field import OlshausenFieldEncoder
from spiking_ven.vocal_error_net import VocalErrorNetV2

OUT = Path("outputs")
MOTIFS = OUT / "motifs.npz"
ENCODER = OUT / "of_encoder.npz"
MODEL = OUT / "of_ven_model_k4max.npz"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (MOTIFS.exists() and ENCODER.exists() and MODEL.exists()),
        reason="reproduction artifacts missing - run `make figure` first",
    ),
]

SEED = 42

# From the seeded 600-rendition reference run.
REFERENCE = {
    "fwd_rev_corr": 0.0036,
    "k1": 7.78,
    "k2": 22.19,
    "k3": 2.85,
    "k4": 2.13,
}


@pytest.fixture(scope="module")
def metrics():
    encoder = OlshausenFieldEncoder.load(str(ENCODER))
    ven = VocalErrorNetV2.load(str(MODEL), seed=SEED)
    st = build_stimuli(encoder, MOTIFS, seed=SEED, verbose=False)
    m = daf_metrics(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                    aud_correct=st["aud_correct"], aud_daf=st["aud_daf"],
                    aud_reversed=st["aud_reversed"])
    m["fwd_rev_corr"] = st["fwd_rev_corr"]
    return m


def test_encoder_separates_forward_from_reversed(metrics):
    """Near-zero forward/reversed correlation is what makes K4 possible at all."""
    assert metrics["fwd_rev_corr"] == pytest.approx(REFERENCE["fwd_rev_corr"], abs=0.002)


def test_k1_matches_reference_and_biology(metrics):
    assert metrics["k1"] == pytest.approx(REFERENCE["k1"], abs=0.3)
    # Also inside the biological distribution: mean 7.7 Hz, SD 8.7 (Fig. 7C).
    mean, sd = BIOLOGICAL_TARGETS["k1_hz"]
    assert abs(metrics["k1"] - mean) < sd


def test_k2_matches_reference(metrics):
    assert metrics["k2"] == pytest.approx(REFERENCE["k2"], abs=1.5)


def test_k3_matches_reference(metrics):
    assert metrics["k3"] == pytest.approx(REFERENCE["k3"], abs=0.2)


def test_k4_matches_reference_and_exceeds_one(metrics):
    assert metrics["k4"] == pytest.approx(REFERENCE["k4"], abs=0.2)
    assert metrics["k4"] > BIOLOGICAL_TARGETS["k4_min"], "K4 must exceed 1x"


def test_cancellation_is_selective(metrics):
    """The headline claim, stated as an inequality rather than a number.

    The trained song must be suppressed well below both novel conditions. This is what
    the figure's excitatory row shows, and it is what a regression would break first.
    """
    assert metrics["k1"] < metrics["k2"] / 2, "white noise should exceed trained song ~2x"
    assert metrics["k1"] < metrics["k4_rate"] / 1.5, "reversed song should clearly exceed it"


def test_figure_excitatory_row_shows_cancellation():
    """E-population rates per figure column: ~6 / 18 / 21 / 17 Hz."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    from spiking_ven.figures.encoding_comparison import compute_encoding_columns

    encoder = OlshausenFieldEncoder.load(str(ENCODER))
    ven = VocalErrorNetV2.load(str(MODEL), seed=SEED)
    # Same template-selection rule as training, from one place.
    corpus = load_corpus(MOTIFS)
    sig_train, _ = corpus.template()

    data = compute_encoding_columns(ven, encoder, sig_train, corpus.T_song, seed=SEED)
    rates = [float(c["sE"].mean() * 1000) for c in data["columns"]]
    trained, others = rates[0], rates[1:]
    print(f"E-row rates: {[round(r, 1) for r in rates]}")

    # 7.8 Hz is what the shipped seeded model gives; the figure rounds it to 7 Hz.
    assert trained == pytest.approx(7.8, abs=1.0)
    for r in others:
        assert r > 2 * trained, f"novel column {r:.1f} Hz should far exceed {trained:.1f} Hz"
