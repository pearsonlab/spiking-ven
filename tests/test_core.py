"""Core-surface tests: import isolation, and the numeric path on numpy + scipy alone.

Deliberately free of downloaded data and trained artifacts, so these run in CI on a
bare checkout. The reproduction assertions that need real song live in the slow suite.
"""

import subprocess
import sys

import numpy as np
import pytest

import spiking_ven as sv


def test_version_matches_package_metadata():
    """pyproject reads the version from __version__; v0.5.0 shipped with the two set by
    hand and disagreeing (0.4.0 vs 0.5.0), so pip and uv reported the wrong release."""
    from importlib.metadata import version

    assert sv.__version__ == version("spiking-ven")


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


def test_hvc_spikes_reject_a_too_short_window():
    """Bursts past the end of T were silently dropped, leaving the drive to stop."""
    with pytest.raises(ValueError, match="too short"):
        sv.generate_hvc_spikes(n_hvc=4, T=100, n_renditions=2, T_song=300,
                               T_burn=50, T_post=50)


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


def test_every_subpackage_is_importable():
    """Guard against a subpackage being missing from the distribution.

    `src/spiking_ven/data/` was once absent from the repo entirely: the .gitignore had an
    unanchored `data/` rule, intended for the raw song corpus, which also matched the
    subpackage. Imports and unit tests still passed, so only a CI run that actually
    invoked the data CLI caught it. This test would have caught it immediately.
    """
    import importlib

    for name in (
        "spiking_ven.cochleagram",
        "spiking_ven.common",
        "spiking_ven.evaluate",
        "spiking_ven.filterbank",
        "spiking_ven.manifest",
        "spiking_ven.causal_filters",
        "spiking_ven.paths",
        "spiking_ven.vocal_error_net",
        "spiking_ven.data",
        "spiking_ven.data.download",
        "spiking_ven.data.motifs",
        "spiking_ven.cli",
        "spiking_ven.cli.prep_data",
        "spiking_ven.cli.train_encoder",
        "spiking_ven.cli.train_ven",
        "spiking_ven.cli.evaluate",
        "spiking_ven.cli.figure",
    ):
        importlib.import_module(name)


def test_figures_subpackage_is_importable():
    """Same guard for `figures/`, which needs the `plots` extra so it cannot go above."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    import importlib

    for name in (
        "spiking_ven.figures",
        "spiking_ven.figures.cancellation",
        "spiking_ven.figures.daf_psth",
        "spiking_ven.figures.causal_filters",
        "spiking_ven.figures.unit_strfs",
    ):
        importlib.import_module(name)


def test_importing_the_figures_does_not_create_directories(tmp_path):
    """Both figure modules ran os.makedirs("outputs") at import, in the process CWD."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, spiking_ven.figures.cancellation, "
         "spiking_ven.figures.daf_psth; print(os.path.exists('outputs'))"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", "importing a figure module created outputs/"


def test_console_script_entry_points_resolve():
    """Every entry point declared in pyproject must actually be importable."""
    import importlib
    from importlib.metadata import entry_points

    eps = [e for e in entry_points(group="console_scripts")
           if e.value.startswith("spiking_ven")]
    assert eps, "no spiking_ven console scripts found"
    for ep in eps:
        module, _, func = ep.value.partition(":")
        mod = importlib.import_module(module)
        assert callable(getattr(mod, func)), f"{ep.name} -> {ep.value} is not callable"


def test_ltp_window_reproduces_the_legacy_rate_threshold():
    """LTP_WINDOW_MS replaced a 6 Hz rate, R_I_TH, whose trace threshold 6 * 20 * 1e-3
    was set to debias a tonic I rate that the network does not have. One I spike k ms
    earlier potentiates iff its trace alpha**k exceeds the threshold, so the window must
    admit the same whole-ms delays the rate did: 0-42 ms."""
    from spiking_ven.constants import LTP_WINDOW_MS, TAU_STDP

    alpha = np.exp(-1.0 / TAU_STDP)
    k = np.arange(100)
    legacy = alpha**k > 6.0 * TAU_STDP * 1e-3
    window = alpha**k > np.exp(-LTP_WINDOW_MS / TAU_STDP)
    np.testing.assert_array_equal(window, legacy)
    assert k[window].max() == 42


@pytest.mark.parametrize(("r_i_th", "expected"), [(6.0, 20.0 * np.log(1 / 0.12)),
                                                  (0.0, np.inf)])
def test_models_saved_with_a_rate_threshold_load_as_a_window(tmp_path, r_i_th, expected):
    """Old .npz files carry r_i_th instead of ltp_window_ms. They must reload with the
    trace threshold they were trained under -- 0 Hz meant no LTD, an infinite window."""
    from spiking_ven.vocal_error_net import VocalErrorNetV2

    ven = VocalErrorNetV2(n_e=8, n_i=4, n_hvc=3, n_aud=5, tau_stdp=20.0, seed=0)
    path = tmp_path / "ven.npz"
    ven.save(str(path))
    d = dict(np.load(path))
    del d["ltp_window_ms"]
    np.savez(path, **d, r_i_th=r_i_th)
    assert VocalErrorNetV2.load(str(path)).ltp_window_ms == pytest.approx(expected)
