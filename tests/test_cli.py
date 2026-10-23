"""The command-line layer: argument parsing, defaults, and dispatch.

Scope is deliberately one layer thick. These tests assert what the CLIs *parse* and
*which function they hand it to* -- never what that function then computes. The layer
below has its own tests, and duplicating them here would make the suite slow and would
not catch the kind of bug this layer actually has.

The defaults tests assert WIRING, not values: that each flag reads the shared constant
rather than restating a number. Restating it would put a second copy of the operating
point in the repository, which is the drift these CLIs exist to prevent.

What catches a WRONG value is not here but in test_reproduction_causal.py, which runs the
model and checks the DAF metrics against the bird. Change B_SCALE and that fails with a
number that means something, where an assertion here could only ever report that 16.0
still equals 16.0.
"""

import argparse

import pytest

from spiking_ven.cli import evaluate as cli_evaluate
from spiking_ven.cli import figure as cli_figure
from spiking_ven.cli import prep_data as cli_prep
from spiking_ven.cli import train_encoder as cli_train_encoder
from spiking_ven.cli import train_ven as cli_train_ven
from spiking_ven import constants as C
from spiking_ven.constants import (
    AUD_DELAY_MS,
    AUD_MEAN_RATE_HZ,
    DAF_WINDOW_S,
    HVC_DELAY_MS,
    SR,
)

PARSER_MODULES = [cli_train_encoder, cli_train_ven, cli_evaluate, cli_figure]
ALL_MODULES = PARSER_MODULES + [cli_prep]


# --------------------------------------------------------------------------- shape


@pytest.mark.parametrize("mod", ALL_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_every_cli_exposes_main_taking_argv(mod):
    """main(argv) is the contract. The console scripts in pyproject call main() with no
    arguments and argparse falls back to sys.argv; tests and `python -m` pass a list."""
    assert callable(mod.main)
    params = mod.main.__code__.co_varnames[:mod.main.__code__.co_argcount]
    assert params[0] == "argv"


@pytest.mark.parametrize("mod", PARSER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_build_parser_is_reusable_and_documented(mod):
    p = mod.build_parser()
    assert isinstance(p, argparse.ArgumentParser)
    assert p.description, "parser description comes from the module docstring"
    # Building twice must not mutate shared state.
    assert {a.dest for a in mod.build_parser()._actions} == {a.dest for a in p._actions}


@pytest.mark.parametrize("mod", PARSER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_parses_an_empty_argv(mod):
    """Every flag has a default; none is required."""
    assert mod.build_parser().parse_args([]) is not None


@pytest.mark.parametrize("mod", ALL_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_help_exits_zero(mod, capsys):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("mod", ALL_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_unknown_flag_exits_two(mod):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--definitely-not-a-flag"])
    assert exc.value.code == 2


# ------------------------------------------------------------------- shared defaults


@pytest.mark.parametrize("mod", PARSER_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_seed_default_is_shared(mod):
    assert mod.build_parser().parse_args([]).seed == 42


@pytest.mark.parametrize(
    "mod",
    [cli_train_ven, cli_evaluate, cli_figure],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_mean_rate_tracks_the_constant(mod):
    """A figure rendered at a different rate than its model was trained at encodes a
    different stimulus, so every consumer must default to the same constant."""
    assert mod.build_parser().parse_args([]).mean_rate_hz == AUD_MEAN_RATE_HZ


@pytest.mark.parametrize(
    ("mod", "path_dests"),
    [
        (cli_train_encoder, ("motifs", "out")),
        (cli_train_ven, ("encoder", "motifs", "out", "traces")),
        (cli_evaluate, ("model", "encoder", "motifs")),
        (cli_figure, ("model", "encoder", "motifs", "out_dir")),
    ],
    ids=lambda v: v.__name__.rsplit(".", 1)[-1] if hasattr(v, "__name__") else "",
)
def test_paths_default_to_none_so_the_paths_module_decides(mod, path_dests):
    """None means "ask paths.py", not "no value". Defaulting to a literal here would put
    a second copy of the output layout in every CLI.

    Listed per module rather than scanned, so a new flag has to be classified deliberately
    rather than swept in."""
    args = mod.build_parser().parse_args([])
    for dest in path_dests:
        assert getattr(args, dest) is None, dest


def test_encoder_flag_is_a_path_everywhere():
    """``--encoder`` is the trained encoder's path in every consumer; train_encoder,
    which produces it, has no such flag."""
    assert "encoder" not in vars(cli_train_encoder.build_parser().parse_args([]))
    for mod in (cli_train_ven, cli_evaluate, cli_figure):
        assert mod.build_parser().parse_args([]).encoder is None
        assert mod.build_parser().parse_args(["--encoder", "x.npz"]).encoder == "x.npz"


# ------------------------------------------------------------------ train_encoder


def test_train_encoder_causal_defaults_come_from_the_operating_point():
    a = cli_train_encoder.build_parser().parse_args([])
    assert a.lam == C.CAUSAL_LAMBDA
    assert a.mp_strength == C.CAUSAL_MP_STRENGTH
    assert a.lr == C.CAUSAL_LR
    assert a.l1 == C.CAUSAL_L1
    assert a.smooth == C.CAUSAL_SMOOTH
    assert a.epochs == C.CAUSAL_EPOCHS
    assert a.n_ista == C.CAUSAL_N_ISTA
    assert a.no_prune is False, "silent filters are pruned by default"


def test_train_encoder_front_end_is_sane():
    """The cochleagram front end is shared with nothing else, so its invariants are
    checked here rather than its literals."""
    a = cli_train_encoder.build_parser().parse_args([])
    assert a.n_bases > 0 and a.k_frames > 0
    assert a.lo_hz < a.hi_hz <= a.sr / 2, "cochleagram band inside Nyquist"
    assert a.sr == SR


# ---------------------------------------------------------------------- train_ven


@pytest.mark.parametrize(
    ("dest", "const"),
    [
        ("n_e", "N_E"), ("n_i", "N_I"), ("n_hvc", "N_HVC"), ("n_rend", "N_RENDITIONS"),
        ("b_scale", "B_SCALE"), ("c_b", "C_B"),
        ("b_hvc_scale", "B_HVC_SCALE"), ("c_hvc", "C_HVC"),
        ("drive_e", "DRIVE_E"), ("drive_i", "DRIVE_I"),
        ("tau_s", "TAU_S"), ("tau_e", "TAU_E"), ("tau_stdp", "TAU_STDP"),
        ("r_i_th", "R_I_TH"), ("j_max_ie", "J_MAX_IE"),
        ("t_post", "T_POST"), ("t_burn", "T_BURN"), ("n_ista", "N_ISTA"),
        ("a_sfa", "A_SFA"), ("tau_sfa", "TAU_SFA"),
    ],
)
def test_train_ven_defaults_come_from_the_operating_point(dest, const):
    assert getattr(cli_train_ven.build_parser().parse_args([]), dest) == getattr(C, const)


def test_train_ven_a_jie_is_derived_not_stored():
    """None means "compute it from the STDP window". The amplitude scales inversely with
    the width so the per-spike integral stays fixed, which is the reason the operating
    point is a module and not a data file -- a stored value would desynchronise the
    learning rule the first time the window changed."""
    assert cli_train_ven.build_parser().parse_args([]).a_jie is None
    assert C.a_jie_for(C.TAU_STDP) == pytest.approx(C.a_jie_for(2 * C.TAU_STDP) * 2)
    assert C.a_jie_for(0.0, tau_s=C.TAU_S) == C.a_jie_for(C.TAU_S)


def test_adaptation_timescale_is_shorter_than_the_burst():
    """A constraint, not a copy of the value: at 100 ms the adaptation current integrates
    over the motif and acts as a tonic gain change instead of truncating the response."""
    burst_ms = (DAF_WINDOW_S[1] - DAF_WINDOW_S[0]) * 1000.0
    assert C.TAU_SFA < burst_ms


def test_train_ven_delays_track_the_constants():
    """The delay is an input; the 23 ms response latency is what the model predicts, so
    the two must stay separate constants rather than one being set to the other."""
    a = cli_train_ven.build_parser().parse_args([])
    assert (a.aud_delay_ms, a.hvc_delay_ms) == (AUD_DELAY_MS, HVC_DELAY_MS)


# ------------------------------------------------------------------------ evaluate


def test_evaluate_shares_the_operating_point():
    a = cli_evaluate.build_parser().parse_args([])
    assert (a.n_hvc, a.n_ista) == (C.N_HVC, C.N_ISTA)
    assert (a.t_post, a.t_burn) == (C.T_POST, C.T_BURN)
    assert a.n_trials == 20, "trial count is an evaluation choice, not a model parameter"
    assert a.json is False
    assert cli_evaluate.build_parser().parse_args(["--json"]).json is True


def test_figure_shares_the_operating_point():
    assert cli_figure.build_parser().parse_args([]).t_post == C.T_POST


# -------------------------------------------------------------------------- figure


def test_figure_views():
    p = cli_figure.build_parser()
    assert p.parse_args([]).view == "raster"
    for view in ("raster", "psth", "filters", "units", "gammatone"):
        assert p.parse_args(["--view", view]).view == view
    with pytest.raises(SystemExit):
        p.parse_args(["--view", "psth-plot"])


def test_figure_type_coercion():
    a = cli_figure.build_parser().parse_args(
        ["--seed", "7", "--sr", "22050", "--t-post", "0", "--mean-rate-hz", "3.5"])
    assert (a.seed, a.sr, a.t_post) == (7, 22050, 0)
    assert isinstance(a.mean_rate_hz, float) and a.mean_rate_hz == 3.5


def test_figure_filters_view_renders_without_a_ven_model(tmp_path):
    """The filter figures describe the encoder alone, so they must render before any
    network exists: no model file is created here, and loading one would fail."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    import numpy as np

    from spiking_ven.causal_filters import CausalFilterEncoder

    K, C, L = 6, 8, 5
    V = np.random.default_rng(0).standard_normal((K, C, L))
    cp = {"n_channels": C, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    enc_path = tmp_path / "enc.npz"
    CausalFilterEncoder(V, np.zeros(K), 0.1, cp).save(enc_path)

    cli_figure.main(["--view", "filters", "--encoder", str(enc_path),
                     "--model", str(tmp_path / "absent.npz"), "--out-dir", str(tmp_path)])
    for name in ("causal_filters_kernels", "causal_filters_summary"):
        assert (tmp_path / f"{name}.png").exists(), name


def test_figure_units_view_renders_without_a_ven_model(tmp_path):
    """Kernels over multi-motif rasters: encoder and corpus only, no network."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    import numpy as np

    from spiking_ven.causal_filters import CausalFilterEncoder

    K, C, L = 6, 8, 5
    V = np.random.default_rng(0).standard_normal((K, C, L))
    cp = {"n_channels": C, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    enc_path = tmp_path / "enc.npz"
    CausalFilterEncoder(V, np.zeros(K), 0.1, cp).save(enc_path)
    sr, n = 16000, 1600                                  # 100 ms renditions
    audio = np.random.default_rng(1).standard_normal((3, n)).astype(np.float32)
    motifs = tmp_path / "motifs.npz"
    np.savez(motifs, audio=audio, lengths=np.full(3, n), song_Ts=np.full(3, 100.0), sr=sr)

    cli_figure.main(["--view", "units", "--units", "0", "3", "--encoder", str(enc_path),
                     "--motifs", str(motifs), "--model", str(tmp_path / "absent.npz"),
                     "--out-dir", str(tmp_path), "--t-post", "0"])
    assert (tmp_path / "unit_strfs.png").exists()


def test_figure_gammatone_view_renders_without_a_ven_model(tmp_path):
    """The front-end figure needs the encoder and the corpus, never a network."""
    pytest.importorskip("matplotlib", reason="requires the 'plots' extra")
    import numpy as np

    from spiking_ven.causal_filters import CausalFilterEncoder

    K, C, L = 6, 8, 5
    V = np.random.default_rng(0).standard_normal((K, C, L))
    cp = {"n_channels": C, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    enc_path = tmp_path / "enc.npz"
    CausalFilterEncoder(V, np.zeros(K), 0.1, cp).save(enc_path)
    n = SR // 5                                        # two 200 ms renditions of noise
    audio = np.random.default_rng(1).standard_normal((2, n)) * [[0.1], [0.2]]
    motifs_path = tmp_path / "motifs.npz"
    np.savez(motifs_path, audio=audio, lengths=np.array([n, n]),
             song_Ts=np.array([200.0, 200.0]), sr=SR)

    cli_figure.main(["--view", "gammatone", "--encoder", str(enc_path),
                     "--motifs", str(motifs_path),
                     "--model", str(tmp_path / "absent.npz"), "--out-dir", str(tmp_path)])
    for ext in ("png", "pdf"):
        assert (tmp_path / f"gammatone_filterbank.{ext}").exists(), ext
