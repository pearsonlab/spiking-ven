"""The command-line layer: argument parsing, defaults, and dispatch.

Scope is deliberately one layer thick. These tests assert what the CLIs *parse* and
*which function they hand it to* -- never what that function then computes. The layer
below has its own tests, and duplicating them here would make the suite slow and would
not catch the kind of bug this layer actually has.

Two of those bugs happened during development and motivate the defaults tests below:

- A sweep constructed the network with library defaults instead of the trained operating
  point, because the real parameters lived only in a scratch script. Four grid points came
  back identical and broken. The values are now CLI defaults, and pinned here.
- The figure entry point hardcoded the auditory rate at 15 Hz, so rendering a model
  trained at any other rate encoded a different stimulus than the metrics beside it.
  ``--mean-rate-hz`` exists for that reason and its default has to track the constant.
"""

import argparse

import pytest

from spiking_ven.cli import evaluate as cli_evaluate
from spiking_ven.cli import figure as cli_figure
from spiking_ven.cli import prep_data as cli_prep
from spiking_ven.cli import train_encoder as cli_train_encoder
from spiking_ven.cli import train_ven as cli_train_ven
from spiking_ven.constants import AUD_DELAY_MS, AUD_MEAN_RATE_HZ, HVC_DELAY_MS, SR

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
    [cli_train_encoder, cli_train_ven, cli_evaluate, cli_figure],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_mean_rate_tracks_the_constant(mod):
    """Hardcoding this is the bug --mean-rate-hz was added to fix: a figure rendered at a
    different rate than its model was trained at encodes a different stimulus."""
    assert mod.build_parser().parse_args([]).mean_rate_hz == AUD_MEAN_RATE_HZ


@pytest.mark.parametrize(
    ("mod", "path_dests"),
    [
        (cli_train_encoder, ("motifs", "out")),
        (cli_train_ven, ("encoder", "causal_encoder", "motifs", "out", "traces")),
        (cli_evaluate, ("model", "encoder", "motifs")),
        (cli_figure, ("model", "of_encoder", "causal_encoder", "motifs", "out_dir")),
    ],
    ids=lambda v: v.__name__.rsplit(".", 1)[-1] if hasattr(v, "__name__") else "",
)
def test_paths_default_to_none_so_the_paths_module_decides(mod, path_dests):
    """None means "ask paths.py", not "no value". Defaulting to a literal here would put
    a second copy of the output layout in every CLI.

    Listed per module rather than scanned, because ``--encoder`` is not a path everywhere:
    in train_encoder it selects the encoder TYPE (patch/causal) and defaults to "patch",
    while in train_ven and evaluate it is a path to an encoder file. Same flag name, two
    meanings. Pinned as-is here rather than reconciled -- renaming a flag is a public
    interface change."""
    args = mod.build_parser().parse_args([])
    for dest in path_dests:
        assert getattr(args, dest) is None, dest


def test_encoder_flag_means_different_things_in_different_clis():
    """Documents the collision above so a future reader meets it deliberately."""
    assert cli_train_encoder.build_parser().parse_args([]).encoder == "patch"
    assert cli_train_ven.build_parser().parse_args([]).encoder is None
    assert cli_evaluate.build_parser().parse_args([]).encoder is None


# ------------------------------------------------------------------ train_encoder


def test_train_encoder_choices_and_rejection():
    p = cli_train_encoder.build_parser()
    assert p.parse_args([]).encoder == "patch"
    assert p.parse_args(["--encoder", "causal"]).encoder == "causal"
    with pytest.raises(SystemExit):
        p.parse_args(["--encoder", "minimum-phase"])


def test_train_encoder_causal_operating_point():
    """The converged causal bank. lam sets both code sparsity and how many filters
    survive; mp_strength is a soft pull because the hard projection collapses kernels to
    impulses; lr_bias is 0 because usage equalisation strangled the gradient."""
    a = cli_train_encoder.build_parser().parse_args([])
    assert (a.causal_lambda, a.causal_mp_strength, a.causal_lr) == (0.3, 0.25, 10.0)
    assert (a.causal_l1, a.causal_smooth) == (7e-5, 1e-3)
    assert (a.causal_lr_bias, a.causal_epochs, a.causal_n_ista) == (0.0, 80, 20)
    assert a.causal_no_prune is False, "silent filters are pruned by default"


def test_train_encoder_patch_defaults_are_untouched():
    """The causal work must not have moved the Olshausen-Field path."""
    a = cli_train_encoder.build_parser().parse_args([])
    assert (a.n_bases, a.n_epochs, a.lr, a.lambda_sparse) == (64, 300, 0.01, 0.05)
    assert (a.target_usage, a.lr_bias, a.batch_size) == (0.10, 0.005, 128)
    assert (a.n_coch_ch, a.k_frames, a.frame_rate) == (40, 16, 500)
    assert (a.lo_hz, a.hi_hz, a.rms_ref, a.sr) == (200.0, 8000.0, 0.1, SR)


# ---------------------------------------------------------------------- train_ven


def test_train_ven_operating_point():
    """These were hardcoded in a scratch script. A sweep that used library defaults
    instead produced four identical broken grid points before anyone noticed."""
    a = cli_train_ven.build_parser().parse_args([])
    assert (a.n_e, a.n_i, a.n_hvc, a.n_rend) == (600, 150, 60, 600)
    assert (a.b_scale, a.c_b, a.b_hvc_scale, a.c_hvc) == (16.0, 0.05, 0.333, 0.1)
    assert (a.drive_e, a.drive_i, a.r_e_target) == (0.06, 0.058, 16.0)
    assert (a.tau_s, a.tau_e, a.tau_stdp) == (20.0, 30.0, 20.0)
    assert (a.c_jie, a.r_i_th, a.j_max_ie) == (0.1, 6.0, 1.0)
    assert (a.t_post, a.t_burn, a.n_ista) == (200, 500, 50)
    assert a.alpha_theta == 0.0, "threshold homeostasis is off at the shipped point"
    assert a.a_jie is None, "None means auto-scale from tau_stdp"


def test_train_ven_adaptation_defaults():
    """tau_sfa must stay shorter than the 50 ms DAF burst: at 100 ms the current
    integrates over the motif and acts as a tonic gain change instead."""
    a = cli_train_ven.build_parser().parse_args([])
    assert (a.a_sfa, a.tau_sfa) == (1.0, 10.0)
    assert a.tau_sfa < 50.0


def test_train_ven_delays_track_the_constants():
    """5/0, not 23/5. At 23 ms the conduction delay consumed the response latency the
    model is supposed to predict, which made the comparison circular."""
    a = cli_train_ven.build_parser().parse_args([])
    assert (a.aud_delay_ms, a.hvc_delay_ms) == (AUD_DELAY_MS, HVC_DELAY_MS)
    assert (a.aud_delay_ms, a.hvc_delay_ms) == (5, 0)


def test_train_ven_takes_either_encoder_flag():
    p = cli_train_ven.build_parser()
    assert p.parse_args(["--causal-encoder", "e.npz"]).causal_encoder == "e.npz"
    assert p.parse_args(["--encoder", "of.npz"]).encoder == "of.npz"
    both = p.parse_args(["--encoder", "a", "--causal-encoder", "b"])
    assert (both.encoder, both.causal_encoder) == ("a", "b")


# ------------------------------------------------------------------------ evaluate


def test_evaluate_defaults():
    a = cli_evaluate.build_parser().parse_args([])
    assert (a.n_trials, a.n_hvc, a.n_ista) == (20, 60, 50)
    assert (a.t_post, a.t_burn, a.r_e_target) == (200, 500, 16.0)
    assert a.json is False
    assert cli_evaluate.build_parser().parse_args(["--json"]).json is True


# -------------------------------------------------------------------------- figure


def test_figure_views():
    p = cli_figure.build_parser()
    assert p.parse_args([]).view == "raster"
    for view in ("raster", "rates", "filters", "psth"):
        assert p.parse_args(["--view", view]).view == view
    with pytest.raises(SystemExit):
        p.parse_args(["--view", "psth-plot"])


def test_figure_type_coercion():
    a = cli_figure.build_parser().parse_args(
        ["--seed", "7", "--sr", "22050", "--t-post", "0", "--mean-rate-hz", "3.5"])
    assert (a.seed, a.sr, a.t_post) == (7, 22050, 0)
    assert isinstance(a.mean_rate_hz, float) and a.mean_rate_hz == 3.5


# ------------------------------------------------------------------------ dispatch


def test_figure_filters_view_needs_no_ven_model(monkeypatch, tmp_path):
    """The filter diagnostics describe the encoder alone. Requiring a model would make
    them undrawable before stage 2 has ever run, so this branch must not load one."""
    pytest.importorskip("matplotlib")
    import spiking_ven.causal_filters as cf
    import spiking_ven.figures.causal_filters as figmod

    seen = {}

    def fake_load(path):
        seen["encoder"] = path
        return object()

    def fake_make(enc, out_dir, *, tag="", sr=16000, patch_encoder=None):
        seen["out_dir"], seen["sr"] = out_dir, sr
        return [tmp_path / "a.png"]

    monkeypatch.setattr(cf.CausalFilterEncoder, "load", staticmethod(fake_load))
    monkeypatch.setattr(figmod, "make_filter_figures", fake_make)
    monkeypatch.setattr(cli_figure, "VocalErrorNetV2", _Exploding("VocalErrorNetV2"))
    monkeypatch.setattr(cli_figure, "load_corpus", _exploding("load_corpus"))

    cli_figure.main(["--view", "filters", "--causal-encoder", "enc.npz",
                     "--out-dir", str(tmp_path)])
    assert seen["encoder"] == "enc.npz"
    assert seen["sr"] == SR


def _exploding(name):
    def _fail(*a, **k):
        raise AssertionError(f"{name} must not be called on this path")
    return _fail


class _Exploding:
    def __init__(self, name):
        self._name = name

    def load(self, *a, **k):
        raise AssertionError(f"{self._name}.load must not be called on this path")
