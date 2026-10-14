"""The command-line layer: argument parsing, defaults, and dispatch.

Scope is deliberately one layer thick. These tests assert what the CLIs *parse* and
*which function they hand it to* -- never what that function then computes. The layer
below has its own tests, and duplicating them here would make the suite slow and would
not catch the kind of bug this layer actually has.

The defaults tests assert WIRING, not values: that each flag reads the shared constant
rather than restating a number. Restating the number here would be worse than redundant --
it would make the test tautological the moment the constant moved, and it would put a
second copy of the operating point in the repository, which is the exact failure these
CLIs exist to prevent. A sweep once built the network from library defaults rather than
the trained point, because the real parameters lived only in a scratch script, and four
grid points came back identical and broken before anyone noticed.

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
from spiking_ven import operating_point as op
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


def test_train_encoder_causal_defaults_come_from_the_operating_point():
    a = cli_train_encoder.build_parser().parse_args([])
    assert a.causal_lambda == op.CAUSAL_LAMBDA
    assert a.causal_mp_strength == op.CAUSAL_MP_STRENGTH
    assert a.causal_lr == op.CAUSAL_LR
    assert a.causal_l1 == op.CAUSAL_L1
    assert a.causal_smooth == op.CAUSAL_SMOOTH
    assert a.causal_lr_bias == op.CAUSAL_LR_BIAS
    assert a.causal_target_usage == op.CAUSAL_TARGET_USAGE
    assert a.causal_epochs == op.CAUSAL_EPOCHS
    assert a.causal_n_ista == op.CAUSAL_N_ISTA
    assert a.causal_no_prune is False, "silent filters are pruned by default"


def test_train_encoder_patch_path_still_parses():
    """The superseded Olshausen-Field path is not part of the operating point, so its
    defaults stay in the CLI. Only that they exist and are sane is checked here; asserting
    each literal would reintroduce the duplication this module removed."""
    a = cli_train_encoder.build_parser().parse_args([])
    assert a.n_bases > 0 and a.n_epochs > 0 and a.batch_size > 0
    assert 0.0 < a.lr < 1.0 and 0.0 < a.lambda_sparse < 1.0
    assert a.lo_hz < a.hi_hz <= a.sr / 2, "cochleagram band inside Nyquist"
    assert a.sr == SR


# ---------------------------------------------------------------------- train_ven


@pytest.mark.parametrize(
    ("dest", "const"),
    [
        ("n_e", "N_E"), ("n_i", "N_I"), ("n_hvc", "N_HVC"), ("n_rend", "N_RENDITIONS"),
        ("b_scale", "B_SCALE"), ("c_b", "C_B"),
        ("b_hvc_scale", "B_HVC_SCALE"), ("c_hvc", "C_HVC"),
        ("drive_e", "DRIVE_E"), ("drive_i", "DRIVE_I"), ("r_e_target", "R_E_TARGET"),
        ("tau_s", "TAU_S"), ("tau_e", "TAU_E"), ("tau_stdp", "TAU_STDP"),
        ("c_jie", "C_JIE"), ("r_i_th", "R_I_TH"), ("j_max_ie", "J_MAX_IE"),
        ("t_post", "T_POST"), ("t_burn", "T_BURN"), ("n_ista", "N_ISTA"),
        ("alpha_theta", "ALPHA_THETA"), ("a_sfa", "A_SFA"), ("tau_sfa", "TAU_SFA"),
    ],
)
def test_train_ven_defaults_come_from_the_operating_point(dest, const):
    assert getattr(cli_train_ven.build_parser().parse_args([]), dest) == getattr(op, const)


def test_train_ven_a_jie_is_derived_not_stored():
    """None means "compute it from the STDP window". The amplitude scales inversely with
    the width so the per-spike integral stays fixed, which is the reason the operating
    point is a module and not a data file -- a stored value would desynchronise the
    learning rule the first time the window changed."""
    assert cli_train_ven.build_parser().parse_args([]).a_jie is None
    assert op.a_jie_for(op.TAU_STDP) == pytest.approx(op.a_jie_for(2 * op.TAU_STDP) * 2)
    assert op.a_jie_for(0.0, tau_s=op.TAU_S) == op.a_jie_for(op.TAU_S)


def test_adaptation_timescale_is_shorter_than_the_burst():
    """A constraint, not a copy of the value: at 100 ms the adaptation current integrates
    over the motif and acts as a tonic gain change instead of truncating the response."""
    burst_ms = (DAF_WINDOW_S[1] - DAF_WINDOW_S[0]) * 1000.0
    assert op.TAU_SFA < burst_ms


def test_train_ven_delays_track_the_constants():
    """At 23 ms the conduction delay consumed the response latency the model is supposed
    to predict, which made the comparison circular."""
    a = cli_train_ven.build_parser().parse_args([])
    assert (a.aud_delay_ms, a.hvc_delay_ms) == (AUD_DELAY_MS, HVC_DELAY_MS)


def test_train_ven_takes_either_encoder_flag():
    p = cli_train_ven.build_parser()
    assert p.parse_args(["--causal-encoder", "e.npz"]).causal_encoder == "e.npz"
    assert p.parse_args(["--encoder", "of.npz"]).encoder == "of.npz"
    both = p.parse_args(["--encoder", "a", "--causal-encoder", "b"])
    assert (both.encoder, both.causal_encoder) == ("a", "b")


# ------------------------------------------------------------------------ evaluate


def test_evaluate_shares_the_operating_point():
    a = cli_evaluate.build_parser().parse_args([])
    assert (a.n_hvc, a.n_ista) == (op.N_HVC, op.N_ISTA)
    assert (a.t_post, a.t_burn, a.r_e_target) == (op.T_POST, op.T_BURN, op.R_E_TARGET)
    assert a.n_trials == 20, "trial count is an evaluation choice, not a model parameter"
    assert a.json is False
    assert cli_evaluate.build_parser().parse_args(["--json"]).json is True


def test_figure_shares_the_operating_point():
    assert cli_figure.build_parser().parse_args([]).t_post == op.T_POST


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
