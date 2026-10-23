"""Unit tests of :class:`VocalErrorNet`: persistence, defaults, delays, and the STDP rule.

Small networks with hand-set weights and the noise off, so each test isolates one piece
of the dynamics in ``_simulate`` and runs in well under a second. The end-to-end check on
the shipped model is ``test_reproduction_causal.py``.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from spiking_ven import constants as C
from spiking_ven import generate_hvc_spikes
from spiking_ven.vocal_error_net import VocalErrorNet, VocalErrorNetV2

SHIPPED_MODEL = Path(__file__).resolve().parent.parent / "outputs" / "ven_model.npz"


def _small_net(**kw) -> VocalErrorNet:
    args = dict(n_e=30, n_i=8, n_hvc=6, n_aud=10, seed=0)
    args.update(kw)
    return VocalErrorNet(**args)


def _resave_without(src: Path, dst: Path, drop) -> None:
    """Copy a saved model, leaving out the keys in ``drop``."""
    d = dict(np.load(src))
    for k in drop:
        del d[k]
    np.savez(dst, **d)


def _state(net: VocalErrorNet) -> dict:
    """Every attribute except the noise generator, which is re-seeded on load."""
    return {k: v for k, v in vars(net).items() if k != "_sim_rng"}


def _assert_same_value(a, b, name):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), name
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)
    else:
        assert a == b, name


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trips_every_attribute(tmp_path):
    net = _small_net(tau_e=25.0, a_sfa=0.7, aud_delay_ms=3, hvc_delay_ms=1, r_i_th=4.0)
    net.JIE = np.random.default_rng(1).random(net.JIE.shape).astype(np.float32)
    net.theta_e = net.theta_e * np.linspace(0.9, 1.1, net.n_e)
    net.stimulus = {"n_ista": 50, "rate_scale": 1.25, "checksums": {"aud": "ab12"}}
    path = tmp_path / "ven.npz"
    net.save(str(path))
    back = VocalErrorNet.load(str(path))

    a, b = _state(net), _state(back)
    assert a.keys() == b.keys(), "save/load must cover every attribute the constructor sets"
    for k in a:
        _assert_same_value(a[k], b[k], k)

    wa, wb = net.to_brian_weights(), back.to_brian_weights()
    assert wa.keys() == wb.keys()
    for k in wa:
        _assert_same_value(wa[k], wb[k], k)


def test_stimulus_round_trips(tmp_path):
    net = _small_net()
    net.stimulus = {"t_post": 200, "nested": {"x": [1, 2, 3]}, "label": "song"}
    net.save(str(tmp_path / "ven.npz"))
    assert VocalErrorNet.load(str(tmp_path / "ven.npz")).stimulus == net.stimulus


def test_load_returns_the_calling_class(tmp_path):
    _small_net().save(str(tmp_path / "ven.npz"))
    assert type(VocalErrorNetV2.load(str(tmp_path / "ven.npz"))) is VocalErrorNetV2


def test_missing_required_keys_raise_naming_all_of_them(tmp_path):
    full, broken = tmp_path / "full.npz", tmp_path / "broken.npz"
    _small_net().save(str(full))
    _resave_without(full, broken, ["tau_e", "JEI"])
    with pytest.raises(ValueError) as exc:
        VocalErrorNet.load(str(broken))
    msg = str(exc.value)
    assert "tau_e" in msg and "JEI" in msg and str(broken) in msg


def test_missing_optional_keys_load_with_documented_defaults(tmp_path):
    # The documented defaults, pinned: each switches its feature off, so a model saved
    # before the feature existed reloads with the dynamics it was trained with.
    expected = {
        "t_refr_ms": 0.0, "a_sfa": 0.0, "tau_sfa": 0.0,
        "aud_delay_ms": 0, "hvc_delay_ms": 0, "tau_stdp": 0.0,
        "g_nmda": 0.0, "nmda_tau_rise": C.NMDA_TAU_RISE, "nmda_tau_decay": C.NMDA_TAU_DECAY,
        "nmda_alpha": C.NMDA_ALPHA, "mg_mm": C.MG_MM, "e_nmda_mv": C.E_NMDA_MV,
        "e_l_mv": C.E_L_MV, "v_th_mv": C.V_TH_MV, "stimulus_json": {},
    }
    assert VocalErrorNet.OPTIONAL_KEY_DEFAULTS == expected

    full, old = tmp_path / "full.npz", tmp_path / "old.npz"
    net = _small_net()
    net.stimulus = {"x": 1}
    net.save(str(full))
    _resave_without(full, old, expected)
    back = VocalErrorNet.load(str(old))
    for k, v in expected.items():
        attr = "stimulus" if k == "stimulus_json" else k
        assert getattr(back, attr) == v, k
        assert type(getattr(back, attr)) is type(v), k
    back.transform(np.zeros((6, 20)), np.zeros((10, 20)))   # still runs


def test_scalar_theta_e_is_broadcast(tmp_path):
    full, scalar = tmp_path / "full.npz", tmp_path / "scalar.npz"
    _small_net().save(str(full))
    d = dict(np.load(full))
    d["theta_e"] = np.array(0.05)
    np.savez(scalar, **d)
    theta = VocalErrorNet.load(str(scalar)).theta_e
    assert theta.shape == (30,) and np.all(theta == 0.05)


@pytest.mark.skipif(not SHIPPED_MODEL.exists(), reason="shipped model missing")
def test_shipped_model_was_trained_at_the_default_operating_point():
    """The constructor defaults are the network ``outputs/ven_model.npz`` was trained as."""
    shipped = VocalErrorNet.load(str(SHIPPED_MODEL))
    fresh = VocalErrorNet(n_aud=shipped.n_aud, seed=0)
    for k, v in _state(fresh).items():
        if k in ("B", "B_hvc", "JEI", "JIE", "stimulus"):
            continue                               # random draws, training, provenance
        _assert_same_value(v, getattr(shipped, k), k)


# ---------------------------------------------------------------------------
# noise streams
# ---------------------------------------------------------------------------

def test_transform_with_rng_leaves_the_network_stream_alone(tmp_path):
    rng = np.random.default_rng(0)
    hvc = (rng.random((6, 200)) < 0.05).astype(np.float32)
    aud = (rng.random((10, 200)) < 0.05).astype(np.float32)
    path = tmp_path / "ven.npz"
    _small_net().save(str(path))

    side = VocalErrorNet.load(str(path), seed=3)
    side_out = side.transform(hvc, aud, rng=np.random.default_rng(99))
    side.transform_all(hvc, aud, rng=np.random.default_rng(98))
    fresh = VocalErrorNet.load(str(path), seed=3)
    np.testing.assert_array_equal(side.transform(hvc, aud), fresh.transform(hvc, aud))
    # Premise: the passed generator is the one actually used.
    assert not np.array_equal(side_out, VocalErrorNet.load(str(path), seed=3).transform(hvc, aud))


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

def test_defaults_are_the_operating_point():
    net = VocalErrorNet(n_aud=7, seed=0)
    expected = {
        "n_e": C.N_E, "n_i": C.N_I, "n_hvc": C.N_HVC,
        "tau_e": C.TAU_E, "tau_i": C.TAU_I, "tau_s": C.TAU_S,
        "J_max_ie": C.J_MAX_IE, "c_B": C.C_B, "B_scale": C.B_SCALE,
        "c_hvc": C.C_HVC, "B_hvc_scale": C.B_HVC_SCALE, "c_JEI": C.C_JEI,
        "t_refr_ms": C.T_REFR_MS, "a_sfa": C.A_SFA, "tau_sfa": C.TAU_SFA,
        "r_i_th": C.R_I_TH, "drive_e": C.DRIVE_E, "drive_i": C.DRIVE_I,
        "noise_e": C.NOISE_E, "noise_i": C.NOISE_I,
        "target_rate": C.TARGET_RATE_E, "target_rate_i": C.TARGET_RATE_I,
        "aud_delay_ms": C.AUD_DELAY_MS, "hvc_delay_ms": C.HVC_DELAY_MS,
        "tau_stdp": C.TAU_STDP,
        "A_jie": C.a_jie_for(C.TAU_STDP, C.TAU_S),
    }
    for k, v in expected.items():
        assert getattr(net, k) == v, k
    assert net.n_aud == 7


def test_default_a_jie_follows_the_stdp_window():
    assert VocalErrorNet(n_aud=3, n_e=4, n_i=2, tau_stdp=10.0).A_jie == C.a_jie_for(10.0, C.TAU_S)


def test_n_aud_is_required():
    with pytest.raises(ValueError, match="n_aud"):
        VocalErrorNet()


def test_v2_legacy_defaults_are_pinned():
    """``VocalErrorNetV2``'s defaults must not change.

    External code (finchsim) builds networks from these exact defaults, so they are
    pinned here value by value rather than read from anywhere.
    """
    legacy = {
        "n_e": 600, "n_i": 150, "n_hvc": 15, "n_aud": 256,
        "tau_e": 30.0, "tau_i": 10.0, "tau_s": 10.0,
        "A_jie": 5e-6, "J_max_ie": 1.0, "c_B": 0.05, "B_scale": 5.0,
        "c_hvc": 0.3, "B_hvc_scale": 2.0, "c_JEI": 0.5,
        "t_refr_ms": 2.0, "a_sfa": 0.0, "tau_sfa": 100.0, "r_i_th": 0.0,
        "drive_e": 0.155, "drive_i": 0.058, "noise_e": 0.1, "noise_i": 0.0,
        "target_rate": 0.025, "target_rate_i": 0.05,
        "aud_delay_ms": 0, "hvc_delay_ms": 0, "tau_stdp": 0.0, "seed": None,
    }
    params = inspect.signature(VocalErrorNetV2.__init__).parameters
    assert {k: p.default for k, p in params.items() if k != "self"} == legacy
    net = VocalErrorNetV2(seed=0)
    for k, v in legacy.items():
        if k != "seed":
            assert getattr(net, k) == v, k


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def _quiet_net(**kw) -> VocalErrorNet:
    """Noise off, no tonic E drive, no recurrent weights: E fires only from audio."""
    net = _small_net(noise_e=0.0, noise_i=0.0, aud_delay_ms=0, hvc_delay_ms=0, **kw)
    net.drive_e = 0.0            # theta_e keeps its value from the tonic-drive calibration
    net.JEI = np.zeros_like(net.JEI)
    return net


def test_aud_delay_shifts_the_e_response_exactly():
    net = _quiet_net()
    net.B = np.zeros_like(net.B)
    net.B[:, 0] = np.linspace(0.3, 3.0, net.n_e)         # a spread of response strengths
    T, t0 = 120, 10
    aud = np.zeros((net.n_aud, T), dtype=np.float32)
    aud[0, t0] = 1.0
    hvc = np.zeros((net.n_hvc, T), dtype=np.float32)

    out = {}
    for d in (0, 3):
        net.aud_delay_ms = d
        out[d] = net.transform(hvc, aud)
    assert out[0].sum() > 0, "the pulse must evoke E spikes"
    assert np.flatnonzero(out[0].any(axis=0))[0] == t0
    assert not out[3][:, : t0 + 3].any()
    np.testing.assert_array_equal(out[3][:, 3:], out[0][:, :-3])


def _stdp_net(**kw) -> VocalErrorNet:
    """Two hand-wired pathways: auditory channel 0 -> E0 and HVC channel 0 -> I0."""
    net = _quiet_net(**kw)
    net.B = np.zeros_like(net.B)
    net.B[0, 0] = 5.0
    net.B_hvc = np.zeros_like(net.B_hvc)
    net.B_hvc[0, 0] = 20.0
    return net


def _pulses(net, T=300, aud_t=(), hvc_t=()):
    aud = np.zeros((net.n_aud, T), dtype=np.float32)
    hvc = np.zeros((net.n_hvc, T), dtype=np.float32)
    aud[0, list(aud_t)] = 1.0
    hvc[0, list(hvc_t)] = 1.0
    return hvc, aud


def test_i_before_gated_e_potentiates_only_that_pair():
    net = _stdp_net(r_i_th=0.0)                        # xi_th = 0: no depression term
    hvc, aud = _pulses(net, hvc_t=[20], aud_t=[25])
    sE = net.fit(hvc, aud)
    assert sE[0].sum() > 0 and sE[1:].sum() == 0
    assert net.JIE[0, 0] > 0.0
    mask = np.ones_like(net.JIE, dtype=bool)
    mask[0, 0] = False
    assert np.all(net.JIE[mask] == 0.0), "only the I0 -> E0 pair was active"


def test_e_long_before_i_does_not_potentiate():
    """The window is causal: an I burst after E has fallen silent changes nothing."""
    net = _stdp_net(r_i_th=0.0)
    hvc, aud = _pulses(net, aud_t=[20], hvc_t=[250])
    sE, sI = net.transform_all(hvc, aud)
    assert sE.sum() > 0 and sI.sum() > 0
    assert sE[:, 250:].sum() == 0, "premise: E is silent when I fires"
    net.fit(hvc, aud)
    assert np.all(net.JIE == 0.0)


@pytest.mark.parametrize("r_i_th", [0.0, 50.0])
def test_closed_audio_gate_blocks_all_plasticity(r_i_th):
    """E fires from tonic drive alone, with I active, but no audio: JIE must not move."""
    net = _stdp_net(r_i_th=r_i_th)
    net.drive_e = 1.0
    net.JIE[:] = 0.3
    hvc, aud = _pulses(net, hvc_t=[20, 60, 100])
    sE, sI = net.transform_all(hvc, aud)
    assert sE.sum() > 0 and sI.sum() > 0
    net.fit(hvc, aud)
    assert np.all(net.JIE == np.float32(0.3))


def test_e_spike_without_i_activity_depresses_and_clips_at_zero():
    # xi_th = r_i_th * tau_stdp * 1e-3 = 50 * 20 * 1e-3 = 1.0; with I silent the trace is
    # 0, so every gated E spike lowers the row by A_jie * xi_th.
    net = _stdp_net(r_i_th=50.0)
    net.JIE[0, 0] = 0.5
    hvc, aud = _pulses(net, aud_t=[20])
    sE = net.fit(hvc, aud)
    n_spikes = sE[0].sum()
    assert n_spikes > 0
    xi_th = net.r_i_th * net.tau_stdp * 1e-3
    assert net.JIE[0, 0] == pytest.approx(0.5 - n_spikes * net.A_jie * xi_th, rel=1e-5)
    assert np.all(net.JIE[0, 1:] == 0.0), "zero weights are clipped, never negative"

    net = _stdp_net(r_i_th=50.0, A_jie=1.0)            # one spike more than erases it
    net.JIE[0, 0] = 0.5
    net.fit(hvc, aud)
    assert net.JIE.min() == 0.0 and net.JIE[0, 0] == 0.0


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------

def test_training_cancels_the_trained_pattern_more_than_a_novel_one():
    """A small network trained on one (HVC, auditory) pairing learns to cancel it.

    The auditory pattern is temporally structured -- each channel bursts once, at its own
    time -- and the novel pattern is the same bursts with the channel order reversed, so
    both deliver the same drive and differ only in timing relative to HVC. The response
    is counted above the HVC-only baseline, so tonic firing does not dilute it.
    """
    n_e, n_i, n_hvc, n_aud = 120, 40, 20, 40
    T_song, T_post, n_rend = 300, 100, 40
    T = T_song + T_post
    net = VocalErrorNet(n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud, seed=0)
    trained = generate_hvc_spikes(n_aud, T, 1, T_song, 0, T_post=T_post, seed=100,
                                  peak_rate=200.0, kernel_width=8.0)
    novel = trained[::-1].copy()
    hvc = generate_hvc_spikes(n_hvc, T, 1, T_song, 0, T_post=T_post, seed=0,
                              peak_rate=300.0, kernel_width=10.0)

    def response(aud):
        silent = net.transform(hvc, np.zeros_like(aud)).sum()
        return net.transform(hvc, aud).sum() - silent

    before_trained, before_novel = response(trained), response(novel)
    assert before_trained == pytest.approx(before_novel, rel=0.1), "premise: matched drive"

    hvc_train = generate_hvc_spikes(n_hvc, n_rend * T, n_rend, T_song, 0, T_post=T_post,
                                    seed=1000, peak_rate=300.0, kernel_width=10.0)
    net.fit(hvc_train, np.tile(trained, (1, n_rend)))

    after_trained, after_novel = response(trained), response(novel)
    assert after_trained < 0.9 * after_novel, (after_trained, after_novel)
    assert after_trained < 0.7 * before_trained, (after_trained, before_trained)


# ------------------------------------------------------------------------------ NMDA


def test_nmda_gating_is_bounded_saturates_and_decays_with_tau_decay():
    from spiking_ven.vocal_error_net import nmda_gating

    aud = np.zeros((2, 1500), dtype=np.float32)
    aud[0, 100] = 1.0                       # one spike
    aud[1, 100:300] = 1.0                   # 1 kHz for 200 ms
    s = nmda_gating(aud, tau_rise=2.0, tau_decay=100.0, alpha=0.5)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[0, :100].max() == 0.0          # causal
    assert s[1, 299] > 0.95                 # saturated, far above a single spike's peak
    assert s[0].max() < 0.7
    t0 = 100 + 20                           # well past the 2 ms rise
    assert s[0, t0 + 100] / s[0, t0] == pytest.approx(np.exp(-1.0), rel=0.02)


def test_mg_block_follows_jahr_stevens():
    from spiking_ven.vocal_error_net import mg_block

    v = np.array([-70.0, -50.0, 0.0])
    np.testing.assert_allclose(mg_block(v, 1.0), 1.0 / (1.0 + np.exp(-0.062 * v) / 3.57))
    assert mg_block(-70.0) < mg_block(-50.0) < mg_block(0.0)
    assert mg_block(-70.0) < 0.06 and mg_block(0.0) > 0.75


def _burst_response(g_nmda):
    net = VocalErrorNet(n_e=40, n_i=10, n_hvc=3, n_aud=4, seed=0, noise_e=0.0,
                        a_sfa=0.0, drive_e=0.0, aud_delay_ms=0, g_nmda=g_nmda)
    net.theta_e = np.full(net.n_e, 0.03)
    aud = np.zeros((4, 400), dtype=np.float32)
    aud[:, 50:100:4] = 1.0                  # a 50 ms burst at 250 Hz per channel
    sE = net.transform(np.zeros((3, 400), dtype=np.float32), aud)
    return sE[:, 50:100].sum(), sE[:, 150:350].sum()


def test_nmda_sustains_firing_after_the_input_ends():
    during0, after0 = _burst_response(0.0)
    during, after = _burst_response(20.0)
    assert after > after0 + 5, (after0, after)
    assert during >= during0


def test_nmda_parameters_round_trip(tmp_path):
    net = VocalErrorNet(n_e=8, n_i=4, n_hvc=3, n_aud=5, seed=0, g_nmda=12.0,
                        nmda_tau_decay=80.0)
    path = tmp_path / "nmda.npz"
    net.save(str(path))
    back = VocalErrorNet.load(str(path))
    assert (back.g_nmda, back.nmda_tau_decay) == (12.0, 80.0)
    w = back.to_brian_weights()
    assert w["g_nmda"] == 12.0 and w["nmda_tau_decay"] == 80.0



# ------------------------------------------------------------------- HVC prelude


def test_a_prelude_is_trimmed_and_changes_no_weights():
    net = VocalErrorNet(n_e=20, n_i=8, n_hvc=6, n_aud=5, seed=0)
    T = 120
    hvc = generate_hvc_spikes(n_hvc=6, T=T, n_renditions=1, T_song=100, T_burn=0,
                              T_post=20, seed=1)
    aud = (np.random.default_rng(2).random((5, T)) < 0.05).astype(np.float32)
    pre = generate_hvc_spikes(n_hvc=6, T=60, n_renditions=1, T_song=50, T_burn=0,
                              T_post=10, seed=3)
    sE, sI = net.transform_all(hvc, aud, hvc_prelude=pre)
    assert sE.shape == (20, T) and sI.shape == (8, T)
    # Learning needs auditory drive, so a prelude alone (no sound) leaves JIE untouched.
    net.fit(np.zeros((6, 0), dtype=np.float32), np.zeros((5, 0), dtype=np.float32),
            hvc_prelude=pre)
    assert not net.JIE.any()
