"""Unit tests of :class:`VocalErrorNet`: persistence, units, defaults, delays, and STDP.

Small networks with hand-set weights and the noise off, so each test isolates one piece
of the dynamics in ``_simulate`` and runs in well under a second. Units: mV, pA, nS, ms;
weights are peak synaptic conductances (nS). The end-to-end check on the shipped model is
``test_reproduction_causal.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from spiking_ven import constants as C
from spiking_ven import generate_hvc_bursts
from spiking_ven.vocal_error_net import VocalErrorNet, stdp_audio_gate

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
# membrane
# ---------------------------------------------------------------------------

def test_tonic_drive_fires_e_at_its_target_rate():
    net = _small_net(noise_e_pa=0.0, a_sfa_pa=0.0, target_rate_hz=25.0, t_refr_ms=0.0)
    T = 1000
    sE = net.transform(np.zeros((6, T), np.float32), np.zeros((10, T), np.float32))
    isi = np.diff(np.flatnonzero(sE[0]))
    assert set(isi) <= {40, 41}, "fires every 1000 / 25 ms (one step later at equality)"


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trips_every_attribute(tmp_path):
    net = _small_net(tau_e=25.0, a_sfa_pa=500.0, aud_delay_ms=3, hvc_delay_ms=1,
                     r_i_th=4.0, v_th_mv=-52.0, c_e_pf=400.0, e_gaba_mv=-75.0)
    net.JIE = np.random.default_rng(1).random(net.JIE.shape).astype(np.float32)
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


def test_missing_required_keys_raise_naming_all_of_them(tmp_path):
    full, broken = tmp_path / "full.npz", tmp_path / "broken.npz"
    _small_net().save(str(full))
    _resave_without(full, broken, ["tau_e", "JEI"])
    with pytest.raises(ValueError) as exc:
        VocalErrorNet.load(str(broken))
    msg = str(exc.value)
    assert "tau_e" in msg and "JEI" in msg and str(broken) in msg


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_tau_stdp_must_be_positive(bad):
    with pytest.raises(ValueError, match="tau_stdp"):
        _small_net(tau_stdp=bad)
    with pytest.raises(ValueError, match="tau_stdp"):
        C.a_jie_for(bad)


def test_current_based_model_files_are_refused(tmp_path):
    """Earlier formats had current-based synapses, which this version cannot run."""
    path = tmp_path / "ven.npz"
    _small_net().save(str(path))
    d = dict(np.load(path))
    d["units"] = np.array("mV-pA")
    np.savez(tmp_path / "old.npz", **d)
    with pytest.raises(ValueError, match="current-based"):
        VocalErrorNet.load(str(tmp_path / "old.npz"))
    del d["units"]
    np.savez(tmp_path / "older.npz", **d)
    with pytest.raises(ValueError, match="current-based"):
        VocalErrorNet.load(str(tmp_path / "older.npz"))


def test_conductance_inhibition_is_shunting_at_rest():
    """With E_GABA at rest, I->E inhibition does nothing to a cell at rest and pulls a
    depolarised one back toward rest -- never below it."""
    from spiking_ven.vocal_error_net import _step
    c, g_l, e_l = 500.0, 500.0 / 30.0, -70.0
    at_rest = _step(np.array([-70.0]), g_l, c, e_l, [(np.array([1000.0]), -70.0)], 0.0)
    assert at_rest[0] == pytest.approx(-70.0)
    up = np.array([-55.0])
    free = _step(up, g_l, c, e_l, [], 0.0)
    inhibited = _step(up, g_l, c, e_l, [(np.array([1000.0]), -70.0)], 0.0)
    assert -70.0 <= inhibited[0] < free[0]


def test_exponential_euler_matches_the_current_based_step_without_synapses():
    from spiking_ven.vocal_error_net import _step
    c, tau, e_l, i_ext = 500.0, 30.0, -70.0, 300.0
    v = np.array([-62.0])
    d = np.exp(-1.0 / tau)
    expected = e_l + d * (v - e_l) + (1 - d) * i_ext / (c / tau)
    np.testing.assert_allclose(_step(v, c / tau, c, e_l, [], i_ext), expected)


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
        "tau_inh": C.TAU_INH, "tau_exc": C.TAU_EXC,
        "e_l_mv": C.E_L_MV, "v_th_mv": C.V_TH_MV, "v_reset_mv": C.V_RESET_MV,
        "c_e_pf": C.C_E_PF, "c_i_pf": C.C_I_PF,
        "e_ampa_mv": C.E_AMPA_MV, "e_gaba_mv": C.E_GABA_MV,
        "i_drive_i_pa": C.I_DRIVE_I_PA, "noise_e_pa": C.NOISE_E_PA,
        "noise_i_pa": C.NOISE_I_PA,
        "c_B": C.C_B, "b_aud_mean_ns": C.B_AUD_MEAN_NS, "b_aud_sd_ns": C.B_AUD_SD_NS,
        "c_hvc": C.C_HVC, "b_hvc_mean_ns": C.B_HVC_MEAN_NS, "b_hvc_sd_ns": C.B_HVC_SD_NS,
        "c_JEI": C.C_JEI, "j_ei_mean_ns": C.J_EI_MEAN_NS, "j_ei_sd_ns": C.J_EI_SD_NS,
        "j_max_ie_ns": C.J_MAX_IE_NS, "a_jie_ns": C.a_jie_for(C.TAU_STDP),
        "t_refr_ms": C.T_REFR_MS, "a_sfa_pa": C.A_SFA_PA, "tau_sfa": C.TAU_SFA,
        "r_i_th": C.R_I_TH, "aud_delay_ms": C.AUD_DELAY_MS, "hvc_delay_ms": C.HVC_DELAY_MS,
        "tau_stdp": C.TAU_STDP, "g_nmda_mean_ns": C.G_NMDA_MEAN_NS,
    }
    for k, v in expected.items():
        assert getattr(net, k) == v, k
    assert net.n_aud == 7


def test_weights_have_the_requested_mean_and_sd():
    net = VocalErrorNet(n_e=400, n_i=200, n_hvc=300, n_aud=200, seed=0)
    for W, mean, sd in ((net.B, C.B_AUD_MEAN_NS, C.B_AUD_SD_NS),
                        (net.B_hvc, C.B_HVC_MEAN_NS, C.B_HVC_SD_NS),
                        (net.JEI, C.J_EI_MEAN_NS, C.J_EI_SD_NS)):
        nz = W[W > 0]
        assert nz.mean() == pytest.approx(mean, rel=0.05)
        assert nz.std() == pytest.approx(sd, rel=0.1)


def test_default_a_jie_follows_the_stdp_window():
    assert VocalErrorNet(n_aud=3, n_e=4, n_i=2, tau_stdp=10.0).a_jie_ns == C.a_jie_for(10.0)


def test_n_aud_is_required():
    with pytest.raises(ValueError, match="n_aud"):
        VocalErrorNet()


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

#: Tonic E current that fires E every few ms (the "drive_e = 1" of the old units).
STRONG_DRIVE_PA = 7500.0


def _quiet_net(**kw) -> VocalErrorNet:
    """Noise off, no tonic E drive, no E->I weights: E fires only from audio."""
    net = _small_net(noise_e_pa=0.0, noise_i_pa=0.0, i_drive_e_pa=0.0,
                     aud_delay_ms=0, hvc_delay_ms=0, **kw)
    net.JEI = np.zeros_like(net.JEI)
    return net


def test_aud_delay_shifts_the_e_response_exactly():
    net = _quiet_net()
    net.B = np.zeros_like(net.B)
    net.B[:, 0] = np.linspace(100.0, 1000.0, net.n_e)    # a spread of conductances (nS)
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
    net.B[0, 0] = 500.0                       # suprathreshold from one spike (nS)
    net.B_hvc = np.zeros_like(net.B_hvc)
    net.B_hvc[0, 0] = 300.0
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
    net = _stdp_net(r_i_th=0.0, g_nmda_mean_ns=0.0)    # no slow NMDA tail keeping E up
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
    net.i_drive_e_pa = STRONG_DRIVE_PA
    net.JIE[:] = 100.0
    hvc, aud = _pulses(net, hvc_t=[20, 60, 100])
    sE, sI = net.transform_all(hvc, aud)
    assert sE.sum() > 0 and sI.sum() > 0
    net.fit(hvc, aud)
    assert np.all(net.JIE == np.float32(100.0))


def test_stdp_audio_gate_is_a_causal_window_of_tau_stdp():
    B = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    aud = np.zeros((2, 100))
    aud[0, 10] = 1.0
    aud[1, 50] = 1.0
    g = stdp_audio_gate(aud, B, 20.0)
    assert g.shape == (3, 100)
    np.testing.assert_array_equal(np.flatnonzero(g[0]), np.arange(10, 30))
    np.testing.assert_array_equal(np.flatnonzero(g[1]), np.arange(50, 70))
    assert not g[2].any(), "a cell with no auditory input is never gated"


def test_gate_closes_tau_stdp_after_the_last_input_spike():
    """E spikes driven by tonic drive long after the input must not move JIE."""
    net = _stdp_net(r_i_th=0.0)
    net.i_drive_e_pa = STRONG_DRIVE_PA
    net.JIE[:] = 100.0
    hvc, aud = _pulses(net, T=300, aud_t=[20], hvc_t=[100, 140, 180])
    sE, sI = net.transform_all(hvc, aud)
    assert sE[0, 100:].sum() > 0 and sI.sum() > 0, "premise: E and I both fire late"
    assert sI[:, :45].sum() == 0, "premise: I is silent while the gate is open"
    net.fit(hvc, aud)
    # I first fires at ~100 ms, long after the gate (20-40 ms) closed: no I trace was
    # ever read by a gated spike, and r_i_th = 0 means no depression either.
    assert np.all(net.JIE == np.float32(100.0))


def test_e_spike_without_i_activity_depresses_and_clips_at_zero():
    # xi_th = r_i_th * tau_stdp * 1e-3 = 50 * 20 * 1e-3 = 1.0; with I silent the trace is
    # 0, so every gated E spike (within tau_stdp of the input) lowers the row by
    # a_jie_ns * xi_th.
    net = _stdp_net(r_i_th=50.0)
    net.JIE[0, 0] = 50.0
    hvc, aud = _pulses(net, aud_t=[20])
    sE = net.fit(hvc, aud)
    n_spikes = sE[0, 20:20 + int(net.tau_stdp)].sum()
    assert n_spikes > 0
    xi_th = net.r_i_th * net.tau_stdp * 1e-3
    expected = 50.0 - n_spikes * net.a_jie_ns * xi_th
    assert expected > 0.0, "premise: the row is lowered, not clipped"
    assert net.JIE[0, 0] == pytest.approx(expected, rel=1e-5)
    assert np.all(net.JIE[0, 1:] == 0.0), "zero weights are clipped, never negative"

    net = _stdp_net(r_i_th=50.0, a_jie_ns=100.0)       # one spike more than erases it
    net.JIE[0, 0] = 50.0
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
    # The HVC weights are set for 600 HVC units; 20 need ~30x the conductance for the
    # same drive to I. The auditory ones for 54 channels: 40 carry 54/40 of it.
    net = VocalErrorNet(n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud, seed=0,
                        b_hvc_mean_ns=30 * C.B_HVC_MEAN_NS, b_hvc_sd_ns=30 * C.B_HVC_SD_NS,
                        b_aud_mean_ns=54 / 40 * C.B_AUD_MEAN_NS,
                        b_aud_sd_ns=54 / 40 * C.B_AUD_SD_NS)
    trained = generate_hvc_bursts(n_aud, T, 1, T_song, 0, T_post=T_post, seed=100)
    novel = trained[::-1].copy()
    hvc = generate_hvc_bursts(n_hvc, T, 1, T_song, 0, T_post=T_post, seed=0)

    def response(aud):
        silent = net.transform(hvc, np.zeros_like(aud)).sum()
        return net.transform(hvc, aud).sum() - silent

    before_trained, before_novel = response(trained), response(novel)
    assert before_trained == pytest.approx(before_novel, rel=0.1), "premise: matched drive"

    hvc_train = generate_hvc_bursts(n_hvc, n_rend * T, n_rend, T_song, 0, T_post=T_post,
                                    seed=0)
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


def _burst_response(g_nmda_mean_ns):
    net = VocalErrorNet(n_e=40, n_i=10, n_hvc=3, n_aud=4, seed=0, noise_e_pa=0.0,
                        a_sfa_pa=0.0, i_drive_e_pa=0.0, aud_delay_ms=0,
                        g_nmda_mean_ns=g_nmda_mean_ns,
                        b_aud_mean_ns=650.0, b_aud_sd_ns=325.0)
    aud = np.zeros((4, 400), dtype=np.float32)
    aud[:, 50:100:4] = 1.0                  # a 50 ms burst at 250 Hz per channel
    sE = net.transform(np.zeros((3, 400), dtype=np.float32), aud)
    return sE[:, 50:100].sum(), sE[:, 150:350].sum()


def test_nmda_sustains_firing_after_the_input_ends():
    during0, after0 = _burst_response(0.0)
    during, after = _burst_response(1200.0)
    assert after > after0 + 5, (after0, after)
    assert during >= during0


def test_nmda_parameters_round_trip(tmp_path):
    net = VocalErrorNet(n_e=8, n_i=4, n_hvc=3, n_aud=5, seed=0, g_nmda_mean_ns=12.0,
                        nmda_tau_decay=80.0)
    path = tmp_path / "nmda.npz"
    net.save(str(path))
    back = VocalErrorNet.load(str(path))
    assert (back.g_nmda_mean_ns, back.nmda_tau_decay) == (12.0, 80.0)
    w = back.to_brian_weights()
    assert w["g_nmda_mean_ns"] == 12.0 and w["nmda_tau_decay"] == 80.0
    np.testing.assert_array_equal(w["G_nmda"], back.nmda_conductances())


def test_nmda_conductances_are_b_rescaled_to_their_mean():
    net = VocalErrorNet(n_e=400, n_i=10, n_hvc=3, n_aud=200, seed=0, g_nmda_mean_ns=900.0)
    G = net.nmda_conductances()
    np.testing.assert_allclose(G, net.B * (900.0 / net.b_aud_mean_ns))
    assert G[G > 0].mean() == pytest.approx(900.0, rel=0.05)
    assert not VocalErrorNet(n_e=4, n_i=2, n_aud=3, g_nmda_mean_ns=0.0).nmda_conductances().any()



# ------------------------------------------------------------------- HVC prelude


def test_a_prelude_is_trimmed_and_changes_no_weights():
    net = VocalErrorNet(n_e=20, n_i=8, n_hvc=6, n_aud=5, seed=0)
    T = 120
    hvc = generate_hvc_bursts(n_hvc=6, T=T, n_renditions=1, T_song=100, T_burn=0,
                              T_post=20, seed=1)
    aud = (np.random.default_rng(2).random((5, T)) < 0.05).astype(np.float32)
    pre = generate_hvc_bursts(n_hvc=6, T=60, n_renditions=1, T_song=50, T_burn=0,
                              T_post=10, seed=3)
    sE, sI = net.transform_all(hvc, aud, hvc_prelude=pre)
    assert sE.shape == (20, T) and sI.shape == (8, T)
    # Learning needs auditory drive, so a prelude alone (no sound) leaves JIE untouched.
    net.fit(np.zeros((6, 0), dtype=np.float32), np.zeros((5, 0), dtype=np.float32),
            hvc_prelude=pre)
    assert not net.JIE.any()


def test_stdp_gate_is_saved():
    net = _stdp_net()
    assert net.stdp_gate == "window"


def test_gaba_decay_lengthens_the_inhibition_from_one_i_spike():
    """A tonically firing E cell is silenced longer by one I spike at tau_inh = 5 ms than
    by the one-step pulse of tau_inh = 0, at the same peak conductance."""
    def silence_after_i_spike(tau_inh):
        net = _quiet_net(tau_inh=tau_inh, tau_exc=0.0)
        net.i_drive_e_pa = STRONG_DRIVE_PA     # E fires tonically
        net.B_hvc = np.zeros_like(net.B_hvc)
        net.B_hvc[0, 0] = 300.0                # HVC 0 -> I0 only
        net.JIE = np.zeros_like(net.JIE)
        net.JIE[0, 0] = 1000.0                 # I0 -> E0 only (peak conductance, nS)
        hvc, aud = _pulses(net, T=150, hvc_t=[60])
        sE, sI = net.transform_all(hvc, aud)
        t_i = int(np.flatnonzero(sI[0])[0])
        assert sI[1:].sum() == 0 and sE[0, :t_i].sum() > 0, "premise: E fires before I"
        after = np.flatnonzero(sE[0, t_i:])
        return int(after[0]) if after.size else 150
    assert silence_after_i_spike(5.0) > silence_after_i_spike(0.0)


def test_recurrent_decay_zero_is_the_instantaneous_pulse():
    from spiking_ven.vocal_error_net import _decay
    assert _decay(0.0) == 0.0
    assert _decay(5.0) == pytest.approx(np.exp(-1 / 5.0))


def test_ei_delay_shifts_when_e_spikes_reach_i():
    """One E spike reaches I's AMPA trace 1 + ei_delay_ms steps later."""
    def first_i_spike(delay):
        net = _quiet_net(ei_delay_ms=delay, tau_exc=0.0)
        net.JEI = np.zeros_like(net.JEI)
        net.JEI[0, 0] = 400.0                  # E0 -> I0, strong enough to fire I0 alone
        net.B_hvc = np.zeros_like(net.B_hvc)
        net.B = np.zeros_like(net.B)
        net.B[0, 0] = 500.0                    # one auditory spike fires E0
        hvc, aud = _pulses(net, T=60, aud_t=[10])
        sE, sI = net.transform_all(hvc, aud)
        t_e = int(np.flatnonzero(sE[0])[0])
        return t_e, int(np.flatnonzero(sI[0])[0])
    t_e0, t_i0 = first_i_spike(0)
    t_e3, t_i3 = first_i_spike(3)
    assert t_e0 == t_e3
    assert t_i3 - t_i0 == 3

