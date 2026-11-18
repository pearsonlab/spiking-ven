"""Unit tests of :class:`VocalErrorNet`: persistence, units, defaults, delays, and STDP.

Small networks with hand-set weights and the noise off, so each test isolates one piece
of the dynamics (the Brian2 equations in :mod:`spiking_ven.network`). Units: mV, pA, nS,
ms; weights are peak synaptic conductances (nS). Tests run at dt = 1 ms unless they say
otherwise. The end-to-end check on the shipped model is ``test_reproduction_causal.py``.

Brian2 timing: an input spike in bin ``t`` with conduction delay ``d`` is delivered at
the end of step ``t + d``, so it first moves the membrane in the step after.
"""

from pathlib import Path

import numpy as np
import pytest

from spiking_ven import constants as C
from spiking_ven import generate_hvc_bursts
from spiking_ven.vocal_error_net import MODEL_FORMAT, VocalErrorNet

SHIPPED_MODEL = Path(__file__).resolve().parent.parent / "outputs" / "ven_model.npz"


def _small_net(**kw) -> VocalErrorNet:
    args = dict(n_e=30, n_i=8, n_hvc=6, n_aud=10, seed=0, dt_ms=1.0)
    args.update(kw)
    return VocalErrorNet(**args)


def _resave_without(src: Path, dst: Path, drop) -> None:
    """Copy a saved model, leaving out the keys in ``drop``."""
    d = dict(np.load(src))
    for k in drop:
        del d[k]
    np.savez(dst, **d)


def _state(net: VocalErrorNet) -> dict:
    """Every attribute except the noise generator (re-seeded on load) and the cached
    Brian2 network (rebuilt on demand)."""
    return {k: v for k, v in vars(net).items() if k not in ("_sim_rng", "_built")}


def _assert_same_value(a, b, name):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), name
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)
    else:
        assert a == b, name


def _run_built(net: VocalErrorNet, T_ms: float, setup=None, monitor=None):
    """Run ``net.build()`` for ``T_ms`` with no input; ``setup(objs)`` sets state first and
    ``monitor`` = (group name, variables) records them. Returns (objs, state monitor)."""
    from brian2 import StateMonitor, ms

    objs = net.build()
    if setup is not None:
        setup(objs)
    mon = None
    if monitor is not None:
        mon = StateMonitor(objs[monitor[0]], monitor[1], record=True, dt=net.dt_ms * ms)
        objs["net"].add(mon)
    objs["net"].run(T_ms * ms)
    return objs, mon


# ---------------------------------------------------------------------------
# membrane
# ---------------------------------------------------------------------------

def test_tonic_drive_fires_e_at_its_target_rate():
    """The derived tonic current alone fires E every 1000 / 25 = 40 ms (41 on the 1 ms
    grid: the spike is detected the step after threshold is reached)."""
    net = _small_net(noise_e=0.0, beta_adapt_pa=0.0, target_rate_hz=25.0, t_refr_ms=0.0)
    T = 1000
    sE = net.transform(np.zeros((6, T), np.float32), np.zeros((10, T), np.float32))
    isi = np.diff(np.flatnonzero(sE[0]))
    assert set(isi) <= {40, 41}
    assert (sE.sum(axis=1) == sE[0].sum()).all(), "every E cell is identical"


def test_exponential_euler_relaxes_exactly_toward_v_inf():
    """No synapses, constant drive: V(t) = V_inf + (V0 - V_inf) exp(-t / tau)."""
    from brian2 import mV

    tau, i_ext = 30.0, 300.0
    net = _small_net(noise_e=0.0, beta_adapt_pa=0.0, tau_e=tau, i_drive_e_pa=i_ext)
    v_inf = net.e_l_mv + i_ext / net.g_l_e          # -52 mV: below threshold

    def setup(o):
        o["E"].v = -62.0 * mV

    objs, _ = _run_built(net, 5.0, setup)
    expected = v_inf + (-62.0 - v_inf) * np.exp(-5.0 / tau)
    np.testing.assert_allclose(np.asarray(objs["E"].v / mV), expected, rtol=1e-9)


def test_conductance_inhibition_is_shunting_at_rest():
    """With E_GABA at rest, I->E inhibition does nothing to a cell at rest and pulls a
    depolarised one back toward rest -- never below it."""
    from brian2 import mV, nS

    net = _small_net(n_e=3, noise_e=0.0, beta_adapt_pa=0.0, i_drive_e_pa=0.0,
                     g_nmda_mean_ns=0.0)

    def setup(o):
        o["E"].v = np.array([-70.0, -55.0, -55.0]) * mV
        o["E"].g_gaba = np.array([1000.0, 1000.0, 0.0]) * nS

    objs, _ = _run_built(net, 1.0, setup)
    v = np.asarray(objs["E"].v / mV)
    assert v[0] == pytest.approx(-70.0)
    assert -70.0 <= v[1] < v[2], "inhibition speeds the return to rest, not past it"


def test_mg_block_follows_jahr_stevens():
    """The block, evaluated at the step's starting potential."""
    from brian2 import mV

    net = _small_net(n_e=3, noise_e=0.0, beta_adapt_pa=0.0, i_drive_e_pa=0.0)
    v0 = np.array([-70.0, -50.0, 0.0])

    def setup(o):
        o["E"].v = v0 * mV

    objs, _ = _run_built(net, net.dt_ms, setup)
    mg = np.asarray(objs["E"].mg_block[:])
    np.testing.assert_allclose(mg, 1.0 / (1.0 + net.mg_mm * np.exp(-0.062 * v0) / 3.57))
    assert mg[0] < mg[1] < mg[2]
    assert mg[0] < 0.06 and mg[2] > 0.75


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trips_every_attribute(tmp_path):
    net = _small_net(tau_e=25.0, beta_adapt_pa=500.0, aud_delay_ms=3, hvc_delay_ms=1,
                     r_i_hz=4.0, v_th_mv=-52.0, c_e_pf=400.0, e_gaba_mv=-75.0, dt_ms=0.5)
    net.J_IE = np.random.default_rng(1).random(net.J_IE.shape).astype(np.float32)
    net.stimulus = {"n_ista": 50, "rate_scale": 1.25, "checksums": {"aud": "ab12"}}
    path = tmp_path / "ven.npz"
    net.save(str(path))
    back = VocalErrorNet.load(str(path))

    a, b = _state(net), _state(back)
    assert a.keys() == b.keys(), "save/load must cover every attribute the constructor sets"
    for k in a:
        _assert_same_value(a[k], b[k], k)

    wa, wb = net.weights(), back.weights()
    assert wa.keys() == wb.keys()
    for k in wa:
        _assert_same_value(wa[k], wb[k], k)
    assert repr(net.namespace()) == repr(back.namespace())


def test_stimulus_round_trips(tmp_path):
    net = _small_net()
    net.stimulus = {"t_post": 200, "nested": {"x": [1, 2, 3]}, "label": "song"}
    net.save(str(tmp_path / "ven.npz"))
    assert VocalErrorNet.load(str(tmp_path / "ven.npz")).stimulus == net.stimulus


def test_missing_required_keys_raise_naming_all_of_them(tmp_path):
    full, broken = tmp_path / "full.npz", tmp_path / "broken.npz"
    _small_net().save(str(full))
    _resave_without(full, broken, ["tau_e", "J_EI"])
    with pytest.raises(ValueError) as exc:
        VocalErrorNet.load(str(broken))
    msg = str(exc.value)
    assert "tau_e" in msg and "J_EI" in msg and str(broken) in msg


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_tau_stdp_must_be_positive(bad):
    with pytest.raises(ValueError, match="tau_stdp"):
        _small_net(tau_stdp=bad)
    with pytest.raises(ValueError, match="tau_stdp"):
        C.rho_for(bad)


def test_earlier_model_formats_are_refused(tmp_path):
    """Files from the numpy engine (no format marker, or another one) cannot be run."""
    path = tmp_path / "ven.npz"
    _small_net().save(str(path))
    d = dict(np.load(path))
    d["model_format"] = np.array("something-else")
    np.savez(tmp_path / "other.npz", **d)
    with pytest.raises(ValueError, match=MODEL_FORMAT):
        VocalErrorNet.load(str(tmp_path / "other.npz"))
    del d["model_format"]
    d["units"] = np.array("mV-pA-nS")            # what the numpy engine wrote
    np.savez(tmp_path / "numpy.npz", **d)
    with pytest.raises(ValueError, match="numpy"):
        VocalErrorNet.load(str(tmp_path / "numpy.npz"))


def _is_current_format(path: Path) -> bool:
    if not path.exists():
        return False
    d = np.load(path)
    return "model_format" in d and str(d["model_format"]) == MODEL_FORMAT


@pytest.mark.skipif(not _is_current_format(SHIPPED_MODEL),
                    reason="shipped model missing or from an earlier engine")
def test_shipped_model_was_trained_at_the_default_operating_point():
    """The constructor defaults are the network ``outputs/ven_model.npz`` was trained as."""
    shipped = VocalErrorNet.load(str(SHIPPED_MODEL))
    fresh = VocalErrorNet(n_aud=shipped.n_aud, seed=0)
    for k, v in _state(fresh).items():
        if k in ("A", "B", "J_EI", "J_IE", "stimulus"):
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


def test_noise_intensity_does_not_depend_on_dt():
    """White noise of intensity sigma (pA sqrt(ms)): the free-membrane voltage SD is the
    same at dt = 1 and 0.1 ms (sigma / C * sqrt(tau / 2) = 5.8 mV for the shipped E)."""
    from brian2 import mV

    sds = {}
    for dt in (1.0, 0.1):
        net = _small_net(n_e=200, dt_ms=dt, beta_adapt_pa=0.0,
                         i_drive_e_pa=0.0, v_th_mv=1000.0)   # never fires
        _, mon = _run_built(net, 300.0, monitor=("E", "v"))
        v = np.asarray(mon.v / mV)[:, int(100 / dt):]      # past the initial transient
        sds[dt] = float(v.std())
    expected = C.NOISE_E / C.C_E_PF * np.sqrt(C.TAU_E / 2.0)
    for dt, sd in sds.items():
        assert sd == pytest.approx(expected, rel=0.1), (dt, sd, expected)


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

def test_defaults_are_the_operating_point():
    net = VocalErrorNet(n_aud=7, seed=0)
    expected = {
        "n_e": C.N_E, "n_i": C.N_I, "n_hvc": C.N_HVC, "dt_ms": C.DT_MS,
        "tau_e": C.TAU_E, "tau_i": C.TAU_I, "tau_ampa": C.TAU_AMPA, "tau_gaba": C.TAU_GABA,
        "e_l_mv": C.E_L_MV, "v_th_mv": C.V_TH_MV, "v_reset_mv": C.V_RESET_MV,
        "c_e_pf": C.C_E_PF, "c_i_pf": C.C_I_PF,
        "e_ampa_mv": C.E_AMPA_MV, "e_gaba_mv": C.E_GABA_MV,
        "i_drive_i_pa": C.I_DRIVE_I_PA, "noise_e": C.NOISE_E, "noise_i": C.NOISE_I,
        "c_aud": C.C_AUD, "aud_mean_ns": C.AUD_MEAN_NS, "aud_sd_ns": C.AUD_SD_NS,
        "c_hvc": C.C_HVC, "hvc_mean_ns": C.HVC_MEAN_NS, "hvc_sd_ns": C.HVC_SD_NS,
        "c_ei": C.C_EI, "ei_mean_ns": C.EI_MEAN_NS, "ei_sd_ns": C.EI_SD_NS,
        "rho_ns": C.rho_for(C.TAU_STDP),
        "t_refr_ms": C.T_REFR_MS, "beta_adapt_pa": C.BETA_ADAPT_PA,
        "tau_adapt": C.TAU_ADAPT, "r_i_hz": C.R_I_HZ, "aud_delay_ms": C.AUD_DELAY_MS,
        "hvc_delay_ms": C.HVC_DELAY_MS, "tau_stdp": C.TAU_STDP,
        "g_nmda_mean_ns": C.G_NMDA_MEAN_NS, "tau_nmda_rise": C.TAU_NMDA_RISE,
        "tau_nmda_decay": C.TAU_NMDA_DECAY, "alpha_nmda": C.ALPHA_NMDA,
        "mg_mm": C.MG_MM, "e_nmda_mv": C.E_NMDA_MV,
    }
    for k, v in expected.items():
        assert getattr(net, k) == v, k
    assert net.n_aud == 7


def test_weights_have_the_requested_mean_and_sd():
    net = VocalErrorNet(n_e=400, n_i=200, n_hvc=300, n_aud=200, seed=0)
    for W, mean, sd in ((net.A, C.AUD_MEAN_NS, C.AUD_SD_NS),
                        (net.B, C.HVC_MEAN_NS, C.HVC_SD_NS),
                        (net.J_EI, C.EI_MEAN_NS, C.EI_SD_NS)):
        nz = W[W > 0]
        assert nz.mean() == pytest.approx(mean, rel=0.05)
        assert nz.std() == pytest.approx(sd, rel=0.1)
    assert not net.J_IE.any(), "I -> E starts at zero"


def test_default_rho_follows_the_stdp_window():
    assert VocalErrorNet(n_aud=3, n_e=4, n_i=2, tau_stdp=10.0).rho_ns == C.rho_for(10.0)


def test_theta_is_the_i_trace_mean_at_r_i():
    net = _small_net(r_i_hz=50.0, tau_stdp=20.0)
    assert net.theta == pytest.approx(1.0)


def test_n_aud_is_required():
    with pytest.raises(ValueError, match="n_aud"):
        VocalErrorNet()


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

#: Tonic E current that fires E every few ms.
STRONG_DRIVE_PA = 7500.0


def _quiet_net(**kw) -> VocalErrorNet:
    """Noise off, no tonic E drive, no E->I weights: E fires only from audio."""
    args = dict(noise_e=0.0, noise_i=0.0, i_drive_e_pa=0.0, aud_delay_ms=0, hvc_delay_ms=0)
    args.update(kw)
    net = _small_net(**args)
    net.J_EI = np.zeros_like(net.J_EI)
    return net


def test_aud_delay_shifts_the_e_response_exactly():
    net = _quiet_net()
    net.A = np.zeros_like(net.A)
    net.A[:, 0] = np.linspace(100.0, 1000.0, net.n_e)    # a spread of conductances (nS)
    T, t0 = 120, 10
    aud = np.zeros((net.n_aud, T), dtype=np.float32)
    aud[0, t0] = 1.0
    hvc = np.zeros((net.n_hvc, T), dtype=np.float32)

    out = {}
    for d in (0, 3):
        net.aud_delay_ms = d
        out[d] = net.transform(hvc, aud)
    assert out[0].sum() > 0, "the pulse must evoke E spikes"
    assert np.flatnonzero(out[0].any(axis=0))[0] == t0 + 1, "first moves v the step after"
    assert not out[3][:, : t0 + 3 + 1].any()
    np.testing.assert_array_equal(out[3][:, 3:], out[0][:, :-3])


def test_hvc_delay_shifts_the_i_response_exactly():
    net = _quiet_net()
    net.B = np.zeros_like(net.B)
    net.B[:, 0] = 300.0
    hvc = np.zeros((net.n_hvc, 100), dtype=np.float32)
    hvc[0, 20] = 1.0
    aud = np.zeros((net.n_aud, 100), dtype=np.float32)
    first = {}
    for d in (0, 4):
        net.hvc_delay_ms = d
        _, sI = net.transform_all(hvc, aud)
        assert sI.sum() > 0
        first[d] = int(np.flatnonzero(sI.any(axis=0))[0])
    assert first[4] - first[0] == 4


def _stdp_net(**kw) -> VocalErrorNet:
    """Two hand-wired pathways: auditory channel 0 -> E0 and HVC channel 0 -> I0."""
    net = _quiet_net(**kw)
    net.A = np.zeros_like(net.A)
    net.A[0, 0] = 500.0                       # suprathreshold from one spike (nS)
    net.B = np.zeros_like(net.B)
    net.B[0, 0] = 300.0
    return net


def _pulses(net, T=300, aud_t=(), hvc_t=()):
    aud = np.zeros((net.n_aud, T), dtype=np.float32)
    hvc = np.zeros((net.n_hvc, T), dtype=np.float32)
    aud[0, list(aud_t)] = 1.0
    hvc[0, list(hvc_t)] = 1.0
    return hvc, aud


def test_i_before_gated_e_potentiates_only_that_pair():
    net = _stdp_net(r_i_hz=0.0)                        # theta = 0: no depression term
    hvc, aud = _pulses(net, hvc_t=[20], aud_t=[25])
    sE = net.fit(hvc, aud)
    assert sE[0].sum() > 0 and sE[1:].sum() == 0
    assert net.J_IE[0, 0] > 0.0
    mask = np.ones_like(net.J_IE, dtype=bool)
    mask[0, 0] = False
    assert np.all(net.J_IE[mask] == 0.0), "only the I0 -> E0 pair was active"


def test_e_long_before_i_does_not_potentiate():
    """The window is causal: an I burst after E has fallen silent changes nothing."""
    net = _stdp_net(r_i_hz=0.0, g_nmda_mean_ns=0.0)    # no slow NMDA tail keeping E up
    hvc, aud = _pulses(net, aud_t=[20], hvc_t=[250])
    sE, sI = net.transform_all(hvc, aud)
    assert sE.sum() > 0 and sI.sum() > 0
    assert sE[:, 250:].sum() == 0, "premise: E is silent when I fires"
    net.fit(hvc, aud)
    assert np.all(net.J_IE == 0.0)


@pytest.mark.parametrize("r_i_hz", [0.0, 50.0])
def test_closed_audio_gate_blocks_all_plasticity(r_i_hz):
    """E fires from tonic drive alone, with I active, but no audio: J_IE must not move."""
    net = _stdp_net(r_i_hz=r_i_hz)
    net.i_drive_e_pa = STRONG_DRIVE_PA
    net.J_IE[:] = 100.0
    hvc, aud = _pulses(net, hvc_t=[20, 60, 100])
    sE, sI = net.transform_all(hvc, aud)
    assert sE.sum() > 0 and sI.sum() > 0
    net.fit(hvc, aud)
    assert np.all(net.J_IE == np.float32(100.0))


def test_gate_closes_tau_stdp_after_the_last_input_spike():
    """E spikes driven by tonic drive long after the input must not move J_IE."""
    net = _stdp_net(r_i_hz=0.0)
    net.i_drive_e_pa = STRONG_DRIVE_PA
    net.J_IE[:] = 100.0
    hvc, aud = _pulses(net, T=300, aud_t=[20], hvc_t=[100, 140, 180])
    sE, sI = net.transform_all(hvc, aud)
    assert sE[0, 100:].sum() > 0 and sI.sum() > 0, "premise: E and I both fire late"
    assert sI[:, :45].sum() == 0, "premise: I is silent while the gate is open"
    net.fit(hvc, aud)
    # I first fires at ~100 ms, long after the gate (20-40 ms) closed: no I trace was
    # ever read by a gated spike, and r_i_hz = 0 means no depression either.
    assert np.all(net.J_IE == np.float32(100.0))


def test_audio_gate_is_open_for_tau_stdp_after_an_input_spike():
    """The gate is the auditory trace above 1/e: open for tau_stdp after an isolated
    input spike, closed before it and after."""
    from brian2 import ms

    net = _stdp_net(tau_stdp=20.0)
    objs = net.build()
    objs["aud"].set_spikes([0], [20] * ms)
    from brian2 import StateMonitor
    mon = StateMonitor(objs["E"], "gate", record=[0, 1])
    objs["net"].add(mon)
    objs["net"].run(80 * ms)
    t_open = np.asarray(mon.t / ms)[np.asarray(mon.gate[0]) > 0.5]
    assert t_open.min() == pytest.approx(21.0, abs=1.0)
    assert t_open.max() == pytest.approx(40.0, abs=1.0)
    assert np.all(np.diff(t_open) == pytest.approx(1.0)), "one contiguous window"
    assert not np.asarray(mon.gate[1]).any(), "a cell with no auditory input never opens"


def test_e_spike_without_i_activity_depresses_and_clips_at_zero():
    # theta = r_i_hz * tau_stdp = 50 Hz * 20 ms = 1.0; with I silent the trace is 0, so
    # every gated E spike (within ~tau_stdp of the input) lowers the weight by rho*theta.
    net = _stdp_net(r_i_hz=50.0)
    net.J_IE[0, 0] = 50.0
    hvc, aud = _pulses(net, aud_t=[20])
    sE = net.fit(hvc, aud)
    step = net.rho_ns * net.theta
    n_gated = int(sE[0, 20:20 + int(net.tau_stdp)].sum())
    assert n_gated > 0
    k = (50.0 - float(net.J_IE[0, 0])) / step
    assert k == pytest.approx(round(k), abs=1e-4), "whole multiples of rho * theta"
    assert abs(round(k) - n_gated) <= 1, (k, n_gated)
    assert net.J_IE[0, 0] > 0.0, "premise: the weight is lowered, not clipped"
    assert np.all(net.J_IE[0, 1:] == 0.0), "zero weights are clipped, never negative"

    net = _stdp_net(r_i_hz=50.0, rho_ns=100.0)       # one spike more than erases it
    net.J_IE[0, 0] = 50.0
    net.fit(hvc, aud)
    assert net.J_IE.min() == 0.0 and net.J_IE[0, 0] == 0.0


def test_transform_never_changes_the_weights():
    net = _stdp_net(r_i_hz=0.0)
    hvc, aud = _pulses(net, hvc_t=[20], aud_t=[25])
    before = net.J_IE.copy()
    net.transform_all(hvc, aud)
    np.testing.assert_array_equal(net.J_IE, before)
    net.fit(hvc, aud)
    assert not np.array_equal(net.J_IE, before), "premise: the same input does learn"


def test_gaba_decay_lengthens_the_inhibition_from_one_i_spike():
    """A tonically firing E cell is silenced longer by one I spike at tau_gaba = 15 ms
    than at 1 ms, at the same peak conductance."""
    def silence_after_i_spike(tau_gaba):
        net = _quiet_net(tau_gaba=tau_gaba)
        net.i_drive_e_pa = STRONG_DRIVE_PA     # E fires tonically
        net.B = np.zeros_like(net.B)
        net.B[0, 0] = 300.0                    # HVC 0 -> I0 only
        net.J_IE = np.zeros_like(net.J_IE)
        net.J_IE[0, 0] = 1000.0                # I0 -> E0 only (peak conductance, nS)
        hvc, aud = _pulses(net, T=150, hvc_t=[60])
        sE, sI = net.transform_all(hvc, aud)
        t_i = int(np.flatnonzero(sI[0])[0])
        assert sI[1:].sum() == 0 and sE[0, :t_i].sum() > 0, "premise: E fires before I"
        after = np.flatnonzero(sE[0, t_i:])
        return int(after[0]) if after.size else 150
    assert silence_after_i_spike(15.0) > silence_after_i_spike(1.0)


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
    net = VocalErrorNet(n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud, seed=0, dt_ms=1.0,
                        hvc_mean_ns=30 * C.HVC_MEAN_NS, hvc_sd_ns=30 * C.HVC_SD_NS,
                        aud_mean_ns=54 / 40 * C.AUD_MEAN_NS,
                        aud_sd_ns=54 / 40 * C.AUD_SD_NS)
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
    from brian2 import StateMonitor, ms

    net = _small_net(n_aud=2, aud_delay_ms=0)
    objs = net.build()
    t_single = [100.0]
    t_train = list(np.arange(100.0, 300.0, 1.0))           # 1 kHz for 200 ms
    objs["aud"].set_spikes([0] * len(t_single) + [1] * len(t_train),
                           np.array(t_single + t_train) * ms)
    mon = StateMonitor(objs["nmda"], "s", record=True)
    objs["net"].add(mon)
    objs["net"].run(600 * ms)
    s = np.asarray(mon.s)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s[0, :100].max() == 0.0          # causal
    assert s[1, 300] > 0.95                 # saturated, far above a single spike's peak
    assert s[0].max() < 0.7
    t0 = 100 + 20                           # well past the 2 ms rise
    assert s[0, t0 + 100] / s[0, t0] == pytest.approx(np.exp(-1.0), rel=0.02)


def _burst_response(g_nmda_mean_ns):
    net = VocalErrorNet(n_e=40, n_i=10, n_hvc=3, n_aud=4, seed=0, dt_ms=1.0, noise_e=0.0,
                        beta_adapt_pa=0.0, i_drive_e_pa=0.0, aud_delay_ms=0,
                        g_nmda_mean_ns=g_nmda_mean_ns,
                        aud_mean_ns=650.0, aud_sd_ns=325.0)
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
                        tau_nmda_decay=80.0)
    path = tmp_path / "nmda.npz"
    net.save(str(path))
    back = VocalErrorNet.load(str(path))
    assert (back.g_nmda_mean_ns, back.tau_nmda_decay) == (12.0, 80.0)
    np.testing.assert_array_equal(back.weights()["G"], back.nmda_conductances())


def test_nmda_conductances_are_a_rescaled_to_their_mean():
    net = VocalErrorNet(n_e=400, n_i=10, n_hvc=3, n_aud=200, seed=0, g_nmda_mean_ns=900.0)
    G = net.nmda_conductances()
    np.testing.assert_allclose(G, net.A * (900.0 / net.aud_mean_ns))
    assert G[G > 0].mean() == pytest.approx(900.0, rel=0.05)
    assert not VocalErrorNet(n_e=4, n_i=2, n_aud=3, g_nmda_mean_ns=0.0).nmda_conductances().any()


def test_network_without_nmda_runs():
    net = _small_net(g_nmda_mean_ns=0.0)
    hvc, aud = np.zeros((6, 50), np.float32), np.zeros((10, 50), np.float32)
    aud[:, 10] = 1.0
    sE, sI = net.transform_all(hvc, aud)
    assert sE.shape == (30, 50) and sI.shape == (8, 50)


# ------------------------------------------------------------------- HVC prelude


def test_a_prelude_is_trimmed_and_changes_no_weights():
    net = VocalErrorNet(n_e=20, n_i=8, n_hvc=6, n_aud=5, seed=0, dt_ms=1.0)
    T = 120
    hvc = generate_hvc_bursts(n_hvc=6, T=T, n_renditions=1, T_song=100, T_burn=0,
                              T_post=20, seed=1)
    aud = (np.random.default_rng(2).random((5, T)) < 0.05).astype(np.float32)
    pre = generate_hvc_bursts(n_hvc=6, T=60, n_renditions=1, T_song=50, T_burn=0,
                              T_post=10, seed=3)
    sE, sI = net.transform_all(hvc, aud, hvc_prelude=pre)
    assert sE.shape == (20, T) and sI.shape == (8, T)
    # Learning needs auditory drive, so a prelude alone (no sound) leaves J_IE untouched.
    net.fit(np.zeros((6, 0), dtype=np.float32), np.zeros((5, 0), dtype=np.float32),
            hvc_prelude=pre)
    assert not net.J_IE.any()


# ------------------------------------------------------------------ network cache


def test_changing_a_fixed_weight_rebuilds_the_network():
    """The Brian2 network bakes in the fixed weights; editing one must take effect."""
    net = _quiet_net()
    hvc, aud = _pulses(net, T=60, aud_t=[10])
    net.A = np.zeros_like(net.A)
    assert net.transform(hvc, aud).sum() == 0
    net.A[0, 0] = 500.0
    assert net.transform(hvc, aud).sum() > 0
