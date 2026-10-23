"""The Brian2 runtime must run the network the numpy model trained.

``brian2_runtime`` takes everything from ``VocalErrorNet.to_brian_weights()``. These
tests pin the parts of the numpy dynamics that have to survive the port -- spike-frequency
adaptation, the refractory period, the I-population noise -- and check end to end that the
same weights and input put both runtimes in the same rate regime. Skipped without the
``brian2`` extra.
"""

import numpy as np
import pytest

brian2 = pytest.importorskip("brian2", reason="requires the 'brian2' extra")

from brian2 import (  # noqa: E402  (after importorskip by design)
    Network,
    SpikeGeneratorGroup,
    SpikeMonitor,
    StateMonitor,
    defaultclock,
    ms,
)

from spiking_ven.brian2_runtime import (  # noqa: E402
    build_ven_groups,
    build_ven_synapses,
)
from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402


@pytest.fixture(autouse=True)
def _scope():
    """Fresh Brian2 scope and the numpy model's 1 ms timestep for every test."""
    brian2.start_scope()
    defaultclock.dt = 1.0 * ms
    yield


def _weights(**overrides):
    w = VocalErrorNet(n_e=20, n_i=5, n_hvc=4, n_aud=6, seed=0).to_brian_weights()
    w.update(overrides)
    return w


def _run_e(w, aud_times_ms=(), T=100):
    """Drive every E cell from auditory channel 0 at the given times; return spikes."""
    groups = build_ven_groups(w)
    aud = SpikeGeneratorGroup(w["n_aud"], [0] * len(aud_times_ms),
                              np.asarray(aud_times_ms, dtype=float) * ms)
    hvc = SpikeGeneratorGroup(w["n_hvc"], [], [] * ms)
    syns = build_ven_synapses(groups, w, aud, hvc)
    mon = SpikeMonitor(groups["e_group"])
    Network(aud, hvc, groups["e_group"], groups["i_group"], *syns, mon).run(T * ms)
    return mon


def _strong_aud(w, scale=50.0):
    """Every E cell driven hard by auditory channel 0, nothing else connected."""
    B = np.zeros_like(w["B"])
    B[:, 0] = scale
    return {"B": B, "JIE": np.zeros_like(w["JIE"]), "JEI": np.zeros_like(w["JEI"]),
            "noise_e": 0.0, "aud_delay_ms": 0}


def test_refractory_period_bounds_the_isi():
    """Under drive that would otherwise fire every step, no ISI is shorter than t_refr_ms."""
    w = _weights(t_refr_ms=2.0, a_sfa=0.0)
    w.update(_strong_aud(w))
    mon = _run_e(w, aud_times_ms=range(0, 100, 1))
    t = np.asarray(mon.t[mon.i == 0] / ms)
    assert len(t) > 10, "the cell should fire repeatedly under this drive"
    assert np.diff(t).min() >= 2.0 - 1e-9


def test_spike_frequency_adaptation_slows_firing():
    """a_sfa must reach the E membrane: the same drive gives fewer spikes with it on."""
    counts = {}
    for a_sfa in (0.0, 1.0):
        brian2.start_scope()
        w = _weights(t_refr_ms=2.0, a_sfa=a_sfa, tau_sfa=10.0)
        # Strong enough to fire well above baseline, weak enough not to sit at the
        # refractory ceiling, where adaptation has nothing left to slow.
        w.update(_strong_aud(w, scale=0.2))
        counts[a_sfa] = _run_e(w, aud_times_ms=range(0, 100, 1)).num_spikes
    assert counts[1.0] < 0.8 * counts[0.0], counts


def test_all_zero_weight_matrices_wire_inactive_synapses():
    """A population with no connections must be inactive, not a ValueError from numpy."""
    w = _weights()
    for k in ("B", "B_hvc", "JIE", "JEI"):
        w[k] = np.zeros_like(w[k])
    groups = build_ven_groups(w)
    aud = SpikeGeneratorGroup(w["n_aud"], [], [] * ms)
    hvc = SpikeGeneratorGroup(w["n_hvc"], [], [] * ms)
    syns = build_ven_synapses(groups, w, aud, hvc)
    assert len(syns) == 4
    assert all(not bool(s.active) for s in syns), "empty matrices should deactivate"


def test_noise_i_reaches_the_i_membrane():
    """noise_i must reach the I equation, and zero noise must stay exactly noiseless."""
    voltages = {}
    for amp in (0.0, 0.5):
        brian2.start_scope()
        groups = build_ven_groups(_weights(n_i=40, noise_i=amp, drive_i=0.0))
        mon = StateMonitor(groups["i_group"], "v", record=range(40))
        Network(groups["e_group"], groups["i_group"], mon).run(50 * ms)
        voltages[amp] = float(np.std(mon.v))
    assert voltages[0.0] == 0.0, "zero amplitude must stay exactly noiseless"
    assert voltages[0.5] > 0.01, "a non-zero noise_i must move the I membrane"


@pytest.mark.parametrize("a_sfa", [0.0, 1.0])
def test_same_weights_give_the_same_e_rate_regime_in_both_runtimes(a_sfa):
    """End-to-end: the same weights and input must not land in different regimes.

    Not an equality check -- the Brian2 port filters the recurrent couplings through a
    1 ms synaptic decay and integrates in continuous time, so the two differ by
    construction. Run with the shipped refractory period and adaptation.
    """
    rng = np.random.default_rng(0)
    n_e, n_i, n_aud, n_hvc, T = 60, 15, 12, 6, 400

    net = VocalErrorNet(n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud, noise_e=0.0,
                        noise_i=0.0, t_refr_ms=2.0, a_sfa=a_sfa, tau_sfa=10.0,
                        seed=0)
    aud_spikes = (rng.random((n_aud, T)) < 0.02).astype(np.float32)
    hvc_spikes = (rng.random((n_hvc, T)) < 0.02).astype(np.float32)
    numpy_rate = float(net.transform(hvc_spikes, aud_spikes).mean() * 1000)

    w = net.to_brian_weights()
    groups = build_ven_groups(w)
    a_i, a_t = np.nonzero(aud_spikes.T)
    h_i, h_t = np.nonzero(hvc_spikes.T)
    aud_g = SpikeGeneratorGroup(n_aud, a_t.astype(int), a_i.astype(float) * ms)
    hvc_g = SpikeGeneratorGroup(n_hvc, h_t.astype(int), h_i.astype(float) * ms)
    syns = build_ven_synapses(groups, w, aud_g, hvc_g)
    mon = SpikeMonitor(groups["e_group"])
    Network(aud_g, hvc_g, groups["e_group"], groups["i_group"], *syns, mon).run(T * ms)
    brian_rate = mon.num_spikes / n_e / (T * 1e-3)

    assert numpy_rate > 0 and brian_rate > 0, (
        f"both runtimes should fire: numpy={numpy_rate:.1f} Hz brian2={brian_rate:.1f} Hz"
    )
    assert brian_rate == pytest.approx(numpy_rate, rel=0.5), (
        f"E rates in different regimes: numpy={numpy_rate:.1f} Hz "
        f"brian2={brian_rate:.1f} Hz"
    )
