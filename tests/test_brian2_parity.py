"""The Brian2 runtime must agree with the numpy model it claims to mirror.

``brian2_runtime`` exists so that weights tuned against :class:`VocalErrorNetV2`
"transfer here without translation". Nothing used to check that: the existing tests
build the groups and count the synapse objects without ever running the network, and
four defects lived in that gap --

* the STDP presynaptic trace was incremented once per outgoing synapse instead of once
  per spike, so its size (and the effective learning rate) scaled with the E population;
* the analytic threshold fallbacks used a superseded formula and ran 61% high;
* an all-zero weight matrix crashed inside Brian2's index handling;
* ``noise_i`` was accepted and silently dropped.

Each test below pins one of those. Skipped without the ``brian2`` extra.
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
from spiking_ven.vocal_error_net import VocalErrorNetV2  # noqa: E402


@pytest.fixture(autouse=True)
def _scope():
    """Fresh Brian2 scope and the numpy model's 1 ms timestep for every test."""
    brian2.start_scope()
    defaultclock.dt = 1.0 * ms
    yield


def _one_i_neuron_net(n_e: int):
    """A single I neuron driven to fire, projecting to every one of n_e E neurons."""
    groups = build_ven_groups(N_e=n_e, N_i=1)
    aud = SpikeGeneratorGroup(2, [], [] * ms)
    hvc = SpikeGeneratorGroup(1, [0], [2] * ms)
    syns = build_ven_synapses(
        groups, aud, hvc,
        B_weights=np.full((n_e, 2), 1e-9),
        B_hvc_weights=np.full((1, 1), 10.0),     # enough to drive the I neuron
        JIE_weights=np.full((n_e, 1), 0.01),
        JEI_weights=np.full((1, n_e), 1e-9),
        learn=True,
    )
    return groups, aud, hvc, syns


@pytest.mark.parametrize("n_e", [10, 200])
def test_stdp_trace_counts_spikes_not_synapses(n_e):
    """x_i must advance once per I spike, whatever the number of E targets.

    Incremented from the synapse's on_pre it advanced once per synapse, so this ratio
    came out at n_e (measured 61x and 1225x for n_e=10 and 200).
    """
    groups, aud, hvc, syns = _one_i_neuron_net(n_e)
    spikes = SpikeMonitor(groups["i_group"])
    trace = StateMonitor(groups["i_group"], "x_i", record=0)
    Network(aud, hvc, groups["e_group"], groups["i_group"], *syns, spikes, trace).run(20 * ms)

    assert spikes.num_spikes > 0, "the I neuron should fire under this drive"
    peak = float(trace.x_i[0].max())
    # A +1-per-spike trace decaying at tau_s can never exceed the spike count.
    assert peak <= spikes.num_spikes + 1e-6, (
        f"x_i peaked at {peak:.1f} after {spikes.num_spikes} spikes with {n_e} outgoing "
        "synapses - it is counting synapses, not spikes"
    )
    assert peak > 1.0, "with several spikes inside tau_s the trace should accumulate"


def test_stdp_trace_is_independent_of_population_size():
    """The same I spike train must give the same trace for any E population size."""
    peaks = []
    for n_e in (10, 50, 200):
        brian2.start_scope()
        groups, aud, hvc, syns = _one_i_neuron_net(n_e)
        trace = StateMonitor(groups["i_group"], "x_i", record=0)
        Network(aud, hvc, groups["e_group"], groups["i_group"], *syns, trace).run(20 * ms)
        peaks.append(float(trace.x_i[0].max()))
    assert peaks[0] == pytest.approx(peaks[1]) == pytest.approx(peaks[2]), (
        f"peak x_i varies with N_e: {peaks}"
    )


def test_threshold_fallbacks_match_the_numpy_model():
    """Defaulted thresholds must equal VocalErrorNetV2's, or the defaults are a trap."""
    net = VocalErrorNetV2()
    groups = build_ven_groups()
    assert float(groups["e_group"].theta[0]) == pytest.approx(float(net.theta_e[0]))
    assert groups["i_group"].namespace["theta_i_ns"] == pytest.approx(net.theta_i)


def test_all_zero_weight_matrices_wire_inactive_synapses():
    """A population with no connections must be inactive, not a ValueError from numpy."""
    groups = build_ven_groups(N_e=5, N_i=2)
    aud = SpikeGeneratorGroup(3, [], [] * ms)
    hvc = SpikeGeneratorGroup(2, [], [] * ms)
    syns = build_ven_synapses(
        groups, aud, hvc,
        B_weights=np.zeros((5, 3)), B_hvc_weights=np.zeros((2, 2)),
        JIE_weights=np.zeros((5, 2)), JEI_weights=np.zeros((2, 5)),
    )
    assert len(syns) == 4
    assert all(not bool(s.active) for s in syns), "empty matrices should deactivate"


def test_noise_i_reaches_the_i_membrane():
    """noise_i was accepted and dropped: the I equation had no noise term at all."""
    voltages = {}
    for amp in (0.0, 0.5):
        brian2.start_scope()
        groups = build_ven_groups(N_e=2, N_i=40, noise_i=amp, drive_i=0.0)
        mon = StateMonitor(groups["i_group"], "v", record=range(40))
        Network(groups["e_group"], groups["i_group"], mon).run(50 * ms)
        voltages[amp] = float(np.std(mon.v))
    assert voltages[0.0] == 0.0, "zero amplitude must stay exactly noiseless"
    assert voltages[0.5] > 0.01, "a non-zero noise_i must move the I membrane"


def test_tau_s_mismatch_is_rejected():
    """tau_s only takes effect via build_ven_groups, so a mismatch must not pass quietly."""
    groups = build_ven_groups(N_e=4, N_i=2, tau_s=10.0)
    aud = SpikeGeneratorGroup(2, [], [] * ms)
    hvc = SpikeGeneratorGroup(2, [], [] * ms)
    with pytest.raises(ValueError, match="tau_s"):
        build_ven_synapses(groups, aud, hvc,
                           B_weights=np.ones((4, 2)) * 0.1,
                           B_hvc_weights=np.ones((2, 2)) * 0.1,
                           tau_s=20.0)


def test_trained_weights_give_a_plausible_e_rate_in_both_runtimes():
    """End-to-end sanity: the same weights and input must not land in different regimes.

    Not an equality check -- the Brian2 port filters I->E through a 1 ms synaptic decay
    and integrates in continuous time, so the two differ by construction. The point is
    that a defect large enough to matter (a 60x learning rate, a 61% threshold offset)
    shows up as the populations sitting in different rate regimes.
    """
    rng = np.random.default_rng(0)
    n_e, n_i, n_aud, n_hvc, T = 60, 15, 12, 6, 400

    net = VocalErrorNetV2(n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud,
                          noise_e=0.0, noise_i=0.0, seed=0)
    aud_spikes = (rng.random((n_aud, T)) < 0.02).astype(np.float32)
    hvc_spikes = (rng.random((n_hvc, T)) < 0.02).astype(np.float32)
    numpy_rate = float(net.transform(hvc_spikes, aud_spikes).mean() * 1000)

    w = net.to_brian_weights()
    groups = build_ven_groups(
        N_e=w["n_e"], N_i=w["n_i"], tau_e=w["tau_e"], tau_i=w["tau_i"], tau_s=w["tau_s"],
        drive_e=w["drive_e"], drive_i=w["drive_i"], noise_e=0.0, noise_i=0.0,
        theta_e=w["theta_e"], theta_i=w["theta_i"], v_reset=w["v_reset"],
    )
    a_i, a_t = np.nonzero(aud_spikes.T)
    h_i, h_t = np.nonzero(hvc_spikes.T)
    aud_g = SpikeGeneratorGroup(n_aud, a_t.astype(int), a_i.astype(float) * ms)
    hvc_g = SpikeGeneratorGroup(n_hvc, h_t.astype(int), h_i.astype(float) * ms)
    syns = build_ven_synapses(
        groups, aud_g, hvc_g, B_weights=w["B"], B_hvc_weights=w["B_hvc"],
        JIE_weights=w["JIE"], JEI_weights=w["JEI"], learn=False,
        tau_s=w["tau_s"], xi_th=w["xi_th"], J_max_ie=w["J_max_ie"],
        aud_delay=0.0, hvc_delay=0.0,
    )
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
