"""The vocal error network as Brian2 equations.

This module *is* the model: the equations below are the ones in the paper's Methods, in
Brian2's notation and with physical units, and everything else in the package (training
loop, metrics, figures) treats it as a black box that maps input spikes to E and I spikes.

Populations
-----------
- ``aud``   auditory channels (spike generator: encoder output)
- ``nmda``  one NMDA gating variable per auditory channel (Wang 2002)
- ``hvc``   HVC premotor neurons (spike generator)
- ``E``     excitatory error neurons: conductance LIF with adaptation
- ``I``     inhibitory interneurons: conductance LIF

Projections (all fixed except I -> E)
-------------------------------------
- aud  -> E   AMPA, weights ``A`` (the auditory "eligibility" trace for STDP rides along)
- nmda -> E   NMDA, weights ``G`` (``A`` rescaled), Mg2+-blocked
- hvc  -> I   AMPA, weights ``B``
- E    -> I   AMPA, weights ``J_EI``
- I    -> E   GABA_A, weights ``J_IE``, learned by an audio-gated three-factor STDP rule

Parameters arrive as a :class:`dict` of Brian2 quantities from
:meth:`spiking_ven.vocal_error_net.VocalErrorNet.namespace`; the weights as dense
``(post, pre)`` arrays in nS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from brian2 import (
    Network,
    NeuronGroup,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Synapses,
    ms,
    nS,
)

__all__ = ["E_EQS", "I_EQS", "NMDA_EQS", "STDP_ON_POST", "STDP_ON_PRE", "build_network", "spike_events",
           "bin_spikes"]

# ---------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------
# Both populations are leaky integrate-and-fire membranes whose synapses are
# conductances acting through their reversal potentials. ``I_noise`` is white current
# noise of intensity ``sigma`` (pA sqrt(ms)), held constant over each step, so a run at
# any dt has the same diffusion. The membrane is integrated by exponential Euler, which
# needs it linear in v: the Mg2+ block is therefore evaluated once per step (at the
# step's starting potential), like the noise.

#: Excitatory error neurons.
E_EQS = """
dv/dt = (g_L*(E_L - v) + g_ampa*(E_AMPA - v) + g_nmda*mg_block*(E_NMDA - v)
         + g_gaba*(E_GABA - v) + I_tonic - I_adapt + I_noise) / C : volt (unless refractory)
dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens
dI_adapt/dt = -I_adapt / tau_adapt : amp
da_aud/dt = -a_aud / tau_stdp : 1
dx_e/dt = -x_e / tau_stdp : 1
dr_loc/dt = -r_loc / tau_loc : Hz
dr_slow/dt = -r_slow / tau_slow : Hz
alpha_e : 1
g_nmda : siemens
mg_block = 1 / (1 + Mg / (3.57*mM) * exp(-0.062 * v/mV)) : 1 (constant over dt)
gate = int(a_aud > exp(-1)) : 1
I_noise = sigma * randn() / sqrt(dt) : amp (constant over dt)
"""
# ``a_aud`` counts recent auditory input spikes, each decaying with tau_stdp; ``gate``
# is open while it exceeds 1/e, i.e. for tau_stdp after an isolated input spike. ``x_e``
# is the cell's own spike trace and ``alpha_e`` its rate target (``2 r tau_stdp``), used
# only by the optional rate-target rule (``VOGELS_ON_PRE``). ``r_loc`` is the cell's own
# rate averaged over ``tau_loc`` (an exponential running mean), for the rule's local
# target (``lam_loc``); ``r_slow`` the same over ``tau_slow``, its optional reference.
# ``g_nmda`` is summed from the NMDA synapses (see ``build_network``).

#: Inhibitory interneurons. ``x_stdp`` is the presynaptic STDP trace.
I_EQS = """
dv/dt = (g_L*(E_L - v) + g_ampa*(E_AMPA - v) + I_tonic + I_noise) / C : volt (unless refractory)
dg_ampa/dt = -g_ampa / tau_ampa : siemens
dx_stdp/dt = -x_stdp / tau_stdp : 1
I_noise = sigma * randn() / sqrt(dt) : amp (constant over dt)
"""

#: NMDA gating, one per auditory channel: a rise variable driven by the channel's spikes
#: and a saturating gate (Wang 2002).
NMDA_EQS = """
dx/dt = -x / tau_nmda_rise : 1
ds/dt = -s / tau_nmda_decay + alpha_nmda * x * (1 - s) : 1
"""

#: Three-factor STDP on I -> E, at each postsynaptic (E) spike: move the weight by
#: ``rho`` times (the presynaptic trace minus its baseline ``theta``), only while the E
#: cell's audio gate is open, and never below zero. ``rho`` is 0 outside training.
STDP_ON_POST = "w = clip(w + rho * gate_post * (x_stdp_pre - theta), 0*nS, inf*nS)"

#: Optional presynaptic depression (after Vogels et al. 2011), at each I spike: the weight
#: falls by ``rho_dep`` while the E cell's audio gate is open, so I activity that does not
#: precede the E cell's spikes costs weight. ``rho_dep`` is 0 by default and outside
#: training.
STDP_ON_PRE = "g_gaba_post += w; w = clip(w - rho_dep * gate_post, 0*nS, inf*nS)"

#: Rate-target inhibitory STDP (Vogels et al. 2011, Science 334:1569), an option for
#: I_HVC -> E: at each I spike the weight moves by ``eta`` times (the E cell's spike trace
#: minus its target ``alpha_e = 2 r_target tau_stdp``); at each E spike it grows by ``eta``
#: times the I trace. Each E cell settles at its target rate; with a sparse, reproducible
#: presynaptic basis, the learned inhibition follows the E cell's excess drive in time.
#: The target is the constant ``alpha_e`` blended (weight ``lam_loc``) with a local one,
#: ``2 tau_stdp r_loc``: the cell's own rate over the last ``tau_loc``. ``lam_loc = 0`` is
#: the constant target of Vogels et al.; ``lam_loc = 1`` asks the rule to flatten the rate
#: on timescales shorter than ``tau_loc`` only. With ``use_slow``, the local term is
#: measured against the cell's rate over ``tau_slow`` instead of its target, so the target
#: becomes ``alpha_e + lam_loc 2 tau_stdp (r_loc - r_slow)``: fast fluctuations are
#: flattened while the mean stays anchored at ``alpha_e`` (a high-pass target).
_REF = "(use_slow * 2 * tau_stdp * r_slow_post + (1 - use_slow) * alpha_e_post)"
_TARGET = f"(alpha_e_post + lam_loc * (2 * tau_stdp * r_loc_post - {_REF}))"
VOGELS_ON_PRE = f"g_gaba_post += w; w = clip(w + eta * (x_e_post - {_TARGET}), 0*nS, inf*nS)"
VOGELS_ON_POST = "w = clip(w + eta * x_stdp_pre, 0*nS, inf*nS)"
#: The same, with both terms multiplied by the E cell's audio gate (learning only while
#: auditory input drives the cell), for comparison.
VOGELS_ON_PRE_GATED = ("g_gaba_post += w; "
                       f"w = clip(w + eta * gate_post * (x_e_post - {_TARGET}), 0*nS, inf*nS)")
VOGELS_ON_POST_GATED = "w = clip(w + eta * gate_post * x_stdp_pre, 0*nS, inf*nS)"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _connect(syn: Synapses, W: np.ndarray, *, mask: np.ndarray | None = None) -> None:
    """Connect ``syn`` from a ``(post, pre)`` weight matrix in nS and set ``syn.w``.

    Only nonzero entries become synapses, unless ``mask`` (same shape, bool) says which
    pairs are connected -- for a plastic matrix, whose weights may start at zero.
    """
    W = np.asarray(W, dtype=np.float64)
    post, pre = np.nonzero(W if mask is None else np.asarray(mask, dtype=bool))
    if pre.size == 0:
        syn.active = False
        return
    syn.connect(i=pre, j=post)
    syn.w[:] = W[post, pre] * nS


def build_network(ns: dict, weights: dict, *, n_aud: int, n_hvc: int,
                  aud_delay, hvc_delay, dt) -> dict:
    """Build every group, synapse and monitor of the network, inside one ``Network``.

    ``ns`` is the namespace of Brian2 quantities (``VocalErrorNet.namespace()``):
    population-specific constants under ``"E"``, ``"I"``, the rest shared. ``weights``
    holds the ``(post, pre)`` matrices ``A``, ``G``, ``B``, ``J_EI``, ``J_IE`` in nS, and
    ``M_IE``, which I -> E pairs are connected (plastic synapses exist only there).
    Returns a dict of the Brian2 objects, with the network under ``"net"``.
    """
    shared = {k: v for k, v in ns.items() if k not in ("E", "I", "IH")}
    n_e, n_i = weights["A"].shape[0], weights["B"].shape[0]
    reset_v = "v = V_reset"

    aud = SpikeGeneratorGroup(n_aud, [], [] * ms, dt=dt, name="aud")
    hvc = SpikeGeneratorGroup(n_hvc, [], [] * ms, dt=dt, name="hvc")

    # E and I are the paper's population names.
    E = NeuronGroup(n_e, E_EQS, threshold="v > V_th",
                    reset=(f"{reset_v}; I_adapt += beta; x_e += 1; r_loc += 1/tau_loc; "
                           "r_slow += 1/tau_slow"),
                    refractory="t_ref",
                    method="exponential_euler", namespace={**shared, **ns["E"]},
                    dt=dt, name="E")
    I = NeuronGroup(n_i, I_EQS, threshold="v > V_th",  # noqa: E741
                    reset=f"{reset_v}; x_stdp += 1", refractory="t_ref",
                    method="exponential_euler", namespace={**shared, **ns["I"]},
                    dt=dt, name="I")
    nmda = NeuronGroup(n_aud, NMDA_EQS, method="exponential_euler", namespace=shared,
                       dt=dt, name="nmda")
    E.v = shared["E_L"]
    I.v = shared["E_L"]

    aud_e = Synapses(aud, E, "w : siemens", on_pre="g_ampa_post += w; a_aud_post += 1",
                     delay=aud_delay, dt=dt, name="aud_e")
    _connect(aud_e, weights["A"])
    aud_nmda = Synapses(aud, nmda, on_pre="x_post += 1", delay=aud_delay, dt=dt,
                        name="aud_nmda")
    aud_nmda.connect(j="i")
    nmda_e = Synapses(nmda, E, "w : siemens\ng_nmda_post = w * s_pre : siemens (summed)",
                      dt=dt, name="nmda_e")
    _connect(nmda_e, weights["G"])
    if not nmda_e.active:      # no NMDA: the summed variable is never written
        E.g_nmda = 0 * nS

    hvc_i = Synapses(hvc, I, "w : siemens", on_pre="g_ampa_post += w", delay=hvc_delay,
                     dt=dt, name="hvc_i")
    _connect(hvc_i, weights["B"])
    e_i = Synapses(E, I, "w : siemens", on_pre="g_ampa_post += w", dt=dt, name="e_i")
    _connect(e_i, weights["J_EI"])
    i_e = Synapses(I, E, "w : siemens\nrho : siemens (shared)\nrho_dep : siemens (shared)",
                   on_pre=STDP_ON_PRE, on_post=STDP_ON_POST,
                   namespace=shared, dt=dt, name="i_e")
    _connect(i_e, weights["J_IE"], mask=weights["M_IE"])

    mon_e = SpikeMonitor(E, name="mon_e")
    mon_i = SpikeMonitor(I, name="mon_i")
    objs: dict[str, Any] = dict(aud=aud, hvc=hvc, E=E, I=I, nmda=nmda, aud_e=aud_e, aud_nmda=aud_nmda,
                nmda_e=nmda_e, hvc_i=hvc_i, e_i=e_i, i_e=i_e, mon_e=mon_e, mon_i=mon_i)

    # Optional second inhibitory population, I_HVC ("IH"): driven only by HVC (no E input),
    # a sparse temporal basis whose plastic synapses onto E learn the cancellation. Same
    # membrane and STDP rule as I; its own baseline ``theta_h``.
    if weights.get("B_H") is not None and weights["B_H"].shape[0] > 0:
        IH = NeuronGroup(weights["B_H"].shape[0], I_EQS, threshold="v > V_th",
                         reset=f"{reset_v}; x_stdp += 1", refractory="t_ref",
                         method="exponential_euler", namespace={**shared, **ns["IH"]},
                         dt=dt, name="IH")
        IH.v = shared["E_L"]
        hvc_ih = Synapses(hvc, IH, "w : siemens", on_pre="g_ampa_post += w",
                          delay=hvc_delay, dt=dt, name="hvc_ih")
        _connect(hvc_ih, weights["B_H"])
        if shared.get("vogels_h", False):
            gated = shared.get("vogels_gate", False)
            ih_e = Synapses(IH, E, "w : siemens\neta : siemens (shared)",
                            on_pre=VOGELS_ON_PRE_GATED if gated else VOGELS_ON_PRE,
                            on_post=VOGELS_ON_POST_GATED if gated else VOGELS_ON_POST,
                            namespace=shared, dt=dt, name="ih_e")
        else:
            ih_e = Synapses(IH, E,
                            "w : siemens\nrho : siemens (shared)\nrho_dep : siemens (shared)",
                            on_pre=STDP_ON_PRE, on_post=STDP_ON_POST,
                            namespace={**shared, "theta": shared["theta_h"]}, dt=dt,
                            name="ih_e")
        _connect(ih_e, weights["J_HE"], mask=weights["M_HE"])
        objs.update(IH=IH, hvc_ih=hvc_ih, ih_e=ih_e, mon_ih=SpikeMonitor(IH, name="mon_ih"))
    objs["net"] = Network(*objs.values())
    return objs


# ---------------------------------------------------------------------------
# Spike-array conversion
# ---------------------------------------------------------------------------

def spike_events(spikes: np.ndarray, bin_ms: float = 1.0):
    """``(n, T)`` spike counts on a ``bin_ms`` grid -> (indices, times) for a generator."""
    idx, t = np.nonzero(np.asarray(spikes) > 0)
    return idx.astype(int), t * bin_ms * ms


def bin_spikes(mon: SpikeMonitor, n: int, T: int, t0_ms: float = 0.0) -> np.ndarray:
    """A monitor's spikes as an ``(n, T)`` float32 0/1 array on the 1 ms grid from ``t0_ms``."""
    out = np.zeros((n, T), dtype=np.float32)
    t = np.floor(np.asarray(mon.t / ms) - t0_ms + 1e-9).astype(int)
    ok = (t >= 0) & (t < T)
    out[np.asarray(mon.i)[ok], t[ok]] = 1.0
    return out
