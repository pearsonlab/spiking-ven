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
g_nmda : siemens
mg_block = 1 / (1 + Mg / (3.57*mM) * exp(-0.062 * v/mV)) : 1 (constant over dt)
gate = int(a_aud > exp(-1)) : 1
I_noise = sigma * randn() / sqrt(dt) : amp (constant over dt)
"""
# ``a_aud`` counts recent auditory input spikes, each decaying with tau_stdp; ``gate``
# is open while it exceeds 1/e, i.e. for tau_stdp after an isolated input spike.
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


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _connect(syn: Synapses, W: np.ndarray, *, dense: bool = False) -> None:
    """Connect ``syn`` from a ``(post, pre)`` weight matrix in nS and set ``syn.w``.

    Only nonzero entries become synapses unless ``dense``, which wires every pair (for a
    plastic matrix that starts at zero).
    """
    W = np.asarray(W, dtype=np.float64)
    post, pre = np.nonzero(np.ones_like(W) if dense else W)
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
    holds the ``(post, pre)`` matrices ``A``, ``G``, ``B``, ``J_EI``, ``J_IE`` in nS.
    Returns a dict of the Brian2 objects, with the network under ``"net"``.
    """
    shared = {k: v for k, v in ns.items() if k not in ("E", "I")}
    n_e, n_i = weights["A"].shape[0], weights["B"].shape[0]
    reset_v = "v = V_reset"

    aud = SpikeGeneratorGroup(n_aud, [], [] * ms, dt=dt, name="aud")
    hvc = SpikeGeneratorGroup(n_hvc, [], [] * ms, dt=dt, name="hvc")

    # E and I are the paper's population names.
    E = NeuronGroup(n_e, E_EQS, threshold="v > V_th",
                    reset=f"{reset_v}; I_adapt += beta", refractory="t_ref",
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
    _connect(i_e, weights["J_IE"], dense=True)

    mon_e = SpikeMonitor(E, name="mon_e")
    mon_i = SpikeMonitor(I, name="mon_i")
    objs: dict[str, Any] = dict(aud=aud, hvc=hvc, E=E, I=I, nmda=nmda, aud_e=aud_e, aud_nmda=aud_nmda,
                nmda_e=nmda_e, hvc_i=hvc_i, e_i=e_i, i_e=i_e, mon_e=mon_e, mon_i=mon_i)
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
