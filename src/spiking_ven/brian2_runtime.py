"""Brian2 runtime for a trained vocal error network.

NeuronGroup/Synapses builders that run a :class:`~spiking_ven.vocal_error_net.VocalErrorNet`
trained in numpy inside a Brian2 network, so a host circuit can wire its E population
onward. Inference only: training happens in numpy, and every parameter here comes from
``ven.to_brian_weights()``.

This module requires the ``brian2`` extra::

    uv sync --extra brian2      # or: pip install "spiking-ven[brian2]"

It is deliberately NOT imported from :mod:`spiking_ven`, so importing the package never
pulls in Brian2. Typical use::

    from spiking_ven.brian2_runtime import build_ven_groups, build_ven_synapses

    w = VocalErrorNet.load(path).to_brian_weights()
    groups = build_ven_groups(w)
    syns = build_ven_synapses(groups, w, aud_group, hvc_group)

Downstream wiring of the E population into a circuit (E -> VTA_DA / VTA_GABA / VP) is
**not** here: those builders take a host model's population dicts, so they belong to the
circuit that owns them (see finchsim's ``aiv_synapses.py``). This package stops at the
error signal.
"""

import numpy as np

from brian2 import NeuronGroup, SpikeSource, Synapses
from brian2.units import ms, second

from .common import uniform_scalar

__all__ = [
    "build_ven_groups",
    "build_ven_synapses",
]

# ---------------------------------------------------------------------------
# VEN equation strings
# ---------------------------------------------------------------------------

# E neurons: dimensionless LIF with spike-frequency adaptation.
# g_aud decays with tau_s (matches numpy spike_to_rate filtering of aud input).
# g_inh decays with the model's tau_inh (GABA_A). A model with tau_inh = 0 has the numpy
# model's instantaneous one-step coupling, approximated here by a 1 ms decay.
# w_sfa steps up by a_sfa on every spike (in the reset) and decays with tau_sfa, as in
# the numpy loop. The membrane is clamped at v_reset during the refractory period.
#
# The noise term is added only when the amplitude is non-zero. A `sigma * xi`
# term with sigma == 0 contributes nothing but still draws a random number per
# neuron per timestep, which shifts the shared stream and so perturbs every
# other group's noise. Omitting it keeps a noiseless population exactly
# noiseless and leaves the rest of the network's draws untouched.
_EQS_VEN_E = """
dv/dt     = (-v + drive_e_ven + g_aud - g_inh - w_sfa{nmda}) / tau_e_ven{noise} : 1 (unless refractory)
dg_aud/dt = -g_aud / tau_s_ven : 1
dg_inh/dt = -g_inh / tau_inh_ven : 1
dw_sfa/dt = -w_sfa / tau_sfa_ven : 1
{nmda_vars}"""

# NMDA on aud -> E (Wang 2002), present only when g_nmda > 0. ``s_nmda`` is the summed
# B-weighted gating from the NMDA synapses; the Mg2+ block is evaluated at the potential in
# mV given by the same map as the numpy model (reset -> e_l_mv, theta_e -> v_th_mv), and
# ``e_nmda_u_ven`` is the reversal potential in membrane units.
_NMDA_TERM = (" + g_nmda_ven * s_nmda * (e_nmda_u_ven - v)"
              " / (1 + mg_mm_ven * exp(-0.062 * (e_l_mv_ven + v * mv_per_unit_ven)) / 3.57)")
_NMDA_VARS = """s_nmda    : 1
"""

# Per-synapse NMDA gating. Every synapse from one auditory channel sees the same spikes,
# so this equals the per-channel gating of the numpy model.
_EQS_NMDA_SYN = """
w_syn : 1
dx_nmda/dt = -x_nmda / tau_nmda_rise : 1 (clock-driven)
ds_gate/dt = -s_gate / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_gate) : 1 (clock-driven)
s_nmda_post = w_syn * s_gate : 1 (summed)
"""

# I neurons: dimensionless LIF. g_hvc filtered at tau_s (spike_to_rate); g_exc decays
# with the model's tau_exc (E→I, AMPA; 1 ms when the model has 0, as for g_inh).
_EQS_VEN_I = """
dv/dt     = (-v + drive_i_ven + g_hvc + g_exc) / tau_i_ven{noise} : 1 (unless refractory)
dg_hvc/dt = -g_hvc / tau_s_ven : 1
dg_exc/dt = -g_exc / tau_exc_ven : 1
"""



def _syn_tau(tau_ms: float) -> float:
    """A recurrent synapse's decay (ms); 0, the numpy one-step pulse, runs as 1 ms."""
    return float(tau_ms) if tau_ms > 0.0 else 1.0

def _eqs(template: str, noise_amp: float, sigma_name: str) -> str:
    """Fill in a population's equation, including the noise term only if it is live."""
    return template.format(noise=f" + {sigma_name} * xi" if noise_amp > 0.0 else "")


# ---------------------------------------------------------------------------
# VEN groups
# ---------------------------------------------------------------------------

def build_ven_groups(w: dict, name_prefix: str = "ven") -> dict:
    """Build the E and I LIF NeuronGroups of a trained network.

    ``w`` is ``VocalErrorNet.to_brian_weights()``. Returns ``{"e_group", "i_group"}``.

    Noise: VocalErrorNet adds ``noise_e`` per 1 ms step, so the SDE sigma is always
    calibrated to dt = 1 ms regardless of Brian2's defaultclock.dt:
    sigma = noise * sqrt(1 ms) / tau. Calibrating to the Brian2 dt instead would give
    10x smaller variance at dt = 0.1 ms and suppress the noise-driven crossings.
    """
    DT_NUMPY_S = 1e-3   # VocalErrorNet reference timestep (1 ms)
    sigma_e = w["noise_e"] * float(np.sqrt(DT_NUMPY_S)) / (w["tau_e"] * 1e-3)
    sigma_i = w["noise_i"] * float(np.sqrt(DT_NUMPY_S)) / (w["tau_i"] * 1e-3)
    refractory = w["t_refr_ms"] * ms

    nmda = float(w.get("g_nmda", 0.0)) > 0.0
    theta_e = uniform_scalar(w["theta_e"], "theta_e")
    mv_per_unit = (w["v_th_mv"] - w["e_l_mv"]) / theta_e if nmda else 0.0
    eqs_e = _EQS_VEN_E.replace("{nmda}", _NMDA_TERM if nmda else "").replace(
        "{nmda_vars}", _NMDA_VARS if nmda else "")
    e_group = NeuronGroup(
        w["n_e"],
        _eqs(eqs_e, w["noise_e"], "sigma_e_ven"),
        threshold="v > theta_e_ven",
        reset="v = v_reset_ven; w_sfa += a_sfa_ven",
        refractory=refractory,
        namespace={
            "tau_e_ven":   w["tau_e"] * ms,
            "tau_s_ven":   w["tau_s"] * ms,
            "tau_inh_ven": _syn_tau(w.get("tau_inh", 0.0)) * ms,
            "tau_sfa_ven": w["tau_sfa"] * ms,
            "a_sfa_ven":   w["a_sfa"],
            "drive_e_ven": w["drive_e"],
            "v_reset_ven": w["v_reset"],
            "theta_e_ven": theta_e,
            "sigma_e_ven": sigma_e / second**0.5,   # SDE noise amplitude [1/sqrt(s)]
            **({"g_nmda_ven": w["g_nmda"], "mg_mm_ven": w["mg_mm"],
                "e_l_mv_ven": w["e_l_mv"], "mv_per_unit_ven": mv_per_unit,
                "e_nmda_u_ven": (w["e_nmda_mv"] - w["e_l_mv"]) / mv_per_unit}
               if nmda else {}),
        },
        method="euler",
        name=f"{name_prefix}_e",
    )
    e_group.v = 0.0

    i_group = NeuronGroup(
        w["n_i"],
        _eqs(_EQS_VEN_I, w["noise_i"], "sigma_i_ven"),
        threshold="v > theta_i_ven",
        reset="v = v_reset_ven",
        refractory=refractory,
        namespace={
            "tau_i_ven":   w["tau_i"] * ms,
            "tau_s_ven":   w["tau_s"] * ms,
            "tau_exc_ven": _syn_tau(w.get("tau_exc", 0.0)) * ms,
            "drive_i_ven": w["drive_i"],
            "v_reset_ven": w["v_reset"],
            "theta_i_ven": w["theta_i"],
            "sigma_i_ven": sigma_i / second**0.5,   # SDE noise amplitude [1/sqrt(s)]
        },
        method="euler",
        name=f"{name_prefix}_i",
    )
    i_group.v = 0.0

    return {"e_group": e_group, "i_group": i_group}


# ---------------------------------------------------------------------------
# VEN synapses
# ---------------------------------------------------------------------------

def _wire(syn: Synapses, W: np.ndarray) -> None:
    """Connect ``syn`` from a dense ``(post, pre)`` weight matrix.

    Every VEN weight matrix is stored post-by-pre (the numpy model's convention), so
    the row index is the Brian2 target and the column index is the source.

    An all-zero matrix is wired as an inactive synapse rather than passed to
    ``connect()``: Brian2 reduces the empty index arrays and fails with a bare
    ``ValueError: zero-size array to reduction operation maximum``, which says nothing
    about which population was empty.
    """
    post_idx, pre_idx = np.nonzero(W)
    if pre_idx.size == 0:
        syn.active = False
        return
    syn.connect(i=pre_idx.astype(int), j=post_idx.astype(int))
    syn.w_syn[:] = W[post_idx, pre_idx]


def build_ven_synapses(
    groups: dict,
    w: dict,
    aud_group: SpikeSource,
    hvc_group: SpikeSource,
    name_prefix: str = "ven",
) -> list:
    """Wire the trained weights: aud→E (B), HVC→I (B_hvc), E→I (JEI), I→E (JIE).

    ``groups`` is from :func:`build_ven_groups`, ``w`` from ``to_brian_weights()``.
    ``aud_group`` must have ``w["n_aud"]`` neurons and ``hvc_group`` ``w["n_hvc"]``.

    The aud→E and HVC→I conduction delays are the ones the network was trained with,
    so the trained JIE weights see the same relative timing here that produced them --
    the causal I-before-E STDP window is built on that offset.

    With ``g_nmda > 0`` the NMDA component of aud→E is a fifth synapse object on the same
    connections, appended to the list.

    Returns ``[syn_b, syn_b_hvc, syn_jei, syn_jie]`` (plus ``syn_nmda``).
    """
    e_group = groups["e_group"]
    i_group = groups["i_group"]

    def _syn(src, tgt, target_var, W, delay_ms, name):
        syn = Synapses(src, tgt, "w_syn : 1", on_pre=f"{target_var}_post += w_syn",
                       delay=delay_ms * ms, name=f"{name_prefix}_{name}")
        _wire(syn, np.asarray(W, dtype=np.float64))
        return syn

    # Use w_syn (not w) to avoid collision with a host model's adaptation variable w.
    syns = [
        _syn(aud_group, e_group, "g_aud", w["B"], w["aud_delay_ms"], "b"),
        _syn(hvc_group, i_group, "g_hvc", w["B_hvc"], w["hvc_delay_ms"], "b_hvc"),
        _syn(e_group, i_group, "g_exc", w["JEI"], 0.0, "jei"),
        _syn(i_group, e_group, "g_inh", w["JIE"], 0.0, "jie"),
    ]
    if float(w.get("g_nmda", 0.0)) > 0.0:
        syn = Synapses(aud_group, e_group, _EQS_NMDA_SYN, on_pre="x_nmda += 1",
                       delay=w["aud_delay_ms"] * ms, method="euler",
                       namespace={"tau_nmda_rise": w["nmda_tau_rise"] * ms,
                                  "tau_nmda_decay": w["nmda_tau_decay"] * ms,
                                  "alpha_nmda": w["nmda_alpha"] / ms},
                       name=f"{name_prefix}_nmda")
        _wire(syn, np.asarray(w["B"], dtype=np.float64))
        syns.append(syn)
    return syns
