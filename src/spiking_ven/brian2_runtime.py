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

from brian2 import NeuronGroup, Synapses
from brian2.units import ms, second

__all__ = [
    "build_ven_groups",
    "build_ven_synapses",
]

# ---------------------------------------------------------------------------
# VEN equation strings
# ---------------------------------------------------------------------------

# E neurons: dimensionless LIF with per-neuron threshold and spike-frequency adaptation.
# g_aud decays with tau_s (matches numpy spike_to_rate filtering of aud input).
# g_inh decays with tau_inh (fast synaptic decay, ~1 ms, approximating the
# instantaneous JIE @ sI_prev coupling in the numpy model at dt=1 ms).
# w_sfa steps up by a_sfa on every spike (in the reset) and decays with tau_sfa, as in
# the numpy loop. The membrane is clamped at v_reset during the refractory period.
#
# The noise term is added only when the amplitude is non-zero. A `sigma * xi`
# term with sigma == 0 contributes nothing but still draws a random number per
# neuron per timestep, which shifts the shared stream and so perturbs every
# other group's noise. Omitting it keeps a noiseless population exactly
# noiseless and leaves the rest of the network's draws untouched.
_EQS_VEN_E = """
dv/dt     = (-v + drive_e_ven + g_aud - g_inh - w_sfa) / tau_e_ven{noise} : 1 (unless refractory)
dg_aud/dt = -g_aud / tau_s_ven : 1
dg_inh/dt = -g_inh / tau_inh_ven : 1
dw_sfa/dt = -w_sfa / tau_sfa_ven : 1
theta     : 1
"""

# I neurons: dimensionless LIF. g_hvc filtered at tau_s (spike_to_rate); g_exc fast
# decay (E→I coupling).
_EQS_VEN_I = """
dv/dt     = (-v + drive_i_ven + g_hvc + g_exc) / tau_i_ven{noise} : 1 (unless refractory)
dg_hvc/dt = -g_hvc / tau_s_ven : 1
dg_exc/dt = -g_exc / tau_exc_ven : 1
"""


def _eqs(template: str, noise_amp: float, sigma_name: str) -> str:
    """Fill in a population's equation, including the noise term only if it is live."""
    return template.format(noise=f" + {sigma_name} * xi" if noise_amp > 0.0 else "")


# ---------------------------------------------------------------------------
# VEN groups
# ---------------------------------------------------------------------------

def _check_supported(w: dict) -> None:
    """Refuse a model using a feature this port does not implement."""
    if float(w.get("g_nmda", 0.0)) > 0.0:
        raise NotImplementedError(
            "the NMDA component of aud -> E (g_nmda > 0) is not implemented in the Brian2 "
            "runtime; the network would run without it"
        )


def build_ven_groups(w: dict, name_prefix: str = "ven") -> dict:
    """Build the E and I LIF NeuronGroups of a trained network.

    ``w`` is ``VocalErrorNet.to_brian_weights()``. Returns ``{"e_group", "i_group"}``.

    Noise: VocalErrorNet adds ``noise_e`` per 1 ms step, so the SDE sigma is always
    calibrated to dt = 1 ms regardless of Brian2's defaultclock.dt:
    sigma = noise * sqrt(1 ms) / tau. Calibrating to the Brian2 dt instead would give
    10x smaller variance at dt = 0.1 ms and suppress the noise-driven crossings.
    """
    _check_supported(w)
    DT_NUMPY_S = 1e-3   # VocalErrorNet reference timestep (1 ms)
    sigma_e = w["noise_e"] * float(np.sqrt(DT_NUMPY_S)) / (w["tau_e"] * 1e-3)
    sigma_i = w["noise_i"] * float(np.sqrt(DT_NUMPY_S)) / (w["tau_i"] * 1e-3)
    refractory = w["t_refr_ms"] * ms

    e_group = NeuronGroup(
        w["n_e"],
        _eqs(_EQS_VEN_E, w["noise_e"], "sigma_e_ven"),
        threshold="v > theta",
        reset="v = v_reset_ven; w_sfa += a_sfa_ven",
        refractory=refractory,
        namespace={
            "tau_e_ven":   w["tau_e"] * ms,
            "tau_s_ven":   w["tau_s"] * ms,
            "tau_inh_ven": 1.0 * ms,   # fast synaptic decay for I→E inhibition
            "tau_sfa_ven": w["tau_sfa"] * ms,
            "a_sfa_ven":   w["a_sfa"],
            "drive_e_ven": w["drive_e"],
            "v_reset_ven": w["v_reset"],
            "sigma_e_ven": sigma_e / second**0.5,   # SDE noise amplitude [1/sqrt(s)]
        },
        method="euler",
        name=f"{name_prefix}_e",
    )
    e_group.v     = 0.0
    e_group.theta = w["theta_e"]

    i_group = NeuronGroup(
        w["n_i"],
        _eqs(_EQS_VEN_I, w["noise_i"], "sigma_i_ven"),
        threshold="v > theta_i_ven",
        reset="v = v_reset_ven",
        refractory=refractory,
        namespace={
            "tau_i_ven":   w["tau_i"] * ms,
            "tau_s_ven":   w["tau_s"] * ms,
            "tau_exc_ven": 1.0 * ms,   # fast synaptic decay for E→I excitation
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
    aud_group: NeuronGroup,
    hvc_group: NeuronGroup,
    name_prefix: str = "ven",
) -> list:
    """Wire the trained weights: aud→E (B), HVC→I (B_hvc), E→I (JEI), I→E (JIE).

    ``groups`` is from :func:`build_ven_groups`, ``w`` from ``to_brian_weights()``.
    ``aud_group`` must have ``w["n_aud"]`` neurons and ``hvc_group`` ``w["n_hvc"]``.

    The aud→E and HVC→I conduction delays are the ones the network was trained with,
    so the trained JIE weights see the same relative timing here that produced them --
    the causal I-before-E STDP window is built on that offset.

    Returns ``[syn_b, syn_b_hvc, syn_jei, syn_jie]``.
    """
    e_group = groups["e_group"]
    i_group = groups["i_group"]

    def _syn(src, tgt, target_var, W, delay_ms, name):
        syn = Synapses(src, tgt, "w_syn : 1", on_pre=f"{target_var}_post += w_syn",
                       delay=delay_ms * ms, name=f"{name_prefix}_{name}")
        _wire(syn, np.asarray(W, dtype=np.float64))
        return syn

    # Use w_syn (not w) to avoid collision with a host model's adaptation variable w.
    return [
        _syn(aud_group, e_group, "g_aud", w["B"], w["aud_delay_ms"], "b"),
        _syn(hvc_group, i_group, "g_hvc", w["B_hvc"], w["hvc_delay_ms"], "b_hvc"),
        _syn(e_group, i_group, "g_exc", w["JEI"], 0.0, "jei"),
        _syn(i_group, e_group, "g_inh", w["JIE"], 0.0, "jie"),
    ]
