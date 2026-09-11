"""Brian2 runtime glue for the audio pipeline and vocal error network.

NeuronGroup/Synapse builders that mirror the numpy reference implementation in
:mod:`spiking_ven.vocal_error_net`. Parameter names and defaults are identical, so
values tuned against the numpy model transfer here without translation.

This module requires the ``brian2`` extra::

    uv sync --extra brian2      # or: pip install "spiking-ven[brian2]"

It is deliberately NOT imported from :mod:`spiking_ven`, so importing the package never
pulls in Brian2. Import it explicitly:

    from spiking_ven.brian2_runtime import build_ven_groups, build_ven_synapses

Architecture
------------
auditory spike train (from the numpy encoder, via a SpikeGeneratorGroup)
    -> build_ven_groups() / build_ven_synapses()   E/I LIF with STDP

Downstream wiring of the E population into a circuit (E -> VTA_DA / VTA_GABA / VP) is
**not** here: those builders take a host model's population dicts, so they belong to the
circuit that owns them (see finchsim's ``aiv_synapses.py``). This package stops at the
error signal.

The VEN builders are pure NeuronGroup + Synapses and compile to Brian2 standalone. An
earlier pair of ``network_operation``-based filterbank builders lived here; they had no
callers and could not compile to standalone, so they were removed.
"""

import numpy as np
from numpy.random import default_rng

from brian2 import NeuronGroup, Synapses
from brian2.units import ms, second

from .constants import AUD_DELAY_MS, HVC_DELAY_MS

__all__ = [
    "build_ven_groups",
    "build_ven_synapses",
]

# ---------------------------------------------------------------------------
# VEN equation strings
# ---------------------------------------------------------------------------

# E neurons: dimensionless LIF with per-neuron adaptive threshold.
# g_aud decays with tau_s (matches numpy spike_to_rate filtering of aud input).
# g_inh decays with tau_inh (fast synaptic decay, ~1 ms, approximating the
# instantaneous JIE @ sI_prev coupling in the numpy model at dt=1 ms).
#
# The noise term is added only when the amplitude is non-zero. A `sigma * xi`
# term with sigma == 0 contributes nothing but still draws a random number per
# neuron per timestep, which shifts the shared stream and so perturbs every
# other group's noise. Omitting it keeps a noiseless population exactly
# noiseless and leaves the rest of the network's draws untouched.
_EQS_VEN_E = """
dv/dt     = (-v + drive_e_ven + g_aud - g_inh) / tau_e_ven{noise} : 1
dg_aud/dt = -g_aud / tau_s_ven : 1
dg_inh/dt = -g_inh / tau_inh_ven : 1
theta     : 1
"""

# I neurons: dimensionless LIF.
# g_hvc filtered at tau_s (spike_to_rate); g_exc fast decay (E→I coupling);
# x_i exponential presynaptic trace for STDP.
#
# x_i is incremented in the I group's own `reset`, NOT in the JIE synapse's
# `on_pre`. It is a neuron-level variable, so an on_pre increment runs once per
# outgoing synapse rather than once per spike: the trace then scales with the
# number of E targets (measured ~60x at the tuned operating point) and the
# effective STDP rate scales with it. The numpy model's `x_i = alpha * x_i + sI`
# adds exactly 1 per spike, which is what the reset reproduces.
_EQS_VEN_I = """
dv/dt     = (-v + drive_i_ven + g_hvc + g_exc) / tau_i_ven{noise} : 1
dg_hvc/dt = -g_hvc / tau_s_ven : 1
dg_exc/dt = -g_exc / tau_exc_ven : 1
dx_i/dt   = -x_i / tau_s_ven : 1
"""


def _eqs(template: str, noise_amp: float, sigma_name: str) -> str:
    """Fill in a population's equation, including the noise term only if it is live."""
    return template.format(noise=f" + {sigma_name} * xi" if noise_amp > 0.0 else "")


# ---------------------------------------------------------------------------
# VEN groups
# ---------------------------------------------------------------------------

def build_ven_groups(
    N_e: int = 600,
    N_i: int = 150,
    tau_e: float = 30.0,
    tau_i: float = 10.0,
    tau_s: float = 10.0,
    drive_e: float = 0.155,
    drive_i: float = 0.058,
    noise_e: float = 0.1,
    noise_i: float = 0.0,
    theta_e: np.ndarray | None = None,
    theta_i: float | None = None,
    target_rate: float = 0.025,
    target_rate_i: float = 0.05,
    v_reset: float = 0.0,
    name_prefix: str = "ven",
) -> dict:
    """Build E and I LIF NeuronGroups for the vocal error network.

    Returns dict with keys 'e_group', 'i_group', 'tau_s', 'v_reset'.
    Parameter names and defaults match VocalErrorNetV2.__init__() exactly.

    noise_e, noise_i : per-step Gaussian noise amplitude (same units as v).
                       Converted to SDE sigma so per-step std matches the numpy
                       model. A zero amplitude omits the term entirely rather than
                       multiplying a wasted random draw by zero.
    theta_e: (N_e,) array of per-neuron E thresholds (from to_brian_weights()).
             If None, computed analytically from target_rate.
    theta_i: scalar I threshold.  If None, computed from target_rate_i.
    """
    # Noise calibration: VocalErrorNetV2 uses 1 ms timesteps, so sigma must
    # always be calibrated to dt=1ms regardless of Brian2 defaultclock.dt.
    # Per-ms variance: sigma^2 (SDE) = (noise_e/tau_e)^2 * dt_numpy (numpy AR(1)).
    # → sigma = noise_e * sqrt(dt_numpy) / tau_e_s  (units: 1/sqrt(s))
    # Using Brian2 dt here would give 10x smaller variance at dt=0.1ms,
    # suppressing all noise-driven threshold crossings.
    DT_NUMPY_S = 1e-3   # VocalErrorNetV2 reference timestep (1 ms)
    tau_e_s = tau_e * 1e-3
    tau_i_s = tau_i * 1e-3
    sigma_e_val = noise_e * float(np.sqrt(DT_NUMPY_S)) / tau_e_s  # 1/sqrt(s)
    sigma_i_val = noise_i * float(np.sqrt(DT_NUMPY_S)) / tau_i_s  # 1/sqrt(s)

    # These fallbacks must match VocalErrorNetV2.__init__ exactly, or a caller that
    # lets them default gets a different operating point from the numpy model.
    # Two traps, both of which this code previously fell into:
    #   * the E threshold is set by the tonic drive, not by target_rate * tau_s.
    #     tau_s is the STDP trace constant and must not appear in a membrane
    #     threshold (the old formula ran 61% high at the default parameters).
    #   * the I threshold is calibrated against a FIXED 10 ms trace reference, so
    #     that changing tau_s does not move theta_i.
    _TAU_S_REF = 10.0
    if theta_e is None:
        decay_e  = float(np.exp(-1.0 / tau_e))
        isi_e    = 1.0 / target_rate
        theta_e0 = float(max(drive_e * (1.0 - decay_e ** isi_e), 1e-3))
    else:
        theta_e0 = None   # will be set per-neuron below

    if theta_i is None:
        decay_i  = float(np.exp(-1.0 / tau_i))
        isi_i    = 1.0 / target_rate_i
        theta_i0 = float(max(target_rate_i * _TAU_S_REF * (1.0 - decay_i ** isi_i), 1e-3))
    else:
        theta_i0 = float(theta_i)

    ns_e = {
        "tau_e_ven":   tau_e * ms,
        "tau_s_ven":   tau_s * ms,
        "tau_inh_ven": 1.0 * ms,   # fast synaptic decay for I→E inhibition
        "drive_e_ven": drive_e,
        "v_reset_e":   v_reset,
        "sigma_e_ven": sigma_e_val / second**0.5,   # SDE noise amplitude [1/sqrt(s)]
    }
    e_group = NeuronGroup(
        N_e,
        _eqs(_EQS_VEN_E, noise_e, "sigma_e_ven"),
        threshold="v > theta",
        reset="v = v_reset_e",
        namespace=ns_e,
        method="euler",
        name=f"{name_prefix}_e",
    )
    e_group.v     = 0.0
    e_group.theta = theta_e if theta_e is not None else theta_e0
    e_group.g_aud = 0.0
    e_group.g_inh = 0.0

    ns_i = {
        "tau_i_ven":   tau_i * ms,
        "tau_s_ven":   tau_s * ms,
        "tau_exc_ven": 1.0 * ms,   # fast synaptic decay for E→I excitation
        "drive_i_ven": drive_i,
        "v_reset_i":   v_reset,
        "theta_i_ns":  theta_i0,
        "sigma_i_ven": sigma_i_val / second**0.5,   # SDE noise amplitude [1/sqrt(s)]
    }
    i_group = NeuronGroup(
        N_i,
        _eqs(_EQS_VEN_I, noise_i, "sigma_i_ven"),
        threshold="v > theta_i_ns",
        # x_i is the STDP presynaptic trace: +1 per spike, here and not in the
        # JIE synapse's on_pre (see the note above _EQS_VEN_I).
        reset="v = v_reset_i; x_i += 1",
        namespace=ns_i,
        method="euler",
        name=f"{name_prefix}_i",
    )
    i_group.v     = 0.0
    i_group.x_i   = 0.0
    i_group.g_hvc = 0.0
    i_group.g_exc = 0.0

    return {
        "e_group": e_group,
        "i_group": i_group,
        "tau_s":   tau_s,
        "v_reset": v_reset,
    }


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
    aud_group: NeuronGroup,
    hvc_group: NeuronGroup,
    B_weights: np.ndarray,
    B_hvc_weights: np.ndarray,
    JIE_weights: np.ndarray | None = None,
    JEI_weights: np.ndarray | None = None,
    learn: bool = True,
    A_jie: float = 5e-6,
    tau_s: float = 10.0,
    xi_th: float = 0.0,
    J_max_ie: float = 1.0,
    aud_delay: float = AUD_DELAY_MS,
    hvc_delay: float = HVC_DELAY_MS,
    name_prefix: str = "ven",
) -> list:
    """Build fixed (B, B_hvc, JEI) and optionally plastic (JIE) synapses for the VEN.

    Parameters
    ----------
    groups        : dict returned by build_ven_groups()
    aud_group     : auditory spike source → E neurons (n_aud neurons)
    hvc_group     : HVC spike source → I neurons (n_hvc neurons)
    B_weights     : (N_e, n_aud) sparse weight matrix for aud→E
    B_hvc_weights : (N_i, n_hvc) sparse weight matrix for HVC→I
    JIE_weights   : (N_e, N_i) I→E weight matrix; if None, random lognormal init
    JEI_weights   : (N_i, N_e) E→I weight matrix; if None, random lognormal init
    learn         : if True, JIE synapse includes STDP on_post (training mode);
                    if False, JIE is a fixed synapse — use after loading trained weights
                    via VocalErrorNetV2.to_brian_weights()
    A_jie         : STDP learning rate for I→E (causal Hebbian); ignored when learn=False
    tau_s         : I presynaptic trace time constant (ms); must match build_ven_groups
    xi_th         : STDP threshold on I trace (= r_i_th * tau_s * 1e-3)
    J_max_ie      : hard upper bound on JIE weights
    aud_delay     : conduction delay (ms) on the aud→E synapse.  Approximates the
                    multi-synaptic cochlea→L1→CM→AIV pathway.  Defaults to
                    ``constants.AUD_DELAY_MS``: AIV DAF response onset 23 ± 12 ms
                    (Mandelblat-Cerf 2014 T5).
    hvc_delay     : conduction delay (ms) on the HVC→I synapse.  Defaults to
                    ``constants.HVC_DELAY_MS`` (5 ms, estimated).  Both defaults are
                    the values VocalErrorNetV2 is trained with, so trained JIE weights
                    see the same relative timing here that produced them -- HVC drive
                    arriving ~18 ms before the auditory response is what the causal
                    I-before-E STDP window is built on. Overriding one without the
                    other changes that offset.
                    TODO: measure HVC-shelf → AIV-I latency directly (currently
                    unmeasured; Mandelblat-Cerf 2014 reports stimulation thresholds
                    only, not spike latencies); revisit this value once data exist.
    name_prefix   : prefix for Brian2 synapse names

    Returns
    -------
    list of Synapses objects: [syn_b, syn_b_hvc, syn_jei, syn_jie]

    Typical inference usage::

        w = trained_ven.to_brian_weights()
        syns = build_ven_synapses(
            groups, aud_group, hvc_group,
            B_weights=w["B"], B_hvc_weights=w["B_hvc"],
            JIE_weights=w["JIE"], JEI_weights=w["JEI"],
            learn=False, xi_th=w["xi_th"],
        )
    """
    e_group = groups["e_group"]
    i_group = groups["i_group"]

    # tau_s sets the decay of the x_i STDP trace, which lives on the I group and so is
    # fixed by build_ven_groups. Passing a different value here cannot change it, so
    # check rather than silently ignore the argument.
    if not np.isclose(tau_s, groups["tau_s"]):
        raise ValueError(
            f"tau_s={tau_s} does not match the groups' tau_s={groups['tau_s']}. The x_i "
            "trace decays at the value given to build_ven_groups; pass the same one here."
        )

    B       = np.asarray(B_weights,     dtype=np.float64)
    B_hvc   = np.asarray(B_hvc_weights, dtype=np.float64)
    N_e, N_aud  = B.shape
    N_i, N_hvc  = B_hvc.shape

    # --- B: aud → E (fixed sparse; increments g_aud which decays at tau_s) ---
    syn_b = Synapses(
        aud_group, e_group,
        "w_syn : 1",
        on_pre="g_aud_post += w_syn",
        delay=aud_delay * ms,
        name=f"{name_prefix}_b",
    )
    _wire(syn_b, B)          # (N_e, N_aud): rows=E (post), cols=aud (pre)

    # --- B_hvc: HVC → I (fixed sparse; increments g_hvc which decays at tau_s) ---
    # Use w_syn (not w) to avoid collision with HVC_X adaptation-current variable w.
    syn_b_hvc = Synapses(
        hvc_group, i_group,
        "w_syn : 1",
        on_pre="g_hvc_post += w_syn",
        delay=hvc_delay * ms,
        name=f"{name_prefix}_b_hvc",
    )
    _wire(syn_b_hvc, B_hvc)  # (N_i, N_hvc): rows=I (post), cols=hvc (pre)

    # --- JEI: E → I (fixed sparse) ---
    if JEI_weights is not None:
        JEI_full = np.asarray(JEI_weights, dtype=np.float64)  # (N_i, N_e)
    else:
        c_JEI    = float(np.count_nonzero(B_hvc)) / (N_i * N_hvc) if N_hvc > 0 else 0.5
        rng_jei  = default_rng(0)
        srKE     = float(np.sqrt(N_e * c_JEI))
        mean_jei = 1.7 / srKE / 10
        JEI_full = np.zeros((N_i, N_e), dtype=np.float64)
        mask_jei = rng_jei.random((N_i, N_e)) < c_JEI
        JEI_full[mask_jei] = rng_jei.lognormal(
            np.log(mean_jei) - 0.5 * np.log(1 + 0.01),
            np.sqrt(np.log(1 + 0.01)),
            size=int(mask_jei.sum()),
        )

    syn_jei = Synapses(
        e_group, i_group,
        "w_syn : 1",
        on_pre="g_exc_post += w_syn",
        name=f"{name_prefix}_jei",
    )
    _wire(syn_jei, JEI_full)  # (N_i, N_e): rows=I (post), cols=E (pre)

    # --- JIE: I → E ---
    if JIE_weights is not None:
        JIE_full = np.asarray(JIE_weights, dtype=np.float64)  # (N_e, N_i)
    else:
        c_JIE    = 0.5
        rng_jie  = default_rng(1)
        srKI     = float(np.sqrt(N_i * c_JIE))
        mean_jie = 1.0 / srKI / 10
        mask_jie = rng_jie.random((N_e, N_i)) < c_JIE
        JIE_full = np.zeros((N_e, N_i), dtype=np.float64)
        JIE_full[mask_jie] = rng_jie.lognormal(
            np.log(mean_jie) - 0.5 * np.log(1 + 0.01),
            np.sqrt(np.log(1 + 0.01)),
            size=int(mask_jie.sum()),
        )

    if learn:
        # Causal Hebbian STDP: the I spike advances x_i (in the I group's reset, so
        # once per spike); the E spike reads that trace and potentiates w_syn.
        syn_jie = Synapses(
            i_group, e_group,
            "w_syn : 1",
            on_pre="g_inh_post += w_syn",
            on_post="w_syn = clip(w_syn + A_jie_syn * (x_i_pre - xi_th_syn), 0, w_max_syn)",
            namespace={"A_jie_syn": A_jie, "xi_th_syn": xi_th, "w_max_syn": J_max_ie},
            name=f"{name_prefix}_jie",
        )
    else:
        # fixed inhibition: no plasticity, no trace update
        syn_jie = Synapses(
            i_group, e_group,
            "w_syn : 1",
            on_pre="g_inh_post += w_syn",
            name=f"{name_prefix}_jie",
        )

    _wire(syn_jie, JIE_full)  # (N_e, N_i): rows=E (post), cols=I (pre)

    return [syn_b, syn_b_hvc, syn_jei, syn_jie]
