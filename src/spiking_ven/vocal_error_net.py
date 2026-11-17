"""Spiking E-I vocal error network (HVC -> I architecture).

Architecture
------------
- Auditory spikes  (n_aud, T)  →  B (n_e × n_aud)    →  E neurons  [fixed sparse]
- E neurons        →  JEI (n_i × n_e)                 →  I neurons  [fixed]
- HVC spikes       (n_hvc, T) →  B_hvc (n_i × n_hvc) →  I neurons  [fixed sparse]
- I neurons        →  JIE (n_e × n_i)                 →  E neurons  [plastic — causal Hebbian]
- Auditory spikes  also drive an NMDA conductance on E (``g_nmda_mean_ns``), on B's
  connections; see
  ``docs/MODEL.md``.

Plasticity
----------
Only JIE learns, by causal STDP, during ``fit()``. Everything else is fixed.

:class:`VocalErrorNet` defaults to the shipped operating point in
:mod:`spiking_ven.constants`, so ``VocalErrorNet(n_aud=encoder.n_channels)`` builds the
network the published model was trained as.

Synapses are conductances (AMPA, NMDA, GABA_A) acting through their reversal
potentials; see ``docs/MODEL.md``.
"""

import json

import numpy as np
from numpy.random import default_rng

from . import constants as C
from . import units as U
from .common import (
    _lognormal_sparse,
    generate_hvc_bursts,  # re-exported for convenience
    spike_to_rate,        # re-exported for convenience
)

__all__ = ["VocalErrorNet", "generate_hvc_bursts", "spike_to_rate",
           "nmda_gating", "mg_block", "stdp_audio_gate"]

#: NMDA parameters a model file stores.
_NMDA_KEYS = ("g_nmda_mean_ns", "nmda_tau_rise", "nmda_tau_decay", "nmda_alpha", "mg_mm",
              "e_nmda_mv")


# ---------------------------------------------------------------------------
# NMDA synapse
# ---------------------------------------------------------------------------

def nmda_gating(aud_spikes, tau_rise=C.NMDA_TAU_RISE, tau_decay=C.NMDA_TAU_DECAY,
                alpha=C.NMDA_ALPHA, dt=1.0):
    """NMDA gating variable per auditory channel, ``(n_aud, T)`` in [0, 1].

    ``dx/dt = -x / tau_rise + spikes`` and ``ds/dt = -s / tau_decay + alpha x (1 - s)``
    (Wang 2002), stepped exactly for the decay and the saturating increment, so ``s``
    stays in [0, 1] at any input rate.
    """
    x = spike_to_rate(aud_spikes, tau_rise, dt).astype(np.float64)
    keep = np.exp(-dt / tau_decay)
    s = np.zeros_like(x)
    prev = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        decayed = prev * keep
        prev = decayed + (1.0 - decayed) * (1.0 - np.exp(-alpha * x[:, t] * dt))
        s[:, t] = prev
    return s


#: The STDP audio gate new networks learn with; see :func:`stdp_audio_gate`.
STDP_GATE = "window"


def _decay(tau_ms: float, dt: float = 1.0) -> float:
    """Per-step decay of a synaptic trace; 0 for ``tau_ms <= 0`` (instantaneous)."""
    return float(np.exp(-dt / tau_ms)) if tau_ms > 0.0 else 0.0


def stdp_audio_gate(aud_spikes, B, window_ms: float) -> np.ndarray:
    """``(n_e, T)`` bool: E cell ``e`` received an auditory spike in ``(t - window_ms, t]``.

    The STDP audio gate. Only E spikes that fall inside it update ``JIE``, so inhibition
    is learned only by cells the song drives, and only while it drives them. It is
    defined on input spike events with the STDP window, not on the filtered synaptic
    drive: an exponential never returns to zero, so ``B @ h_aud > 0`` stayed open for
    the rest of the run after a cell's first input spike, whatever ``tau_s`` was.
    """
    aud = np.asarray(aud_spikes, dtype=np.float64)
    w = max(1, int(round(window_ms)))
    cs = np.concatenate([np.zeros((aud.shape[0], 1)), np.cumsum(aud, axis=1)], axis=1)
    t = np.arange(aud.shape[1])
    recent = cs[:, t + 1] - cs[:, np.maximum(t + 1 - w, 0)]     # spikes in (t - w, t]
    return (np.asarray(B, dtype=np.float64) @ recent) > 0.0


def mg_block(v_mv, mg_mm=C.MG_MM):
    """Fraction of NMDA conductance not blocked by Mg2+ at ``v_mv`` (Jahr & Stevens 1990)."""
    return 1.0 / (1.0 + mg_mm * np.exp(-0.062 * np.asarray(v_mv)) / 3.57)


# ---------------------------------------------------------------------------
# simulation loop
# ---------------------------------------------------------------------------

def _step(V, g_l, c, e_l, g_syn_e, I_ext):
    """Exponential-Euler membrane step over 1 ms, conductances held for the step.

    ``C dV/dt = -g_L (V - E_L) - sum_k g_k (V - E_k) + I_ext``: V relaxes toward
    ``V_inf = (g_L E_L + sum g_k E_k + I_ext) / g_tot`` with time constant ``C / g_tot``.
    Exact for constant conductances, and stable however large they get. ``g_syn_e`` is a
    list of (conductance, reversal) pairs, each conductance one value per cell.
    """
    g_tot = g_l + sum(g for g, _ in g_syn_e)
    v_inf = (g_l * e_l + sum(g * e for g, e in g_syn_e) + I_ext) / g_tot
    return v_inf + (V - v_inf) * np.exp(-g_tot / c)


def _simulate(
    h_aud, h_hvc, B, B_hvc, JEI, JIE,
    c_e, c_i, g_l_e, g_l_i, e_l, v_th, v_reset, e_ampa, e_gaba,
    i_drive_e, i_drive_i, noise_e, noise_i, t_refr_ms, a_sfa, tau_sfa,
    A_jie, xi_th,
    learn_weights, alpha_stdp, rng,
    aud_delay_ms: int = 0,
    hvc_delay_ms: int = 0,
    s_nmda: np.ndarray | None = None,
    G_nmda: np.ndarray | None = None,
    nmda: tuple | None = None,
    stdp_gate: np.ndarray | None = None,
    decay_inh: float = 0.0,
    decay_exc: float = 0.0,
):
    # Potentials in mV, currents in pA, conductances in nS, capacitances in pF (nS / pF =
    # 1 / ms). Every weight matrix is a peak conductance (nS) per unit of its trace;
    # ``G_nmda`` likewise. ``nmda`` is (e_nmda_mv, mg_mm). Membrane noise comes from the
    # caller's seeded generator, never the global np.random.
    n_e = B.shape[0]
    n_i = JEI.shape[0]
    T   = h_aud.shape[1]

    V_e     = np.full(n_e, float(e_l))
    V_i     = np.full(n_i, float(e_l))
    sE_prev = np.zeros(n_e)
    sI_prev = np.zeros(n_i)
    x_i     = np.zeros(n_i)
    # Recurrent synaptic traces, one per presynaptic cell: each spike adds 1, which
    # decays with tau_inh (I->E, GABA_A) or tau_exc (E->I, AMPA). A decay of 0 is a
    # one-step pulse: the previous step's spikes only.
    g_inh   = np.zeros(n_i)
    g_exc   = np.zeros(n_e)

    sE_mat  = np.zeros((n_e, T), dtype=np.float32)
    sI_mat  = np.zeros((n_i, T), dtype=np.float32)

    _zero_aud = np.zeros(h_aud.shape[0])
    _zero_hvc = np.zeros(h_hvc.shape[0])

    # Absolute refractory period, in 1 ms steps. Without it a cell can fire on every
    # consecutive bin (a 1000 Hz ceiling), which strong DAF drive reaches; the baseline
    # fires far slower, so the constraint clips only what is firing impossibly fast.
    refr_e = np.zeros(n_e)
    refr_i = np.zeros(n_i)

    # Spike-frequency adaptation on E: a hyperpolarising current (pA) that steps up by
    # ``a_sfa`` on every spike and decays with ``tau_sfa``. It limits the DAF response
    # where weakening the auditory drive cannot: adaptation bites hardest on the cells
    # firing most and barely touches a cell adding one spike, whereas a weaker drive
    # scales every cell alike. See docs/design-notes.md.
    w_sfa = np.zeros(n_e)
    decay_sfa = float(np.exp(-1.0 / tau_sfa)) if tau_sfa > 0.0 else 0.0

    for t in range(T):
        h_aud_t = h_aud[:, t - aud_delay_ms] if t >= aud_delay_ms else _zero_aud
        h_hvc_t = h_hvc[:, t - hvc_delay_ms] if t >= hvc_delay_ms else _zero_hvc

        # --- E dynamics: conductances (nS) through their reversals, currents (pA) ---
        g_inh = decay_inh * g_inh + sI_prev
        syn_e = [(B @ h_aud_t, e_ampa), (JIE @ g_inh, e_gaba)]
        if G_nmda is not None:
            # NMDA conductance through the same connections, Mg2+-blocked at the
            # membrane potential of the previous step.
            assert s_nmda is not None and nmda is not None, "G_nmda needs s_nmda, nmda"
            e_nmda, mg_mm = nmda
            s_t = s_nmda[:, t - aud_delay_ms] if t >= aud_delay_ms else _zero_aud
            syn_e.append(((G_nmda @ s_t) * mg_block(V_e, mg_mm), e_nmda))
        I_e = np.full(n_e, float(i_drive_e))
        if a_sfa > 0.0:
            w_sfa *= decay_sfa
            I_e = I_e - w_sfa
        if noise_e > 0.0:
            I_e += rng.normal(0.0, noise_e, n_e)
        # Countdown is decremented BEFORE the spike test, so a cell with t_refr_ms = 2
        # may fire again 2 ms later, not 3.
        if t_refr_ms > 0.0:
            refr_e = np.maximum(refr_e - 1.0, 0.0)
        V_e = _step(V_e, g_l_e, c_e, e_l, syn_e, I_e)
        if t_refr_ms > 0.0:
            V_e[refr_e > 0.0] = v_reset       # clamped while refractory, as in a LIF
        sE  = ((V_e > v_th) & (refr_e <= 0.0)).astype(np.float64)
        V_e[sE > 0.5] = v_reset
        if t_refr_ms > 0.0:
            refr_e[sE > 0.5] = t_refr_ms
        if a_sfa > 0.0:
            w_sfa += a_sfa * sE
        sE_mat[:, t] = sE

        # --- I dynamics ---
        g_exc = decay_exc * g_exc + sE_prev
        syn_i = [(JEI @ g_exc + B_hvc @ h_hvc_t, e_ampa)]
        I_i = np.full(n_i, float(i_drive_i))
        if noise_i > 0.0:
            I_i += rng.normal(0.0, noise_i, n_i)
        if t_refr_ms > 0.0:
            refr_i = np.maximum(refr_i - 1.0, 0.0)
        V_i = _step(V_i, g_l_i, c_i, e_l, syn_i, I_i)
        if t_refr_ms > 0.0:
            V_i[refr_i > 0.0] = v_reset
        sI  = ((V_i > v_th) & (refr_i <= 0.0)).astype(np.float64)
        V_i[sI > 0.5] = v_reset
        if t_refr_ms > 0.0:
            refr_i[sI > 0.5] = t_refr_ms
        sI_mat[:, t] = sI

        if learn_weights:
            # Causal STDP: I-then-E order (I precedes E -> potentiate JIE). Update at
            # each audio-gated E spike using the current I trace, which decays with
            # alpha_stdp (tau_stdp); xi_th debiases it above the I baseline. The audio
            # gate: an E spike counts only if the cell received an auditory input spike
            # within the last tau_stdp ms (stdp_audio_gate), read with the same
            # conduction delay as the drive.
            assert stdp_gate is not None, "learn_weights needs stdp_gate"
            gate_sE = sE * (stdp_gate[:, t - aud_delay_ms] if t >= aud_delay_ms
                            else 0.0)
            x_i = alpha_stdp * x_i + sI
            JIE += A_jie * np.outer(gate_sE, x_i - xi_th)
            np.maximum(JIE, 0.0, out=JIE)         # depression stops at zero

        sE_prev = sE
        sI_prev = sI

    return sE_mat, sI_mat, JIE


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

#: Scalar parameters a model file stores, all floats (units in the class docstring).
_FLOAT_KEYS = (
    "tau_e", "tau_i", "tau_s", "tau_inh", "tau_exc", "tau_stdp", "tau_sfa",
    "e_l_mv", "v_th_mv", "v_reset_mv", "c_e_pf", "c_i_pf", "e_ampa_mv", "e_gaba_mv",
    "i_drive_e_pa", "i_drive_i_pa", "noise_e_pa", "noise_i_pa", "a_sfa_pa",
    "c_B", "b_aud_mean_ns", "b_aud_sd_ns", "c_hvc", "b_hvc_mean_ns", "b_hvc_sd_ns",
    "c_JEI", "j_ei_mean_ns", "j_ei_sd_ns", "a_jie_ns", "r_i_th",
    "t_refr_ms",
) + _NMDA_KEYS
_INT_KEYS = ("n_e", "n_i", "n_hvc", "n_aud", "aud_delay_ms", "hvc_delay_ms")
_ARRAY_KEYS = ("B", "B_hvc", "JEI", "JIE")

#: Marks a file in these units with conductance synapses. Files with another marker, or
#: none, used current-based synapses and cannot be run by this version.
UNITS = "mV-pA-nS"
_LAST_CURRENT_BASED = "50f5304"


class VocalErrorNet:
    """Spiking LIF E-I vocal error network with causal Hebbian STDP on I→E.

    HVC premotor input drives I neurons; auditory input drives E neurons.
    I→E (JIE) strengthens when I precedes E (causal window only), implementing
    song-matched inhibition that cancels trained auditory patterns.

    Units: ms, mV, pA, nS, pF, Hz. The membrane is
    ``C dV/dt = -g_L (V - E_L) - sum_k g_k (V - E_k) + I`` with ``g_L = C / tau``: every
    synapse is a conductance acting through its reversal potential (AMPA and NMDA at
    ``e_ampa_mv`` / ``e_nmda_mv``, GABA_A at ``e_gaba_mv``), and ``I`` holds the tonic drive,
    adaptation and noise currents. Weight matrices ``B``, ``B_hvc``, ``JEI`` and ``JIE`` are
    the peak conductance (nS) one presynaptic spike opens.

    Parameters
    ----------
    n_e, n_i, n_hvc, n_aud : population sizes. ``n_aud`` has no default: it must equal
                   the encoder's ``n_channels``.
    tau_e, tau_i : membrane time constants (ms), at rest (leak only)
    tau_s        : aud->E and HVC->I synaptic decay (ms), AMPA. It plays no part in
                   learning: ``tau_stdp`` sets both the I trace and the audio gate.
    tau_inh      : I->E synaptic decay (ms), GABA_A; 0 = a one-step pulse
    tau_exc      : E->I synaptic decay (ms), AMPA; 0 = a one-step pulse
    e_l_mv, v_th_mv, v_reset_mv : rest, threshold and reset (mV), shared by E and I
    c_e_pf, c_i_pf : membrane capacitances (pF)
    e_ampa_mv, e_gaba_mv : AMPA and GABA_A reversal potentials (mV)
    target_rate_hz : rate (Hz) the tonic drive alone fires E at; sets ``i_drive_e_pa``
    i_drive_e_pa : tonic E current (pA); ``None`` derives it from ``target_rate_hz``
    i_drive_i_pa : tonic I current (pA)
    noise_e_pa, noise_i_pa : SD (pA) of the Gaussian current noise added every 1 ms step
    c_B, b_aud_mean_ns, b_aud_sd_ns : aud->E connection probability and log-normal peak
                   conductance mean and SD (nS)
    c_hvc, b_hvc_mean_ns, b_hvc_sd_ns : the same for HVC->I
    c_JEI, j_ei_mean_ns, j_ei_sd_ns : the same for E->I (fixed)
    a_jie_ns     : STDP step (nS); ``None`` is ``constants.a_jie_for(tau_stdp)``
    r_i_th       : I rate (Hz) the STDP trace is debiased against; x_i below it -> LTD
    tau_stdp     : STDP window (ms), > 0: the I trace's time constant and the audio
                   gate's width
    t_refr_ms    : absolute refractory period (ms); 0 disables it
    a_sfa_pa, tau_sfa : adaptation current step per E spike (pA) and its decay (ms)
    aud_delay_ms, hvc_delay_ms : conduction delays (ms)
    g_nmda_mean_ns : mean peak NMDA conductance (nS) per aud->E connection, at full
                   gating and unblocked; the matrix is ``B`` rescaled to this mean (0 = off)
    nmda_tau_rise, nmda_tau_decay, nmda_alpha : NMDA gating kinetics (ms, ms, 1/ms)
    mg_mm, e_nmda_mv : Mg2+ concentration (mM) and NMDA reversal potential (mV)
    seed         : RNG seed

    Every default is the shipped operating point (:mod:`spiking_ven.constants`).

    ``stimulus`` is a dict describing how the training stimuli were built (rate
    calibration, ISTA iterations, protocol timing, input checksums). It is empty on a new
    network, is saved with the model, and lets later evaluation rebuild the same stimuli.
    """

    def __init__(
        self,
        n_e: int = C.N_E,
        n_i: int = C.N_I,
        n_hvc: int = C.N_HVC,
        n_aud: int | None = None,
        tau_e: float = C.TAU_E,
        tau_i: float = C.TAU_I,
        tau_s: float = C.TAU_S,
        tau_inh: float = C.TAU_INH,
        tau_exc: float = C.TAU_EXC,
        e_l_mv: float = C.E_L_MV,
        v_th_mv: float = C.V_TH_MV,
        v_reset_mv: float = C.V_RESET_MV,
        c_e_pf: float = C.C_E_PF,
        c_i_pf: float = C.C_I_PF,
        e_ampa_mv: float = C.E_AMPA_MV,
        e_gaba_mv: float = C.E_GABA_MV,
        target_rate_hz: float = C.TARGET_RATE_E_HZ,
        i_drive_e_pa: float | None = None,
        i_drive_i_pa: float = C.I_DRIVE_I_PA,
        noise_e_pa: float = C.NOISE_E_PA,
        noise_i_pa: float = C.NOISE_I_PA,
        c_B: float = C.C_B,
        b_aud_mean_ns: float = C.B_AUD_MEAN_NS,
        b_aud_sd_ns: float = C.B_AUD_SD_NS,
        c_hvc: float = C.C_HVC,
        b_hvc_mean_ns: float = C.B_HVC_MEAN_NS,
        b_hvc_sd_ns: float = C.B_HVC_SD_NS,
        c_JEI: float = C.C_JEI,
        j_ei_mean_ns: float = C.J_EI_MEAN_NS,
        j_ei_sd_ns: float = C.J_EI_SD_NS,
        a_jie_ns: float | None = None,
        r_i_th: float = C.R_I_TH,
        tau_stdp: float = C.TAU_STDP,
        t_refr_ms: float = C.T_REFR_MS,
        a_sfa_pa: float = C.A_SFA_PA,
        tau_sfa: float = C.TAU_SFA,
        aud_delay_ms: int = C.AUD_DELAY_MS,
        hvc_delay_ms: int = C.HVC_DELAY_MS,
        g_nmda_mean_ns: float = C.G_NMDA_MEAN_NS,
        nmda_tau_rise: float = C.NMDA_TAU_RISE,
        nmda_tau_decay: float = C.NMDA_TAU_DECAY,
        nmda_alpha: float = C.NMDA_ALPHA,
        mg_mm: float = C.MG_MM,
        e_nmda_mv: float = C.E_NMDA_MV,
        seed: int | None = None,
    ) -> None:
        if n_aud is None:
            raise ValueError("n_aud is required: pass the encoder's n_channels")
        if tau_stdp <= 0.0:
            raise ValueError(f"tau_stdp must be > 0 ms (the STDP window), got {tau_stdp}")
        if not v_reset_mv < v_th_mv:
            raise ValueError(f"v_reset_mv ({v_reset_mv}) must be below v_th_mv ({v_th_mv})")
        rng = default_rng(seed)
        # Separate stream for simulation noise, so weight initialisation and the
        # simulation cannot interleave and perturb each other.
        self._sim_rng = default_rng(None if seed is None else seed + 10_000)
        self.n_e, self.n_i, self.n_hvc, self.n_aud = int(n_e), int(n_i), int(n_hvc), int(n_aud)
        self.tau_e, self.tau_i, self.tau_s = float(tau_e), float(tau_i), float(tau_s)
        self.tau_inh, self.tau_exc = float(tau_inh), float(tau_exc)
        self.e_l_mv, self.v_th_mv, self.v_reset_mv = float(e_l_mv), float(v_th_mv), float(v_reset_mv)
        self.c_e_pf, self.c_i_pf = float(c_e_pf), float(c_i_pf)
        self.e_ampa_mv, self.e_gaba_mv = float(e_ampa_mv), float(e_gaba_mv)
        self.i_drive_e_pa = float(
            i_drive_e_pa if i_drive_e_pa is not None else
            U.tonic_drive_pa(target_rate_hz, self.tau_e, self.g_l_e,
                             self.e_l_mv, self.v_th_mv, self.v_reset_mv))
        self.i_drive_i_pa = float(i_drive_i_pa)
        self.noise_e_pa, self.noise_i_pa = float(noise_e_pa), float(noise_i_pa)
        self.c_B, self.b_aud_mean_ns, self.b_aud_sd_ns = float(c_B), float(b_aud_mean_ns), float(b_aud_sd_ns)
        self.c_hvc, self.b_hvc_mean_ns, self.b_hvc_sd_ns = float(c_hvc), float(b_hvc_mean_ns), float(b_hvc_sd_ns)
        self.c_JEI, self.j_ei_mean_ns, self.j_ei_sd_ns = float(c_JEI), float(j_ei_mean_ns), float(j_ei_sd_ns)
        self.a_jie_ns = float(C.a_jie_for(tau_stdp) if a_jie_ns is None else a_jie_ns)
        self.r_i_th = float(r_i_th)
        self.tau_stdp = float(tau_stdp)
        #: Absolute refractory period (ms). 2 ms is the standard cortical value and caps
        #: the instantaneous rate at 500 Hz; 0 disables it (ceiling 1000 Hz).
        self.t_refr_ms = float(t_refr_ms)
        self.a_sfa_pa, self.tau_sfa = float(a_sfa_pa), float(tau_sfa)
        self.aud_delay_ms, self.hvc_delay_ms = int(aud_delay_ms), int(hvc_delay_ms)
        #: NMDA component of aud -> E (see constants.G_NMDA_MEAN_NS); off when 0.
        self.g_nmda_mean_ns = float(g_nmda_mean_ns)
        self.nmda_tau_rise  = float(nmda_tau_rise)
        self.nmda_tau_decay = float(nmda_tau_decay)
        self.nmda_alpha     = float(nmda_alpha)
        self.mg_mm          = float(mg_mm)
        self.e_nmda_mv      = float(e_nmda_mv)
        self.stimulus: dict = {}
        #: Which STDP audio gate this network learns with; see :func:`stdp_audio_gate`.
        self.stdp_gate: str = STDP_GATE

        # Fixed weights: sparse log-normal peak conductances (nS), drawn in this order.
        self.B = _lognormal_sparse((n_e, n_aud), c_B, b_aud_mean_ns, b_aud_sd_ns, rng)
        self.B_hvc = _lognormal_sparse((n_i, n_hvc), c_hvc, b_hvc_mean_ns, b_hvc_sd_ns, rng)
        self.JEI = _lognormal_sparse((n_i, n_e), c_JEI, j_ei_mean_ns, j_ei_sd_ns, rng)

        # I→E: zero-initialised. Starting from zero means only song-driven E neurons ever
        # get trained (via STDP in fit()), leaving novel-E rows at zero.
        self.JIE = np.zeros((n_e, n_i), dtype=np.float32)

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------

    @property
    def g_l_e(self) -> float:
        """E leak conductance (nS)."""
        return U.leak_conductance(self.c_e_pf, self.tau_e)

    @property
    def g_l_i(self) -> float:
        """I leak conductance (nS)."""
        return U.leak_conductance(self.c_i_pf, self.tau_i)

    def nmda_conductances(self) -> np.ndarray:
        """Peak NMDA conductance (nS) per aud->E connection: ``B`` rescaled so its mean is
        ``g_nmda_mean_ns`` (at full gating, before the Mg2+ block)."""
        B = np.asarray(self.B, dtype=np.float64)
        if self.g_nmda_mean_ns <= 0.0 or self.b_aud_mean_ns <= 0.0:
            return np.zeros_like(B)
        return B * (self.g_nmda_mean_ns / self.b_aud_mean_ns)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, hvc_spikes: np.ndarray, aud_spikes: np.ndarray, *,
            hvc_prelude: np.ndarray | None = None) -> np.ndarray:
        """Train on paired HVC+auditory input: updates JIE by STDP. Returns sE_mat.

        ``hvc_prelude`` is HVC activity run first, with no sound (introductory notes,
        say); the returned spikes start after it. No learning happens during it, because
        the STDP rule needs auditory drive.
        """
        sE, _sI = self._sim(hvc_spikes, aud_spikes, learn_weights=True,
                            hvc_prelude=hvc_prelude)
        return sE

    def transform(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        hvc_prelude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inference pass (no weight updates). Returns (n_e, T) float32 E spike matrix.

        Membrane noise is drawn from ``rng`` if given, otherwise from the network's own
        seeded stream. Pass a separate generator for side computations (monitoring during
        training, say) so they do not change the noise the next ``fit`` sees.
        ``hvc_prelude`` is HVC activity run first, with no sound; the returned spikes
        start after it, so they align with ``hvc_spikes``.
        """
        sE, _sI = self._sim(hvc_spikes, aud_spikes, learn_weights=False, rng=rng,
                            hvc_prelude=hvc_prelude)
        return sE

    def transform_all(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        hvc_prelude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference pass. Returns (sE_mat, sI_mat) both (n, T) float32; ``rng`` and
        ``hvc_prelude`` as in :meth:`transform`."""
        return self._sim(hvc_spikes, aud_spikes, learn_weights=False, rng=rng,
                         hvc_prelude=hvc_prelude)

    def save(self, path: str) -> None:
        np.savez(
            path,
            units=np.array(UNITS),
            **{k: getattr(self, k) for k in _ARRAY_KEYS},
            **{k: getattr(self, k) for k in _INT_KEYS},
            **{k: getattr(self, k) for k in _FLOAT_KEYS},
            stimulus_json=np.array(json.dumps(self.stimulus, sort_keys=True)),
            stdp_gate=np.array(self.stdp_gate),
        )

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "VocalErrorNet":
        """Load a model written by :meth:`save`.

        Raises ``ValueError`` naming every missing key, or for a model with current-based
        synapses (any earlier format), which this version cannot run: use spiking-ven at
        commit ``50f5304`` or earlier for those, or retrain. ``seed`` seeds the
        simulation noise.
        """
        d = np.load(path)
        units = str(d["units"]) if "units" in d else "dimensionless"
        if units != UNITS:
            raise ValueError(
                f"{path}: a model with current-based synapses ({units!r}); this version "
                f"has conductance synapses and cannot run it. Retrain, or use spiking-ven "
                f"at commit {_LAST_CURRENT_BASED} or earlier.")
        missing = [k for k in _INT_KEYS + _FLOAT_KEYS + _ARRAY_KEYS if k not in d]
        if missing:
            raise ValueError(f"{path}: not a complete VocalErrorNet model; missing keys "
                             f"{', '.join(missing)}")
        obj = cls.__new__(cls)
        obj._sim_rng = default_rng(None if seed is None else seed + 10_000)
        for k in _INT_KEYS:
            setattr(obj, k, int(d[k]))
        for k in _FLOAT_KEYS:
            setattr(obj, k, float(d[k]))
        for k in _ARRAY_KEYS:
            setattr(obj, k, d[k])
        obj.stimulus = json.loads(str(d["stimulus_json"])) if "stimulus_json" in d else {}
        obj.stdp_gate = str(d["stdp_gate"]) if "stdp_gate" in d else STDP_GATE
        return obj

    def to_brian_weights(self) -> dict:
        """Everything :mod:`spiking_ven.brian2_runtime` needs to run this network.

        Weight matrices (float64, post x pre) as peak conductances (nS), the NMDA
        conductances (nS, ``G_nmda``), and every scalar the dynamics use -- potentials
        and reversals (mV), capacitances (pF), leak conductances (nS), drives, noise and
        adaptation (pA), time constants, the refractory period and the conduction delays.
        """
        return {
            **{k: np.asarray(getattr(self, k), dtype=np.float64) for k in _ARRAY_KEYS},
            "G_nmda": self.nmda_conductances(),
            **{k: float(getattr(self, k)) for k in _FLOAT_KEYS},
            "g_l_e": self.g_l_e,
            "g_l_i": self.g_l_i,
            "aud_delay_ms": int(self.aud_delay_ms),
            "hvc_delay_ms": int(self.hvc_delay_ms),
            "n_e": int(self.n_e),
            "n_i": int(self.n_i),
            "n_aud": int(self.n_aud),
            "n_hvc": int(self.n_hvc),
        }

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _sim(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        learn_weights: bool = False,
        rng: np.random.Generator | None = None,
        hvc_prelude: np.ndarray | None = None,
    ):
        n_pre = 0 if hvc_prelude is None else int(np.asarray(hvc_prelude).shape[1])
        if n_pre:
            hvc_spikes = np.concatenate(
                [np.asarray(hvc_prelude, dtype=np.float32), np.asarray(hvc_spikes)], axis=1)
            aud_spikes = np.concatenate(
                [np.zeros((aud_spikes.shape[0], n_pre), dtype=np.float32),
                 np.asarray(aud_spikes)], axis=1)
        dt = 1.0
        # The STDP trace decays at tau_stdp; xi_th is calibrated to the same constant so
        # the I-baseline debiasing stays correct whatever the window.
        alpha_stdp = float(np.exp(-dt / self.tau_stdp))
        xi_th      = float(self.r_i_th * self.tau_stdp * 1e-3)

        h_hvc = spike_to_rate(hvc_spikes, self.tau_s, dt).astype(np.float64)
        h_aud = spike_to_rate(aud_spikes,  self.tau_s, dt).astype(np.float64)
        s_nmda = None
        if self.g_nmda_mean_ns > 0.0:
            s_nmda = nmda_gating(aud_spikes, self.nmda_tau_rise, self.nmda_tau_decay,
                                 self.nmda_alpha, dt)

        sE_mat, sI_mat, JIE = _simulate(
            h_aud, h_hvc,
            *(np.asarray(getattr(self, k), dtype=np.float64) for k in _ARRAY_KEYS),
            self.c_e_pf, self.c_i_pf, self.g_l_e, self.g_l_i,
            self.e_l_mv, self.v_th_mv, self.v_reset_mv, self.e_ampa_mv, self.e_gaba_mv,
            self.i_drive_e_pa, self.i_drive_i_pa, self.noise_e_pa, self.noise_i_pa,
            self.t_refr_ms, self.a_sfa_pa, self.tau_sfa,
            self.a_jie_ns, xi_th,
            learn_weights, alpha_stdp, self._sim_rng if rng is None else rng,
            aud_delay_ms=self.aud_delay_ms,
            hvc_delay_ms=self.hvc_delay_ms,
            s_nmda=s_nmda,
            G_nmda=self.nmda_conductances() if self.g_nmda_mean_ns > 0.0 else None,
            nmda=(self.e_nmda_mv, self.mg_mm),
            stdp_gate=(stdp_audio_gate(aud_spikes, self.B, self.tau_stdp)
                       if learn_weights else None),
            decay_inh=_decay(self.tau_inh), decay_exc=_decay(self.tau_exc),
        )

        if learn_weights:
            self.JIE = JIE.astype(np.float32)
        return sE_mat[:, n_pre:], sI_mat[:, n_pre:]
