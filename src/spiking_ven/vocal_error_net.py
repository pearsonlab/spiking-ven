"""Spiking E-I vocal error network (HVC -> I architecture).

Architecture
------------
- Auditory spikes  (n_aud, T)  →  B (n_e × n_aud)    →  E neurons  [fixed sparse]
- E neurons        →  JEI (n_i × n_e)                 →  I neurons  [fixed]
- HVC spikes       (n_hvc, T) →  B_hvc (n_i × n_hvc) →  I neurons  [fixed sparse]
- I neurons        →  JIE (n_e × n_i)                 →  E neurons  [plastic — causal Hebbian]
- Auditory spikes  also drive an NMDA conductance on E (``g_nmda``), through B; see
  ``docs/MODEL.md``.

Plasticity
----------
Only JIE learns, by causal STDP, during ``fit()``. Everything else is fixed.

:class:`VocalErrorNet` defaults to the shipped operating point in
:mod:`spiking_ven.constants`, so ``VocalErrorNet(n_aud=encoder.n_channels)`` builds the
network the published model was trained as. :class:`VocalErrorNetV2` is the earlier
name, kept with its earlier (non-operating-point) defaults so existing callers behave as
before; new code should use :class:`VocalErrorNet`.
"""

import json
from typing import Any

import numpy as np
from numpy.random import default_rng

from . import constants as C
from . import units as U
from .common import (
    _lognormal_sparse,
    generate_hvc_bursts,  # re-exported for convenience
    spike_to_rate,        # re-exported for convenience
    uniform_scalar,
)

__all__ = ["VocalErrorNet", "VocalErrorNetV2", "generate_hvc_bursts", "spike_to_rate",
           "nmda_gating", "mg_block", "stdp_audio_gate"]

#: NMDA parameters a model file stores.
_NMDA_KEYS = ("g_nmda", "nmda_tau_rise", "nmda_tau_decay", "nmda_alpha", "mg_mm",
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

def _simulate(
    h_aud, h_hvc, B, B_hvc, JEI, JIE,
    decay_e, decay_i, g_l_e, g_l_i, e_l, v_th, v_reset,
    i_drive_e, i_drive_i, noise_e, noise_i, t_refr_ms, a_sfa, tau_sfa,
    A_jie, J_max_ie, xi_th,
    learn_weights, alpha_stdp, rng,
    aud_delay_ms: int = 0,
    hvc_delay_ms: int = 0,
    s_nmda: np.ndarray | None = None,
    g_nmda: float = 0.0,
    nmda: tuple | None = None,
    stdp_gate: np.ndarray | None = None,
    decay_inh: float = 0.0,
    decay_exc: float = 0.0,
):
    # Potentials in mV, currents in pA, conductances in nS. Every weight matrix here is
    # the synaptic current (pA) per unit of its trace, i.e. the peak EPSC/IPSC per spike.
    # ``nmda`` is (e_nmda_mv, mg_mm). Membrane noise comes from the caller's seeded
    # generator, never the global np.random.
    n_e = B.shape[0]
    n_i = JEI.shape[0]
    T   = h_aud.shape[1]

    V_e     = np.full(n_e, float(e_l))
    V_i     = np.full(n_i, float(e_l))
    sE_prev = np.zeros(n_e)
    sI_prev = np.zeros(n_i)
    x_i     = np.zeros(n_i)
    # Recurrent synaptic traces, one per presynaptic cell: each spike adds 1, which
    # decays with tau_inh (I->E, GABA_A) or tau_exc (E->I, AMPA). A decay of 0 is the
    # former instantaneous coupling: the previous step's spikes only.
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

        # --- E dynamics (currents in pA) ---
        g_inh = decay_inh * g_inh + sI_prev
        I_e = i_drive_e + B @ h_aud_t - JIE @ g_inh
        if g_nmda > 0.0:
            # NMDA conductance through the same connections, Mg2+-blocked at the
            # membrane potential, with its driving force normalised to 1 at rest.
            assert s_nmda is not None and nmda is not None, "g_nmda > 0 needs s_nmda, nmda"
            e_nmda, mg_mm = nmda
            s_t = s_nmda[:, t - aud_delay_ms] if t >= aud_delay_ms else _zero_aud
            I_e = I_e + (g_nmda * (B @ s_t) * mg_block(V_e, mg_mm)
                         * (e_nmda - V_e) / (e_nmda - e_l))
        if a_sfa > 0.0:
            w_sfa *= decay_sfa
            I_e = I_e - w_sfa
        if noise_e > 0.0:
            I_e += rng.normal(0.0, noise_e, n_e)
        # Countdown is decremented BEFORE the spike test, so a cell with t_refr_ms = 2
        # may fire again 2 ms later, not 3.
        if t_refr_ms > 0.0:
            refr_e = np.maximum(refr_e - 1.0, 0.0)
        V_e = e_l + decay_e * (V_e - e_l) + (1.0 - decay_e) * I_e / g_l_e
        if t_refr_ms > 0.0:
            V_e[refr_e > 0.0] = v_reset       # clamped while refractory, as in a LIF
        sE  = ((V_e > v_th) & (refr_e <= 0.0)).astype(np.float64)
        V_e[sE > 0.5] = v_reset
        if t_refr_ms > 0.0:
            refr_e[sE > 0.5] = t_refr_ms
        if a_sfa > 0.0:
            w_sfa += a_sfa * sE
        sE_mat[:, t] = sE

        # --- I dynamics (currents in pA) ---
        g_exc = decay_exc * g_exc + sE_prev
        I_i = i_drive_i + JEI @ g_exc + B_hvc @ h_hvc_t
        if noise_i > 0.0:
            I_i += rng.normal(0.0, noise_i, n_i)
        if t_refr_ms > 0.0:
            refr_i = np.maximum(refr_i - 1.0, 0.0)
        V_i = e_l + decay_i * (V_i - e_l) + (1.0 - decay_i) * I_i / g_l_i
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
            np.clip(JIE, 0.0, J_max_ie, out=JIE)

        sE_prev = sE
        sI_prev = sI

    return sE_mat, sI_mat, JIE


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

#: Scalar parameters a model file stores, all floats (units in the class docstring).
_FLOAT_KEYS = (
    "tau_e", "tau_i", "tau_s", "tau_inh", "tau_exc", "tau_stdp", "tau_sfa",
    "e_l_mv", "v_th_mv", "v_reset_mv", "c_e_pf", "c_i_pf",
    "i_drive_e_pa", "i_drive_i_pa", "noise_e_pa", "noise_i_pa", "a_sfa_pa",
    "c_B", "b_aud_mean_mv", "b_aud_sd_mv", "c_hvc", "b_hvc_mean_mv", "b_hvc_sd_mv",
    "c_JEI", "j_ei_mean_mv", "j_ei_sd_mv", "j_max_ie_mv", "a_jie_mv", "r_i_th",
    "t_refr_ms",
) + _NMDA_KEYS
_INT_KEYS = ("n_e", "n_i", "n_hvc", "n_aud", "aud_delay_ms", "hvc_delay_ms")
_ARRAY_KEYS = ("B", "B_hvc", "JEI", "JIE")

#: Marks a file in these units. Files without it are the earlier dimensionless format.
UNITS = "mV-pA"


class VocalErrorNet:
    """Spiking LIF E-I vocal error network with causal Hebbian STDP on I→E.

    HVC premotor input drives I neurons; auditory input drives E neurons.
    I→E (JIE) strengthens when I precedes E (causal window only), implementing
    song-matched inhibition that cancels trained auditory patterns.

    Units: ms, mV, pA, pF, Hz. The membrane is ``C dV/dt = -g_L (V - E_L) + I`` with
    ``g_L = C / tau`` (see :mod:`spiking_ven.units`). Weight matrices ``B``, ``B_hvc``,
    ``JEI`` and ``JIE`` are stored as the peak PSP (mV) one presynaptic spike produces at
    rest; the simulation turns them into synaptic currents (pA).

    Parameters
    ----------
    n_e, n_i, n_hvc, n_aud : population sizes. ``n_aud`` has no default: it must equal
                   the encoder's ``n_channels``.
    tau_e, tau_i : membrane time constants (ms)
    tau_s        : aud->E and HVC->I synaptic decay (ms), AMPA. It plays no part in
                   learning: ``tau_stdp`` sets both the I trace and the audio gate.
    tau_inh      : I->E synaptic decay (ms), GABA_A; 0 = instantaneous (one-step pulse)
    tau_exc      : E->I synaptic decay (ms), AMPA; 0 = instantaneous
    e_l_mv, v_th_mv, v_reset_mv : rest, threshold and reset (mV), shared by E and I
    c_e_pf, c_i_pf : membrane capacitances (pF)
    target_rate_hz : rate (Hz) the tonic drive alone fires E at; sets ``i_drive_e_pa``
    i_drive_e_pa : tonic E current (pA); ``None`` derives it from ``target_rate_hz``
    i_drive_i_pa : tonic I current (pA)
    noise_e_pa, noise_i_pa : SD (pA) of the Gaussian current noise added every 1 ms step
    c_B, b_aud_mean_mv, b_aud_sd_mv : aud->E connection probability and log-normal weight
                   mean and SD (peak EPSP, mV)
    c_hvc, b_hvc_mean_mv, b_hvc_sd_mv : the same for HVC->I
    c_JEI, j_ei_mean_mv, j_ei_sd_mv : the same for E->I (fixed)
    j_max_ie_mv  : upper bound on each I->E weight (peak IPSP, mV)
    a_jie_mv     : STDP step (peak IPSP, mV); ``None`` is ``constants.a_jie_for(tau_stdp)``
    r_i_th       : I rate (Hz) the STDP trace is debiased against; x_i below it -> LTD
    tau_stdp     : STDP window (ms), > 0: the I trace's time constant and the audio
                   gate's width
    t_refr_ms    : absolute refractory period (ms); 0 disables it
    a_sfa_pa, tau_sfa : adaptation current step per E spike (pA) and its decay (ms)
    aud_delay_ms, hvc_delay_ms : conduction delays (ms)
    g_nmda       : NMDA current at full gating, at rest and unblocked, relative to the
                   fast EPSC (0 = off)
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
        target_rate_hz: float = C.TARGET_RATE_E_HZ,
        i_drive_e_pa: float | None = None,
        i_drive_i_pa: float = C.I_DRIVE_I_PA,
        noise_e_pa: float = C.NOISE_E_PA,
        noise_i_pa: float = C.NOISE_I_PA,
        c_B: float = C.C_B,
        b_aud_mean_mv: float = C.B_AUD_MEAN_MV,
        b_aud_sd_mv: float = C.B_AUD_SD_MV,
        c_hvc: float = C.C_HVC,
        b_hvc_mean_mv: float = C.B_HVC_MEAN_MV,
        b_hvc_sd_mv: float = C.B_HVC_SD_MV,
        c_JEI: float = C.C_JEI,
        j_ei_mean_mv: float = C.J_EI_MEAN_MV,
        j_ei_sd_mv: float = C.J_EI_SD_MV,
        j_max_ie_mv: float = C.J_MAX_IE_MV,
        a_jie_mv: float | None = None,
        r_i_th: float = C.R_I_TH,
        tau_stdp: float = C.TAU_STDP,
        t_refr_ms: float = C.T_REFR_MS,
        a_sfa_pa: float = C.A_SFA_PA,
        tau_sfa: float = C.TAU_SFA,
        aud_delay_ms: int = C.AUD_DELAY_MS,
        hvc_delay_ms: int = C.HVC_DELAY_MS,
        g_nmda: float = C.G_NMDA,
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
        self.i_drive_e_pa = float(
            i_drive_e_pa if i_drive_e_pa is not None else
            U.tonic_drive_pa(target_rate_hz, self.tau_e, self.g_l_e,
                             self.e_l_mv, self.v_th_mv, self.v_reset_mv))
        self.i_drive_i_pa = float(i_drive_i_pa)
        self.noise_e_pa, self.noise_i_pa = float(noise_e_pa), float(noise_i_pa)
        self.c_B, self.b_aud_mean_mv, self.b_aud_sd_mv = float(c_B), float(b_aud_mean_mv), float(b_aud_sd_mv)
        self.c_hvc, self.b_hvc_mean_mv, self.b_hvc_sd_mv = float(c_hvc), float(b_hvc_mean_mv), float(b_hvc_sd_mv)
        self.c_JEI, self.j_ei_mean_mv, self.j_ei_sd_mv = float(c_JEI), float(j_ei_mean_mv), float(j_ei_sd_mv)
        self.j_max_ie_mv = float(j_max_ie_mv)
        self.a_jie_mv = float(C.a_jie_for(tau_stdp) if a_jie_mv is None else a_jie_mv)
        self.r_i_th = float(r_i_th)
        self.tau_stdp = float(tau_stdp)
        #: Absolute refractory period (ms). 2 ms is the standard cortical value and caps
        #: the instantaneous rate at 500 Hz; 0 disables it (ceiling 1000 Hz).
        self.t_refr_ms = float(t_refr_ms)
        self.a_sfa_pa, self.tau_sfa = float(a_sfa_pa), float(tau_sfa)
        self.aud_delay_ms, self.hvc_delay_ms = int(aud_delay_ms), int(hvc_delay_ms)
        #: NMDA component of aud -> E (see constants.G_NMDA); off when g_nmda is 0.
        self.g_nmda         = float(g_nmda)
        self.nmda_tau_rise  = float(nmda_tau_rise)
        self.nmda_tau_decay = float(nmda_tau_decay)
        self.nmda_alpha     = float(nmda_alpha)
        self.mg_mm          = float(mg_mm)
        self.e_nmda_mv      = float(e_nmda_mv)
        self.stimulus: dict = {}
        #: Which STDP audio gate this network learns with. "window" is
        #: :func:`stdp_audio_gate`; models saved before it load as "drive" (the old
        #: ``B @ h_aud > 0``) and can be evaluated but not trained further.
        self.stdp_gate: str = STDP_GATE

        # Fixed weights: sparse log-normal, peak PSP (mV), drawn in this order.
        self.B = _lognormal_sparse((n_e, n_aud), c_B, b_aud_mean_mv, b_aud_sd_mv, rng)
        self.B_hvc = _lognormal_sparse((n_i, n_hvc), c_hvc, b_hvc_mean_mv, b_hvc_sd_mv, rng)
        self.JEI = _lognormal_sparse((n_i, n_e), c_JEI, j_ei_mean_mv, j_ei_sd_mv, rng)

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

    def synaptic_currents(self) -> dict[str, np.ndarray]:
        """Each weight matrix as the synaptic current (pA) one spike produces at its peak."""
        f = self._current_factors()
        return {k: np.asarray(getattr(self, k), dtype=np.float64) * f[k] for k in _ARRAY_KEYS}

    def _current_factors(self) -> dict[str, float]:
        """pA of peak synaptic current per mV of peak PSP, per connection."""
        return {
            "B": U.current_per_psp(self.tau_s, self.tau_e, self.g_l_e),
            "B_hvc": U.current_per_psp(self.tau_s, self.tau_i, self.g_l_i),
            "JEI": U.current_per_psp(self.tau_exc, self.tau_i, self.g_l_i),
            "JIE": U.current_per_psp(self.tau_inh, self.tau_e, self.g_l_e),
        }

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
        """Load a model written by :meth:`save`, or by the earlier dimensionless format.

        Raises ``ValueError`` naming every missing key. A file without the ``units`` marker
        is the earlier dimensionless format, converted on load (:func:`_load_legacy`) so
        it keeps the dynamics it was trained with. ``seed`` seeds the simulation noise.
        """
        d = np.load(path)
        if "units" not in d:
            return _load_legacy(cls, d, path, seed)
        if str(d["units"]) != UNITS:
            raise ValueError(f"{path}: unknown units {str(d['units'])!r}")
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

        Weight matrices (float64, post x pre) as synaptic currents (pA per spike, at the
        peak), and every scalar the dynamics use -- potentials (mV), leak conductances
        (nS), drives, noise and adaptation (pA), time constants, the refractory period and
        the conduction delays, without which the Brian2 network would not be the one that
        was trained.
        """
        return {
            **self.synaptic_currents(),
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
        if learn_weights and self.stdp_gate != STDP_GATE:
            raise ValueError(
                f"this model was trained with the {self.stdp_gate!r} STDP gate, which no "
                f"longer exists; retrain it from scratch (the gate is now {STDP_GATE!r})")
        n_pre = 0 if hvc_prelude is None else int(np.asarray(hvc_prelude).shape[1])
        if n_pre:
            hvc_spikes = np.concatenate(
                [np.asarray(hvc_prelude, dtype=np.float32), np.asarray(hvc_spikes)], axis=1)
            aud_spikes = np.concatenate(
                [np.zeros((aud_spikes.shape[0], n_pre), dtype=np.float32),
                 np.asarray(aud_spikes)], axis=1)
        dt      = 1.0
        decay_e = float(np.exp(-dt / self.tau_e))
        decay_i = float(np.exp(-dt / self.tau_i))
        # The STDP trace decays at tau_stdp; xi_th is calibrated to the same constant so
        # the I-baseline debiasing stays correct whatever the window.
        alpha_stdp   = float(np.exp(-dt / self.tau_stdp))
        xi_th        = float(self.r_i_th * self.tau_stdp * 1e-3)

        h_hvc = spike_to_rate(hvc_spikes, self.tau_s, dt).astype(np.float64)
        h_aud = spike_to_rate(aud_spikes,  self.tau_s, dt).astype(np.float64)
        s_nmda = None
        if self.g_nmda > 0.0:
            s_nmda = nmda_gating(aud_spikes, self.nmda_tau_rise, self.nmda_tau_decay,
                                 self.nmda_alpha, dt)

        # Learning runs on the I->E current (pA); JIE is stored as peak IPSP (mV).
        f = self._current_factors()
        cur = self.synaptic_currents()
        sE_mat, sI_mat, JIE_pa = _simulate(
            h_aud, h_hvc, cur["B"], cur["B_hvc"], cur["JEI"], cur["JIE"],
            decay_e, decay_i, self.g_l_e, self.g_l_i,
            self.e_l_mv, self.v_th_mv, self.v_reset_mv,
            self.i_drive_e_pa, self.i_drive_i_pa, self.noise_e_pa, self.noise_i_pa,
            self.t_refr_ms, self.a_sfa_pa, self.tau_sfa,
            self.a_jie_mv * f["JIE"], self.j_max_ie_mv * f["JIE"], xi_th,
            learn_weights, alpha_stdp, self._sim_rng if rng is None else rng,
            aud_delay_ms=self.aud_delay_ms,
            hvc_delay_ms=self.hvc_delay_ms,
            s_nmda=s_nmda, g_nmda=self.g_nmda, nmda=(self.e_nmda_mv, self.mg_mm),
            stdp_gate=(stdp_audio_gate(aud_spikes, self.B, self.tau_stdp)
                       if learn_weights else None),
            decay_inh=_decay(self.tau_inh), decay_exc=_decay(self.tau_exc),
        )

        if learn_weights:
            self.JIE = (JIE_pa / f["JIE"]).astype(np.float32)
        return sE_mat[:, n_pre:], sI_mat[:, n_pre:]


# ---------------------------------------------------------------------------
# the earlier dimensionless format
# ---------------------------------------------------------------------------

#: Keys of the earlier dimensionless file format, and the defaults of keys added to it
#: later: each default switches its feature off, so such a model reloads with the
#: dynamics it was trained with. ``tau_stdp`` is not listed: a file without it (or with
#: 0, the former "reuse the synaptic window") loads it as its own ``tau_s``.
LEGACY_REQUIRED_KEYS = (
    "n_e", "n_i", "n_hvc", "n_aud",
    "tau_e", "tau_i", "tau_s", "A_jie", "J_max_ie", "c_B", "c_hvc", "c_JEI", "r_i_th",
    "drive_e", "drive_i", "noise_e", "noise_i", "theta_e", "theta_i",
    "B", "B_hvc", "JEI", "JIE",
)
LEGACY_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "t_refr_ms": 0.0,
    "a_sfa": 0.0,
    "tau_sfa": 0.0,
    "tau_inh": 0.0,
    "tau_exc": 0.0,
    "aud_delay_ms": 0,
    "hvc_delay_ms": 0,
    "g_nmda": 0.0,
    "nmda_tau_rise": C.NMDA_TAU_RISE,
    "nmda_tau_decay": C.NMDA_TAU_DECAY,
    "nmda_alpha": C.NMDA_ALPHA,
    "mg_mm": C.MG_MM,
    "e_nmda_mv": C.E_NMDA_MV,
    "e_l_mv": C.E_L_MV,
    "v_th_mv": C.V_TH_MV,
    "stimulus_json": {},
    "stdp_gate": "drive",
}


def _load_legacy(cls, d, path, seed):
    """A dimensionless model file, converted to mV / pA (:func:`units.from_dimensionless`).

    Weight statistics (mean / SD) are not stored in that format; they are recorded as
    the converted matrices' own nonzero mean and SD.
    """
    missing = [k for k in LEGACY_REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"{path}: not a complete VocalErrorNet model; missing keys "
                         f"{', '.join(missing)}")
    opt = LEGACY_OPTIONAL_DEFAULTS
    p = {k: uniform_scalar(d[k], k) for k in LEGACY_REQUIRED_KEYS
         if k not in ("n_e", "n_i", "n_hvc", "n_aud", "B", "B_hvc", "JEI", "JIE")}
    for k in ("t_refr_ms", "a_sfa", "tau_sfa", "tau_inh", "tau_exc") + _NMDA_KEYS + (
            "e_l_mv", "v_th_mv"):
        p[k] = float(d[k]) if k in d else opt[k]
    saved_stdp = float(d["tau_stdp"]) if "tau_stdp" in d else 0.0
    p["tau_stdp"] = saved_stdp if saved_stdp > 0.0 else p["tau_s"]
    new = U.from_dimensionless(p, C.C_E_PF, C.C_I_PF)
    wf = new.pop("w_factor")

    obj = cls.__new__(cls)
    obj._sim_rng = default_rng(None if seed is None else seed + 10_000)
    for k in ("n_e", "n_i", "n_hvc", "n_aud"):
        setattr(obj, k, int(d[k]))
    for k in ("aud_delay_ms", "hvc_delay_ms"):
        setattr(obj, k, int(d[k]) if k in d else opt[k])
    for k in ("tau_e", "tau_i", "tau_s", "tau_inh", "tau_exc", "tau_stdp", "tau_sfa",
              "r_i_th", "t_refr_ms", "c_B", "c_hvc", "c_JEI", "nmda_tau_rise",
              "nmda_tau_decay", "nmda_alpha", "mg_mm", "e_nmda_mv"):
        setattr(obj, k, float(p[k]))
    for k, v in new.items():
        setattr(obj, k, float(v))
    for k in _ARRAY_KEYS:
        setattr(obj, k, (np.asarray(d[k], dtype=np.float64) * wf[k]).astype(np.float32))
    for k, (mk, sk) in {"B": ("b_aud_mean_mv", "b_aud_sd_mv"),
                        "B_hvc": ("b_hvc_mean_mv", "b_hvc_sd_mv"),
                        "JEI": ("j_ei_mean_mv", "j_ei_sd_mv")}.items():
        nz = np.asarray(getattr(obj, k))[np.asarray(getattr(obj, k)) > 0]
        setattr(obj, mk, float(nz.mean()) if nz.size else 0.0)
        setattr(obj, sk, float(nz.std()) if nz.size else 0.0)
    obj.stimulus = (json.loads(str(d["stimulus_json"])) if "stimulus_json" in d
                    else dict(opt["stimulus_json"]))
    obj.stdp_gate = str(d["stdp_gate"]) if "stdp_gate" in d else opt["stdp_gate"]
    return obj


class VocalErrorNetV2(VocalErrorNet):
    """Earlier name for :class:`VocalErrorNet`, with its earlier dimensionless arguments.

    Takes the parameters the class had before the operating point became its default
    (no adaptation, no conduction delay, no NMDA, ``n_aud=256``), in the earlier
    dimensionless units, and converts them (:func:`units.from_dimensionless`): the network
    behaves as it did, but its attributes are in mV / pA like any other. Kept so code
    written against it keeps behaving as before. Prefer :class:`VocalErrorNet`.
    """

    def __init__(
        self,
        n_e: int = 600,
        n_i: int = 150,
        n_hvc: int = 15,
        n_aud: int = 256,
        tau_e: float = 30.0,
        tau_i: float = 10.0,
        tau_s: float = 10.0,
        A_jie: float = 5e-6,
        J_max_ie: float = 1.0,
        c_B: float = 0.05,
        B_scale: float = 5.0,
        c_hvc: float = 0.3,
        B_hvc_scale: float = 2.0,
        c_JEI: float = 0.5,
        t_refr_ms: float = 2.0,
        a_sfa: float = 0.0,
        tau_sfa: float = 100.0,
        r_i_th: float = 0.0,
        drive_e: float = 0.155,
        drive_i: float = 0.058,
        noise_e: float = 0.1,
        noise_i: float = 0.0,
        target_rate: float = 0.025,
        target_rate_i: float = 0.05,
        aud_delay_ms: int = 0,
        hvc_delay_ms: int = 0,
        tau_stdp: float = 10.0,     # its tau_s: the window it once reused from it
        seed: int | None = None,
    ) -> None:
        # The earlier thresholds: tonic drive alone fires E at target_rate (per ms); I's
        # from a fixed reference input target_rate_i * 10.
        theta_e = max(drive_e * (1.0 - np.exp(-1.0 / tau_e) ** (1.0 / target_rate)), 1e-3)
        theta_i = max(target_rate_i * 10.0
                      * (1.0 - np.exp(-1.0 / tau_i) ** (1.0 / target_rate_i)), 1e-3)
        new = U.from_dimensionless(
            dict(theta_e=theta_e, theta_i=theta_i, tau_e=tau_e, tau_i=tau_i, tau_s=tau_s,
                 tau_inh=0.0, tau_exc=0.0, drive_e=drive_e, drive_i=drive_i,
                 noise_e=noise_e, noise_i=noise_i, a_sfa=a_sfa, A_jie=A_jie,
                 J_max_ie=J_max_ie, g_nmda=0.0, e_l_mv=C.E_L_MV, v_th_mv=C.V_TH_MV,
                 e_nmda_mv=C.E_NMDA_MV),
            C.C_E_PF, C.C_I_PF)
        wf = new["w_factor"]
        b = B_scale / n_aud * wf["B"]
        bh = B_hvc_scale * wf["B_hvc"]
        je = 0.17 / np.sqrt(n_e * c_JEI) * wf["JEI"]
        super().__init__(
            n_e=n_e, n_i=n_i, n_hvc=n_hvc, n_aud=n_aud, tau_e=tau_e, tau_i=tau_i,
            tau_s=tau_s, tau_inh=0.0, tau_exc=0.0,
            e_l_mv=new["e_l_mv"], v_th_mv=new["v_th_mv"], v_reset_mv=new["v_reset_mv"],
            i_drive_e_pa=new["i_drive_e_pa"], i_drive_i_pa=new["i_drive_i_pa"],
            noise_e_pa=new["noise_e_pa"], noise_i_pa=new["noise_i_pa"],
            c_B=c_B, b_aud_mean_mv=b, b_aud_sd_mv=0.5 * b,
            c_hvc=c_hvc, b_hvc_mean_mv=bh, b_hvc_sd_mv=0.5 * bh,
            c_JEI=c_JEI, j_ei_mean_mv=je, j_ei_sd_mv=0.1 * je,
            j_max_ie_mv=new["j_max_ie_mv"], a_jie_mv=new["a_jie_mv"],
            r_i_th=r_i_th, tau_stdp=tau_stdp, t_refr_ms=t_refr_ms,
            a_sfa_pa=new["a_sfa_pa"], tau_sfa=tau_sfa,
            aud_delay_ms=aud_delay_ms, hvc_delay_ms=hvc_delay_ms, g_nmda=0.0, seed=seed,
        )
