"""Spiking E-I vocal error network: parameters, weights, training and inference.

The dynamics are Brian2 equations in :mod:`spiking_ven.network`; this module owns what
they run on -- the parameters (with their units), the weight matrices -- and the
train / infer / save / load cycle around them.

Architecture (weight matrices are ``(post, pre)``, peak conductances in nS)
----------------------------------------------------------------------------
- ``A``    auditory channels -> E   AMPA, and NMDA through ``G`` (``A`` rescaled)  [fixed]
- ``B``    HVC -> I                AMPA                                          [fixed]
- ``J_EI`` E -> I                  AMPA                                          [fixed]
- ``J_IE`` I -> E                  GABA_A, zero-initialised, learned during ``fit``

:class:`VocalErrorNet` defaults to the shipped operating point in
:mod:`spiking_ven.constants`, so ``VocalErrorNet(n_aud=encoder.n_channels)`` builds the
network the published model was trained as.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from numpy.random import default_rng

from . import constants as C
from . import units as U
from .common import _lognormal_sparse, generate_hvc_bursts  # noqa: F401 (re-export)

__all__ = ["VocalErrorNet", "generate_hvc_bursts", "MODEL_FORMAT"]

#: Marks a model file written by this version: Brian2 dynamics, mV-pA-nS units, the
#: paper's weight names. Files with another marker cannot be run by it.
MODEL_FORMAT = "brian2-v1"

#: Scalar parameters a model file stores, all floats (units in the class docstring).
_FLOAT_KEYS = (
    "dt_ms",
    "tau_e", "tau_i", "c_e_pf", "c_i_pf", "e_l_mv", "v_th_mv", "v_reset_mv", "t_refr_ms",
    "i_drive_e_pa", "i_drive_i_pa", "noise_e", "noise_i",
    "tau_ampa", "tau_gaba", "e_ampa_mv", "e_gaba_mv",
    "beta_adapt_pa", "tau_adapt",
    "c_aud", "aud_mean_ns", "aud_sd_ns",
    "c_hvc", "hvc_mean_ns", "hvc_sd_ns",
    "c_ei", "ei_mean_ns", "ei_sd_ns",
    "c_ie", "ie_init_mean_ns", "ie_init_sd_ns",
    "c_hvc_h", "hvc_h_mean_ns", "hvc_h_sd_ns", "rho_h_ns", "r_ih_hz", "i_drive_ih_pa",
    "vogels_h", "eta_h_ns", "vogels_gate", "lam_loc", "tau_loc_ms", "tau_slow_ms",
    "vogels_post",
    "rho_ns", "rho_pre_ns", "j_ie_row_max_ns", "r_i_hz", "tau_stdp",
    "aud_delay_ms", "hvc_delay_ms",
    "g_nmda_mean_ns", "tau_nmda_rise", "tau_nmda_decay", "alpha_nmda", "mg_mm", "e_nmda_mv",
)
_INT_KEYS = ("n_e", "n_i", "n_hvc", "n_aud", "n_ih")
_ARRAY_KEYS = ("A", "B", "J_EI", "J_IE", "M_IE", "B_H", "J_HE", "M_HE", "R_TARGET")


class VocalErrorNet:
    """Spiking conductance-LIF E-I vocal error network with audio-gated STDP on I -> E.

    HVC premotor input drives I; auditory input drives E. During :meth:`fit`, I -> E
    weights strengthen where I activity precedes auditory-driven E spikes, so trained
    song comes to be cancelled by HVC-timed inhibition. The equations are in
    :mod:`spiking_ven.network`.

    Units: ms, mV, pA, nS, pF, Hz.

    Parameters
    ----------
    n_e, n_i, n_hvc, n_aud : population sizes. ``n_aud`` has no default: it must equal
                   the encoder's ``n_channels``.
    dt_ms        : simulation step (ms)
    tau_e, tau_i : membrane time constants at rest (ms); ``g_L = C / tau``
    c_e_pf, c_i_pf : membrane capacitances (pF)
    e_l_mv, v_th_mv, v_reset_mv : rest, threshold, reset (mV), shared by E and I
    t_refr_ms    : absolute refractory period (ms)
    target_rate_hz : rate (Hz) the tonic drive alone fires E at; sets ``i_drive_e_pa``
    i_drive_e_pa, i_drive_i_pa : tonic currents (pA); E's ``None`` derives it from
                   ``target_rate_hz``
    noise_e, noise_i : white current-noise intensity (pA sqrt(ms)); numerically, the SD
                   of the current noise over a 1 ms step
    tau_ampa, tau_gaba : synaptic decays (ms): AMPA for aud -> E, HVC -> I and E -> I;
                   GABA_A for I -> E
    e_ampa_mv, e_gaba_mv : reversal potentials (mV)
    beta_adapt_pa, tau_adapt : adaptation current step per E spike (pA) and its decay (ms)
    c_aud, aud_mean_ns, aud_sd_ns : aud -> E connection probability and log-normal peak
                   conductance mean and SD (nS) -- matrix ``A``
    c_hvc, hvc_mean_ns, hvc_sd_ns : the same for HVC -> I -- ``B``
    c_ei, ei_mean_ns, ei_sd_ns : the same for E -> I -- ``J_EI``
    c_ie         : I -> E connection probability -- the plastic synapses ``M_IE`` (1 = all
                   pairs, the shipped model)
    ie_init_mean_ns, ie_init_sd_ns : log-normal initial I -> E weights on those synapses
                   (0 = start from zero, the shipped model)
    n_ih         : size of an optional second inhibitory population, I_HVC, driven only by
                   HVC (0 = none, the shipped model). Its synapses onto E (``J_HE``, all
                   pairs, from zero) are plastic by the same rule as I -> E.
    c_hvc_h, hvc_h_mean_ns, hvc_h_sd_ns : HVC -> I_HVC probability and weights (``B_H``)
    rho_h_ns, r_ih_hz : STDP step (``None`` = ``rho_ns``) and baseline rate for I_HVC -> E
    i_drive_ih_pa : tonic current on I_HVC (pA)
    vogels_h     : 1 = I_HVC -> E learns by the rate-target rule of Vogels et al. 2011
                   instead (each E cell held at its rate in ``R_TARGET``); 0 = the STDP rule
    eta_h_ns     : its learning rate (nS)
    vogels_gate  : 1 = gate that rule by the E cell's auditory input (as the STDP rule is)
    lam_loc, tau_loc_ms : that rule's target is ``(1 - lam_loc) R_TARGET + lam_loc r_loc``,
                   with ``r_loc`` the cell's own rate averaged over ``tau_loc_ms`` -- a local
                   target that flattens the rate on shorter timescales (0 = constant)
    vogels_post  : scale of that rule's postsynaptic term (1 = Vogels et al.; 0 = off)
    tau_slow_ms  : > 0 measures that local term against the cell's rate over this longer
                   time instead of ``R_TARGET``: target ``R_TARGET + lam_loc (r_loc -
                   r_slow)``, flattening fast fluctuations with the mean anchored
    rho_ns       : STDP step (nS); ``None`` is ``constants.rho_for(tau_stdp)``
    rho_pre_ns   : presynaptic depression per I spike while the audio gate is open (nS)
    j_ie_row_max_ns : cap on each E cell's total I -> E weight (nS), enforced by scaling
                   the row down after each ``fit``; 0 = no cap
    r_i_hz       : I rate (Hz) setting the STDP baseline ``theta = r_i tau_stdp``
    tau_stdp     : STDP window (ms): the I trace's decay and the audio gate's
    aud_delay_ms, hvc_delay_ms : conduction delays (ms)
    g_nmda_mean_ns : mean peak NMDA conductance (nS) per aud -> E connection, at full
                   gating and unblocked; ``G`` is ``A`` rescaled to this mean (0 = off)
    tau_nmda_rise, tau_nmda_decay, alpha_nmda : NMDA gating kinetics (ms, ms, 1/ms)
    mg_mm, e_nmda_mv : Mg2+ concentration (mM) and NMDA reversal potential (mV)
    seed         : RNG seed (weights, and the simulation noise stream)

    ``stimulus`` describes how the training stimuli were built; it is saved with the
    model so later evaluation can rebuild them.
    """

    # Every parameter is an attribute (units in the docstring above), set by __init__
    # or load.
    n_e: int
    n_i: int
    n_hvc: int
    n_aud: int
    dt_ms: float
    tau_e: float
    tau_i: float
    c_e_pf: float
    c_i_pf: float
    e_l_mv: float
    v_th_mv: float
    v_reset_mv: float
    t_refr_ms: float
    i_drive_e_pa: float
    i_drive_i_pa: float
    noise_e: float
    noise_i: float
    tau_ampa: float
    tau_gaba: float
    e_ampa_mv: float
    e_gaba_mv: float
    beta_adapt_pa: float
    tau_adapt: float
    c_aud: float
    aud_mean_ns: float
    aud_sd_ns: float
    c_hvc: float
    hvc_mean_ns: float
    hvc_sd_ns: float
    c_ei: float
    ei_mean_ns: float
    ei_sd_ns: float
    rho_ns: float
    rho_pre_ns: float
    j_ie_row_max_ns: float
    r_i_hz: float
    tau_stdp: float
    aud_delay_ms: float
    hvc_delay_ms: float
    g_nmda_mean_ns: float
    tau_nmda_rise: float
    tau_nmda_decay: float
    alpha_nmda: float
    mg_mm: float
    e_nmda_mv: float
    A: np.ndarray
    B: np.ndarray
    J_EI: np.ndarray
    J_IE: np.ndarray
    M_IE: np.ndarray
    n_ih: int
    B_H: np.ndarray
    J_HE: np.ndarray
    M_HE: np.ndarray
    c_hvc_h: float
    hvc_h_mean_ns: float
    hvc_h_sd_ns: float
    rho_h_ns: float
    r_ih_hz: float
    i_drive_ih_pa: float
    vogels_h: float
    eta_h_ns: float
    vogels_gate: float
    lam_loc: float
    tau_loc_ms: float
    tau_slow_ms: float
    vogels_post: float
    R_TARGET: np.ndarray
    c_ie: float
    ie_init_mean_ns: float
    ie_init_sd_ns: float
    stimulus: dict
    _built: tuple[str, dict] | None

    def __init__(
        self,
        n_e: int = C.N_E,
        n_i: int = C.N_I,
        n_hvc: int = C.N_HVC,
        n_aud: int | None = None,
        dt_ms: float = C.DT_MS,
        tau_e: float = C.TAU_E,
        tau_i: float = C.TAU_I,
        c_e_pf: float = C.C_E_PF,
        c_i_pf: float = C.C_I_PF,
        e_l_mv: float = C.E_L_MV,
        v_th_mv: float = C.V_TH_MV,
        v_reset_mv: float = C.V_RESET_MV,
        t_refr_ms: float = C.T_REFR_MS,
        target_rate_hz: float = C.TARGET_RATE_E_HZ,
        i_drive_e_pa: float | None = None,
        i_drive_i_pa: float = C.I_DRIVE_I_PA,
        noise_e: float = C.NOISE_E,
        noise_i: float = C.NOISE_I,
        tau_ampa: float = C.TAU_AMPA,
        tau_gaba: float = C.TAU_GABA,
        e_ampa_mv: float = C.E_AMPA_MV,
        e_gaba_mv: float = C.E_GABA_MV,
        beta_adapt_pa: float = C.BETA_ADAPT_PA,
        tau_adapt: float = C.TAU_ADAPT,
        c_aud: float = C.C_AUD,
        aud_mean_ns: float = C.AUD_MEAN_NS,
        aud_sd_ns: float = C.AUD_SD_NS,
        c_hvc: float = C.C_HVC,
        hvc_mean_ns: float = C.HVC_MEAN_NS,
        hvc_sd_ns: float = C.HVC_SD_NS,
        c_ei: float = C.C_EI,
        ei_mean_ns: float = C.EI_MEAN_NS,
        ei_sd_ns: float = C.EI_SD_NS,
        c_ie: float = C.C_IE,
        ie_init_mean_ns: float = C.IE_INIT_MEAN_NS,
        ie_init_sd_ns: float = C.IE_INIT_SD_NS,
        n_ih: int = 0,
        c_hvc_h: float = 0.02,
        hvc_h_mean_ns: float = 20.0,
        hvc_h_sd_ns: float = 10.0,
        rho_h_ns: float | None = None,
        r_ih_hz: float = 5.0,
        i_drive_ih_pa: float = 0.0,
        vogels_h: float = 0.0,
        eta_h_ns: float = 1.0,
        vogels_gate: float = 0.0,
        lam_loc: float = 0.0,
        tau_loc_ms: float = 100.0,
        tau_slow_ms: float = 0.0,
        vogels_post: float = 1.0,
        rho_ns: float | None = None,
        rho_pre_ns: float = C.RHO_PRE_NS,
        j_ie_row_max_ns: float = C.J_IE_ROW_MAX_NS,
        r_i_hz: float = C.R_I_HZ,
        tau_stdp: float = C.TAU_STDP,
        aud_delay_ms: float = C.AUD_DELAY_MS,
        hvc_delay_ms: float = C.HVC_DELAY_MS,
        g_nmda_mean_ns: float = C.G_NMDA_MEAN_NS,
        tau_nmda_rise: float = C.TAU_NMDA_RISE,
        tau_nmda_decay: float = C.TAU_NMDA_DECAY,
        alpha_nmda: float = C.ALPHA_NMDA,
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
        # A separate stream for the simulation noise, so weight initialisation and
        # simulation cannot perturb each other.
        self._sim_rng = default_rng(None if seed is None else seed + 10_000)
        self.n_e, self.n_i, self.n_hvc, self.n_aud = int(n_e), int(n_i), int(n_hvc), int(n_aud)
        args = dict(locals())                        # every float parameter, by name
        for k in _FLOAT_KEYS:
            if args.get(k) is not None:
                setattr(self, k, float(args[k]))
        self.i_drive_e_pa = float(
            i_drive_e_pa if i_drive_e_pa is not None else
            U.tonic_drive_pa(target_rate_hz, self.tau_e, self.g_l_e,
                             self.e_l_mv, self.v_th_mv, self.v_reset_mv))
        self.rho_ns = float(C.rho_for(tau_stdp) if rho_ns is None else rho_ns)
        self.stimulus = {}

        # Fixed weights: sparse log-normal peak conductances (nS), drawn in this order.
        self.A = _lognormal_sparse((n_e, n_aud), c_aud, aud_mean_ns, aud_sd_ns, rng)
        self.B = _lognormal_sparse((n_i, n_hvc), c_hvc, hvc_mean_ns, hvc_sd_ns, rng)
        self.J_EI = _lognormal_sparse((n_i, n_e), c_ei, ei_mean_ns, ei_sd_ns, rng)
        # I -> E: plastic synapses on the pairs in M_IE. The shipped model connects every
        # pair and starts at zero, so only song-driven E cells ever acquire inhibition; a
        # balanced start draws sparse connections with nonzero initial weights. (Drawn
        # after the fixed weights, and only when used, so they do not change.)
        if c_ie >= 1.0 and ie_init_mean_ns <= 0.0:
            self.M_IE = np.ones((n_e, n_i), dtype=bool)
            self.J_IE = np.zeros((n_e, n_i), dtype=np.float32)
        else:
            self.M_IE = rng.random((n_e, n_i)) < c_ie
            if ie_init_mean_ns > 0.0:
                sd = ie_init_sd_ns if ie_init_sd_ns > 0 else 0.5 * ie_init_mean_ns
                s2 = np.log(1 + (sd / ie_init_mean_ns) ** 2)
                W = rng.lognormal(np.log(ie_init_mean_ns) - 0.5 * s2, np.sqrt(s2),
                                  (n_e, n_i))
            else:
                W = np.zeros((n_e, n_i))
            self.J_IE = (W * self.M_IE).astype(np.float32)
        # Optional I_HVC: drawn last, and only when used, so nothing above changes.
        self.n_ih = int(n_ih)
        self.rho_h_ns = float(self.rho_ns if rho_h_ns is None else rho_h_ns)
        if self.n_ih > 0:
            self.B_H = _lognormal_sparse((self.n_ih, n_hvc), c_hvc_h, hvc_h_mean_ns,
                                         hvc_h_sd_ns, rng)
        else:
            self.B_H = np.zeros((0, n_hvc), dtype=np.float32)
        self.J_HE = np.zeros((n_e, self.n_ih), dtype=np.float32)
        self.M_HE = np.ones((n_e, self.n_ih), dtype=bool)
        self.R_TARGET = np.zeros(n_e, dtype=np.float32)
        self.last_ih = None
        self._built = None

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

    @property
    def theta(self) -> float:
        """STDP baseline: the I trace's mean at rate ``r_i_hz`` (dimensionless)."""
        return self.r_i_hz * self.tau_stdp * 1e-3

    def nmda_conductances(self) -> np.ndarray:
        """``G``: peak NMDA conductance (nS) per aud -> E connection, ``A`` rescaled so its
        mean is ``g_nmda_mean_ns`` (full gating, before the Mg2+ block)."""
        A = np.asarray(self.A, dtype=np.float64)
        if self.g_nmda_mean_ns <= 0.0 or self.aud_mean_ns <= 0.0:
            return np.zeros_like(A)
        return A * (self.g_nmda_mean_ns / self.aud_mean_ns)

    def namespace(self) -> dict:
        """Every constant of the equations, as Brian2 quantities.

        Population-specific values sit under ``"E"`` and ``"I"``; the rest are shared.
        """
        from brian2 import Hz, mM, ms, mV, nS, pA, pF  # noqa: F401
        per_ms = 1.0 / ms
        sqrt_ms = ms ** 0.5
        return {
            "E_L": self.e_l_mv * mV, "V_th": self.v_th_mv * mV,
            "V_reset": self.v_reset_mv * mV, "t_ref": self.t_refr_ms * ms,
            "E_AMPA": self.e_ampa_mv * mV, "E_GABA": self.e_gaba_mv * mV,
            "E_NMDA": self.e_nmda_mv * mV,
            "tau_ampa": self.tau_ampa * ms, "tau_gaba": self.tau_gaba * ms,
            "tau_adapt": self.tau_adapt * ms, "tau_stdp": self.tau_stdp * ms,
            "tau_nmda_rise": self.tau_nmda_rise * ms,
            "tau_nmda_decay": self.tau_nmda_decay * ms,
            "alpha_nmda": self.alpha_nmda * per_ms, "Mg": self.mg_mm * mM,
            "theta": self.theta,
            "theta_h": self.r_ih_hz * self.tau_stdp * 1e-3,
            "vogels_h": bool(self.vogels_h),
            "vogels_gate": bool(self.vogels_gate),
            "lam_loc": float(self.lam_loc), "tau_loc": self.tau_loc_ms * ms,
            "use_slow": float(self.tau_slow_ms > 0),
            "post_gain": float(self.vogels_post),
            "tau_slow": (self.tau_slow_ms if self.tau_slow_ms > 0 else 1000.0) * ms,
            "E": {"C": self.c_e_pf * pF, "g_L": self.g_l_e * nS,
                  "I_tonic": self.i_drive_e_pa * pA, "sigma": self.noise_e * pA * sqrt_ms,
                  "beta": self.beta_adapt_pa * pA},
            "IH": {"C": self.c_i_pf * pF, "g_L": self.g_l_i * nS,
                   "I_tonic": self.i_drive_ih_pa * pA, "sigma": self.noise_i * pA * sqrt_ms},
            "I": {"C": self.c_i_pf * pF, "g_L": self.g_l_i * nS,
                  "I_tonic": self.i_drive_i_pa * pA, "sigma": self.noise_i * pA * sqrt_ms},
        }

    def weights(self) -> dict:
        """The ``(post, pre)`` weight matrices (nS) the network is built from."""
        return {"A": np.asarray(self.A, dtype=np.float64), "G": self.nmda_conductances(),
                "B": np.asarray(self.B, dtype=np.float64),
                "J_EI": np.asarray(self.J_EI, dtype=np.float64),
                "J_IE": np.asarray(self.J_IE, dtype=np.float64),
                "M_IE": np.asarray(self.M_IE, dtype=bool),
                "B_H": np.asarray(self.B_H, dtype=np.float64),
                "J_HE": np.asarray(self.J_HE, dtype=np.float64),
                "M_HE": np.asarray(self.M_HE, dtype=bool)}

    def build(self) -> dict:
        """The network's Brian2 objects (:func:`spiking_ven.network.build_network`).

        For hosting the network inside a larger Brian2 circuit; :meth:`fit` and
        :meth:`transform` build and cache their own.
        """
        from brian2 import ms
        from .network import build_network
        return build_network(self.namespace(), self.weights(), n_aud=self.n_aud,
                             n_hvc=self.n_hvc, aud_delay=self.aud_delay_ms * ms,
                             hvc_delay=self.hvc_delay_ms * ms, dt=self.dt_ms * ms)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def set_rate_targets(self, n_hvc_t: int, n_aud: int, T: int = 1000,
                         n_trials: int = 10, seed: int = 0) -> np.ndarray:
        """Set ``R_TARGET`` to each E cell's rate (Hz) in silence, not singing.

        Measured on the network as it stands (before training, for the rate-target rule),
        over ``n_trials`` runs of ``T`` ms with no HVC or auditory input.
        """
        rng = np.random.default_rng(seed)
        z_h = np.zeros((n_hvc_t, T), dtype=np.float32)
        z_a = np.zeros((n_aud, T), dtype=np.float32)
        r = np.mean([self.transform(z_h, z_a, rng=rng) for _ in range(n_trials)],
                    axis=(0, 2)) * 1000.0
        self.R_TARGET = r.astype(np.float32)
        return self.R_TARGET

    def fit(self, hvc_spikes: np.ndarray, aud_spikes: np.ndarray, *,
            hvc_prelude: np.ndarray | None = None) -> np.ndarray:
        """Train on paired HVC + auditory input (1 ms bins): updates ``J_IE`` by STDP.

        Returns the E spikes ``(n_e, T)``. ``hvc_prelude`` is HVC activity run first,
        with no sound (introductory notes); the returned spikes start after it.
        """
        sE, _ = self._sim(hvc_spikes, aud_spikes, learn=True, hvc_prelude=hvc_prelude)
        return sE

    def transform(self, hvc_spikes: np.ndarray, aud_spikes: np.ndarray, *,
                  rng: np.random.Generator | None = None,
                  hvc_prelude: np.ndarray | None = None) -> np.ndarray:
        """Inference (no learning). Returns the E spikes, ``(n_e, T)`` float32 on 1 ms bins.

        The noise seed is drawn from ``rng`` if given, otherwise from the network's own
        stream; pass a separate generator for side computations (monitoring during
        training) so they do not change what the next ``fit`` sees.
        """
        sE, _ = self._sim(hvc_spikes, aud_spikes, learn=False, rng=rng,
                          hvc_prelude=hvc_prelude)
        return sE

    def transform_all(self, hvc_spikes: np.ndarray, aud_spikes: np.ndarray, *,
                      rng: np.random.Generator | None = None,
                      hvc_prelude: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Inference. Returns ``(E spikes, I spikes)``; arguments as :meth:`transform`."""
        return self._sim(hvc_spikes, aud_spikes, learn=False, rng=rng,
                         hvc_prelude=hvc_prelude)

    def save(self, path: str) -> None:
        np.savez(
            path,
            model_format=np.array(MODEL_FORMAT),
            **{k: getattr(self, k) for k in _ARRAY_KEYS},
            **{k: getattr(self, k) for k in _INT_KEYS},
            **{k: getattr(self, k) for k in _FLOAT_KEYS},
            stimulus_json=np.array(json.dumps(self.stimulus, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "VocalErrorNet":
        """Load a model written by :meth:`save`; ``seed`` seeds the simulation noise.

        Raises ``ValueError`` for a file from an earlier version (the numpy engine), or
        one missing keys.
        """
        d = np.load(path)
        fmt = str(d["model_format"]) if "model_format" in d else "numpy"
        if fmt != MODEL_FORMAT:
            raise ValueError(f"{path}: a {fmt!r} model; this version runs {MODEL_FORMAT!r} "
                             f"models only. Retrain with sven-train-ven.")
        # Added after brian2-v1 first shipped: defaults reproduce the shipped model.
        defaults = {"rho_pre_ns": 0.0, "j_ie_row_max_ns": 0.0, "c_ie": 1.0,
                    "ie_init_mean_ns": 0.0, "ie_init_sd_ns": 0.0, "n_ih": 0, "c_hvc_h": 0.02,
                    "hvc_h_mean_ns": 20.0, "hvc_h_sd_ns": 10.0,
                    "rho_h_ns": float(d["rho_ns"]) if "rho_ns" in d else 0.0,
                    "r_ih_hz": 5.0, "i_drive_ih_pa": 0.0, "vogels_h": 0.0,
                    "eta_h_ns": 1.0, "vogels_gate": 0.0, "lam_loc": 0.0,
                    "tau_loc_ms": 100.0, "tau_slow_ms": 0.0,
                    "vogels_post": 1.0}   # the constructor's defaults
        optional_arrays = ("M_IE", "B_H", "J_HE", "M_HE", "R_TARGET")
        missing = [k for k in _INT_KEYS + _FLOAT_KEYS + _ARRAY_KEYS
                   if k not in d and k not in defaults and k not in optional_arrays]
        if missing:
            raise ValueError(f"{path}: not a complete VocalErrorNet model; missing keys "
                             f"{', '.join(missing)}")
        obj = cls.__new__(cls)
        obj._sim_rng = default_rng(None if seed is None else seed + 10_000)
        obj._built = None
        obj.last_ih = None
        for k in _INT_KEYS:
            setattr(obj, k, int(d[k]) if k in d else int(defaults[k]))
        for k in _FLOAT_KEYS:
            setattr(obj, k, float(d[k]) if k in d else defaults[k])
        for k in _ARRAY_KEYS:
            if k in d:
                setattr(obj, k, d[k])
            elif k == "M_IE":
                obj.M_IE = np.ones(d["J_IE"].shape, dtype=bool)
            elif k == "R_TARGET":
                obj.R_TARGET = np.zeros(int(d["n_e"]), dtype=np.float32)
            elif k == "B_H":
                obj.B_H = np.zeros((0, int(d["n_hvc"])), dtype=np.float32)
            else:                                    # J_HE, M_HE: no I_HVC
                obj.J_HE = np.zeros((int(d["n_e"]), 0), dtype=np.float32)
                obj.M_HE = np.ones((int(d["n_e"]), 0), dtype=bool)
        obj.stimulus = json.loads(str(d["stimulus_json"])) if "stimulus_json" in d else {}
        return obj

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _structure_key(self) -> str:
        """Everything a built network bakes in: parameters and the fixed weights."""
        h = hashlib.sha1()
        h.update(repr([(k, getattr(self, k)) for k in _INT_KEYS + _FLOAT_KEYS]).encode())
        for k in ("A", "B", "J_EI", "M_IE", "B_H", "M_HE"):
            h.update(np.ascontiguousarray(getattr(self, k), dtype=np.float32).tobytes())
        return h.hexdigest()

    def _network(self) -> dict:
        """The cached Brian2 network, rebuilt when a parameter or fixed weight changed."""
        key = self._structure_key()
        built = getattr(self, "_built", None)
        if built is None or built[0] != key:
            objs = self.build()
            objs["net"].store("initial")
            built = (key, objs)
            self._built = built
        return built[1]

    def _sim(self, hvc_spikes, aud_spikes, learn: bool = False,
             rng: np.random.Generator | None = None, hvc_prelude=None):
        from brian2 import Hz, ms, nS, seed as brian_seed
        from .network import bin_spikes, spike_events

        n_pre = 0 if hvc_prelude is None else int(np.asarray(hvc_prelude).shape[1])
        if n_pre:
            hvc_spikes = np.concatenate([np.asarray(hvc_prelude, dtype=np.float32),
                                         np.asarray(hvc_spikes, dtype=np.float32)], axis=1)
            aud_spikes = np.concatenate(
                [np.zeros((aud_spikes.shape[0], n_pre), dtype=np.float32),
                 np.asarray(aud_spikes, dtype=np.float32)], axis=1)
        T = int(hvc_spikes.shape[1])

        o = self._network()
        o["net"].restore("initial")
        o["aud"].set_spikes(*spike_events(aud_spikes))
        o["hvc"].set_spikes(*spike_events(hvc_spikes))
        i_e = o["i_e"]
        i_e.w[:] = np.asarray(self.J_IE, dtype=np.float64)[np.asarray(i_e.j),
                                                           np.asarray(i_e.i)] * nS
        i_e.rho = (self.rho_ns if learn else 0.0) * nS
        i_e.rho_dep = (self.rho_pre_ns if learn else 0.0) * nS
        ih_e = o.get("ih_e")
        if ih_e is not None:
            ih_e.w[:] = np.asarray(self.J_HE, dtype=np.float64)[np.asarray(ih_e.j),
                                                               np.asarray(ih_e.i)] * nS
            if self.vogels_h:
                ih_e.eta = (self.eta_h_ns if learn else 0.0) * nS
                o["E"].alpha_e = 2.0 * np.asarray(self.R_TARGET, dtype=np.float64) \
                    * self.tau_stdp * 1e-3
                # The local rate estimate starts at the cell's target rate.
                o["E"].r_loc = np.asarray(self.R_TARGET, dtype=np.float64) * Hz
                o["E"].r_slow = np.asarray(self.R_TARGET, dtype=np.float64) * Hz
            else:
                ih_e.rho = (self.rho_h_ns if learn else 0.0) * nS
                ih_e.rho_dep = (self.rho_pre_ns if learn else 0.0) * nS
        brian_seed(int((self._sim_rng if rng is None else rng).integers(2**31)))
        o["net"].run(T * ms)

        if learn:
            J = np.zeros((self.n_e, self.n_i), dtype=np.float32)
            J[np.asarray(i_e.j), np.asarray(i_e.i)] = np.asarray(i_e.w / nS)
            if self.j_ie_row_max_ns > 0.0:
                # Homeostatic cap on each cell's total inhibition, between renditions:
                # rows above it are scaled down, preserving their pattern.
                tot = J.sum(axis=1, keepdims=True)
                J *= np.minimum(1.0, self.j_ie_row_max_ns / np.maximum(tot, 1e-12))
            self.J_IE = J
            if ih_e is not None:
                H = np.zeros((self.n_e, self.n_ih), dtype=np.float32)
                H[np.asarray(ih_e.j), np.asarray(ih_e.i)] = np.asarray(ih_e.w / nS)
                self.J_HE = H
        #: I_HVC spikes of the last run (after any prelude), or None without I_HVC.
        self.last_ih = (bin_spikes(o["mon_ih"], self.n_ih, T)[:, n_pre:]
                        if "mon_ih" in o else None)
        sE = bin_spikes(o["mon_e"], self.n_e, T)
        sI = bin_spikes(o["mon_i"], self.n_i, T)
        return sE[:, n_pre:], sI[:, n_pre:]
