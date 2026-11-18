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
    "rho_ns", "rho_pre_ns", "j_ie_row_max_ns", "r_i_hz", "tau_stdp",
    "aud_delay_ms", "hvc_delay_ms",
    "g_nmda_mean_ns", "tau_nmda_rise", "tau_nmda_decay", "alpha_nmda", "mg_mm", "e_nmda_mv",
)
_INT_KEYS = ("n_e", "n_i", "n_hvc", "n_aud")
_ARRAY_KEYS = ("A", "B", "J_EI", "J_IE")


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
        p = dict(locals())
        for k in _FLOAT_KEYS:
            if k in p and p[k] is not None:
                setattr(self, k, float(p[k]))
        self.i_drive_e_pa = float(
            i_drive_e_pa if i_drive_e_pa is not None else
            U.tonic_drive_pa(target_rate_hz, self.tau_e, self.g_l_e,
                             self.e_l_mv, self.v_th_mv, self.v_reset_mv))
        self.rho_ns = float(C.rho_for(tau_stdp) if rho_ns is None else rho_ns)
        self.stimulus: dict = {}

        # Fixed weights: sparse log-normal peak conductances (nS), drawn in this order.
        self.A = _lognormal_sparse((n_e, n_aud), c_aud, aud_mean_ns, aud_sd_ns, rng)
        self.B = _lognormal_sparse((n_i, n_hvc), c_hvc, hvc_mean_ns, hvc_sd_ns, rng)
        self.J_EI = _lognormal_sparse((n_i, n_e), c_ei, ei_mean_ns, ei_sd_ns, rng)
        # I -> E starts at zero, so only song-driven E cells ever acquire inhibition.
        self.J_IE = np.zeros((n_e, n_i), dtype=np.float32)
        self._built: tuple | None = None

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
            "E": {"C": self.c_e_pf * pF, "g_L": self.g_l_e * nS,
                  "I_tonic": self.i_drive_e_pa * pA, "sigma": self.noise_e * pA * sqrt_ms,
                  "beta": self.beta_adapt_pa * pA},
            "I": {"C": self.c_i_pf * pF, "g_L": self.g_l_i * nS,
                  "I_tonic": self.i_drive_i_pa * pA, "sigma": self.noise_i * pA * sqrt_ms},
        }

    def weights(self) -> dict:
        """The ``(post, pre)`` weight matrices (nS) the network is built from."""
        return {"A": np.asarray(self.A, dtype=np.float64), "G": self.nmda_conductances(),
                "B": np.asarray(self.B, dtype=np.float64),
                "J_EI": np.asarray(self.J_EI, dtype=np.float64),
                "J_IE": np.asarray(self.J_IE, dtype=np.float64)}

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
        defaults = {"rho_pre_ns": 0.0, "j_ie_row_max_ns": 0.0}   # added after brian2-v1
        missing = [k for k in _INT_KEYS + _FLOAT_KEYS + _ARRAY_KEYS
                   if k not in d and k not in defaults]
        if missing:
            raise ValueError(f"{path}: not a complete VocalErrorNet model; missing keys "
                             f"{', '.join(missing)}")
        obj = cls.__new__(cls)
        obj._sim_rng = default_rng(None if seed is None else seed + 10_000)
        obj._built = None
        for k in _INT_KEYS:
            setattr(obj, k, int(d[k]))
        for k in _FLOAT_KEYS:
            setattr(obj, k, float(d[k]) if k in d else defaults[k])
        for k in _ARRAY_KEYS:
            setattr(obj, k, d[k])
        obj.stimulus = json.loads(str(d["stimulus_json"])) if "stimulus_json" in d else {}
        return obj

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _structure_key(self) -> str:
        """Everything a built network bakes in: parameters and the fixed weights."""
        h = hashlib.sha1()
        h.update(repr([(k, getattr(self, k)) for k in _INT_KEYS + _FLOAT_KEYS]).encode())
        for k in ("A", "B", "J_EI"):
            h.update(np.ascontiguousarray(getattr(self, k), dtype=np.float32).tobytes())
        return h.hexdigest()

    def _network(self) -> dict:
        """The cached Brian2 network, rebuilt when a parameter or fixed weight changed."""
        key = self._structure_key()
        if getattr(self, "_built", None) is None or self._built[0] != key:
            objs = self.build()
            objs["net"].store("initial")
            self._built = (key, objs)
        assert self._built is not None
        return self._built[1]

    def _sim(self, hvc_spikes, aud_spikes, learn: bool = False,
             rng: np.random.Generator | None = None, hvc_prelude=None):
        from brian2 import ms, nS, seed as brian_seed
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
        sE = bin_spikes(o["mon_e"], self.n_e, T)
        sI = bin_spikes(o["mon_i"], self.n_i, T)
        return sE[:, n_pre:], sI[:, n_pre:]
