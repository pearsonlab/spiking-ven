"""Spiking E-I vocal error network — v2 (HVC→I architecture).

Architecture
------------
- Auditory spikes  (n_aud, T)  →  B (n_e × n_aud)    →  E neurons  [fixed sparse]
- E neurons        →  JEI (n_i × n_e)                 →  I neurons  [fixed]
- HVC spikes       (n_hvc, T) →  B_hvc (n_i × n_hvc) →  I neurons  [fixed sparse]
- I neurons        →  JIE (n_e × n_i)                 →  E neurons  [plastic — causal Hebbian]

Plasticity
----------
Only JIE learns, by causal STDP, during ``fit()``. Everything else is fixed.
"""

import numpy as np
from numpy.random import default_rng

from .common import (
    _lognormal_sparse,
    generate_hvc_spikes,  # re-exported for convenience
    spike_to_rate,        # re-exported for convenience
)

__all__ = ["VocalErrorNetV2", "generate_hvc_spikes", "spike_to_rate"]


# ---------------------------------------------------------------------------
# simulation loop
# ---------------------------------------------------------------------------

def _sim_loop_v2(
    h_aud, h_hvc, B, B_hvc, JEI, JIE,
    decay_e, decay_i,
    theta_e, theta_i, v_reset, noise_e, noise_i, t_refr_ms, a_sfa, tau_sfa,
    A_jie, J_max_ie, xi_th,
    drive_e, drive_i,
    learn_weights, alpha_stdp, rng,
    aud_delay_ms: int = 0,
    hvc_delay_ms: int = 0,
):
    # Membrane noise comes from the caller's seeded generator, never the global np.random.
    n_e = B.shape[0]
    n_i = JEI.shape[0]
    T   = h_aud.shape[1]

    V_e     = np.zeros(n_e)
    V_i     = np.zeros(n_i)
    sE_prev = np.zeros(n_e)
    sI_prev = np.zeros(n_i)
    x_i     = np.zeros(n_i)

    sE_mat  = np.zeros((n_e, T), dtype=np.float32)
    sI_mat  = np.zeros((n_i, T), dtype=np.float32)

    _zero_aud = np.zeros(h_aud.shape[0])
    _zero_hvc = np.zeros(h_hvc.shape[0])

    # Absolute refractory period, in 1 ms steps. Without it a cell can fire on every
    # consecutive bin whenever (1 - decay) * u clears threshold from v_reset, so the
    # ceiling is 1000 Hz. It was being reached: during a DAF burst the median E ISI was
    # 1 ms, with 73% of burst spikes following the previous one in the very next bin. The
    # baseline was unaffected (median ISI 31 ms), which is what makes this the right
    # constraint rather than a knob -- it clips only what is firing impossibly fast.
    refr_e = np.zeros(n_e)
    refr_i = np.zeros(n_i)

    # Spike-frequency adaptation on E: a per-neuron hyperpolarising current that steps up
    # by ``a_sfa`` on every spike and decays with ``tau_sfa``. Chosen over any further
    # reduction of the auditory drive because the two are not interchangeable. Extra
    # spikes per burst turned out to depend only on the PRODUCT of auditory rate and
    # B_scale, as drive^0.36 -- two runs at drive 149 and 161 gave 9.04 and 9.36 spikes
    # despite a 4x difference in rate and a 4.4x difference in weight -- so reaching the
    # bird's ~1-2 spikes that way needs an auditory rate near 0.3 Hz, an order of
    # magnitude under the 2.4 +/- 2.7 Hz Schneider & Woolley measured.
    #
    # More to the point, a gain change is the wrong SHAPE. Per responder the model already
    # spans 1.1 to 19.9 extra spikes: the weak end is inside the bird's range and only the
    # tail is wrong. Scaling the drive moves both. Adaptation is activity-dependent, so it
    # bites hardest on exactly the cells firing most and barely touches a cell adding one
    # spike.
    w_sfa = np.zeros(n_e)
    decay_sfa = float(np.exp(-1.0 / tau_sfa)) if tau_sfa > 0.0 else 0.0

    for t in range(T):
        h_aud_t = h_aud[:, t - aud_delay_ms] if t >= aud_delay_ms else _zero_aud
        h_hvc_t = h_hvc[:, t - hvc_delay_ms] if t >= hvc_delay_ms else _zero_hvc

        # --- E dynamics ---
        audio_drive_e = B @ h_aud_t
        u_e = drive_e + audio_drive_e - JIE @ sI_prev
        if a_sfa > 0.0:
            w_sfa *= decay_sfa
            u_e = u_e - w_sfa
        if noise_e > 0.0:
            u_e += rng.normal(0.0, noise_e, n_e)
        # Countdown is decremented BEFORE the spike test, so a cell with t_refr_ms = 2
        # may fire again 2 ms later, not 3. Testing first and decrementing after costs an
        # extra millisecond and would silently make the refractory period t_refr_ms + 1.
        if t_refr_ms > 0.0:
            refr_e = np.maximum(refr_e - 1.0, 0.0)
        V_e = decay_e * V_e + (1.0 - decay_e) * u_e
        if t_refr_ms > 0.0:
            V_e[refr_e > 0.0] = v_reset       # clamped while refractory, as in a LIF
        sE  = ((V_e > theta_e) & (refr_e <= 0.0)).astype(np.float64)
        V_e[sE > 0.5] = v_reset
        if t_refr_ms > 0.0:
            refr_e[sE > 0.5] = t_refr_ms
        if a_sfa > 0.0:
            w_sfa += a_sfa * sE
        sE_mat[:, t] = sE

        # --- I dynamics ---
        u_i = drive_i + JEI @ sE_prev + B_hvc @ h_hvc_t
        if noise_i > 0.0:
            u_i += rng.normal(0.0, noise_i, n_i)
        if t_refr_ms > 0.0:
            refr_i = np.maximum(refr_i - 1.0, 0.0)
        V_i = decay_i * V_i + (1.0 - decay_i) * u_i
        if t_refr_ms > 0.0:
            V_i[refr_i > 0.0] = v_reset
        sI  = ((V_i > theta_i) & (refr_i <= 0.0)).astype(np.float64)
        V_i[sI > 0.5] = v_reset
        if t_refr_ms > 0.0:
            refr_i[sI > 0.5] = t_refr_ms
        sI_mat[:, t] = sI

        if learn_weights:
            # Causal STDP: I-then-E order (I precedes E → potentiate JIE).
            # Update at the moment of each audio-gated E spike using the current
            # I trace — captures recent I firing before this E spike occurred.
            # xi_th is the trace one I spike leaves after ltp_window_ms, so I cells that
            # spiked within the window potentiate and the rest depress. The I trace
            # decays with alpha_stdp (tau_stdp), independently of the synaptic tau_s.
            audio_gate_e = (audio_drive_e > 0.0).astype(np.float64)
            gate_sE = sE * audio_gate_e
            x_i = alpha_stdp * x_i + sI
            JIE += A_jie * np.outer(gate_sE, x_i - xi_th)
            np.clip(JIE, 0.0, J_max_ie, out=JIE)

        sE_prev = sE
        sI_prev = sI

    return sE_mat, sI_mat, JIE


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class VocalErrorNetV2:
    """Spiking LIF E-I vocal error network with causal Hebbian STDP on I→E.

    HVC premotor input drives I neurons; auditory input drives E neurons.
    I→E (JIE) strengthens when I precedes E (causal window only), implementing
    song-matched inhibition that cancels trained auditory patterns.

    Parameters
    ----------
    n_e, n_i, n_hvc, n_aud : population sizes
    tau_e, tau_i : membrane time constants (ms)
    tau_s        : input synaptic time constant (ms). Sets the decay of the conductance
                   each auditory and HVC spike produces -- `spike_to_rate` on aud_spikes
                   and hvc_spikes -- so it is the EPSC time course of the aud->E and
                   HVC->I synapses, not a free smoothing parameter.

                   It is NOT the STDP trace constant. It used to double as one, which
                   tied the coincidence window to the synaptic one; set ``tau_stdp`` to
                   pin the STDP window independently. One coupling survives that split:
                   the LTP audio gate is ``B @ h_aud > 0``, so shortening tau_s still
                   narrows the window in which LTP can occur.
    A_jie        : causal Hebbian STDP amplitude for JIE (I→E)
    J_max_ie     : hard upper bound on JIE weights
    c_B          : auditory→E connection probability
    B_scale      : mean weight scale for auditory→E (compensates for sparsity)
    c_hvc        : HVC→I connection probability
    B_hvc_scale  : mean weight scale for HVC→I (tune so HVC burst drives I reliably)
    c_JEI        : E→I connection probability (fixed throughout)
    ltp_window_ms: causal LTP window (ms) for JIE. An audio-gated E spike potentiates
                   JIE from I cells that spiked within this many ms before it and
                   depresses it from the rest. A tuned parameter. ``inf`` (the default)
                   disables LTD, leaving pure potentiation.
    tau_stdp     : decay constant (ms) of the I trace; 0 falls back to tau_s
    drive_e      : tonic E drive
    drive_i      : tonic I drive
    noise_e      : std of additive Gaussian noise on E input
    noise_i      : std of additive Gaussian noise on I input
    target_rate  : E firing rate used to analytically initialise theta_e (Hz-like units)
    target_rate_i: I firing rate used to analytically initialise theta_i
    seed         : RNG seed
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
        ltp_window_ms: float = np.inf,
        drive_e: float = 0.155,
        drive_i: float = 0.058,
        noise_e: float = 0.1,
        noise_i: float = 0.0,
        target_rate: float = 0.025,
        target_rate_i: float = 0.05,
        aud_delay_ms: int = 0,
        hvc_delay_ms: int = 0,
        tau_stdp: float = 0.0,
        seed: int | None = None,
    ) -> None:
        rng = default_rng(seed)
        # Separate stream for simulation noise, so weight initialisation and the
        # simulation cannot interleave and perturb each other.
        self._sim_rng      = default_rng(None if seed is None else seed + 10_000)
        self.n_e           = n_e
        self.n_i           = n_i
        self.n_hvc         = n_hvc
        self.n_aud         = n_aud
        self.tau_e         = tau_e
        self.tau_i         = tau_i
        self.tau_s         = tau_s
        self.A_jie         = A_jie
        self.J_max_ie      = J_max_ie
        self.c_B           = c_B
        self.B_scale       = B_scale
        self.c_hvc         = c_hvc
        self.B_hvc_scale   = B_hvc_scale
        self.c_JEI         = c_JEI
        #: Absolute refractory period (ms). 2 ms is the standard cortical value and caps
        #: the instantaneous rate at 500 Hz. Set to 0 to recover the old unbounded
        #: dynamics, which allowed 1000 Hz.
        self.t_refr_ms     = float(t_refr_ms)
        #: Spike-frequency adaptation: increment per spike and decay constant (ms).
        #: ``a_sfa`` is in the same units as ``theta_e`` (~0.044), so an increment of
        #: 0.01 means ~4 spikes in quick succession suffice to silence a cell.
        self.a_sfa         = float(a_sfa)
        self.tau_sfa       = float(tau_sfa)
        self.ltp_window_ms = float(ltp_window_ms)
        self.drive_e       = drive_e
        self.drive_i       = drive_i
        self.noise_e       = noise_e
        self.noise_i       = noise_i
        self.target_rate   = target_rate
        self.target_rate_i = target_rate_i
        self.aud_delay_ms  = int(aud_delay_ms)
        self.hvc_delay_ms  = int(hvc_delay_ms)
        self.tau_stdp      = float(tau_stdp)
        self.v_reset: float = 0.0

        # aud→E: sparse log-normal, fixed
        mean_B = B_scale / n_aud
        self.B = _lognormal_sparse((n_e, n_aud), c_B, mean_B, mean_B * 0.5, rng)

        # HVC→I: sparse log-normal, fixed.
        # B_hvc_scale is the mean weight PER CONNECTION (not divided by n_hvc).
        # This keeps per-burst drive constant as n_hvc changes, so adding more HVC
        # neurons improves temporal coverage without reducing per-burst I drive.
        mean_Bh = B_hvc_scale
        self.B_hvc = _lognormal_sparse((n_i, n_hvc), c_hvc, mean_Bh, mean_Bh * 0.5, rng)

        # E→I: sparse log-normal, fixed (not plastic).
        srKE = np.sqrt(n_e * c_JEI)
        self.JEI = _lognormal_sparse(
            (n_i, n_e), c_JEI, 1.7 / srKE / 10, 1.7 / srKE / 10 * 0.1, rng
        )

        # I→E: zero-initialised. Starting from zero means only song-driven E neurons ever
        # get trained (via STDP in fit()), leaving novel-E rows at zero.
        self.JIE = np.zeros((n_e, n_i), dtype=np.float32)

        # E threshold, per neuron.
        # Effective steady-state input is drive_e (tonic), not target_rate * tau_s —
        # tau_s is the STDP trace constant and must not appear in the membrane threshold.
        _decay_e  = float(np.exp(-1.0 / tau_e))
        _isi_e    = 1.0 / target_rate
        _theta_e0 = float(max(drive_e * (1.0 - _decay_e ** _isi_e), 1e-3))
        self.theta_e = np.full(n_e, _theta_e0, dtype=np.float64)

        # I threshold: use a fixed tau_s reference of 10 ms so theta_i stays at the
        # calibrated value regardless of the STDP trace tau_s.  theta_i >> drive_i
        # ensures I only fires when HVC drives it, not from tonic E feedback alone.
        _TAU_S_REF = 10.0
        _decay_i   = float(np.exp(-1.0 / tau_i))
        _isi_i     = 1.0 / target_rate_i
        self.theta_i = float(max(target_rate_i * _TAU_S_REF * (1.0 - _decay_i ** _isi_i), 1e-3))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, hvc_spikes: np.ndarray, aud_spikes: np.ndarray) -> np.ndarray:
        """Train on paired HVC+auditory input: updates JIE by STDP. Returns sE_mat."""
        sE, _sI = self._sim(hvc_spikes, aud_spikes, learn_weights=True)
        return sE

    def transform(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
    ) -> np.ndarray:
        """Inference pass (no weight updates). Returns (n_e, T) float32 E spike matrix."""
        sE, _sI = self._sim(hvc_spikes, aud_spikes, learn_weights=False)
        return sE

    def transform_all(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference pass. Returns (sE_mat, sI_mat) both (n, T) float32."""
        return self._sim(hvc_spikes, aud_spikes, learn_weights=False)

    def save(self, path: str) -> None:
        np.savez(
            path,
            B=self.B, B_hvc=self.B_hvc, JEI=self.JEI, JIE=self.JIE,
            n_e=self.n_e, n_i=self.n_i, n_hvc=self.n_hvc, n_aud=self.n_aud,
            tau_e=self.tau_e, tau_i=self.tau_i, tau_s=self.tau_s,
            A_jie=self.A_jie, J_max_ie=self.J_max_ie,
            c_B=self.c_B, B_scale=self.B_scale,
            c_hvc=self.c_hvc, B_hvc_scale=self.B_hvc_scale,
            c_JEI=self.c_JEI, ltp_window_ms=self.ltp_window_ms,
            drive_e=self.drive_e, drive_i=self.drive_i,
            noise_e=self.noise_e, noise_i=self.noise_i,
            t_refr_ms=self.t_refr_ms, a_sfa=self.a_sfa, tau_sfa=self.tau_sfa,
            target_rate=self.target_rate, target_rate_i=self.target_rate_i,
            aud_delay_ms=self.aud_delay_ms, hvc_delay_ms=self.hvc_delay_ms,
            tau_stdp=self.tau_stdp,
            theta_e=self.theta_e, theta_i=self.theta_i,
        )

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "VocalErrorNetV2":
        d = np.load(path)
        obj = cls.__new__(cls)
        obj._sim_rng = default_rng(None if seed is None else seed + 10_000)
        for k in ("n_e", "n_i", "n_hvc", "n_aud"):
            setattr(obj, k, int(d[k]))
        scalar_keys = (
            "tau_e", "tau_i", "tau_s", "A_jie", "J_max_ie",
            "c_B", "B_scale", "c_hvc", "B_hvc_scale", "c_JEI",
            "drive_e", "drive_i", "noise_e", "noise_i",
            "target_rate", "target_rate_i",
            "tau_stdp", "theta_i", "t_refr_ms", "a_sfa", "tau_sfa",
        )
        # Models saved before these knobs were removed carry extra keys; ignoring them
        # rather than reading them is the point. The missing-key default of 0.0 also does
        # the right thing for ``t_refr_ms``: a model trained before the refractory period
        # existed had none, so it reloads with the dynamics it was actually trained under
        # rather than silently acquiring the new constructor default.
        for k in scalar_keys:
            setattr(obj, k, float(d[k]) if k in d else 0.0)
        # Models saved before the window replaced the rate carry r_i_th, whose trace
        # threshold r * tau * 1e-3 is the trace one I spike leaves after
        # tau * ln(1 / (r * tau * 1e-3)) ms. A rate of 0 meant no LTD: an infinite window.
        if "ltp_window_ms" in d:
            obj.ltp_window_ms = float(d["ltp_window_ms"])
        else:
            tau = obj.tau_stdp if obj.tau_stdp > 0.0 else obj.tau_s
            xi_th = float(d["r_i_th"]) * tau * 1e-3 if "r_i_th" in d else 0.0
            obj.ltp_window_ms = -tau * float(np.log(xi_th)) if xi_th > 0.0 else np.inf
        obj.aud_delay_ms  = int(d["aud_delay_ms"]) if "aud_delay_ms" in d else 0
        obj.hvc_delay_ms  = int(d["hvc_delay_ms"]) if "hvc_delay_ms" in d else 0
        raw_theta = d["theta_e"] if "theta_e" in d else np.array(0.0)
        obj.theta_e = (raw_theta.copy() if raw_theta.ndim > 0
                       else np.full(obj.n_e, float(raw_theta), dtype=np.float64))
        obj.B        = d["B"]
        obj.B_hvc    = d["B_hvc"]
        obj.JEI      = d["JEI"]
        obj.JIE      = d["JIE"]
        obj.v_reset  = 0.0
        return obj

    def to_brian_weights(self) -> dict:
        """Everything :mod:`spiking_ven.brian2_runtime` needs to run this network.

        Weight matrices (float64, post x pre), the per-neuron E thresholds, and every
        scalar the dynamics use -- including spike-frequency adaptation, the refractory
        period and the conduction delays, without which the Brian2 network would not be
        the one that was trained.
        """
        return {
            "B":            self.B.astype(np.float64),
            "B_hvc":        self.B_hvc.astype(np.float64),
            "JIE":          self.JIE.astype(np.float64),
            "JEI":          self.JEI.astype(np.float64),
            "theta_e":      self.theta_e.astype(np.float64),
            "theta_i":      float(self.theta_i),
            "tau_s":        float(self.tau_s),
            "tau_e":        float(self.tau_e),
            "tau_i":        float(self.tau_i),
            "drive_e":      float(self.drive_e),
            "drive_i":      float(self.drive_i),
            "noise_e":      float(self.noise_e),
            "noise_i":      float(self.noise_i),
            "v_reset":      float(self.v_reset),
            "a_sfa":        float(self.a_sfa),
            "tau_sfa":      float(self.tau_sfa),
            "t_refr_ms":    float(self.t_refr_ms),
            "aud_delay_ms": int(self.aud_delay_ms),
            "hvc_delay_ms": int(self.hvc_delay_ms),
            "n_e":          int(self.n_e),
            "n_i":          int(self.n_i),
            "n_aud":        int(self.n_aud),
            "n_hvc":        int(self.n_hvc),
        }

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _sim(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        learn_weights: bool = False,
    ):
        dt      = 1.0
        decay_e = float(np.exp(-dt / self.tau_e))
        decay_i = float(np.exp(-dt / self.tau_i))
        # The STDP trace decays at tau_stdp; xi_th is where one I spike's trace has decayed
        # to after ltp_window_ms, so the window holds whatever the trace constant.
        tau_stdp_eff = self.tau_stdp if self.tau_stdp > 0.0 else self.tau_s
        alpha_stdp   = float(np.exp(-dt / tau_stdp_eff))
        xi_th        = float(np.exp(-self.ltp_window_ms / tau_stdp_eff))

        h_hvc = spike_to_rate(hvc_spikes, self.tau_s, dt).astype(np.float64)
        h_aud = spike_to_rate(aud_spikes,  self.tau_s, dt).astype(np.float64)

        sE_mat, sI_mat, JIE = _sim_loop_v2(
            h_aud, h_hvc,
            self.B.astype(np.float64),
            self.B_hvc.astype(np.float64),
            self.JEI.astype(np.float64),
            self.JIE.astype(np.float64),
            decay_e, decay_i,
            self.theta_e, float(self.theta_i),
            float(self.v_reset), float(self.noise_e), float(self.noise_i),
            self.t_refr_ms, self.a_sfa, self.tau_sfa,
            float(self.A_jie), float(self.J_max_ie), xi_th,
            float(self.drive_e), float(self.drive_i),
            learn_weights, alpha_stdp, self._sim_rng,
            aud_delay_ms=self.aud_delay_ms,
            hvc_delay_ms=self.hvc_delay_ms,
        )

        if learn_weights:
            self.JIE = JIE.astype(np.float32)
        return sE_mat, sI_mat
