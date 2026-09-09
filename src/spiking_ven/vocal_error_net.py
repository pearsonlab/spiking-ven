"""Spiking E-I vocal error network — v2 (HVC→I architecture).

Architecture
------------
- Auditory spikes  (n_aud, T)  →  B (n_e × n_aud)    →  E neurons  [fixed sparse]
- E neurons        →  JEI (n_i × n_e)                 →  I neurons  [fixed]
- HVC spikes       (n_hvc, T) →  B_hvc (n_i × n_hvc) →  I neurons  [fixed sparse]
- I neurons        →  JIE (n_e × n_i)                 →  E neurons  [plastic — causal Hebbian]

Plasticity
----------
- fit()   : song stimulus — JIE STDP on, theta adaptation off
- adapt() : novel stimulus — JIE STDP off, per-neuron theta adaptation on
            theta_e[k] += alpha_theta * (sE[k] - r_e_target_per_bin)
            Gated to novel so homeostasis doesn't fight JIE learning.
- use_mask: optional structural mask keeps JIE at initial sparsity (default off)
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
    decay_e, decay_i, alpha_s,
    theta_e, theta_i, v_reset, noise_e, noise_i,
    A_jie, J_max_ie, xi_th, xe_th,
    drive_e, drive_i,
    learn_weights,
    JIE_mask=None,
    learn_theta=False,
    alpha_theta=0.0,
    r_e_target_per_bin=0.0,
    learn_jei=False,
    A_jei=0.0,
    r0_xe_jei=0.0,
    aud_delay_ms: int = 0,
    hvc_delay_ms: int = 0,
    decay_jie: float = 0.0,
    W_max_ie: float = 0.0,
    epsilon_hltd: float = 0.0,
    epsilon_col: float = 0.0,
    alpha_stdp: float = 0.0,
):
    n_e = B.shape[0]
    n_i = JEI.shape[0]
    T   = h_aud.shape[1]

    V_e     = np.zeros(n_e)
    V_i     = np.zeros(n_i)
    sE_prev = np.zeros(n_e)
    sI_prev = np.zeros(n_i)
    x_e     = np.zeros(n_e)
    x_i     = np.zeros(n_i)

    sE_mat  = np.zeros((n_e, T), dtype=np.float32)
    sI_mat  = np.zeros((n_i, T), dtype=np.float32)
    x_e_acc = np.zeros(n_e)   # accumulates audio-gated x_e trace for BCM update
    x_e_all = np.zeros(n_e)   # unfiltered E trace for JEI homeostasis (all spikes, no audio gate)

    _zero_aud = np.zeros(h_aud.shape[0])
    _zero_hvc = np.zeros(h_hvc.shape[0])

    for t in range(T):
        h_aud_t = h_aud[:, t - aud_delay_ms] if t >= aud_delay_ms else _zero_aud
        h_hvc_t = h_hvc[:, t - hvc_delay_ms] if t >= hvc_delay_ms else _zero_hvc

        # --- E dynamics ---
        audio_drive_e = B @ h_aud_t
        u_e = drive_e + audio_drive_e - JIE @ sI_prev
        if noise_e > 0.0:
            u_e += np.random.normal(0.0, noise_e, n_e)
        V_e = decay_e * V_e + (1.0 - decay_e) * u_e
        sE  = (V_e > theta_e).astype(np.float64)
        V_e[sE > 0.5] = v_reset
        sE_mat[:, t] = sE

        # --- I dynamics ---
        u_i = drive_i + JEI @ sE_prev + B_hvc @ h_hvc_t
        if noise_i > 0.0:
            u_i += np.random.normal(0.0, noise_i, n_i)
        V_i = decay_i * V_i + (1.0 - decay_i) * u_i
        sI  = (V_i > theta_i).astype(np.float64)
        V_i[sI > 0.5] = v_reset
        sI_mat[:, t] = sI

        # Unfiltered E trace used for JEI homeostasis (always updated for efficiency)
        x_e_all = alpha_s * x_e_all + sE

        if learn_weights:
            # Causal STDP: I-then-E order (I precedes E → potentiate JIE).
            # Update at the moment of each audio-gated E spike using the current
            # I trace — captures recent I firing before this E spike occurred.
            # xe_th is unused here; xi_th debiases x_i above tonic I baseline.
            # alpha_stdp (if > 0) gives a separate, narrower coincidence window
            # for x_i only — allowing temporal specificity without affecting audio
            # smoothing (alpha_s) or homeostasis.
            _alpha_xi = alpha_stdp if alpha_stdp > 0.0 else alpha_s
            audio_gate_e = (audio_drive_e > 0.0).astype(np.float64)
            gate_sE = sE * audio_gate_e
            x_e = alpha_s * x_e + gate_sE  # retained for BCM
            x_i = _alpha_xi * x_i + sI
            x_e_acc += x_e
            JIE += A_jie * np.outer(gate_sE, x_i - xi_th)
            # Heterosynaptic competition — Fiete et al. 2010 summed-weight limit rule,
            # applied symmetrically at both pre- and post-synaptic neurons:
            #
            #   Incoming (per E neuron, row of JIE): when E neuron e fired with
            #   audio gate AND its total incoming I weight ≥ W_max_ie, depress all
            #   of its incoming I→E synapses.  Drives WTA over which I neuron
            #   inhibits each E neuron.
            #
            #   Outgoing (per I neuron, column of JIE): when I neuron i had an
            #   elevated trace (contributed LTP) AND any E neuron fired with audio
            #   gate AND its total outgoing weight ≥ W_max_ie, depress all of its
            #   outgoing I→E synapses.  Drives WTA over which E neuron each I
            #   neuron targets — producing the time-matched diagonal structure.
            #
            # Sums are computed once before either update to avoid order-dependence.
            if W_max_ie > 0.0:
                _do_row = epsilon_hltd > 0.0
                _do_col = epsilon_col > 0.0
                if _do_row or _do_col:
                    row_sum = JIE.sum(axis=1) if _do_row else None
                    col_sum = JIE.sum(axis=0) if _do_col else None
                    ltp_e = gate_sE > 0.0
                    if _do_row:
                        compete_e = ltp_e & (row_sum >= W_max_ie)
                        if compete_e.any():
                            JIE[compete_e, :] -= epsilon_hltd
                    if _do_col and ltp_e.any():
                        ltp_i = (x_i - xi_th) > 0.0
                        compete_i = ltp_i & (col_sum >= W_max_ie)
                        if compete_i.any():
                            JIE[:, compete_i] -= epsilon_col
            if decay_jie > 0.0:
                JIE *= decay_jie
            if JIE_mask is not None:
                JIE *= JIE_mask
            np.clip(JIE, 0.0, J_max_ie, out=JIE)

        if learn_jei:
            # iSTDP on E→I: grow JEI when I fires and E was above target; shrink otherwise.
            # Rule: ΔJEI = A_jei * outer(sI, x_e_all − r0_xe)
            # r0_xe is the steady-state x_e trace at the target E rate; when E fires at
            # exactly target, the rule averages to zero → stable homeostatic fixed point.
            JEI += A_jei * np.outer(sI, x_e_all - r0_xe_jei)
            np.clip(JEI, 0.0, None, out=JEI)  # E→I weights non-negative

        if learn_theta:
            theta_e += alpha_theta * (sE - r_e_target_per_bin)

        sE_prev = sE
        sI_prev = sI

    return sE_mat, sI_mat, JIE, JEI, theta_e, x_e_acc / T


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
    tau_s        : I pre-synaptic trace time constant for STDP (ms)
    A_jie        : causal Hebbian STDP amplitude for JIE (I→E)
    J_max_ie     : hard upper bound on JIE weights
    c_B          : auditory→E connection probability
    B_scale      : mean weight scale for auditory→E (compensates for sparsity)
    c_hvc        : HVC→I connection probability
    B_hvc_scale  : mean weight scale for HVC→I (tune so HVC burst drives I reliably)
    c_JEI        : E→I connection probability (fixed throughout)
    c_JIE        : I→E initial connection probability (plastic weights)
    r_e_th       : E baseline rate threshold for STDP (Hz); x_e below this → no LTP on JIE.
                   Set to the expected tonic E firing rate so plasticity only occurs for
                   audio-evoked activity above baseline.  Ignored when alpha_bcm > 0 (BCM
                   sliding threshold replaces this scalar).
    r_i_th       : I baseline rate threshold for STDP (Hz); x_i below this → LTD on JIE.
                   Set to mean I rate during song so rule is zero-mean for uncorrelated activity.
    alpha_bcm    : BCM sliding-threshold learning rate.  When > 0, a per-neuron threshold
                   theta_bcm tracks the running mean audio-gated x_e during fit() calls and
                   replaces the scalar xe_th.  E neurons firing above their mean get JIE
                   potentiation; below-mean neurons get de-potentiation.  alpha_bcm=0 disables.
    W_max_ie     : summed incoming JIE weight limit per E neuron for heterosynaptic competition
                   (Fiete et al. 2010).  When > 0 and an E neuron fires with audio gate AND
                   its row sum of JIE exceeds this limit, all its incoming I→E weights are
                   decremented by epsilon_hltd.  Combined with causal STDP this drives WTA:
                   each E neuron converges to strong inhibition from its most time-correlated
                   I neuron.  Set to 0 to disable (default).
    epsilon_hltd : uniform hLTD decrement applied to all incoming JIE synapses of a competing
                   E neuron per timestep.  Should be comparable to a typical single-timestep
                   STDP increment: A_jie × peak_xi ≈ A_jie × tau_s × 1e-3.
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
        c_JIE: float = 0.5,
        r_e_th: float = 0.0,
        r_i_th: float = 0.0,
        drive_e: float = 0.155,
        drive_i: float = 0.058,
        noise_e: float = 0.1,
        noise_i: float = 0.0,
        target_rate: float = 0.025,
        target_rate_i: float = 0.05,
        alpha_theta: float = 0.0,
        r_e_target: float = 4.0,
        r_e_song_target: float = 0.0,
        alpha_bcm: float = 0.0,
        A_jei_plastic: float = 0.0,
        use_mask: bool = False,
        aud_delay_ms: int = 0,
        hvc_delay_ms: int = 0,
        tau_w_ie: float = 0.0,
        W_max_ie: float = 0.0,
        epsilon_hltd: float = 0.0,
        epsilon_col: float = 0.0,
        tau_stdp: float = 0.0,
        lambda_huber: float = 0.0,
        huber_delta: float = 0.1,
        seed: int | None = None,
    ) -> None:
        rng = default_rng(seed)
        self._rng          = rng
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
        self.c_JIE         = c_JIE
        self.r_e_th        = r_e_th
        self.r_i_th        = r_i_th
        self.drive_e       = drive_e
        self.drive_i       = drive_i
        self.noise_e       = noise_e
        self.noise_i       = noise_i
        self.target_rate   = target_rate
        self.target_rate_i = target_rate_i
        self.alpha_theta      = alpha_theta
        self.r_e_target       = r_e_target
        self.r_e_song_target  = float(r_e_song_target)
        self.alpha_bcm        = alpha_bcm
        self.A_jei_plastic    = A_jei_plastic
        self.use_mask         = use_mask
        self.aud_delay_ms  = int(aud_delay_ms)
        self.hvc_delay_ms  = int(hvc_delay_ms)
        self.tau_w_ie      = float(tau_w_ie)
        self.W_max_ie      = float(W_max_ie)
        self.epsilon_hltd  = float(epsilon_hltd)
        self.epsilon_col   = float(epsilon_col)
        self.tau_stdp      = float(tau_stdp)
        self.lambda_huber  = float(lambda_huber)
        self.huber_delta   = float(huber_delta)
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

        # E→I: sparse log-normal, fixed (not plastic)
        srKE = np.sqrt(n_e * c_JEI)
        self.JEI = _lognormal_sparse(
            (n_i, n_e), c_JEI, 1.7 / srKE / 10, 1.7 / srKE / 10 * 0.1, rng
        )

        # I→E: zero-initialized; structural mask defines allowed synapses.
        # Starting from zero means only song-driven E neurons ever get trained
        # (via STDP in fit()), leaving novel-E rows at zero.  The mask is sampled
        # independently so it is not derived from the initial weights.
        self.JIE = np.zeros((n_e, n_i), dtype=np.float32)
        self.JIE_mask = (rng.random((n_e, n_i)) < c_JIE).astype(np.float32)

        # E threshold: per-neuron array (adapts via adapt() when alpha_theta > 0).
        # Effective steady-state input is drive_e (tonic), not target_rate * tau_s —
        # tau_s is the STDP trace constant and must not appear in the membrane threshold.
        _decay_e  = float(np.exp(-1.0 / tau_e))
        _isi_e    = 1.0 / target_rate
        _theta_e0 = float(max(drive_e * (1.0 - _decay_e ** _isi_e), 1e-3))
        self.theta_e = np.full(n_e, _theta_e0, dtype=np.float64)

        # BCM sliding threshold: per-neuron running mean of audio-gated x_e during fit()
        self.theta_bcm = np.zeros(n_e, dtype=np.float64)

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

    def fit(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        n_renditions: int,
        T_song: int,
        T_burn: int,
        T_post: int = 200,
        verbose: bool = True,
    ) -> list[float]:
        """Train on paired HVC+auditory input. Updates JIE; also updates theta_e
        toward r_e_song_target if set (bidirectional homeostasis)."""
        song_theta = self.r_e_song_target > 0.0
        sE_mat, _sI, rate_history = self._sim(
            hvc_spikes, aud_spikes,
            learn_weights=True,
            learn_theta=song_theta,
            r_e_target_override=self.r_e_song_target if song_theta else 0.0,
            n_renditions=n_renditions, T_song=T_song,
            T_burn=T_burn, T_post=T_post, verbose=verbose,
        )
        return rate_history

    def adapt(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
    ) -> np.ndarray:
        """Novel-stimulus pass: homeostasis on, JIE STDP off. Returns sE_mat.

        Homeostasis is via JEI iSTDP when A_jei_plastic > 0, else via per-neuron
        theta adaptation (learn_theta).  Both can be enabled simultaneously.
        """
        use_jei  = self.A_jei_plastic > 0.0
        use_theta = self.alpha_theta > 0.0
        sE, _sI, _ = self._sim(hvc_spikes, aud_spikes,
                                learn_weights=False,
                                learn_theta=use_theta,
                                learn_jei=use_jei)
        return sE

    def transform(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
    ) -> np.ndarray:
        """Inference pass (no weight updates). Returns (n_e, T) float32 E spike matrix."""
        sE, _sI, _ = self._sim(hvc_spikes, aud_spikes, learn_weights=False)
        return sE

    def transform_all(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference pass. Returns (sE_mat, sI_mat) both (n, T) float32."""
        sE, sI, _ = self._sim(hvc_spikes, aud_spikes, learn_weights=False)
        return sE, sI

    def save(self, path: str) -> None:
        np.savez(
            path,
            B=self.B, B_hvc=self.B_hvc, JEI=self.JEI, JIE=self.JIE, JIE_mask=self.JIE_mask,
            n_e=self.n_e, n_i=self.n_i, n_hvc=self.n_hvc, n_aud=self.n_aud,
            tau_e=self.tau_e, tau_i=self.tau_i, tau_s=self.tau_s,
            A_jie=self.A_jie, J_max_ie=self.J_max_ie,
            c_B=self.c_B, B_scale=self.B_scale,
            c_hvc=self.c_hvc, B_hvc_scale=self.B_hvc_scale,
            c_JEI=self.c_JEI, c_JIE=self.c_JIE, r_e_th=self.r_e_th, r_i_th=self.r_i_th,
            drive_e=self.drive_e, drive_i=self.drive_i,
            noise_e=self.noise_e, noise_i=self.noise_i,
            target_rate=self.target_rate, target_rate_i=self.target_rate_i,
            alpha_theta=self.alpha_theta, r_e_target=self.r_e_target,
            r_e_song_target=self.r_e_song_target,
            alpha_bcm=self.alpha_bcm, theta_bcm=self.theta_bcm,
            A_jei_plastic=self.A_jei_plastic,
            use_mask=self.use_mask,
            aud_delay_ms=self.aud_delay_ms, hvc_delay_ms=self.hvc_delay_ms,
            tau_w_ie=self.tau_w_ie,
            W_max_ie=self.W_max_ie, epsilon_hltd=self.epsilon_hltd,
            epsilon_col=self.epsilon_col, tau_stdp=self.tau_stdp,
            lambda_huber=self.lambda_huber, huber_delta=self.huber_delta,
            theta_e=self.theta_e, theta_i=self.theta_i,
        )

    @classmethod
    def load(cls, path: str) -> "VocalErrorNetV2":
        d = np.load(path)
        obj = cls.__new__(cls)
        obj._rng = default_rng()
        for k in ("n_e", "n_i", "n_hvc", "n_aud"):
            setattr(obj, k, int(d[k]))
        scalar_keys = (
            "tau_e", "tau_i", "tau_s", "A_jie", "J_max_ie",
            "c_B", "B_scale", "c_hvc", "B_hvc_scale", "c_JEI", "c_JIE", "r_e_th", "r_i_th",
            "drive_e", "drive_i", "noise_e", "noise_i",
            "target_rate", "target_rate_i", "alpha_theta", "r_e_target", "r_e_song_target",
            "alpha_bcm", "A_jei_plastic", "tau_w_ie", "W_max_ie", "epsilon_hltd", "epsilon_col", "tau_stdp", "lambda_huber", "huber_delta", "theta_i",
        )
        for k in scalar_keys:
            setattr(obj, k, float(d[k]) if k in d else 0.0)
        obj.use_mask      = bool(d["use_mask"]) if "use_mask" in d else False
        obj.aud_delay_ms  = int(d["aud_delay_ms"]) if "aud_delay_ms" in d else 0
        obj.hvc_delay_ms  = int(d["hvc_delay_ms"]) if "hvc_delay_ms" in d else 0
        raw_theta = d["theta_e"] if "theta_e" in d else np.array(0.0)
        obj.theta_e = (raw_theta.copy() if raw_theta.ndim > 0
                       else np.full(obj.n_e, float(raw_theta), dtype=np.float64))
        raw_bcm = d["theta_bcm"] if "theta_bcm" in d else np.array(0.0)
        obj.theta_bcm = (raw_bcm.copy() if raw_bcm.ndim > 0
                         else np.zeros(obj.n_e, dtype=np.float64))
        obj.B        = d["B"]
        obj.B_hvc    = d["B_hvc"]
        obj.JEI      = d["JEI"]
        obj.JIE      = d["JIE"]
        obj.JIE_mask = d["JIE_mask"] if "JIE_mask" in d else (obj.JIE > 0).astype(np.float32)
        obj.v_reset  = 0.0
        return obj

    def to_brian_weights(self) -> dict:
        """Return trained weight matrices ready for build_ven_synapses(learn=False).

        Returns
        -------
        dict with keys:
            B, B_hvc, JIE, JEI  — float64 numpy arrays
            theta_e             — (n_e,) float64 array of per-neuron E thresholds
            xi_th               — dimensionless I-trace threshold (= r_i_th * tau_s * 1e-3)
            tau_s, tau_e, tau_i, drive_e, drive_i, A_jie, J_max_ie — scalar floats
            theta_i, v_reset    — scalar floats
            n_e, n_i, n_aud, n_hvc — ints
        """
        return {
            "B":        self.B.astype(np.float64),
            "B_hvc":    self.B_hvc.astype(np.float64),
            "JIE":      self.JIE.astype(np.float64),
            "JEI":      self.JEI.astype(np.float64),
            "theta_e":  self.theta_e.astype(np.float64),
            "theta_i":  float(self.theta_i),
            "xi_th":    float(self.r_i_th * self.tau_s * 1e-3),
            "tau_s":    float(self.tau_s),
            "tau_e":    float(self.tau_e),
            "tau_i":    float(self.tau_i),
            "drive_e":  float(self.drive_e),
            "drive_i":  float(self.drive_i),
            "noise_e":  float(self.noise_e),
            "noise_i":  float(self.noise_i),
            "v_reset":  float(self.v_reset),
            "A_jie":    float(self.A_jie),
            "J_max_ie": float(self.J_max_ie),
            "n_e":      int(self.n_e),
            "n_i":      int(self.n_i),
            "n_aud":    int(self.n_aud),
            "n_hvc":    int(self.n_hvc),
        }

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _sim(
        self,
        hvc_spikes: np.ndarray,
        aud_spikes: np.ndarray,
        learn_weights: bool = False,
        learn_theta: bool = False,
        learn_jei: bool = False,
        r_e_target_override: float = 0.0,
        n_renditions: int = 0,
        T_song: int = 0,
        T_burn: int = 0,
        T_post: int = 200,
        verbose: bool = True,
    ):
        dt      = 1.0
        decay_e = float(np.exp(-dt / self.tau_e))
        decay_i = float(np.exp(-dt / self.tau_i))
        alpha_s = float(np.exp(-dt / self.tau_s))
        # If tau_stdp is set, use a narrower kernel for the x_i STDP trace only.
        # xi_th is recalibrated to the effective kernel width so the I-baseline
        # debiasing stays correct regardless of tau_stdp.
        tau_stdp_eff = self.tau_stdp if self.tau_stdp > 0.0 else self.tau_s
        alpha_stdp   = float(np.exp(-dt / tau_stdp_eff))
        # BCM mode: use per-neuron sliding threshold; otherwise scalar.
        xe_th: float | np.ndarray = (
            self.theta_bcm.copy() if self.alpha_bcm > 0.0
            else float(self.r_e_th * self.tau_s * 1e-3)
        )
        xi_th     = float(self.r_i_th * tau_stdp_eff * 1e-3)
        mask      = self.JIE_mask.astype(np.float64) if self.use_mask else None
        decay_jie = float(np.exp(-1.0 / self.tau_w_ie)) if self.tau_w_ie > 0.0 else 0.0
        W_max_ie     = float(self.W_max_ie)
        epsilon_hltd = float(self.epsilon_hltd)
        epsilon_col  = float(self.epsilon_col)

        h_hvc = spike_to_rate(hvc_spikes, self.tau_s, dt).astype(np.float64)
        h_aud = spike_to_rate(aud_spikes,  self.tau_s, dt).astype(np.float64)

        r_e_target_eff = (r_e_target_override if r_e_target_override > 0.0
                          else float(self.r_e_target))
        # r0_xe_jei: steady-state x_e trace at target E rate; iSTDP zero-mean at target
        r0_xe_jei = r_e_target_eff * float(self.tau_s) * 1e-3

        theta_e = self.theta_e.copy()
        sE_mat, sI_mat, JIE, JEI, theta_e, x_e_mean = _sim_loop_v2(
            h_aud, h_hvc,
            self.B.astype(np.float64),
            self.B_hvc.astype(np.float64),
            self.JEI.astype(np.float64),
            self.JIE.astype(np.float64),
            decay_e, decay_i, alpha_s,
            theta_e, float(self.theta_i),
            float(self.v_reset), float(self.noise_e), float(self.noise_i),
            float(self.A_jie), float(self.J_max_ie), xi_th, xe_th,
            float(self.drive_e), float(self.drive_i),
            learn_weights,
            JIE_mask=mask,
            learn_theta=learn_theta,
            alpha_theta=float(self.alpha_theta),
            r_e_target_per_bin=r_e_target_eff * 1e-3,
            learn_jei=learn_jei,
            A_jei=float(self.A_jei_plastic),
            r0_xe_jei=r0_xe_jei,
            aud_delay_ms=self.aud_delay_ms,
            hvc_delay_ms=self.hvc_delay_ms,
            decay_jie=decay_jie,
            W_max_ie=W_max_ie,
            epsilon_hltd=epsilon_hltd,
            epsilon_col=epsilon_col,
            alpha_stdp=alpha_stdp,
        )

        if learn_weights:
            self.JIE = JIE.astype(np.float32)
            if self.lambda_huber > 0.0:
                # Huber regularization: L2 for small weights, L1 for large.
                # Applied once per _sim() call (i.e. per rendition).
                reg = np.where(self.JIE <= self.huber_delta,
                               self.lambda_huber * self.JIE / self.huber_delta,
                               self.lambda_huber)
                self.JIE = np.clip(self.JIE - reg, 0.0, self.J_max_ie).astype(np.float32)
            if self.alpha_bcm > 0.0:
                self.theta_bcm += self.alpha_bcm * (x_e_mean - self.theta_bcm)
        if learn_jei:
            self.JEI = JEI.astype(np.float32)
        if learn_theta:
            self.theta_e = theta_e

        rate_history: list[float] = []
        T_rend = T_song + T_post if T_song > 0 else 0
        if T_rend > 0:
            for r in range(n_renditions):
                t0   = T_burn + r * T_rend
                t1   = t0 + T_rend
                rate = float(sE_mat[:, t0:t1].mean())
                rate_history.append(rate)
                if verbose:
                    print(f"  rendition {r + 1}/{n_renditions}"
                          f"  mean_E={rate * 1000:.1f} Hz", flush=True)

        return sE_mat, sI_mat, rate_history
