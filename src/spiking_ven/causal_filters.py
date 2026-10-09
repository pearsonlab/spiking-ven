"""Causal minimum-phase filter bank -- strictly past in, sparse code out.

Each filter is one spacetime kernel ``V[k]`` of shape (C, L), spanning the L frames
before now. Inference at time t touches only frames < t, so an activation depends on
nothing at or after t.

    drive:       a_k(t) = <V[k], X[:, t-L:t]>                one matmul, all t at once
    competition: s(t) = argmin_s 0.5||P_t - sum_k s_k V_k||^2 + sum_k (lam+b_k)|s_k|
                 by ISTA on the precomputed drive and Gram, so correlated filters
                 explain each other away rather than all firing together

The kernels are constrained to be **minimum phase** (:func:`min_phase`). That is what
makes the code prompt. An unconstrained kernel drifts toward near-linear phase, whose
group delay is ~L/2; the patch encoder's atoms sat at an energy centroid of 7.5 of 16
frames, which is exactly that, and it cost ~30 ms of encoder latency. Minimum phase
places the energy as early as the magnitude response allows -- it constrains *when* a
filter responds without constraining *what* it is tuned to. Measured: centroid 7.5 ->
10.9 of 16, energy in the newest 4 frames 0.26 -> 0.50, coder latency 30 ms -> 2 ms,
with no loss of forward/reversed selectivity.

Historical note, because the idea is tempting and does not work as stated: these kernels
used to carry an extra "prediction" column V[k][:, L] trained on one-step-ahead error,
the intent being to reward kernels informative about what comes next. It was inert.
``infer`` reads only the analysis columns, the learning gradient is elementwise in the
lag index so prediction error never reached them, and :func:`min_phase` did not mix it
in either -- verified bit-identical codes under perturbation of that column alone. Every
bit of the latency improvement came from minimum phase. Making the bank genuinely
predictive would mean training the analysis kernels so their code predicts the next
frame, not bolting a readout onto a reconstruction-trained code.
"""

import numpy as np

__all__ = ["causal_meansub", "infer", "learn", "min_phase", "prune_silent"]


def causal_meansub(X, w):
    """Subtract the mean over the trailing w frames. Causal, and matches the patch
    encoder's per-window DC removal -- without it the code is level-dominated and goes
    direction-blind (measured forward/reversed correlation 0.998)."""
    C, T = X.shape
    cs = np.cumsum(np.concatenate([np.zeros((C, 1)), X], axis=1), axis=1)
    idx = np.arange(T)
    lo = np.maximum(idx - w + 1, 0)
    means = (cs[:, idx + 1] - cs[:, lo]) / (idx - lo + 1)
    return X - means.mean(axis=0, keepdims=True)


def _past_windows(X, L):
    """(T, C*L) matrix whose row t is the strictly-past window X[:, t-L:t], zero-padded."""
    C, T = X.shape
    padded = np.concatenate([np.zeros((C, L)), X[:, :-1]], axis=1) if T else X
    win = np.lib.stride_tricks.sliding_window_view(padded, L, axis=1)[:, :T, :]
    return win.transpose(1, 0, 2).reshape(T, C * L)


def _roughness_grad(V):
    """Gradient of 0.5 * ||D V||^2, D the first difference along channel and lag.

    This is the smoothness prior, not weight decay. A plain L2 penalty on the entries
    cannot do anything here: every kernel is renormalised to unit norm after each update,
    and L2 is a uniform rescale, so it cancels exactly (and elastic net collapses to its
    L1 soft-threshold for the same reason). Penalising *differences* changes entries
    relative to each other, so it survives the renormalisation.

    Neumann (reflecting) boundaries, so the edges are not pulled toward zero.
    """
    g = np.zeros_like(V)
    for axis in (1, 2):
        d = np.diff(V, axis=axis)
        g -= np.diff(d, axis=axis, prepend=0.0, append=0.0)
    return g


def infer(X, V, lam, n_iter=20, bias=None):
    """Strictly causal sparse activations. ``V`` is (K, C, L); returns S (K, T).

    ``bias`` is a per-filter threshold offset from :func:`learn`'s usage equalisation.
    It must be carried to inference, exactly as the patch encoder carries its ``_bias``
    -- without it the thresholds are wrong and the usage balance is lost.
    """
    K, C, L = V.shape
    T = X.shape[1]
    W = V.reshape(K, C * L)
    G = W @ W.T
    step = 1.0 / max(float(np.linalg.eigvalsh(G).max()), 1e-12)
    b = np.zeros(K) if bias is None else np.asarray(bias, dtype=float)
    thr = np.maximum((lam + b) * step, 0.0)                 # per-filter threshold
    drive = _past_windows(X, L) @ W.T                       # (T, K), all t at once
    S = np.zeros((T, K))
    for t in range(T):
        s = np.zeros(K)
        a = drive[t]
        for _ in range(n_iter):
            s = s + step * (a - G @ s)
            s = np.sign(s) * np.maximum(np.abs(s) - thr, 0.0)
        S[t] = s
    return S.T


def learn(cochs, K=64, L=16, lam=0.3, n_epochs=30, n_iter=20, lr=0.1, seed=42, log=print,
          target_usage=0.10, lr_bias=0.0, ema_decay=0.9, mp_strength=0.25, l1_V=7e-5,
          smooth_V=0.0):
    """Alternate causal inference with a gradient step on the kernels.

    Objective is reconstruction of the strictly-past window from the sparse code, with
    the kernels pulled toward minimum phase after each update. Returns ``(V, bias)``;
    the bias must be passed to :func:`infer`.

    ``smooth_V`` weights a roughness penalty on the kernels -- L2 on first differences
    along channel and lag (see :func:`_roughness_grad`). Use this, not weight decay: a
    plain L2 penalty is exactly cancelled by the per-filter unit-norm renormalisation.

    Both penalties have to be scaled against the data gradient, which is small: at
    ``lr = 0.1`` the median gradient step on an entry is 2.3e-6 against entries of
    magnitude 2.6e-2. An ``l1_V`` of 0.03 puts the soft-threshold 1300x above that and
    annihilates the dictionary -- 99.8% of entries zero, one surviving tap per kernel.
    The useful range is a threshold of order the gradient step, so ``lr * l1_V`` around
    1e-5 to 1e-4.

    ``lam`` now carries the code sparsity, and with it how many filters survive: 0.3 ->
    10.7% activation and 56 of 64 filters, 0.6 -> 3.2% and 23, 1.2 -> 0.15% and 1. The
    default is 0.3, chosen on song cancellation (song/reversed 2.47 against 1.72 at 0.6),
    which is what the network exists to do. It costs responder fraction (52% against 43%,
    bird 41%) and peak change (316 Hz against 262); DAF latency is inside the bird's
    23 +/- 12 ms either way. ``lam`` is also the only thing rejecting broadband noise --
    at 0.3 white noise drives the dictionary 17x harder than at 0.6 -- but there is no
    experimental measurement of continuous noise during singing, so that is a diagnostic
    and was not allowed to set the default.

    ``l1_V`` puts an L1 penalty on the kernel entries, applied as a soft-threshold after
    each update. Without it the kernels do not converge to anything: at 3% activation a
    filter fires too rarely to accumulate gradient, and over 80 epochs the past frames
    move almost not at all -- measured median correlation with the random initialisation
    +0.67 overall and +0.81 over the 15 older frames, with reconstruction error improving
    5%. The visible "speckle" in those kernels was untouched initialisation, not a
    learned receptive field. L1 drives those entries to zero instead of leaving noise
    there, so what survives is what the data paid for.

    Note the tension with ``mp_strength``: the minimum-phase step is an FFT resynthesis
    and returns a dense kernel, so it partly refills what L1 zeroed. The order here is
    gradient -> minimum-phase pull -> soft-threshold -> renormalise, which leaves the
    stored kernels genuinely sparse and the minimum-phase constraint approximate. Both
    are soft pulls; neither is imposed exactly.

    ``lr_bias`` drives the per-filter threshold bias toward ``target_usage`` (lifetime
    sparsity equalisation). It defaults to OFF. Forcing all K filters to fire equally on
    a corpus this stereotyped buys nothing: it does not reduce redundancy (effective rank
    29/64 on, 32/64 off) and it pins the code so sparse that the dictionary cannot train.
    Letting filters die is the honest outcome -- if there were residual variance left
    unexplained, the spare filters would go find it.

    ``mp_strength`` is how hard that pull is: ``V <- (1-a) V + a * min_phase(V)``, a
    proximal step on the distance to the minimum-phase manifold rather than a hard
    projection. ``a = 0`` disables it; ``a = 1`` is the strict projection, and the strict
    projection has a **degenerate fixed point**. Minimum phase turns a white magnitude
    response into an impulse, so each strict projection concentrates energy at lag 0,
    which whitens the lag-axis spectrum, which makes the next projection concentrate
    harder. Measured at ``a = 1``: lag-axis spectral flatness 0.08 -> 0.57 and 58% of
    every kernel's energy in the single most recent frame -- an instantaneous spectral
    code. Latency is fine there (it was already solved) but the temporal integration that
    sharpens song cancellation is gone: song/reversed contrast fell 2.29 -> 1.83 and the
    DAF responder fraction went 51% -> 74% against the bird's 41%.

    The default is not tuned. Measured over ``a`` = 0.10 / 0.25 / 0.50, the bank lands on
    the same place every time -- kernel energy centroid 10.1 of 16 (sd 0.5), newest four
    frames holding 0.49 / 0.49 / 0.48 of the energy. The soft pull has one stable fixed
    point and only ``a = 1`` escapes it into the impulse. So the choice here is between
    pulling toward minimum phase and projecting onto it, not a value to sweep.

    This was invisible for a while because :func:`min_phase` defaulted to an FFT short
    enough to alias the cepstrum, which made the "projection" accidentally soft. That is
    a bad way to set a hyperparameter; the softness is explicit here and the projection
    itself is now accurate.

    Usage equalisation must be strong enough to REVERSE a dropout, not just discourage
    one. At ``lr_bias=0.005`` with ``ema_decay=0.99`` it was not: a filter that stopped
    winning had its usage EMA decay away while its bias crept down at most
    ``lr_bias * target_usage`` per step, so over 191 epochs the bias floor only reached
    -0.056 while it needed to reach ``-lam`` = -0.1 before that filter's threshold could
    return to zero. Dropout was effectively one-way and the dictionary bled filters:
    64/64 used at epoch 41, 45/64 by epoch 191, while the error had converged by epoch
    20. Recovery now outpaces the decay, and the bias is floored at ``-lam`` so a starved
    filter always reaches a zero threshold rather than an unreachable one. Without any
    equalisation this collapsed outright: 7 filters carried 90% of the activation energy
    and 35 of 64 never fired, so they never received gradient and stayed at their random
    initialisation.
    """
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    bias = np.zeros(K)
    usage_ema = np.full(K, target_usage)
    for ep in range(n_epochs):
        recon_err, act = 0.0, 0.0
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            S = infer(X, V, lam, n_iter, bias=bias)
            P = _past_windows(X, L)                          # (T, C*L)
            R = (P - S.T @ V.reshape(K, C * L)).reshape(T, C, L)
            V = V + lr * np.einsum("kt,tcl->kcl", S, R) / max(T, 1)
            if smooth_V > 0:
                V = V - lr * smooth_V * _roughness_grad(V)
            if mp_strength > 0:
                V = (1.0 - mp_strength) * V + mp_strength * min_phase(V)
            if l1_V > 0:
                V = np.sign(V) * np.maximum(np.abs(V) - lr * l1_V, 0.0)
            V /= np.maximum(np.linalg.norm(V.reshape(K, -1), axis=1), 1e-9)[:, None, None]
            usage = (S != 0).mean(axis=1)
            usage_ema = ema_decay * usage_ema + (1.0 - ema_decay) * usage
            bias += lr_bias * (usage_ema - target_usage)
            np.clip(bias, -lam, None, out=bias)   # threshold can reach 0, never below
            recon_err += float(np.mean(R ** 2))
            act += float(np.mean(S != 0))
        if ep % 5 == 0 or ep == n_epochs - 1:
            log(f"  epoch {ep+1:3d}/{n_epochs}  recon={recon_err/len(cochs):.6f}  "
                f"active={act/len(cochs)*100:.2f}%  used={(usage_ema > 1e-4).sum()}/{K}  "
                f"zeros={100 * float(np.mean(V == 0)):.0f}%  "
                f"bias=[{bias.min():+.3f},{bias.max():+.3f}]")
    return V, bias


def min_phase(V, n_fft=None, eps=1e-9):
    """Project every channel's temporal kernel onto its minimum-phase counterpart.

    Among all causal filters sharing a magnitude response, the minimum-phase one has its
    energy as early as possible -- minimum group delay, maximal partial energy sums.

    Standard real-cepstrum construction: fold the anticausal part of the log-spectrum.
    ``V`` is (K, C, L). Column 0 is the OLDEST frame, so kernels are reversed into lag
    order before projection and back afterwards.

    ``n_fft`` has to be generous. The cepstrum of a short kernel is long, so a short FFT
    aliases it and the projection stops being magnitude-preserving -- which would mean
    quietly retuning the filters, the one thing this is supposed not to do. Measured
    relative magnitude error at L=16: 7.7% at N=128 (the old ``8*L`` default), 1.2% at
    512, 0.5% at 1024. The FFT is over (K, C) kernels once per gradient step and is
    negligible against ISTA, so the default is sized for accuracy, not speed. A residual
    error remains at any N because the minimum-phase counterpart of a finite kernel is in
    general IIR and is truncated back to L frames here.
    """
    K, C, L = V.shape
    N = int(n_fft or max(64 * L, 1024))
    g = V[:, :, ::-1]                                 # lag order: g[..., 0] = most recent
    G = np.fft.rfft(g, N, axis=2)
    logmag = np.log(np.abs(G) + eps)
    full = np.concatenate([logmag, logmag[:, :, -2:0:-1]], axis=2)
    ceps = np.fft.ifft(full, N, axis=2).real
    fold = np.zeros_like(ceps)
    fold[:, :, 0] = ceps[:, :, 0]
    fold[:, :, 1:N // 2] = 2.0 * ceps[:, :, 1:N // 2]
    fold[:, :, N // 2] = ceps[:, :, N // 2]
    g_mp = np.fft.irfft(np.exp(np.fft.rfft(fold, N, axis=2)), N, axis=2)[:, :, :L]
    return np.ascontiguousarray(g_mp[:, :, ::-1])     # back to oldest-first


def prune_silent(V, bias, cochs, lam, n_iter=20):
    """Drop filters that never win on the training corpus. Returns (V, bias, kept).

    Without usage equalisation the dictionary is free to shed filters, which on a corpus
    this stereotyped it does: at ``lam = 0.6`` only 19 of 64 ever fire. The other 45 do
    not vanish -- they decay into near-identical stubs that lose every competition -- so
    they have to be removed explicitly. Leaving them in is not harmless: they become
    permanently silent auditory channels, and the VEN samples each E cell's auditory
    inputs from that width, so they dilute its convergence.
    """
    fired = np.zeros(V.shape[0], dtype=bool)
    for X in cochs:
        fired |= (infer(X, V, lam, n_iter, bias=bias) != 0).any(axis=1)
    kept = np.flatnonzero(fired)
    return V[kept], np.asarray(bias)[kept], kept


class CausalFilterEncoder:
    """Adapter so the VEN pipeline can consume a causal filter bank.

    Mirrors the surface of :class:`~spiking_ven.olshausen_field.OlshausenFieldEncoder`
    that ``evaluate.encode_acts`` actually uses -- ``n_channels`` and an encode call with
    the same normalisation contract -- so the two encoders are swappable and the
    downstream stimulus construction, spike conversion and VEN training are untouched.
    """

    def __init__(self, V, bias, lam, coch_params):
        self.V = np.asarray(V)
        self.bias = np.asarray(bias)
        self.lam = float(lam)
        self.coch_params = dict(coch_params)
        self.L = self.V.shape[2]

    @property
    def n_channels(self) -> int:
        """Output width: one activation per filter, matching the patch encoder's n_bases."""
        return int(self.V.shape[0])

    def encode_signal(self, signal, *, sr, n_ista=20, upsample_to_ms=False,
                      T_out_ms=None, rms_from=None, divisive_gain=False, dn_sigma=1e-6):
        from .cochleagram import cochleagram

        cp = self.coch_params
        sig = np.asarray(signal, dtype=np.float64)
        ref = sig if rms_from is None else np.asarray(rms_from, dtype=np.float64)
        rms = float(np.sqrt(np.mean(ref ** 2)))
        sig = sig / max(rms, 1e-12) * cp["rms_ref"]

        X = cochleagram(sig, sr, n_channels=cp["n_channels"], frame_rate=cp["frame_rate"],
                        lo_hz=cp["lo_hz"], hi_hz=cp["hi_hz"]).astype(np.float64)
        Xc = causal_meansub(X, self.L)
        acts = infer(Xc, self.V, self.lam, n_iter=n_ista, bias=self.bias)

        if divisive_gain:
            # Same two corrections as the patch path, with the causal analogues: divide by
            # the norm of the reconstructed past window so magnitude stops reporting how
            # well the filters matched, then restore level with the window mean that
            # causal_meansub removed.
            recon = np.linalg.norm(acts.T @ self.V.reshape(self.V.shape[0], -1), axis=1)
            mu = (X - Xc).mean(axis=0)
            acts = acts * (mu / (dn_sigma + recon))[None, :]

        if upsample_to_ms:
            acts = np.repeat(acts, 1000 // cp["frame_rate"], axis=1)
        if T_out_ms is not None:
            T = T_out_ms if upsample_to_ms else T_out_ms // (1000 // cp["frame_rate"])
            acts = (acts[:, :T] if acts.shape[1] >= T else
                    np.concatenate([acts, np.zeros((acts.shape[0], T - acts.shape[1]))], axis=1))
        return acts.astype(np.float32)

    def save(self, path):
        np.savez(path, V=self.V, bias=self.bias, lam=self.lam,
                 **{f"cp_{k}": v for k, v in self.coch_params.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        cp = {k[3:]: (v.item() if v.ndim == 0 else v) for k, v in d.items()
              if k.startswith("cp_")}
        return cls(d["V"], d["bias"], float(d["lam"]), cp)
