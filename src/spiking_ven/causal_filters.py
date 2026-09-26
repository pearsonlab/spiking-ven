"""Causal predictive filter bank -- strictly past in, present out.

Each filter is one spacetime kernel V[k] spanning [t-L, t]:

    V[k][:, :L]  analysis   -- matched against the OBSERVED past window
    V[k][:,  L]  prediction -- what the filter says the present will be

Inference at time t touches only the analysis columns, so an activation depends on
nothing at or after t. Learning is offline and does see the present, which is what
shapes the prediction column; the split is deliberate -- the animal learns over
development but must respond causally.

    drive:       a_k(t) = <V[k][:, :L], X[:, t-L:t]>            one matmul, all t at once
    competition: s(t) = argmin_s 0.5||P_t - sum_k s_k W_k||^2 + lam||s(t)||_1
                 by ISTA on the precomputed drive and Gram, so correlated filters
                 explain each other away rather than all firing together
    prediction:  Xhat[:, t] = sum_k s_k(t) V[k][:, L]

Why this shape, rather than sparse coding a window: the learning signal is prediction
error on the *next* frame, which rewards kernels that are informative about what comes
next. Reconstructing a trailing window instead rewards kernels that explain the window's
bulk, which is what left the patch encoder's atoms flat (effective width 13.4 of 16,
energy centroid mid-window) and cost ~15 ms of lag.
"""

import numpy as np

__all__ = ["causal_meansub", "infer", "learn", "predict"]


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


def infer(X, V, lam, n_iter=20, bias=None):
    """Strictly causal sparse activations. Returns S (K, T).

    ``bias`` is a per-filter threshold offset from :func:`learn`'s usage equalisation.
    It must be carried to inference, exactly as the patch encoder carries its ``_bias``
    -- without it the thresholds are wrong and the usage balance is lost.
    """
    K, C, Lp = V.shape
    L = Lp - 1
    T = X.shape[1]
    W = V[:, :, :L].reshape(K, C * L)
    G = W @ W.T
    step = 1.0 / max(float(np.linalg.eigvalsh(G).max()), 1e-12)
    b = np.zeros(K) if bias is None else np.asarray(bias, dtype=float)
    thr = np.maximum((lam + b) * step, 0.0)                 # per-filter threshold
    drive = _past_windows(X, L) @ W.T                       # (T, K), all t at once
    S = np.zeros((T, K))
    s = np.zeros(K)
    for t in range(T):
        s = np.zeros(K)
        a = drive[t]
        for _ in range(n_iter):
            s = s + step * (a - G @ s)
            s = np.sign(s) * np.maximum(np.abs(s) - thr, 0.0)
        S[t] = s
    return S.T


def predict(S, V):
    """Xhat (C, T) from activations and the prediction columns."""
    return V[:, :, -1].T @ S


def learn(cochs, K=64, L=16, lam=0.05, n_epochs=30, n_iter=20, lr=0.1, seed=42, log=print,
          target_usage=0.10, lr_bias=0.05, ema_decay=0.9, min_phase_project=False):
    """Alternate causal inference with a gradient step on the kernels.

    The gradient uses the residual over the whole extended window -- past reconstruction
    and present prediction together -- so the analysis columns stay matched to what they
    are correlated against while the prediction column is driven by prediction error.

    ``min_phase_project`` projects every kernel onto its minimum-phase counterpart after
    each update (see :func:`min_phase`). This constrains when a filter responds without
    constraining what it is tuned to, which is the principled alternative to shortening
    the window: unconstrained kernels drift to near-linear phase and carry ~L/2 of delay.

    Usage equalisation must also be strong enough to REVERSE a dropout, not just
    discourage one. At lr_bias=0.005 with ema_decay=0.99 it was not: a filter that stopped
    winning had its usage EMA decay away while its bias crept down at most
    ``lr_bias * target_usage`` per step, so over 191 epochs the bias floor only reached
    -0.056 while it needed to reach ``-lam`` = -0.1 before that filter's threshold could
    return to zero. Dropout was effectively one-way and the dictionary bled filters:
    64/64 used at epoch 41, 45/64 by epoch 191, while prediction error had converged by
    epoch 20. Recovery now outpaces the decay, and the bias is floored at -lam so a
    starved filter always reaches a zero threshold rather than an unreachable one. Without it this collapsed: 7 filters carried
    90% of the activation energy and 35 of 64 never fired at all, so they never received
    gradient and stayed at their random initialisation. A per-filter threshold bias rises
    for over-used filters and falls for under-used ones, which is the same mechanism
    OlshausenFieldEncoder.fit uses via ``target_usage``/``lr_bias``. Returns (V, bias);
    the bias must be passed to :func:`infer`.
    """
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((K, C, L + 1))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    bias = np.zeros(K)
    usage_ema = np.full(K, target_usage)
    for ep in range(n_epochs):
        pred_err, act = 0.0, 0.0
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            S = infer(X, V, lam, n_iter, bias=bias)
            P = _past_windows(X, L)                          # (T, C*L)
            recon_past = S.T @ V[:, :, :L].reshape(K, C * L)  # (T, C*L)
            R_past = (P - recon_past).reshape(T, C, L)
            R_now = (X - predict(S, V)).T                     # (T, C)
            R = np.concatenate([R_past, R_now[:, :, None]], axis=2)   # (T, C, L+1)
            V += lr * np.einsum("kt,tcl->kcl", S, R) / max(T, 1)
            if min_phase_project:
                V = min_phase(V)
            V /= np.maximum(np.linalg.norm(V.reshape(K, -1), axis=1), 1e-9)[:, None, None]
            usage = (S != 0).mean(axis=1)
            usage_ema = ema_decay * usage_ema + (1.0 - ema_decay) * usage
            bias += lr_bias * (usage_ema - target_usage)
            np.clip(bias, -lam, None, out=bias)   # threshold can reach 0, never below
            pred_err += float(np.mean(R_now ** 2))
            act += float(np.mean(S != 0))
        if ep % 5 == 0 or ep == n_epochs - 1:
            log(f"  epoch {ep+1:3d}/{n_epochs}  pred_err={pred_err/len(cochs):.6f}  "
                f"active={act/len(cochs)*100:.2f}%  used={(usage_ema > 1e-4).sum()}/{K}  "
                f"bias=[{bias.min():+.3f},{bias.max():+.3f}]")
    return V, bias


def min_phase(V, n_fft=None, eps=1e-9):
    """Project every channel's temporal kernel onto its minimum-phase counterpart.

    Among all causal filters sharing a magnitude response, the minimum-phase one has its
    energy as early as possible -- minimum group delay, maximal partial energy sums. An
    unconstrained kernel drifts toward near-linear phase instead, whose delay is ~L/2;
    measured centroid was 7.5 of 16 frames, which is exactly that, and it is where ~30 ms
    of encoder latency comes from.

    This constrains *when* a filter responds without constraining *what* it is tuned to:
    the magnitude response is left untouched, so no spectral selectivity is given up.
    That is the difference between imposing a principle and truncating the window.

    Standard real-cepstrum construction: fold the anticausal part of the log-spectrum.

    ``V`` is (K, C, L+1). Only the analysis columns [:, :, :L] are projected; the
    prediction column is a read-out, not a filter. Note column 0 is the OLDEST frame, so
    kernels are reversed into lag order before projection and back afterwards.
    """
    K, C, Lp = V.shape
    L = Lp - 1
    N = int(n_fft or max(8 * L, 64))
    g = V[:, :, :L][:, :, ::-1]                       # lag order: g[..., 0] = most recent
    G = np.fft.rfft(g, N, axis=2)
    logmag = np.log(np.abs(G) + eps)
    full = np.concatenate([logmag, logmag[:, :, -2:0:-1]], axis=2)
    ceps = np.fft.ifft(full, N, axis=2).real
    fold = np.zeros_like(ceps)
    fold[:, :, 0] = ceps[:, :, 0]
    fold[:, :, 1:N // 2] = 2.0 * ceps[:, :, 1:N // 2]
    fold[:, :, N // 2] = ceps[:, :, N // 2]
    g_mp = np.fft.irfft(np.exp(np.fft.rfft(fold, N, axis=2)), N, axis=2)[:, :, :L]
    out = V.copy()
    out[:, :, :L] = g_mp[:, :, ::-1]                  # back to oldest-first
    return out


class CausalFilterEncoder:
    """Adapter so the VEN pipeline can consume a causal filter bank.

    Mirrors the surface of :class:`~spiking_ven.olshausen_field.OlshausenFieldEncoder`
    that ``evaluate.build_stimuli`` actually uses -- ``n_channels`` and an encode call
    with the same normalisation contract -- so the two encoders are swappable and the
    downstream stimulus construction, spike conversion and VEN training are untouched.
    """

    def __init__(self, V, bias, lam, coch_params):
        self.V = np.asarray(V)
        self.bias = np.asarray(bias)
        self.lam = float(lam)
        self.coch_params = dict(coch_params)
        self.L = self.V.shape[2] - 1

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
            K = self.V.shape[0]
            W = self.V[:, :, :self.L].reshape(K, -1)
            recon = np.linalg.norm(acts.T @ W, axis=1)
            mu = (X - Xc).mean(axis=0)
            acts = acts * (mu / (dn_sigma + recon))[None, :]

        if upsample_to_ms:
            acts = np.repeat(acts, 1000 // cp["frame_rate"], axis=1)
        if T_out_ms is not None:
            T = T_out_ms if upsample_to_ms else T_out_ms // (1000 // cp["frame_rate"])
            acts = (acts[:, :T] if acts.shape[1] >= T else
                    np.concatenate([acts, np.zeros((acts.shape[0], T - acts.shape[1]))], axis=1))
        return acts.astype(np.float32)
