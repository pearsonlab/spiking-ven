"""Causal minimum-phase filter bank -- strictly past in, sparse code out.

Each filter is one spacetime kernel ``V[k]`` of shape (C, L), spanning the L frames
before now. Inference at time t touches only frames < t, and the cochleagram frames are
timed so that those end by t (see :func:`spiking_ven.filterbank.gammatone_spectrogram`),
so an activation depends on no audio at or after t. The one input gain applied before
the cochleagram, :meth:`CausalFilterEncoder.encode_signal`'s ``rms_from``, is a fixed
sound-level calibration; pass a reference signal, as the pipeline does, rather than
letting each stimulus set its own level.

    drive:       a_k(t) = <V[k], X[:, t-L:t]>                one matmul, all t at once
    competition: s(t) = argmin_s 0.5||P_t - sum_k s_k V_k||^2 + lam sum_k |s_k|
                 by ISTA on the precomputed drive and Gram, so correlated filters
                 explain each other away rather than all firing together

The kernels are constrained to be **minimum phase** (:func:`min_phase`). That is what
makes the code prompt. An unconstrained kernel drifts toward near-linear phase, whose
group delay is ~L/2 -- an energy centroid of ~7.5 of 16 frames, costing ~30 ms of encoder
latency. Minimum phase places the energy as early as the magnitude response allows -- it
constrains *when* a filter responds without constraining *what* it is tuned to. In the
shipped bank the energy centroid is 10.9 of 16 frames, and the newest 4 frames hold 59%
of the energy.
"""

import numpy as np

__all__ = ["causal_meansub", "infer", "learn", "min_phase", "prune_silent"]


def causal_meansub(X, w, tau=None):
    """Subtract a causal running level from every frame.

    For each frame, the mean over the trailing ``w`` frames (including that frame) is taken
    per channel and then averaged across channels; that one number is subtracted from
    every channel. Without it the code is dominated by overall level and cannot tell
    forward from reversed song.

    With ``tau`` (frames), the running mean is exponential instead -- ``m[t] = m[t-1] +
    (x[t] - m[t-1]) / tau``, started at the first frame -- so a sound's contribution fades
    smoothly rather than dropping out ``w`` frames later, which a boxcar does (the abrupt
    exit rings: suppression after a loud sound, then an overshoot when it leaves).
    """
    C, T = X.shape
    if tau is not None:
        from scipy.signal import lfilter
        a = 1.0 / float(tau)
        means = lfilter([a], [1.0, -(1.0 - a)], X, axis=1,
                        zi=(1.0 - a) * X[:, :1])[0]
        return X - means.mean(axis=0, keepdims=True)
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


def infer(X, V, lam, n_iter=20):
    """Strictly causal sparse activations. ``V`` is (K, C, L); returns S (K, T)."""
    K, C, L = V.shape
    T = X.shape[1]
    W = V.reshape(K, C * L)
    G = W @ W.T
    step = 1.0 / max(float(np.linalg.eigvalsh(G).max()), 1e-12)
    thr = lam * step                                        # soft threshold
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
          mp_strength=0.25, l1_V=7e-5, smooth_V=0.0):
    """Alternate causal inference with a gradient step on the kernels; returns ``V``.

    The objective is reconstruction of the strictly-past window from the sparse code. Each
    update runs, in order: gradient step -> roughness penalty -> pull toward minimum phase
    -> L1 soft-threshold -> renormalise every kernel to unit norm.

    ``lam``         code sparsity, and with it how many filters survive training.
    ``l1_V``        L1 on kernel entries, applied as a soft threshold of ``lr * l1_V``. It
                    zeroes entries the data never shaped, so what survives is what the
                    data paid for. It has to be scaled against the gradient step: a
                    threshold of the same order as the median step is useful, one far
                    above it zeroes the whole dictionary.
    ``smooth_V``    roughness penalty: L2 on first differences along channel and lag (see
                    :func:`_roughness_grad`). Plain L2 weight decay would do nothing here,
                    because the unit-norm renormalisation cancels a uniform rescale.
    ``mp_strength`` how hard each update pulls toward minimum phase:
                    ``V <- (1 - a) V + a * min_phase(V)``. ``a = 0`` disables it. ``a = 1``
                    is the strict projection, which has a degenerate fixed point: minimum
                    phase turns a white magnitude response into an impulse, so repeated
                    projection collapses every kernel onto its newest frame and loses the
                    temporal integration that song cancellation relies on. Values between
                    0.1 and 0.5 converge to the same bank.

    The minimum-phase step is an FFT resynthesis and returns dense kernels, so it partly
    refills what L1 zeroed. Running L1 after it keeps the stored kernels sparse and leaves
    the minimum-phase constraint approximate; both are soft pulls.

    There is no usage equalisation: filters that stop winning are allowed to die, and
    :func:`prune_silent` removes them afterwards.

    The measurements behind these choices are in ``docs/design-notes.md``.
    """
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    for ep in range(n_epochs):
        recon_err, act = 0.0, 0.0
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            S = infer(X, V, lam, n_iter)
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
            recon_err += float(np.mean(R ** 2))
            act += float(np.mean(S != 0))
        if ep % 5 == 0 or ep == n_epochs - 1:
            log(f"  epoch {ep+1:3d}/{n_epochs}  recon={recon_err/len(cochs):.6f}  "
                f"active={act/len(cochs)*100:.2f}%  "
                f"zeros={100 * float(np.mean(V == 0)):.0f}%")
    return V


def min_phase(V, n_fft=None, eps=1e-9):
    """Project every channel's temporal kernel onto its minimum-phase counterpart.

    Among all causal filters sharing a magnitude response, the minimum-phase one has its
    energy as early as possible -- minimum group delay, maximal partial energy sums.

    Standard real-cepstrum construction: fold the anticausal part of the log-spectrum.
    ``V`` is (K, C, L). Column 0 is the OLDEST frame, so kernels are reversed into lag
    order before projection and back afterwards.

    ``n_fft`` has to be generous. The cepstrum of a short kernel is long, so a short FFT
    aliases it and the projection stops preserving the magnitude response -- quietly
    retuning the filters. At L=16 the relative magnitude error is 7.7% at N=128 and 0.5%
    at the default 1024; the cost is negligible against ISTA. A residual error remains at
    any N because the minimum-phase counterpart of a finite kernel is in general IIR and
    is truncated back to L frames here.
    """
    K, C, L = V.shape
    N = int(n_fft or max(64 * L, 1024))
    g = V[:, :, ::-1]                                 # lag order: g[..., 0] = most recent
    
    # compute the Fourier transform of g; doing rfft and mirroring is more efficient
    # and preserves reality of g by construction
    G = np.fft.rfft(g, N, axis=2)
    logmag = np.log(np.abs(G) + eps)
    full = np.concatenate([logmag, logmag[:, :, -2:0:-1]], axis=2)

    # ensure minimum phase by constructing the imagingary part of the cepstrum
    # as -i H[Re[cepstrum]], where H is the Hilbert transform
    # this is equivalent to doubling the positive elements and zeroing out negative ones
    # in the Fourier transform of the cepstrum
    ceps = np.fft.ifft(full, N, axis=2).real
    fold = np.zeros_like(ceps)
    fold[:, :, 0] = ceps[:, :, 0]
    fold[:, :, 1:N // 2] = 2.0 * ceps[:, :, 1:N // 2]
    fold[:, :, N // 2] = ceps[:, :, N // 2]

    # now back to the time domain
    g_mp = np.fft.irfft(np.exp(np.fft.rfft(fold, N, axis=2)), N, axis=2)[:, :, :L]
    return np.ascontiguousarray(g_mp[:, :, ::-1])     # back to oldest-first


def prune_silent(V, cochs, lam, n_iter=20):
    """Drop filters that never win on the training corpus. Returns (V, kept).

    Without usage equalisation the dictionary sheds filters on a corpus this stereotyped.
    The losers do not vanish -- they decay into near-identical stubs that lose every
    competition -- so they are removed explicitly. Left in, they would be permanently
    silent auditory channels diluting each E cell's sampled inputs.
    """
    fired = np.zeros(V.shape[0], dtype=bool)
    for X in cochs:
        fired |= (infer(X, V, lam, n_iter) != 0).any(axis=1)
    kept = np.flatnonzero(fired)
    return V[kept], kept


class CausalFilterEncoder:
    """A trained causal filter bank: waveform in, sparse activations out.

    ``n_channels`` and ``encode_signal`` are the whole contract the rest of the pipeline
    relies on (see ``stimuli.encode_acts``).
    """

    def __init__(self, V, lam, coch_params):
        self.V = np.asarray(V)
        self.lam = float(lam)
        self.coch_params = dict(coch_params)
        self.L = self.V.shape[2]

    @property
    def n_channels(self) -> int:
        """Output width: one activation per filter."""
        return int(self.V.shape[0])

    def encode_signal(self, signal, *, sr, n_ista=20, upsample_to_ms=False,
                      T_out_ms=None, rms_from=None):
        """Waveform -> (K, T) sparse activations.

        The waveform is scaled by ``rms_ref / RMS(rms_from)`` (the stimulus's own RMS if
        ``rms_from`` is None), turned into a cochleagram, mean-subtracted over the trailing
        L frames, and sparse-coded by :func:`infer`. The activations are the ISTA
        coefficients themselves: sound level reaches them only through the spectral
        contrast that survives mean subtraction, which grows with loudness.
        """
        from .cochleagram import cochleagram

        cp = self.coch_params
        sig = np.asarray(signal, dtype=np.float64)
        ref = sig if rms_from is None else np.asarray(rms_from, dtype=np.float64)
        rms = float(np.sqrt(np.mean(ref ** 2)))
        sig = sig / max(rms, 1e-12) * cp["rms_ref"]

        X = cochleagram(sig, sr, n_channels=cp["n_channels"], frame_rate=cp["frame_rate"],
                        lo_hz=cp["lo_hz"], hi_hz=cp["hi_hz"]).astype(np.float64)
        # The running level is taken over ``meansub_frames`` (default: the filter length
        # L). A shorter window is a higher high-pass cutoff: the code forgets a loud sound
        # sooner after it ends.
        Xc = causal_meansub(X, getattr(self, "meansub_frames", None) or self.L,
                            tau=getattr(self, "meansub_tau", None))
        acts = infer(Xc, self.V, self.lam, n_iter=n_ista)

        if upsample_to_ms:
            acts = np.repeat(acts, 1000 // cp["frame_rate"], axis=1)
        if T_out_ms is not None:
            T = T_out_ms if upsample_to_ms else T_out_ms // (1000 // cp["frame_rate"])
            acts = (acts[:, :T] if acts.shape[1] >= T else
                    np.concatenate([acts, np.zeros((acts.shape[0], T - acts.shape[1]))], axis=1))
        return acts.astype(np.float32)

    def save(self, path):
        np.savez(path, V=self.V, lam=self.lam,
                 **{f"cp_{k}": v for k, v in self.coch_params.items()})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        cp = {k[3:]: (v.item() if v.ndim == 0 else v) for k, v in d.items()
              if k.startswith("cp_")}
        # Encoders saved before the per-filter bias was removed carry an all-zero
        # ``bias`` key; it is not read.
        return cls(d["V"], float(d["lam"]), cp)
