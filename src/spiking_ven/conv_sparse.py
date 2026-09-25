"""Convolutional sparse coding on cochleagrams -- EXPERIMENTAL.

Why this exists: the patch encoder in :mod:`spiking_ven.olshausen_field` extracts
overlapping trailing windows at stride 1, which makes its training set shift-invariant
*within* the window. Nothing in the objective then distinguishes early from late, so the
learned atoms spread their energy flat across the window -- measured effective width 13.4
of 16 frames, centroid at frame 7.5, only 3 of 64 atoms weighted toward the recent end.
A feature therefore matches best once it has travelled to the middle of the window,
costing ~15 ms of lag, and a 2 ms click produces 45 ms of activation.

Convolution handles shift explicitly, so an atom need not hedge across alignments and is
free to localise. That is the hypothesis this module tests.

    min_{S,A}  0.5 ||X - sum_k S_k * A_k||^2 + lam ||S||_1,   ||A_k|| = 1

Alternating FISTA inference with a projected gradient step on the atoms. This is the
simple baseline of that literature, not a competitive algorithm: Garcia-Cardona &
Wohlberg ("Convolutional Dictionary Learning: A Comparative Review and New Algorithms",
arXiv:1709.02893) note the dictionary-learning half is substantially harder than the
coding half, and their ADMM/Fourier methods are what to reach for if this trains poorly.
For the audio framing see Grosse, Raina, Kwong & Ng, "Shift-Invariance Sparse Coding for
Audio Classification", UAI 2007.

Prior art in this project: the Smith-Lewicki encoder removed in v0.2.0 was also
shift-invariant coding for audio, and it failed as a VEN front end because it is
amplitude-invariant -- forward/reversed activation correlation ~0.99, capping the
reversed-song contrast at 1.22x. This differs (cochleagram rather than waveform,
L1-penalised coefficients rather than matching pursuit, divisive gain downstream), but
**check the forward/reversed correlation on any learned dictionary before spending a VEN
retrain on it.**

Reconstruction convention:  Xhat[c, t] = sum_k sum_tau S[k, t-tau] * A[k, c, tau]
so S[k, t0] places atom k starting at t0 and spanning [t0, t0+L-1]. A coefficient is
therefore an event *onset*, and cannot be finalised until t0+L-1 has been observed --
correctly timed, but not causal within one atom length.
"""
import numpy as np


def reconstruct(S, A, T):
    """S (K, T), A (K, C, L) -> Xhat (C, T)."""
    K, C, L = A.shape
    N = T + L - 1
    FS = np.fft.rfft(S, N, axis=1)                    # (K, F)
    FA = np.fft.rfft(A, N, axis=2)                    # (K, C, F)
    return np.fft.irfft((FS[:, None, :] * FA).sum(axis=0), N, axis=1)[:, :T]


def corr_with_atoms(R, A, T):
    """grad wrt S: g[k, t] = sum_c sum_tau R[c, t+tau] A[k, c, tau]."""
    K, C, L = A.shape
    N = T + L - 1
    FR = np.fft.rfft(R, N, axis=1)                    # (C, F)
    FA = np.fft.rfft(A[:, :, ::-1], N, axis=2)        # time-reversed -> correlation
    g = np.fft.irfft((FA * FR[None]).sum(axis=1), N, axis=1)
    return g[:, L - 1:L - 1 + T]


def corr_with_code(R, S, L, T):
    """grad wrt A: g[k, c, tau] = sum_t R[c, t+tau] S[k, t]."""
    N = T + L - 1
    FR = np.fft.rfft(R, N, axis=1)                    # (C, F)
    FS = np.fft.rfft(S[:, ::-1], N, axis=1)           # (K, F)
    g = np.fft.irfft(FS[:, None, :] * FR[None], N, axis=2)
    return g[:, :, T - 1:T - 1 + L]


def infer(X, A, lam, n_iter=60, step=None):
    """FISTA for min_S 0.5||X - conv(S,A)||^2 + lam||S||_1."""
    C, T = X.shape
    K = A.shape[0]
    if step is None:
        FA = np.fft.rfft(A, T + A.shape[2] - 1, axis=2)
        step = 1.0 / max((np.abs(FA) ** 2).sum(axis=(1, 2)).max(), 1e-9)
    S = np.zeros((K, T))
    Z = S.copy()
    t_k = 1.0
    for _ in range(n_iter):
        R = X - reconstruct(Z, A, T)
        S_new = Z + step * corr_with_atoms(R, A, T)
        S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - lam * step, 0.0)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
        Z = S_new + ((t_k - 1) / t_next) * (S_new - S)
        S, t_k = S_new, t_next
    return S


def learn(cochs, K=64, L=16, lam=0.05, n_epochs=40, n_iter=40, lr=0.1, seed=42, log=print):
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((K, C, L))
    A /= np.linalg.norm(A.reshape(K, -1), axis=1)[:, None, None]
    for ep in range(n_epochs):
        rec, spars = 0.0, 0.0
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            S = infer(X, A, lam, n_iter)
            R = X - reconstruct(S, A, T)
            A += lr * corr_with_code(R, S, L, T) / max(T, 1)
            A /= np.maximum(np.linalg.norm(A.reshape(K, -1), axis=1), 1e-9)[:, None, None]
            rec += float(np.mean(R ** 2))
            spars += float(np.mean(S != 0))
        if ep % 5 == 0 or ep == n_epochs - 1:
            log(f"  epoch {ep+1:3d}/{n_epochs}  recon={rec/len(cochs):.6f}  "
                f"active={spars/len(cochs)*100:.2f}%")
    return A
