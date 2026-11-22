"""A two-stage auditory front end after Schneider & Woolley (2013, Neuron 79:141).

**Primary AC.** Dense spectro-temporal receptive fields (STRFs): each unit linearly filters
the strictly-past ``L`` frames of the raw cochleagram (no mean subtraction), and its output
is rectified (ReLU). The STRFs are trained as the encoder of a rectified autoencoder whose
linear decoder reconstructs that window -- dense codes, no sparsity penalty, a small L2 on
the weights. The rectified outputs are scaled to a mean of ``PRIMARY_RATE_HZ`` (Field L:
32.4 +/- 20.1 Hz in S&W) and sampled as Poisson spike trains.

**Higher AC** (broad-spiking neurons). Each unit takes ONE primary unit's spike train and
filters it with a fixed causal temporal kernel: a narrow excitatory Gaussian minus a
delayed, wide inhibitory one (difference of Gaussians, standing in for feedforward
inhibition from narrow-spiking neurons; S&W used ~15 ms and ~100 ms widths). The result,
minus a threshold, is rectified. These are the activations the pipeline then converts to
spikes (``common.acts_to_spikes``, calibrated to the song, ~2.4-2.5 Hz as S&W's BS neurons).

:class:`SWEncoder` has the interface of :class:`~spiking_ven.causal_filters.
CausalFilterEncoder` (``n_channels``, ``encode_signal``, ``save``/``load``), so the stimulus
pipeline, training and metrics use it unchanged.
"""

from __future__ import annotations

import json
import zlib
from typing import Any

import numpy as np

from .cochleagram import cochleagram

__all__ = ["SWEncoder", "learn_relu", "train_autoencoder", "dog_kernel", "PRIMARY_RATE_HZ"]

#: Primary AC (Field L) mean rate, Schneider & Woolley 2013: 32.4 +/- 20.1 Hz.
PRIMARY_RATE_HZ = 32.0


def unfold(X: np.ndarray, L: int) -> np.ndarray:
    """``(C, T)`` frames -> ``(C * L, T)``: column t is frames ``t-L .. t-1`` (zero-padded),
    channel-major, oldest first -- the strictly-past window the STRF sees."""
    C, T = X.shape
    Xp = np.concatenate([np.zeros((C, L)), X], axis=1)
    return np.concatenate([Xp[:, lag:lag + T] for lag in range(L)], axis=0) \
        .reshape(L, C, T).transpose(1, 0, 2).reshape(C * L, T)


def train_autoencoder(cochs, *, K: int = 128, L: int = 32, n_epochs: int = 1500,
                      lr: float = 1e-3, l2: float = 1e-4, seed: int = 42, log=print):
    """Train STRFs ``W`` (K, C*L) and bias ``b`` with a linear decoder ``D`` (C*L, K) so that
    ``D relu(W x + b) + d0`` reconstructs each past window ``x`` (squared error, L2 on
    ``W`` and ``D``). The windows are centred (mean ``mu``) and scaled (``scale``) first;
    biases start slightly positive so units start active. Full-batch Adam over all windows
    of all cochleagrams. Returns ``(W, b, D, d0, mu, scale)``."""
    Xw = np.concatenate([unfold(X, L) for X in cochs], axis=1)
    mu = Xw.mean(axis=1, keepdims=True)
    Xw = Xw - mu
    scale = float(Xw.std()) or 1.0
    Xw = Xw / scale
    D_in, N = Xw.shape
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, D_in)) / np.sqrt(D_in)
    b = np.full((K, 1), 0.1)
    D = rng.standard_normal((D_in, K)) / np.sqrt(K) * 0.1
    d0 = np.zeros((D_in, 1))
    params = [W, b, D, d0]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    var = float(((Xw - d0) ** 2).mean())
    for ep in range(1, n_epochs + 1):
        Z = W @ Xw + b
        R = np.maximum(Z, 0.0)
        E = D @ R + d0 - Xw
        loss = float((E ** 2).mean())
        gE = 2.0 * E / N
        gD = gE @ R.T + 2 * l2 * D
        gd0 = gE.sum(axis=1, keepdims=True)
        gR = D.T @ gE
        gZ = gR * (Z > 0)
        gW = gZ @ Xw.T + 2 * l2 * W
        gb = gZ.sum(axis=1, keepdims=True)
        for i, (p, g) in enumerate(zip(params, [gW, gb, gD, gd0])):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            p -= lr * (m[i] / (1 - b1 ** ep)) / (np.sqrt(v[i] / (1 - b2 ** ep)) + eps)
        if ep == 1 or ep % 100 == 0 or ep == n_epochs:
            log(f"  epoch {ep:4d}/{n_epochs}  recon MSE {loss:.4f} "
                f"({1 - loss / var:.1%} of variance explained)  "
                f"active {(R > 0).mean():.1%}  units alive {(R > 0).any(axis=1).sum()}/{K}")
    return W, b, D, d0, mu, scale


def window_scale(cochs, L: int) -> float:
    """RMS per-element std of the centred strictly-past windows of a corpus (a global input
    normalisation: dividing by it gives a random unit-norm STRF output variance ~1)."""
    from .causal_filters import _past_windows

    P = np.concatenate([_past_windows(X, L) for X in cochs], axis=0)
    return float(np.sqrt(np.mean(P.var(axis=0))))


def tm_basis(taus_ms, L: int, frame_ms: float = 2.0) -> np.ndarray:
    """Discrete-time Takenaka-Malmquist basis on the strictly-past window, (nb, L).

    One real pole per time constant, ``xi_k = exp(-frame_ms / tau_k)``:

        F_n(z) = sqrt(1 - xi_n^2) z^-1 / (1 - xi_n z^-1)
                 * prod_{k<n} (z^-1 - xi_k) / (1 - xi_k z^-1)

    (a first-order low-pass after the all-pass sections of the earlier poles). The F_n are
    orthonormal on the infinite horizon; all poles equal gives the Laguerre basis. Every
    kernel in their span is ``sum_k c_k exp(-t/tau_k)`` (distinct poles), i.e. the impulse
    response of a stable causal linear ODE built from these time constants. Row n is F_n's
    impulse response at lags 1..L frames, laid out like an STRF window: column 0 is the
    OLDEST lag, column L-1 lag 1 (the newest past frame)."""
    from scipy.signal import lfilter

    xi = np.exp(-float(frame_ms) / np.asarray(taus_ms, dtype=np.float64))
    ap = np.zeros(L + 1)
    ap[0] = 1.0                                  # unit impulse through the all-pass chain
    rows = []
    for x in xi:
        h = np.sqrt(1.0 - x ** 2) * lfilter([0.0, 1.0], [1.0, -x], ap)
        rows.append(h[1:L + 1])                  # lags 1..L
        ap = lfilter([-x, 1.0], [1.0, -x], ap)
    return np.array(rows)[:, ::-1].copy()


def learn_relu(cochs, *, K: int = 128, L: int = 32, n_epochs: int = 80, lr: float = 0.1,
               reg_lr: float | None = None, seed: int = 42, mp_strength: float = 0.25,
               l1_V: float = 7e-5, smooth_V: float = 1e-3, tied: bool = False, log=print,
               val_cochs=None, patience: int = 0, min_delta: float = 0.002,
               log_every: int = 5, frame_ms: float = 2.0, smooth_axes=(1, 2),
               recon_tau_ms: float | None = None, predict_frames: int = 0,
               checkpoint=None, resume=None, init_scale: float = 1.0,
               init_bias: float = 0.0, conv_frames: int = 0, code_l1: float = 0.0,
               l2_V: float = 0.0, basis: np.ndarray | None = None,
               act_target: float | None = None, act_lr: float = 0.1,
               act_beta: float = 0.01, act_mode: str = "threshold",
               act_dual_lr: float = 1e-3, act_mu: float = 0.2, act_mu_growth: float = 2.0,
               act_mu_max: float = 2.0, softplus_beta0: float | None = None,
               softplus_growth: float = 1.25, softplus_beta_max: float = 3000.0,
               act_warmup: int = 3, unit_norm_strf: bool = False,
               input_scale: float = 1.0, softplus_hold: float | None = None,
               softplus_hold_patience: int = 5, softplus_anneal: float = 1.15,
               input_noise: float = 0.0, act_update_every: int = 0,
               output_noise: float = 0.0, al_exact: bool = False, al_inner_min: int = 2,
               al_inner_max: int = 8, al_inner_tol: float = 0.01,
               al_mu_growth: float = 4.0, scale_free_shape: bool = False,
               al_plain: bool = False, restart_dead: float = 0.0,
               decorr: float = 0.0, init_data: bool = False):
    """The causal encoder's learning rule (``causal_filters.learn``) with ISTA replaced by a
    ReLU: a rectified autoencoder whose encoder kernels ``V`` are the STRFs.

    Untied (default): reconstruction ``D^T r`` with a linear decoder ``D`` (K, C*L) whose
    rows are renormalised to unit norm after every update. That pins the scale (ReLU is
    positively homogeneous, so otherwise ``V`` could shrink under the penalties while ``D``
    grew), and the penalties shape ``V`` rather than shrink it. Tied (``tied=True``): the
    description below with ``D = V``.

    Code ``r = relu(V x + b)`` on each strictly-past window ``x`` (centred by the corpus's
    mean window ``mu``), reconstruction ``V^T r``. The gradient of the squared error (step
    ``lr``) flows through both roles of ``V`` and into ``b``. Each update then runs, as in
    ``learn``: roughness penalty (L2 on first differences along channel and lag) -> pull
    toward minimum phase -> L1 soft-threshold. Unlike ``learn`` there is no renormalisation:
    with no L1 on the code there is no code/kernel scale trade-off, the kernel norm sets the
    code amplitude, and the two penalties bound it. They are applied with step ``reg_lr``
    (default ``lr``: proximal gradient on MSE/2 + l1_V |V|_1 + smooth_V/2 |DV|^2). With no
    renormalisation, the causal encoder's step (10) wipes out every kernel.
    Returns ``(V, b, mu, D)`` with ``V`` (K, C, L) and ``D`` (K, C*L) (``D`` is ``V``
    flattened when tied).

    Monitoring: with ``val_cochs`` (held-out cochleagrams), every ``log_every`` epochs the
    log reports the held-out unexplained variance and STRF compactness (share of kernel
    energy at lags <= 20 ms, energy-weighted mean lag). With ``patience`` > 0, training
    stops once the held-out error has not improved by ``min_delta`` for ``patience``
    epochs, and the best epoch's weights are returned. ``smooth_axes`` picks the
    directions the roughness penalty acts along: (1, 2) channel and lag, (1,) channel only
    (smoothing along lag favours long horizontal bands). ``init_data`` (with
    ``unit_norm_strf``): start each shape from a random training window (projected onto the
    basis) instead of white noise, so no initial structure sits in directions the data never
    drives (with no penalty, such noise never decays). ``recon_tau_ms`` weights each
    lag's reconstruction error by ``exp(-lag / recon_tau_ms)`` (1 at the newest frame):
    the code is then asked to describe the recent past, so STRF energy at long lags
    survives only where it helps; reported errors are weighted the same way.
    ``predict_frames`` = h > 0: instead of reconstructing the window, the (untied) decoder
    predicts the NEXT h frames ``X[:, t:t+h]`` (each centred by the corpus mean frame;
    zero past the end of a bout) from the code of the strictly-past window; ``D`` is then
    (K, C*h). STRFs keep the history that predicts what comes next.

    ``checkpoint``: called after every epoch with a dict of the full training state (current
    ``V``, ``b``, ``D``, ``mu``, ``epoch`` (1-based), held-out error, best weights so far,
    patience counter, RNG state), so a run can be stopped at any time, its latest STRFs
    inspected, and training continued by passing that dict back as ``resume`` (same corpus
    and settings): the continuation is identical to an uninterrupted run.

    ``init_scale`` scales the random unit-norm initial STRFs (0 = start from zero, so the
    STRFs hold only what was learned; needs ``init_bias`` > 0 to keep the ReLUs alive, and
    the random decoder breaks the symmetry between units). ``init_bias``: initial bias.

    ``conv_frames`` = M > 0: convolutional (overlap-add) reconstruction instead of the
    sliding-window one. Each centred frame is reconstructed ONCE, jointly by the codes of the
    M following time steps (all of which had it in their past window):
    ``xhat_s = sum_k sum_{m=1..M} D_k(m) r_k(s+m)``, loss
    ``1/2 sum_s |x_s - xhat_s|^2 + code_l1 * sum r``. ``D`` is (K, M*C) (lag-major, each
    unit's kernel unit-norm). A unit only explains the part of the signal near its own time,
    so short STRFs suffice; ``code_l1`` keeps the overcomplete code sparse.

    ``l2_V``: weight decay on the STRFs (``V -= reg_lr * l2_V * V`` after each update).

    ``basis`` (nb, L), e.g. ``tm_basis``: each channel's temporal kernel is constrained to
    the span of the basis rows, ``V[k, c] = A[k, c] @ basis``, and trained by gradient
    descent on the coefficients A (the V-gradient G becomes ``G @ basis.T @ basis``).
    Anything that leaves the span (smoothing along lag, min-phase, L1) is projected back
    onto it after each update.

    ``act_target`` = rho: per-unit activity constraint (fraction of frames each unit is
    active), enforced on the threshold by projection: after every bout each unit's bias
    moves toward minus the (1 - rho) quantile of its input ``V x`` on that bout,
    ``b_k <- (1 - act_lr) b_k + act_lr * (-q_k)`` (a running quantile, so it is scale-free
    and revives silent units). Reconstruction then trains only V and D; the bias is set by
    the constraint alone. This fixes how often each unit fires, not how sparse the
    population is at a given moment. (``act_beta`` is unused.)

    ``act_mode="penalty"``: instead, the activity cost is part of the objective, so it also
    reshapes the STRFs (as in Karklin & Simoncelli 2011; Jun, Field & Pearson): loss
    ``+ sum_k lambda_k sum_t r_k(t)`` with per-unit multipliers updated by dual ascent
    ``lambda_k += act_dual_lr * (fraction of frames unit k was active - rho)`` after every
    bout (an equality constraint: lambda may go negative). The bias is learned by gradient
    (including the cost); a unit silent for a whole bout gets a small bias nudge.

    ``act_mode="auglag"``: augmented Lagrangian (method of multipliers) on the per-unit
    constraint c_k = (fraction of frames active) - rho, measured exactly; its gradient uses
    the smooth indicator sigmoid(z / h_k), h_k = act_beta * std_t(z_k) (scale-free; never
    vanishes, so silent units are pushed back on). Each frame's gradient on z_k is
    -(lambda_k + mu c_k) d sigmoid / dz, which flows into the STRFs as well as the bias;
    lambda_k += mu * (epoch-mean c_k) after every epoch, and mu is
    multiplied by ``act_mu_growth`` (up to ``act_mu_max``) whenever the constraint violation
    fails to shrink by a quarter while still above 10% of the target (rms per unit). The constraint switches on after ``act_warmup`` epochs
    (zero-init STRFs first need some structure), and its gradient is preconditioned by
    h_k^2 so the step follows each unit's own input scale.

    ``act_mode="rate"``: the same augmented Lagrangian on each unit's MEAN NONLINEAR OUTPUT,
    c_k = mean_t r_k(t) - act_target (differentiable; gradient -(lambda_k + mu c_k) dr/dz).
    Use with ``unit_norm_strf``: the decoder rows stay unit-norm, so shrinking a unit's gain
    costs reconstruction, which pins the rate scale; the constraint only sets the mean. A
    learned decoder offset absorbs the reconstruction of the code's mean (otherwise a
    positive mean rate fights the mean-subtracted target and silences the units). The
    offset is not saved with the encoder (it only matters for training).

    ``unit_norm_strf``: each STRF is a unit-norm shape times a learned per-unit gain
    (V_k = g_k u_k), updated separately; penalties act on the shape only (L2 is ignored).
    Starts from random unit-norm shapes with gain 0.01.

    ``input_scale``: centred windows (and reconstruction targets) are divided by this global
    scalar, e.g. the rms per-element std of the corpus windows (``window_scale``), so that a
    random unit-norm STRF's output has variance ~1. Save it as the encoder's ``scale``.

    ``softplus_beta0``: train with r = softplus_beta(z) = log(1 + exp(beta z)) / beta instead
    of the ReLU, beta starting at ``softplus_beta0`` and multiplied by ``softplus_growth``
    each epoch; once it reaches ``softplus_beta_max`` the exact ReLU is used. Early stopping
    waits for the ReLU (and, with an activity target, for the constraint).
    ``softplus_hold``: three phases instead -- ramp beta up to ``softplus_hold``, hold it
    there until the activity constraint is met AND the held-out error has plateaued (no
    improvement by ``min_delta`` for ``softplus_hold_patience`` epochs), then anneal slowly
    (x ``softplus_anneal`` per epoch) to the ReLU.

    ``input_noise``: std of Gaussian noise added to each unit's input before the
    nonlinearity, fresh every frame, during training only (held-out error is noise-free).
    Fixed in absolute terms (e.g. sqrt(0.1) on normalised input = SNR 10 for a unit-norm
    STRF at gain 1), so shrinking a unit's gain lowers its SNR and costs reconstruction --
    the collapse to a flat, low-rate output no longer pays (Karklin & Simoncelli).

    ``act_update_every`` = N > 0: in the augmented-Lagrangian modes, update the multipliers
    every N bouts (lambda += mu * sum of the per-bout violations / bouts per epoch, so the
    per-epoch change is unchanged) instead of once per epoch.

    ``output_noise``: std of Gaussian noise added to each unit's rate before the decoder,
    during training. Every unit then adds noise to the reconstruction, so a silent or flat
    unit is pure cost and units must carry signal well above the noise (a gain floor).

    ``al_exact`` (rate mode): a faithful augmented Lagrangian (method of multipliers).
    lambda and mu are held fixed for an inner loop of epochs; the constraint gradient on
    each frame is -(lambda_k + mu c_hat_k) dr/dz with c_hat a running (EMA over bouts)
    estimate of the full-data violation. The inner loop ends when the training augmented
    objective (reconstruction + mean_k[lambda_k c_k + mu/2 c_k^2]) stops improving by
    ``al_inner_tol`` (relative), after at least ``al_inner_min`` and at most
    ``al_inner_max`` epochs. Then lambda <- lambda + mu c_bar (full-data violation of the
    last inner epoch), and mu <- min(mu * ``al_mu_growth``, ``act_mu_max``) if |c_bar| fell
    by less than 75%. Early stopping is counted per outer iteration. ``softplus_anneal``
    <= 1 with ``softplus_hold`` keeps the softplus indefinitely (no ReLU), and early
    stopping is then allowed on the softplus.

    ``scale_free_shape`` (with ``unit_norm_strf``): the shape step uses the population median
    gain instead of each unit's own gain (u += lr * median(g) * dL/dV), so a unit whose gain
    has collapsed can still re-tune its shape and recover (with the exact gradient
    g_k * dL/dV_k the shape freezes once g_k reaches its floor).

    ``al_plain`` (rate mode): plain first-order dual ascent instead of the augmented
    Lagrangian -- the constraint gradient is -lambda_k dr/dz (no quadratic term), and every
    ``act_update_every`` bouts lambda_k += ``act_dual_lr`` * (mean rate over those bouts -
    target). No mu, no per-epoch multiplier update.

    ``restart_dead`` = f > 0 (with ``unit_norm_strf``): after every epoch, units whose
    activity std over the epoch is below f x the median unit's are re-seeded: shape from a
    random training window (projected onto the basis, unit norm), gain = median gain, bias 0,
    lambda 0, fresh random unit-norm decoder row.

    ``decorr`` = w > 0: decorrelation penalty on the code within each bout,
    w/2 * sum_{j != k} C_jk^2 / vbar^2 with C the covariance of the rates over the bout's
    frames and vbar its mean diagonal (scale-free); its gradient -2w (S - mean) C_off /
    (T vbar^2) dr/dz flows into the STRFs and biases."""
    from .causal_filters import _past_windows, _roughness_grad, min_phase

    if basis is not None:
        Bm = np.asarray(basis, dtype=np.float64)
        if Bm.shape[1] != L:
            raise ValueError(f"basis has {Bm.shape[1]} lags, need L={L}")
        metric = Bm.T @ Bm                                   # (L, L): coefficient gradient
        proj = Bm.T @ np.linalg.solve(Bm @ Bm.T, Bm)         # (L, L): projector onto span

    def to_span(V):
        return V if basis is None else V @ proj

    def step(G):
        return G if basis is None else G @ metric

    if reg_lr is None:
        reg_lr = lr
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    mu = np.concatenate([_past_windows(X, L) for X in cochs], axis=0).mean(axis=0)
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    V *= init_scale
    V = to_span(V)
    if unit_norm_strf:                    # unit-norm shapes (random, in span), small gains
        if init_data:                     # ... or random training windows (no noise)
            picks = [(int(rng.integers(len(cochs))), 0) for _ in range(K)]
            U0 = np.empty((K, C, L))
            for k, (i, _) in enumerate(picks):
                Pi = (_past_windows(cochs[i], L) - mu) / input_scale
                U0[k] = Pi[int(rng.integers(len(Pi)))].reshape(C, L)
            U0 = to_span(U0)
        else:
            U0 = to_span(rng.standard_normal((K, C, L)))
        V = 0.01 * U0 / np.linalg.norm(U0.reshape(K, -1), axis=1)[:, None, None]
    b = np.full(K, float(init_bias))
    if init_scale == 0 and init_bias <= 0:
        raise ValueError("zero-init STRFs need init_bias > 0 (otherwise every ReLU is dead)")
    if (predict_frames or conv_frames) and tied:
        raise ValueError("predict_frames / conv_frames need the untied decoder")
    if predict_frames and conv_frames:
        raise ValueError("choose predict_frames or conv_frames, not both")
    M = int(conv_frames)
    h = int(predict_frames)
    mu_y = np.concatenate([X for X in cochs], axis=1).mean(axis=1)           # (C,)
    if h or M:
        D = rng.standard_normal((K, C * (h or M)))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    else:
        D = None if tied else V.reshape(K, C * L).copy()
        if D is not None and init_scale == 0:
            D = rng.standard_normal((K, C * L))
            D /= np.linalg.norm(D, axis=1, keepdims=True)

    def target(X, P):
        """What the code must reproduce: the window, or the next h frames (T, C*h)."""
        if not h:
            return P
        Xc = (X - mu_y[:, None]) / input_scale
        Xp = np.concatenate([Xc, np.zeros((C, h - 1))], axis=1)
        return np.concatenate([Xp[:, j:j + X.shape[1]].T for j in range(h)], axis=1)
    lag_ms = (L - 1 - np.arange(L)) * frame_ms + frame_ms / 2     # oldest column first
    # Reconstruction weight per window column (channel-major, lag inner): exp(-lag/tau),
    # 1 at the newest frame. Uniform without recon_tau_ms.
    wcol = (np.tile(np.exp(-(lag_ms - frame_ms / 2) / recon_tau_ms), C)
            if recon_tau_ms else np.ones(C * L))

    beta_now: list[float | None] = [None]                # softplus sharpness; None = ReLU
    # softplus schedule state: [beta, phase (ramp/hold/anneal), best held-out at hold, stale]
    sp: list[Any] = [float(softplus_beta0 or 0.0), "ramp", np.inf, 0]
    # Learned decoder offset (rate mode): absorbs the reconstruction of the code's mean, so
    # a positive mean-rate target does not fight the mean-subtracted reconstruction target.
    use_d0 = act_target is not None and act_mode == "rate"
    dec0: list[Any] = [0.0]

    def nl(Z):
        """(rate, d rate / dz, soft 'active' indicator) for the current nonlinearity."""
        bta = beta_now[0]
        if bta is None:
            on = (Z > 0).astype(np.float64)
            return np.maximum(Z, 0.0), on, on
        u = bta * Z
        sg = 0.5 * (1.0 + np.tanh(0.5 * u))                # sigmoid(beta z), overflow-safe
        return np.logaddexp(0.0, u) / bta, sg, sg

    def conv_pass(X, P, V, b, D, noise=False):
        """Convolutional reconstruction of bout X: (Z, S, F, Y, E) with F the future-code
        windows (T, K*M), Y the centred frames (T, C) and E = Y - F @ D (T, C)."""
        T = X.shape[1]
        Z = P @ V.reshape(K, C * L).T + b
        if noise and input_noise > 0:
            Z = Z + rng.normal(0.0, input_noise, Z.shape)
        S = nl(Z)[0]
        Sf = S + rng.normal(0.0, output_noise, S.shape) if noise and output_noise > 0 else S
        Sp = np.concatenate([Sf, np.zeros((M, K))], axis=0)
        F = np.stack([Sp[m:m + T] for m in range(1, M + 1)], axis=2).reshape(T, K * M)
        D3 = D.reshape(K, M, C)
        Y = (X - mu_y[:, None]).T / input_scale
        E = Y - F @ D3.reshape(K * M, C) - dec0[0]
        return Z, S, F, Y, E

    def val_error(V, b, D):
        if M:
            num = den = 0.0
            for X in val_cochs or []:
                _, _, _, Y, E = conv_pass(X, (_past_windows(X, L) - mu) / input_scale, V, b,
                                          D)
                num += float((E ** 2).sum())
                den += float((Y ** 2).sum())
            return num / max(den, 1e-12)
        W = V.reshape(K, C * L)
        Dk = W if D is None else D
        num = den = 0.0
        for X in val_cochs or []:
            P = (_past_windows(X, L) - mu) / input_scale
            Y = target(X, P)
            R = Y - nl(P @ W.T + b)[0] @ Dk - dec0[0]
            w = 1.0 if predict_frames else wcol
            num += float((w * R ** 2).sum())
            den += float((w * Y ** 2).sum())
        return num / max(den, 1e-12)

    def compactness(V):
        e = (V ** 2).sum(axis=1)
        e = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        return float(np.median(e[:, lag_ms <= 20].sum(axis=1))), \
            float(np.median((e * lag_ms).sum(axis=1)))

    best = (np.inf, V.copy(), b.copy(), None if D is None else D.copy(), -1)
    stale, start = 0, 0

    gZ_input: list[np.ndarray] = [np.zeros(0)]
    penalty = act_target is not None and act_mode == "penalty"
    rate_mode = act_target is not None and act_mode == "rate"
    auglag = act_target is not None and act_mode in ("auglag", "rate")
    if act_target is not None and act_mode not in ("threshold", "penalty", "auglag", "rate"):
        raise ValueError(f"act_mode must be 'threshold', 'penalty', 'auglag' or 'rate', "
                         f"not {act_mode!r}")

    def norm_rows(D):
        """Unit-norm decoder rows."""
        return D / np.maximum(np.linalg.norm(D, axis=1), 1e-9)[:, None]

    def penalties(V):
        """Smoothing / min-phase / L2 / L1 steps on the STRFs, then back onto the basis."""
        if smooth_V > 0:
            V = V - reg_lr * smooth_V * _roughness_grad(V, smooth_axes)
        if mp_strength > 0:
            V = (1.0 - mp_strength) * V + mp_strength * min_phase(V)
        if l2_V > 0 and not unit_norm_strf:
            V = V * (1.0 - reg_lr * l2_V)
        if l1_V > 0:
            V = np.sign(V) * np.maximum(np.abs(V) - reg_lr * l1_V, 0.0)
        return to_span(V)

    def strf_step(V, Gv, n):
        """Gradient step on the STRFs (Gv = -dLoss/dV), then the penalties. With
        ``unit_norm_strf``, V_k = g_k * u_k with |u_k| = 1: the gain g_k and the shape u_k
        are updated separately (dL/dg_k = <dL/dV_k, u_k>, dL/du_k = g_k dL/dV_k) and the
        penalties act on the shape only."""
        if not unit_norm_strf:
            return penalties(V + lr * Gv / n)
        g = np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
        u = V / np.maximum(g, 1e-12)
        g_new = np.maximum(g + lr * (Gv * u).sum(axis=(1, 2), keepdims=True) / n, 1e-6)
        g_shape = np.median(g) if scale_free_shape else g
        u = penalties(u + lr * g_shape * Gv / n)
        u /= np.maximum(np.linalg.norm(u.reshape(K, -1), axis=1), 1e-12)[:, None, None]
        return g_new * u
    lam = np.zeros(K)
    mu_al = [float(act_mu), np.inf]                      # [mu, previous violation norm]
    # exact augmented Lagrangian state: running violation estimate and inner-loop progress
    al_st: dict[str, Any] = {"c_est": np.zeros(K), "inner": 0, "best": np.inf, "stale": 0,
                             "outer": 0}
    no_anneal = softplus_hold is not None and softplus_anneal <= 1.0

    al_on = [False]                                      # constraint active (after warm-up)

    def soft_active(Z):
        """Smooth 'active' indicator sigmoid(z / h_k), h_k = act_beta * std_t(z_k): scale-free,
        and its gradient never vanishes (silent units can be pushed back on)."""
        h = act_beta * Z.std(axis=0) + 1e-12
        sg = 0.5 * (1.0 + np.tanh(0.5 * Z / h))
        return sg, sg * (1.0 - sg) * h          # gradient preconditioned by h^2 (scale-free)

    def decorr_grad(S, dS):
        """-d/dZ of the decorrelation penalty (0 if decorr == 0)."""
        if decorr <= 0:
            return 0.0
        Sc = S - S.mean(axis=0)
        Cv = Sc.T @ Sc / max(len(S), 1)
        vbar = max(float(np.mean(np.diag(Cv))), 1e-12)
        Coff = Cv - np.diag(np.diag(Cv))
        return -2.0 * decorr * (Sc @ Coff) / (max(len(S), 1) * vbar ** 2) * dS

    def cost_grad(Z, dS, on):
        """-d/dZ of the activity cost (penalty / augmented-Lagrangian modes; else 0)."""
        if penalty:
            return -lam * (Z > 0)
        if auglag and al_on[0]:
            assert act_target is not None
            if rate_mode and al_plain:                   # first-order: lambda only
                return -lam * dS
            if rate_mode and al_exact:                   # running full-data violation
                return -(lam + mu_al[0] * al_st["c_est"]) * dS
            if rate_mode:                                # mean rate per unit = target
                c = nl(Z)[0].mean(axis=0) - act_target
                return -(lam + mu_al[0] * c) * dS
            _, dsg = soft_active(Z)
            c = (Z > 0).mean(axis=0) - act_target        # this bout's (exact) violation
            return -(lam + mu_al[0] * c) * dsg
        return 0.0

    def bias_step(b, gZ, S, n):
        """Gradient step on b; with act_target, the activity constraint."""
        nonlocal lam
        if act_target is None:
            return b + lr * gZ.sum(axis=0) / n
        if penalty or auglag:
            frac = (gZ_input[0] + b > 0).mean(axis=0)
            if penalty:
                lam = lam + act_dual_lr * (frac - act_target)
            return b + lr * gZ.sum(axis=0) / n + 1e-3 * (frac == 0)
        U = gZ_input[0]                                   # V x (pre-bias), set by caller
        q = np.quantile(U, 1.0 - act_target, axis=0)
        return (1.0 - act_lr) * b + act_lr * (1e-9 - q)   # ties count as active
    if resume is not None:
        V, b = np.array(resume["V"], dtype=np.float64), np.array(resume["b"], dtype=np.float64)
        if D is not None:
            D = np.array(resume["D"], dtype=np.float64)
        best = (float(resume["best_val"]), np.array(resume["best_V"]), np.array(resume["best_b"]),
                None if D is None else np.array(resume["best_D"]), int(resume["best_epoch"]))
        stale, start = int(resume["stale"]), int(resume["epoch"])
        if bool(resume["stopped"]):
            start = n_epochs                  # already early-stopped: just return the best
        rng.bit_generator.state = json.loads(str(resume["rng_state"]))
        if "lam" in resume:
            lam = np.array(resume["lam"], dtype=np.float64)
        if "mu_al" in resume:
            mu_al = [float(resume["mu_al"]), float(resume["viol_prev"])]
        if "al_state" in resume:
            al_st.update(json.loads(str(resume["al_state"])))
            al_st["c_est"] = np.array(resume["al_c_est"], dtype=np.float64)
        if "sp_beta" in resume:
            sp[:] = [float(resume["sp_beta"]), str(resume["sp_phase"]),
                     float(resume["sp_best"]), int(resume["sp_stale"])]
        log(f"  resuming after epoch {start} (best epoch {best[4] + 1}, {best[0]:.3f}; "
            f"{stale} stale)")
    for ep in range(start, n_epochs):
        if softplus_beta0 is not None:
            beta_now[0] = None if sp[0] >= softplus_beta_max else sp[0]
        al_on[0] = ep >= act_warmup
        err, act, alive = 0.0, 0.0, np.zeros(K, dtype=bool)
        act_unit = np.zeros(K)
        soft_unit = np.zeros(K)
        s1, s2, nfr = np.zeros(K), np.zeros(K), 0       # per-unit activity moments
        c_acc: list[Any] = [np.zeros(K), 0]              # violations since last lambda update
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            P = (_past_windows(X, L) - mu) / input_scale       # (T, C*L)
            n = max(T, 1)
            if M:
                assert D is not None                   # conv decoder is always untied
                Z, S, F, Y, E = conv_pass(X, P, V, b, D, noise=True)
                gZ_input[0] = Z - b
                # -dLoss/dr_k(t) = sum_m D_k(m) . e_{t-m} - code_l1 (residual over the past)
                Ep = np.concatenate([np.zeros((M, C)), E], axis=0)
                Ew = np.stack([Ep[M - m:M - m + T] for m in range(1, M + 1)], axis=1)
                _, dS, on = nl(Z)
                gZ = ((Ew.reshape(T, M * C) @ D.T - code_l1) * dS + cost_grad(Z, dS, on)
                      + decorr_grad(S, dS))
                soft_unit += (S.mean(axis=0) if rate_mode else (Z > 0).mean(axis=0)
                              if auglag else on.mean(axis=0))
                if al_exact and auglag and al_on[0]:
                    assert act_target is not None
                    al_st["c_est"] = 0.95 * al_st["c_est"] + 0.05 * (S.mean(axis=0) - act_target)
                if auglag and al_on[0] and act_update_every > 0:
                    assert act_target is not None
                    c_acc[0] = c_acc[0] + ((S.mean(axis=0) if rate_mode else (Z > 0).mean(axis=0))
                                           - act_target)
                    c_acc[1] += 1
                    if c_acc[1] >= act_update_every:
                        lam = lam + (act_dual_lr * c_acc[0] / c_acc[1] if al_plain
                                     else mu_al[0] * c_acc[0] / len(cochs))
                        c_acc[0], c_acc[1] = np.zeros(K), 0
                D = norm_rows(D + lr * (F.T @ E).reshape(K, M * C) / n)
                if use_d0:
                    dec0[0] = dec0[0] + lr * E.mean(axis=0)
                V = strf_step(V, step((gZ.T @ P).reshape(K, C, L)), n)
                b = bias_step(b, gZ, S, n)
                err += float((E ** 2).sum()) / max(float((Y ** 2).sum()), 1e-12)
                act += float(np.mean(Z > 0))
                alive |= (Z > 0).any(axis=0)
                act_unit += (Z > 0).mean(axis=0)
                s1, s2, nfr = s1 + S.sum(axis=0), s2 + (S ** 2).sum(axis=0), nfr + T
                continue
            W = V.reshape(K, C * L)
            Z = P @ W.T + b
            if input_noise > 0:
                Z = Z + rng.normal(0.0, input_noise, Z.shape)
            gZ_input[0] = Z - b
            S, dS, on = nl(Z)                                  # (T, K)
            soft_unit += (S.mean(axis=0) if rate_mode else (Z > 0).mean(axis=0)
                          if auglag else on.mean(axis=0))
            if al_exact and auglag and al_on[0]:
                assert act_target is not None
                al_st["c_est"] = 0.95 * al_st["c_est"] + 0.05 * (S.mean(axis=0) - act_target)
            if auglag and al_on[0] and act_update_every > 0:
                assert act_target is not None
                c_acc[0] = c_acc[0] + ((S.mean(axis=0) if rate_mode else (Z > 0).mean(axis=0))
                                       - act_target)
                c_acc[1] += 1
                if c_acc[1] >= act_update_every:
                    lam = lam + (act_dual_lr * c_acc[0] / c_acc[1] if al_plain
                                 else mu_al[0] * c_acc[0] / len(cochs))
                    c_acc[0], c_acc[1] = np.zeros(K), 0
            Dk = W if D is None else D
            Y = target(X, P)
            Sn = S + rng.normal(0.0, output_noise, S.shape) if output_noise > 0 else S
            R = Y - Sn @ Dk - dec0[0]                          # residual
            if use_d0:
                dec0[0] = dec0[0] + lr * R.mean(axis=0)
            Rw = R if predict_frames else R * wcol             # lag-weighted residual
            gZ = (Rw @ Dk.T) * dS + cost_grad(Z, dS, on) + decorr_grad(S, dS)
            G = gZ.T @ P                                       # -dLoss/dW (encoder)
            if D is None:
                G = G + S.T @ Rw                               # tied: decoder role too
            else:
                D = norm_rows(D + lr * (Sn.T @ Rw) / n)
            V = strf_step(V, step(G.reshape(K, C, L)), n)
            b = bias_step(b, gZ, S, n)
            w = 1.0 if predict_frames else wcol
            err += float(np.mean(w * R ** 2)) / float(np.mean(w * Y ** 2))
            act += float(np.mean(Z > 0))
            alive |= (Z > 0).any(axis=0)
            act_unit += (Z > 0).mean(axis=0)
            s1, s2, nfr = s1 + S.sum(axis=0), s2 + (S ** 2).sum(axis=0), nfr + T
        if restart_dead > 0 and unit_norm_strf and nfr > 0:
            sd = np.sqrt(np.maximum(s2 / nfr - (s1 / nfr) ** 2, 0.0))
            dead = np.flatnonzero(sd < restart_dead * np.median(sd))
            if dead.size:
                g = np.linalg.norm(V.reshape(K, -1), axis=1)
                gmed = float(np.median(g))
                for k in dead:
                    Xr = cochs[int(rng.integers(len(cochs)))]
                    Pr = (_past_windows(Xr, L) - mu) / input_scale
                    u = to_span(Pr[int(rng.integers(len(Pr)))].reshape(1, C, L))[0]
                    V[k] = gmed * u / max(float(np.linalg.norm(u)), 1e-12)
                    b[k] = 0.0
                    lam[k] = 0.0
                    if D is not None:
                        dk = rng.standard_normal(D.shape[1])
                        D[k] = dk / np.linalg.norm(dk)
                log(f"  restarted {dead.size} dead units (activity std < {restart_dead:g} x "
                    f"median)")
        outer_done = False
        if al_exact and auglag and al_on[0]:
            assert act_target is not None
            cbar = soft_unit / len(cochs) - act_target   # full-data violation, this epoch
            obj = err / len(cochs) + float(np.mean(lam * cbar + 0.5 * mu_al[0] * cbar ** 2))
            al_st["inner"] += 1
            if obj < al_st["best"] * (1.0 - al_inner_tol):
                al_st["best"], al_st["stale"] = obj, 0
            else:
                al_st["stale"] += 1
            if ((al_st["inner"] >= al_inner_min and al_st["stale"] >= 1)
                    or al_st["inner"] >= al_inner_max):
                lam = lam + mu_al[0] * cbar
                viol = float(np.linalg.norm(cbar))
                grew = viol > 0.25 * mu_al[1]
                if grew:
                    mu_al[0] = min(mu_al[0] * al_mu_growth, act_mu_max)
                mu_al[1] = viol
                al_st["outer"] += 1
                log(f"  AL outer {al_st['outer']} after {al_st['inner']} inner epochs: "
                    f"rms violation {viol / np.sqrt(K):.3g}, lambda median "
                    f"{np.median(lam):.3g}, mu {mu_al[0]:.3g}{' (grown)' if grew else ''}")
                al_st.update(inner=0, best=np.inf, stale=0)
                outer_done = True
        elif auglag and al_on[0] and al_plain:
            pass                                         # lambda updated every N bouts
        elif auglag and al_on[0]:
            assert act_target is not None
            cbar = soft_unit / len(cochs) - act_target   # epoch-mean violation per unit
            if act_update_every > 0:                     # flush the remaining bouts
                lam = lam + mu_al[0] * c_acc[0] / len(cochs)
            else:
                lam = lam + mu_al[0] * cbar
            viol = float(np.linalg.norm(cbar))
            # grow mu only while the violation is still large (rms > 10% of the target);
            # growing it on the residual oscillation amplifies the oscillation
            if viol > 0.75 * mu_al[1] and viol / np.sqrt(K) > 0.1 * act_target:
                mu_al[0] = min(mu_al[0] * act_mu_growth, act_mu_max)
            mu_al[1] = viol
        ve = val_error(V, b, D) if val_cochs is not None else np.nan
        # With an activity target, tightening the constraint raises the reconstruction error
        # at first; only start early stopping once >= 90% of units are within +-50% of it.
        f_unit = soft_unit / len(cochs) if rate_mode else act_unit / len(cochs)
        act_ok = (act_target is None
                  or float(np.mean(np.abs(f_unit - act_target) <= 0.5 * act_target)) >= 0.9)
        es_epoch = (not al_exact) or outer_done          # exact AL: per outer iteration
        if (val_cochs is not None and act_ok and es_epoch
                and (beta_now[0] is None or no_anneal)):
            if ve < best[0] - min_delta:
                best, stale = (ve, V.copy(), b.copy(), None if D is None else D.copy(), ep), 0
            else:
                stale += 1
        stop = patience > 0 and stale >= patience
        if softplus_beta0 is not None:                   # softplus schedule for next epoch
            if softplus_hold is None:
                sp[0] *= softplus_growth
            elif sp[1] == "ramp":
                sp[0] = min(sp[0] * softplus_growth, softplus_hold)
                if sp[0] >= softplus_hold:
                    sp[1] = "hold"
            elif sp[1] == "hold" and no_anneal:
                pass                                     # keep the softplus indefinitely
            elif sp[1] == "hold" and not act_ok:
                sp[2], sp[3] = np.inf, 0                 # constraint unmet: keep holding
            elif sp[1] == "hold":
                if ve < sp[2] - min_delta:
                    sp[2], sp[3] = ve, 0
                else:
                    sp[3] += 1
                if sp[3] >= softplus_hold_patience:
                    sp[1] = "anneal"
                    log(f"  softplus plateau at beta {sp[0]:.3g} (held-out {sp[2]:.3f}); "
                        f"annealing x{softplus_anneal}/epoch to ReLU")
            else:
                sp[0] *= softplus_anneal
        if checkpoint is not None:
            state = {"V": V, "b": b, "mu": mu, "epoch": ep + 1, "val_error": ve,
                     "stale": stale, "stopped": stop, "best_val": best[0],
                     "best_V": best[1], "best_b": best[2], "best_epoch": best[4],
                     "rng_state": json.dumps(rng.bit_generator.state), "lam": lam,
                     "mu_al": mu_al[0], "viol_prev": mu_al[1], "sp_beta": sp[0],
                     "sp_phase": sp[1], "sp_best": sp[2], "sp_stale": sp[3],
                     "al_c_est": al_st["c_est"],
                     "al_state": json.dumps({k: (float(v) if k == "best" else int(v))
                                             for k, v in al_st.items() if k != "c_est"})}
            if D is not None:
                state |= {"D": D, "best_D": best[3]}
            checkpoint(state)
        if ep % log_every == 0 or ep == n_epochs - 1 or stop:
            near, mlag = compactness(V)
            msg = (f"  epoch {ep + 1:3d}/{n_epochs}  unexplained variance train "
                   f"{err / len(cochs):.3f} held-out {ve:.3f}  active {act / len(cochs):.1%}  "
                   f"units alive {alive.sum()}/{K}  zeros {float(np.mean(V == 0)):.0%}  "
                   f"STRF energy <=20 ms {near:.0%}, mean lag {mlag:.0f} ms")
            if beta_now[0] is not None:
                msg += f"  softplus beta {beta_now[0]:.3g}"
            elif softplus_beta0 is not None:
                msg += "  ReLU"
            if act_target is not None:
                pct = np.percentile(f_unit, [10, 50, 90])
                met = "" if act_ok else ", not yet met"
                if rate_mode:
                    gain = float(np.linalg.norm(D, axis=1).mean()) if D is not None else 1.0
                    msg += (f"  unit mean rate 10/50/90% {pct[0]:.3g}/{pct[1]:.3g}/{pct[2]:.3g}"
                            f" (target {act_target:g}{met}), decoder gain {gain:.3g}")
                else:
                    msg += (f"  unit active 10/50/90% {pct[0]:.0%}/{pct[1]:.0%}/{pct[2]:.0%}"
                            f" (target {act_target:.0%}{met})")
                msg += f", bias med {np.median(b):.3g}"
                if penalty or auglag:
                    lp = np.percentile(lam, [10, 50, 90])
                    msg += f", lambda 10/50/90% {lp[0]:.3g}/{lp[1]:.3g}/{lp[2]:.3g}"
                if auglag:
                    msg += f", mu {mu_al[0]:.3g}"
            log(msg)
        if stop:
            log(f"  stopping: held-out error has not improved by {min_delta} for "
                f"{patience} epochs; best epoch {best[4] + 1} ({best[0]:.3f})")
            break
    # Without early stopping (patience 0) the final weights are returned: held-out
    # reconstruction is then only a diagnostic (e.g. constrained runs, where the goal is the
    # STRF shape and a stronger constraint is expected to cost reconstruction).
    if val_cochs is not None and best[4] >= 0 and patience > 0:
        _, V, b, D, _ = best
    return V, b, mu, (V.reshape(K, C * L).copy() if D is None else D)


def dog_kernel(sigma_e: float = 6.4, mu_e: float = 13.0, sigma_i: float = 42.0,
               mu_i: float = 60.0, a_i: float = 1.0, length: int = 300) -> np.ndarray:
    """Causal difference-of-Gaussians temporal kernel at 1 ms: a narrow excitatory Gaussian
    (default FWHM 15 ms) minus ``a_i`` times a delayed wide inhibitory one (FWHM 100 ms),
    each normalised to unit area, so with ``a_i = 1`` sustained input cancels."""
    t = np.arange(length, dtype=np.float64)
    ge = np.exp(-0.5 * ((t - mu_e) / sigma_e) ** 2)
    gi = np.exp(-0.5 * ((t - mu_i) / sigma_i) ** 2)
    return ge / ge.sum() - a_i * gi / gi.sum()


class SWEncoder:
    """Primary AC (dense rectified STRFs) -> higher AC (difference-of-Gaussians on each
    primary unit's rate -- or its Poisson spike train with ``primary_noise`` -> per-unit
    threshold -> ReLU). See the module docstring."""

    kind = "sw"

    def __init__(self, W, b, scale, coch_params, *, primary_gain, dog=None,
                 threshold=0.0, keep=None, D=None, d0=None, mu=None,
                 primary_noise=False, primary_beta=None):
        self.W = np.asarray(W, dtype=np.float64)
        #: Primary nonlinearity: None = ReLU, else softplus(beta z) / beta (as trained).
        self.primary_beta = None if primary_beta is None else float(primary_beta)
        #: Mean window subtracted before the STRFs (the autoencoder's centring).
        self.mu = (np.zeros((self.W.shape[1], 1)) if mu is None
                   else np.asarray(mu, dtype=np.float64).reshape(-1, 1))
        self.b = np.asarray(b, dtype=np.float64).reshape(-1, 1)
        self.scale = float(scale)
        self.coch_params = dict(coch_params)
        self.L = int(self.coch_params["K_frames"])
        self.primary_gain = float(primary_gain)          # Hz per unit of ReLU output
        self.dog = dict(dog or {})
        #: Higher-AC threshold: a scalar, or one value per kept unit (shape (n_kept, 1)).
        th = np.asarray(threshold, dtype=np.float64)
        self.threshold = float(th) if th.ndim == 0 else th.reshape(-1, 1)
        self.keep = (np.arange(self.W.shape[0]) if keep is None
                     else np.asarray(keep, dtype=int))
        self.lam = 0.0                                    # dense code (interface parity)
        #: False (default) feeds the DoG the primary rates (as expected spikes per 1 ms bin),
        #: as in Schneider & Woolley's model; True feeds it Poisson primary spikes.
        self.primary_noise = bool(primary_noise)
        #: The autoencoder's decoder (for diagnostics: reconstructions), if saved.
        self.D = None if D is None else np.asarray(D, dtype=np.float64)
        self.d0 = None if d0 is None else np.asarray(d0, dtype=np.float64).reshape(-1, 1)

    @property
    def n_channels(self) -> int:
        return int(self.keep.size)

    @property
    def V(self) -> np.ndarray:
        """(n_kept, C, L) STRFs of the kept primary units, oldest lag first: output channel
        ``k`` (a higher-AC unit) is the DoG of primary unit ``keep[k]``."""
        C = int(self.coch_params["n_channels"])
        return self.W[self.keep].reshape(self.keep.size, C, self.L)

    # ---- stages -------------------------------------------------------------------------
    def cochleagram(self, signal, *, sr, rms_from=None):
        cp = self.coch_params
        sig = np.asarray(signal, dtype=np.float64)
        ref = sig if rms_from is None else np.asarray(rms_from, dtype=np.float64)
        rms = float(np.sqrt(np.mean(ref ** 2)))
        sig = sig / max(rms, 1e-12) * cp["rms_ref"]
        return cochleagram(sig, sr, n_channels=cp["n_channels"], frame_rate=cp["frame_rate"],
                           lo_hz=cp["lo_hz"], hi_hz=cp["hi_hz"]).astype(np.float64)

    def _nl(self, Z):
        if self.primary_beta is None:
            return np.maximum(Z, 0.0)
        return np.logaddexp(0.0, self.primary_beta * Z) / self.primary_beta

    def primary_rate(self, X) -> np.ndarray:
        """Primary AC rates (Hz) per frame, ``(n_channels, n_frames)``."""
        R = self._nl(self.W @ ((unfold(X, self.L) - self.mu) / self.scale) + self.b)
        return R[self.keep] * self.primary_gain

    def reconstruct(self, X) -> np.ndarray:
        """The decoder's reconstruction of each frame ``t-1`` from the code at ``t``
        (the newest frame of the past window), in cochleagram units: ``(C, T)``."""
        assert self.D is not None and self.d0 is not None, "no decoder saved"
        R = self._nl(self.W @ ((unfold(X, self.L) - self.mu) / self.scale) + self.b)
        Xw = (self.D @ R + self.d0) * self.scale + self.mu   # (C*L, T), channel-major
        C = int(self.coch_params["n_channels"])
        return Xw.reshape(C, self.L, -1)[:, -1, :]

    def primary_spikes(self, rate_frames, *, seed) -> np.ndarray:
        """Poisson primary spike trains at 1 ms (at most one per bin)."""
        ms = 1000 // int(self.coch_params["frame_rate"])
        lam = np.repeat(rate_frames, ms, axis=1) / 1000.0
        if not self.primary_noise:
            return 1.0 - np.exp(-lam)                       # P(spike in bin)
        return (np.random.default_rng(seed).poisson(lam) > 0).astype(np.float64)

    def higher_drive(self, spikes) -> np.ndarray:
        """Each primary spike train filtered by the DoG kernel (before threshold)."""
        k = dog_kernel(**self.dog)
        T = spikes.shape[1]
        out = np.empty_like(spikes)
        for i, s in enumerate(spikes):
            out[i] = np.convolve(s, k)[:T]
        return out

    def drive(self, X, *, seed) -> np.ndarray:
        """Higher-AC drive (DoG output before threshold) at 1 ms for cochleagram ``X``.

        The DoG starts from steady state: its history is the primary rate for silence (a
        dense code is not zero in silence), not zero. Without this, every unit sees a
        spurious step at the start of the sound -- an onset transient larger than anything
        the song itself drives."""
        ms = 1000 // int(self.coch_params["frame_rate"])
        n_pad = -(-dog_kernel(**self.dog).size // ms)            # frames, rounded up
        silent = self.primary_rate(np.zeros((X.shape[0], 1)))     # (n_kept, 1)
        rate = np.concatenate([np.repeat(silent, n_pad, axis=1), self.primary_rate(X)],
                              axis=1)
        return self.higher_drive(self.primary_spikes(rate, seed=seed))[:, n_pad * ms:]

    def encode_signal(self, signal, *, sr, n_ista=None, upsample_to_ms=False, T_out_ms=None,
                      rms_from=None, seed=None):
        """Waveform -> higher-AC activations ``(n_channels, T)`` at 1 ms (``upsample_to_ms``
        is implied: the higher stage works on 1 ms spike trains). The primary spikes are
        drawn with ``seed``, by default derived from the waveform, so the same sound gives
        the same primary spikes."""
        sig = np.asarray(signal, dtype=np.float64)
        if seed is None:
            seed = zlib.crc32(np.ascontiguousarray(sig, dtype=np.float32).tobytes())
        X = self.cochleagram(sig, sr=sr, rms_from=rms_from)
        acts = np.maximum(self.drive(X, seed=seed) - self.threshold, 0.0)
        if T_out_ms is not None:
            T = int(T_out_ms)
            acts = (acts[:, :T] if acts.shape[1] >= T else
                    np.concatenate([acts, np.zeros((acts.shape[0], T - acts.shape[1]))],
                                   axis=1))
        return acts.astype(np.float32)

    # ---- persistence --------------------------------------------------------------------
    def save(self, path):
        arrays: dict[str, Any] = {
            "kind": np.array("sw"), "W": self.W, "b": self.b, "scale": self.scale,
            "primary_gain": self.primary_gain, "threshold": self.threshold,
            "keep": self.keep, "mu": self.mu, "primary_noise": self.primary_noise,
            **{f"dog_{k}": v for k, v in self.dog.items()},
            **{f"cp_{k}": v for k, v in self.coch_params.items()}}
        if self.D is not None:
            arrays.update(D=self.D, d0=self.d0)
        if self.primary_beta is not None:
            arrays["primary_beta"] = self.primary_beta
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        cp = {k[3:]: (v.item() if v.ndim == 0 else v) for k, v in d.items()
              if k.startswith("cp_")}
        dog = {k[4:]: float(v) for k, v in d.items() if k.startswith("dog_")}
        return cls(d["W"], d["b"], float(d["scale"]), cp, primary_gain=float(d["primary_gain"]),
                   dog=dog, threshold=d["threshold"], keep=d["keep"],
                   D=d["D"] if "D" in d else None, d0=d["d0"] if "d0" in d else None,
                   mu=d["mu"] if "mu" in d else None,
                   primary_noise=bool(d["primary_noise"]) if "primary_noise" in d else True,
                   primary_beta=float(d["primary_beta"]) if "primary_beta" in d else None)
