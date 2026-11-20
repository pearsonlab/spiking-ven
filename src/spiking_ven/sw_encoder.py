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


def learn_relu(cochs, *, K: int = 128, L: int = 32, n_epochs: int = 80, lr: float = 0.1,
               reg_lr: float | None = None, seed: int = 42, mp_strength: float = 0.25,
               l1_V: float = 7e-5, smooth_V: float = 1e-3, tied: bool = False, log=print):
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
    flattened when tied)."""
    from .causal_filters import _past_windows, _roughness_grad, min_phase

    if reg_lr is None:
        reg_lr = lr
    C = cochs[0].shape[0]
    rng = np.random.default_rng(seed)
    mu = np.concatenate([_past_windows(X, L) for X in cochs], axis=0).mean(axis=0)
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    b = np.zeros(K)
    D = None if tied else V.reshape(K, C * L).copy()
    for ep in range(n_epochs):
        err, act, alive = 0.0, 0.0, np.zeros(K, dtype=bool)
        for X in [cochs[i] for i in rng.permutation(len(cochs))]:
            T = X.shape[1]
            P = _past_windows(X, L) - mu                       # (T, C*L)
            W = V.reshape(K, C * L)
            Z = P @ W.T + b
            S = np.maximum(Z, 0.0)                             # (T, K)
            Dk = W if D is None else D
            R = P - S @ Dk                                     # residual
            gZ = (R @ Dk.T) * (Z > 0)                          # -dLoss/dZ
            G = gZ.T @ P                                       # -dLoss/dW (encoder)
            n = max(T, 1)
            if D is None:
                G = G + S.T @ R                                # tied: decoder role too
            else:
                D = D + lr * (S.T @ R) / n
                D /= np.maximum(np.linalg.norm(D, axis=1), 1e-9)[:, None]
            V = V + lr * G.reshape(K, C, L) / n
            b = b + lr * gZ.sum(axis=0) / n
            if smooth_V > 0:
                V = V - reg_lr * smooth_V * _roughness_grad(V)
            if mp_strength > 0:
                V = (1.0 - mp_strength) * V + mp_strength * min_phase(V)
            if l1_V > 0:
                V = np.sign(V) * np.maximum(np.abs(V) - reg_lr * l1_V, 0.0)
            err += float(np.mean(R ** 2)) / float(np.mean(P ** 2))
            act += float(np.mean(S > 0))
            alive |= (S > 0).any(axis=0)
        if ep % 5 == 0 or ep == n_epochs - 1:
            log(f"  epoch {ep + 1:3d}/{n_epochs}  unexplained variance "
                f"{err / len(cochs):.3f}  active {act / len(cochs):.1%}  "
                f"units alive {alive.sum()}/{K}  zeros {float(np.mean(V == 0)):.0%}  "
                f"mean |V_k| {float(np.linalg.norm(V.reshape(K, -1), axis=1).mean()):.3f}")
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
                 primary_noise=False):
        self.W = np.asarray(W, dtype=np.float64)
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

    def primary_rate(self, X) -> np.ndarray:
        """Primary AC rates (Hz) per frame, ``(n_channels, n_frames)``."""
        R = np.maximum(self.W @ ((unfold(X, self.L) - self.mu) / self.scale) + self.b, 0.0)
        return R[self.keep] * self.primary_gain

    def reconstruct(self, X) -> np.ndarray:
        """The decoder's reconstruction of each frame ``t-1`` from the code at ``t``
        (the newest frame of the past window), in cochleagram units: ``(C, T)``."""
        assert self.D is not None and self.d0 is not None, "no decoder saved"
        R = np.maximum(self.W @ ((unfold(X, self.L) - self.mu) / self.scale) + self.b, 0.0)
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
                   primary_noise=bool(d["primary_noise"]) if "primary_noise" in d else True)
