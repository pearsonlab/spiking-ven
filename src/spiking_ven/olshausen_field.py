"""Olshausen-Field (1997) sparse coding — audio or cochleagram patches.

Two operating modes
-------------------
Audio mode  (original)
    Learns a dictionary of raw-waveform snippets.
    Use: encoder.fit(audio_list=...) / encoder.encode(signal)

Cochleagram mode  (preferred — matches Gong et al.)
    Input signals are first converted to a gammatone cochleagram, then
    overlapping spectrotemporal patches (n_channels × K_frames) are
    extracted and fed to the same O&F machinery.  Each learned basis atom
    is a spectrotemporal receptive field (STRF), visualised by reshaping
    the flat (n_channels * K_frames) vector to (n_channels, K_frames).
    Use: encoder.fit(patches=...) with patches from coch_extract_patches()
         encoder.encode_patches(X) for inference
         coch_encode(signal, encoder, sr) for the full pipeline

Patches are always mean-subtracted before the inner product so that atoms
respond to spectral shape rather than absolute energy level.  The
cochleagram itself is NOT peak-normalised (causal / streaming safe).

Lifetime sparsity
-----------------
When lifetime_sparsity=True, each atom maintains a per-basis threshold
bias adapted to equalise usage rates.  See fit() docstring.

References
----------
Olshausen & Field, Nature 1996; Neural Comp 1997.
Gong et al. 2026 (SI Methods, sparse coding section).
"""

import numpy as np
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# ISTA core
# ---------------------------------------------------------------------------

def _ista(
    X: np.ndarray,           # (B, W)  patches (already mean-subtracted)
    A: np.ndarray,           # (L, W)  dictionary (unit-norm rows)
    thresh_vec: np.ndarray,  # (L,)    per-basis threshold = (λ + bias) * step
    n_iter: int,
    step: float,
) -> np.ndarray:
    """ISTA for  min_S (1/2)||X - S A||² + ||thresh_vec ⊙ S||₁."""
    S = np.zeros((len(X), len(A)), dtype=np.float64)
    for _ in range(n_iter):
        R = X - S @ A
        S = S + step * (R @ A.T)
        S = np.sign(S) * np.maximum(np.abs(S) - thresh_vec, 0.0)
    return S


# ---------------------------------------------------------------------------
# Cochleagram patch helpers (module-level, usable independently)
# ---------------------------------------------------------------------------

def coch_extract_patches(
    coch: np.ndarray,
    K_frames: int,
    stride: int = 1,
    mean_subtract: bool = True,
) -> np.ndarray:
    """Extract causal spectrotemporal patches from a cochleagram.

    Each patch covers the trailing K_frames frames ending at the current
    position: coch[:, t-K+1 : t+1] flattened to (n_channels * K_frames,).
    The first K-1 output frames are zero-padded on the left (causal boundary).

    Parameters
    ----------
    coch          : (n_channels, T_frames) cochleagram
    K_frames      : context window width in frames
    stride        : output stride in frames (1 = every frame)
    mean_subtract : subtract per-patch mean (removes DC / overall level)

    Returns
    -------
    patches : (T_out, n_channels * K_frames) float32
    """
    n_ch, T = coch.shape
    W = n_ch * K_frames
    # Causal zero-padding: K-1 empty frames before the signal
    padded = np.concatenate(
        [np.zeros((n_ch, K_frames - 1), dtype=np.float64),
         coch.astype(np.float64)],
        axis=1,
    )  # (n_ch, T + K - 1)

    T_out = (T + stride - 1) // stride
    t_starts = np.arange(T_out) * stride               # (T_out,)
    k_offsets = np.arange(K_frames)                    # (K_frames,)
    col_idx = t_starts[:, np.newaxis] + k_offsets      # (T_out, K_frames)

    # padded[:, col_idx] → (n_ch, T_out, K_frames)
    all_patches = padded[:, col_idx]                   # (n_ch, T_out, K_frames)
    patches = all_patches.transpose(1, 0, 2).reshape(T_out, W)  # (T_out, W)

    if mean_subtract:
        patches -= patches.mean(axis=1, keepdims=True)

    return patches.astype(np.float32)


def coch_encode(
    signal: np.ndarray,
    encoder: "OlshausenFieldEncoder",
    sr: int,
    n_ista: int = 100,
    upsample_to_ms: bool = False,
    T_out_ms: int | None = None,
) -> np.ndarray:
    """Full pipeline: audio → cochleagram → ISTA → (n_bases, T) activations.

    Parameters
    ----------
    signal        : 1-D float audio array
    encoder       : trained OlshausenFieldEncoder with coch_params set
    sr            : audio sample rate (Hz)
    n_ista        : ISTA iterations
    upsample_to_ms: if True, repeat each frame so output is at 1 ms resolution
    T_out_ms      : if set, trim/pad output to exactly this many ms columns

    Returns
    -------
    activations : (n_bases, T) float32; T in coch frames unless upsample_to_ms
    """
    from .cochleagram import cochleagram

    cp = encoder.coch_params
    if cp is None:
        raise ValueError("encoder.coch_params is None — use fit(patches=...) with "
                         "cochleagram patches and set encoder.coch_params manually, "
                         "or use coch_encode only with cochleagram-trained encoders.")

    sig = signal.astype(np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    sig = sig / max(rms, 1e-12) * cp["rms_ref"]

    coch = cochleagram(
        sig, sr,
        n_channels=cp["n_channels"],
        frame_rate=cp["frame_rate"],
        lo_hz=cp["lo_hz"],
        hi_hz=cp["hi_hz"],
    )  # (n_ch, T_frames)

    patches = coch_extract_patches(coch, cp["K_frames"], stride=1, mean_subtract=True)
    acts = encoder.encode_patches(patches, n_ista=n_ista)  # (n_bases, T_frames)

    if upsample_to_ms:
        frame_ms = 1000 // cp["frame_rate"]   # ms per frame (e.g. 10)
        acts = np.repeat(acts, frame_ms, axis=1)

    if T_out_ms is not None:
        if upsample_to_ms:
            T_target = T_out_ms
        else:
            T_target = T_out_ms // (1000 // cp["frame_rate"])
        if acts.shape[1] >= T_target:
            acts = acts[:, :T_target]
        else:
            pad = np.zeros((acts.shape[0], T_target - acts.shape[1]), dtype=acts.dtype)
            acts = np.concatenate([acts, pad], axis=1)

    return acts


# ---------------------------------------------------------------------------
# Encoder class
# ---------------------------------------------------------------------------

class OlshausenFieldEncoder:
    """Olshausen-Field sparse coder on arbitrary patches.

    Patch source is external (audio clips or cochleagram slices).
    The core ISTA + dict-update loop is the same regardless.

    Parameters
    ----------
    n_bases  : number of dictionary atoms (L)
    patch_len: flattened patch dimension (W); for cochleagram mode
               this is n_channels * K_frames
    seed     : RNG seed for dictionary initialisation
    """

    def __init__(
        self,
        n_bases: int = 64,
        patch_len: int = 320,
        seed: int | None = None,
        # kept for back-compat with audio-mode construction
        patch_ms: float | None = None,
        sr: int = 16000,
    ) -> None:
        self.n_bases = n_bases
        self.patch_len = patch_len
        self.sr = sr
        self.patch_ms = patch_ms           # only meaningful in audio mode
        self.lambda_sparse: float = 0.08
        self.rms_ref: float = 0.1
        self._bias: np.ndarray = np.zeros(n_bases, dtype=np.float64)
        self.coch_params: dict | None = None   # set externally for cochleagram mode

        rng = default_rng(seed)
        A = rng.standard_normal((n_bases, patch_len))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        self.A: np.ndarray = A.astype(np.float64)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        audio_list: list | None = None,
        *,
        patches: np.ndarray | None = None,
        n_epochs: int = 200,
        lr: float = 0.01,
        lambda_sparse: float = 0.08,
        n_ista: int = 30,
        batch_size: int = 64,
        rms_ref: float = 0.1,
        train_stride_ms: float = 10.0,
        lifetime_sparsity: bool = True,
        target_usage: float = 0.10,
        lr_bias: float = 0.005,
        ema_decay: float = 0.99,
        seed: int = 42,
    ) -> "OlshausenFieldEncoder":
        """Train dictionary.

        Pass either audio_list (raw audio, audio mode) or patches (pre-extracted
        patch matrix of shape (N, patch_len), e.g. from coch_extract_patches).

        lifetime_sparsity : adapt per-basis thresholds to equalise usage rates
        target_usage      : fraction of patches each atom should be active for
        """
        self.lambda_sparse = lambda_sparse
        self.rms_ref = rms_ref

        if patches is None:
            if audio_list is None:
                raise ValueError("Provide audio_list or patches=")
            patches = self._extract_audio_patches(audio_list, rms_ref, train_stride_ms)

        N = len(patches)
        print(
            f"  O&F: {N} patches × {self.patch_len}  L={self.n_bases}  "
            f"λ={lambda_sparse}  {n_epochs} epochs  "
            f"lifetime={lifetime_sparsity}"
            + (f"  target={target_usage:.0%}" if lifetime_sparsity else "")
        )

        step = self._lipschitz_step()
        usage_ema = np.full(self.n_bases, target_usage)
        rng = default_rng(seed)
        self._bias[:] = 0.0

        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            recon_acc, spar_acc, nb = 0.0, 0.0, 0

            for i in range(0, N, batch_size):
                X = patches[perm[i : i + batch_size]].astype(np.float64)
                thresh_vec = np.maximum((lambda_sparse + self._bias) * step, 0.0)

                S = _ista(X, self.A, thresh_vec, n_ista, step)

                R = X - S @ self.A
                self.A += lr * (S.T @ R) / len(X)
                norms = np.linalg.norm(self.A, axis=1, keepdims=True)
                self.A /= np.maximum(norms, 1e-8)

                if lifetime_sparsity:
                    usage_batch = (S != 0).mean(axis=0)
                    usage_ema = ema_decay * usage_ema + (1.0 - ema_decay) * usage_batch
                    self._bias += lr_bias * (usage_ema - target_usage)

                recon_acc += float(np.mean(R ** 2))
                spar_acc  += float(np.mean(S != 0.0))
                nb += 1

            step = self._lipschitz_step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch+1:3d}/{n_epochs}  "
                    f"recon={recon_acc/nb:.5f}  "
                    f"sparsity={spar_acc/nb*100:.1f}%  "
                    f"bias=[{self._bias.min():.3f},{self._bias.max():.3f}]  "
                    f"step={step:.4f}"
                )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode_patches(
        self,
        X: np.ndarray,
        n_ista: int = 100,
    ) -> np.ndarray:
        """Encode pre-extracted patch matrix.

        Parameters
        ----------
        X : (T, patch_len) float array
        Returns
        -------
        activations : (n_bases, T) float32
        """
        step = self._lipschitz_step()
        thresh_vec = np.maximum((self.lambda_sparse + self._bias) * step, 0.0)
        S = _ista(X.astype(np.float64), self.A, thresh_vec, n_ista, step)
        return S.T.astype(np.float32)

    def encode(
        self,
        signal: np.ndarray,
        frame_rate: int = 1000,
        n_ista: int = 100,
        rms_ref: float | None = None,
    ) -> np.ndarray:
        """Audio-mode encode (raw waveform → patches → ISTA).

        Returns (n_bases, T_frames) at frame_rate Hz.
        """
        if rms_ref is None:
            rms_ref = self.rms_ref

        sig = signal.astype(np.float64)
        rms = float(np.sqrt(np.mean(sig ** 2)))
        sig = sig / max(rms, 1e-12) * rms_ref

        W = self.patch_len
        stride = max(1, self.sr // frame_rate)
        n_frames = max(0, (len(sig) - W) // stride + 1)
        if n_frames == 0:
            return np.zeros((self.n_bases, 0), dtype=np.float32)

        row_idx = np.arange(W)[np.newaxis, :] + stride * np.arange(n_frames)[:, np.newaxis]
        X = sig[row_idx]
        X = (X - X.mean(axis=1, keepdims=True)).astype(np.float32)
        return self.encode_patches(X, n_ista)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        kw: dict = dict(
            A=self.A,
            bias=self._bias,
            n_bases=self.n_bases,
            patch_len=self.patch_len,
            sr=self.sr,
            lambda_sparse=self.lambda_sparse,
            rms_ref=self.rms_ref,
        )
        if self.patch_ms is not None:
            kw["patch_ms"] = self.patch_ms
        if self.coch_params is not None:
            cp = self.coch_params
            kw["coch_n_channels"] = cp["n_channels"]
            kw["coch_K_frames"]   = cp["K_frames"]
            kw["coch_frame_rate"] = cp["frame_rate"]
            kw["coch_lo_hz"]      = cp["lo_hz"]
            kw["coch_hi_hz"]      = cp["hi_hz"]
            kw["coch_rms_ref"]    = cp["rms_ref"]
        np.savez(path, **kw)

    @classmethod
    def load(cls, path: str) -> "OlshausenFieldEncoder":
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.A             = data["A"].astype(np.float64)
        obj.n_bases       = int(data["n_bases"])
        obj.patch_len     = int(data["patch_len"])
        obj.sr            = int(data["sr"])
        obj.lambda_sparse = float(data["lambda_sparse"])
        obj.rms_ref       = float(data["rms_ref"])
        obj._bias         = data["bias"].astype(np.float64) if "bias" in data else np.zeros(obj.n_bases)
        obj.patch_ms      = float(data["patch_ms"]) if "patch_ms" in data else None
        if "coch_n_channels" in data:
            obj.coch_params = dict(
                n_channels = int(data["coch_n_channels"]),
                K_frames   = int(data["coch_K_frames"]),
                frame_rate = int(data["coch_frame_rate"]),
                lo_hz      = float(data["coch_lo_hz"]),
                hi_hz      = float(data["coch_hi_hz"]),
                rms_ref    = float(data["coch_rms_ref"]),
            )
        else:
            obj.coch_params = None
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lipschitz_step(self) -> float:
        G = self.A @ self.A.T
        ev_max = float(np.linalg.eigvalsh(G).max())
        return 1.0 / max(ev_max, 1e-6)

    def _extract_audio_patches(
        self,
        audio_list: list,
        rms_ref: float,
        stride_ms: float,
    ) -> np.ndarray:
        stride = max(1, int(self.sr * stride_ms / 1000))
        W = self.patch_len
        all_patches: list[np.ndarray] = []
        for sig in audio_list:
            sig = np.asarray(sig, dtype=np.float64)
            rms = float(np.sqrt(np.mean(sig ** 2)))
            sig = sig / max(rms, 1e-12) * rms_ref
            n = max(0, (len(sig) - W) // stride + 1)
            if n == 0:
                continue
            idx = np.arange(W)[np.newaxis, :] + stride * np.arange(n)[:, np.newaxis]
            p = sig[idx].astype(np.float32)
            p -= p.mean(axis=1, keepdims=True)
            all_patches.append(p)
        if not all_patches:
            return np.empty((0, W), dtype=np.float32)
        return np.concatenate(all_patches, axis=0)


# ---------------------------------------------------------------------------
# Spike conversion (shared by both modes)
# ---------------------------------------------------------------------------

def of_to_spikes(
    activations: np.ndarray,
    mean_rate_hz: float = 15.0,
    frame_rate: int = 1000,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    per_neuron: bool = False,
    sign_split: bool = False,
) -> np.ndarray:
    """Convert O&F activations to Poisson spike trains.

    Parameters
    ----------
    activations  : (n_bases, T) float, signed ISTA coefficients
    mean_rate_hz : target mean firing rate (Hz); applied per-neuron when
                   per_neuron=True, else over the whole population
    frame_rate   : temporal resolution of activations in Hz
    per_neuron   : if True, calibrate each neuron independently to
                   mean_rate_hz rather than using a single global scale
    sign_split   : if True, return (2*n_bases, T) where the first n_bases
                   rows encode positive loadings and the last n_bases rows
                   encode negative loadings (each independently calibrated
                   when per_neuron=True)
    Returns
    -------
    spikes : (n_bases, T) or (2*n_bases, T) float32 spike counts per bin.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    acts = activations.astype(np.float64)
    if sign_split:
        act = np.concatenate([np.maximum(acts, 0.0), np.maximum(-acts, 0.0)], axis=0)
    else:
        act = np.maximum(acts, 0.0)

    dt_s = 1.0 / frame_rate
    if per_neuron:
        act_mean = act.mean(axis=1, keepdims=True)          # (n_neurons, 1)
        rate_scale = mean_rate_hz / np.maximum(act_mean, 1e-12)
    else:
        rate_scale = mean_rate_hz / max(float(act.mean()), 1e-12)

    lam = act * rate_scale * dt_s
    return rng.poisson(lam).astype(np.float32)


# ---------------------------------------------------------------------------
# Back-compat alias for audio mode
# ---------------------------------------------------------------------------

def of_encode(
    signal: np.ndarray,
    encoder: OlshausenFieldEncoder,
    frame_rate: int = 1000,
    n_ista: int = 100,
) -> np.ndarray:
    """Audio-mode convenience wrapper.  Drop-in for sl_gram()."""
    return encoder.encode(signal, frame_rate=frame_rate, n_ista=n_ista)
