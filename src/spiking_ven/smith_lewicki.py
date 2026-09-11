"""Smith-Lewicki matching-pursuit dictionary learning for raw audio.

Trains a bank of 1-D kernels on raw audio via matching pursuit + gradient
descent (Smith & Lewicki, Nature 2006). Learned kernels converge to
gammatone-like cochlear filters and serve as a drop-in replacement for the
fixed gammatone front-end.

Usage
-----
    from spiking_ven.smith_lewicki import SmithLewickiDictionary, sl_gram

    ld = SmithLewickiDictionary(n_kernels=64, kernel_ms=10.0)
    ld.fit(signal, sr=16000)
    gram = sl_gram(signal, ld.kernels, sr=16000)
    # gram shape: (n_kernels, n_frames) — same interface as cochleagram()

Performance notes
-----------------
- All segments are pre-sliced into a (n_segs, T) array once.
- Each MP iteration processes all segments in mini-batches via batched
  rfft / irfft (scipy.fft workers=-1 → all CPU cores).
- Kernel FFTs are computed once per epoch and reused across all spikes.

References
----------
Smith & Lewicki 2006, Nature 439:978
Smith & Lewicki 2005, Neural Comp. 17(1):19-45
"""

import numpy as np
from math import gcd
from numpy.random import default_rng

from scipy.fft import rfft, irfft
from scipy.signal import resample_poly


class SmithLewickiDictionary:
    def __init__(
        self,
        n_kernels: int = 64,
        kernel_ms: float = 10.0,
        sr: int = 16000,
        seed: int | None = None,
    ) -> None:
        rng = default_rng(seed)
        self.n_kernels = n_kernels
        self.kernel_len = int(sr * kernel_ms / 1000)
        self.sr = sr

        K = rng.standard_normal((n_kernels, self.kernel_len)).astype(np.float32)
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        self.kernels = K

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------


    @property
    def n_channels(self) -> int:
        """Number of output channels, i.e. kernels.

        Shared name with
        :class:`~spiking_ven.olshausen_field.OlshausenFieldEncoder` (which counts
        bases); see the note there.
        """
        return int(self.n_kernels)

    def save(self, path: str) -> None:
        """Save kernels and metadata to a .npz file."""
        np.savez(
            path,
            kernels=self.kernels,
            n_kernels=self.n_kernels,
            kernel_len=self.kernel_len,
            sr=self.sr,
        )

    @classmethod
    def load(cls, path: str) -> "SmithLewickiDictionary":
        """Load a previously saved SmithLewickiDictionary from a .npz file."""
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.kernels = data["kernels"]
        obj.n_kernels = int(data["n_kernels"])
        obj.kernel_len = int(data["kernel_len"])
        obj.sr = int(data["sr"])
        return obj

    def fit(
        self,
        signal: np.ndarray,
        n_epochs: int = 5,
        seg_ms: float = 160.0,
        n_spikes: int = 20,
        lr: float = 0.02,
        chunk_size: int = 32,
    ) -> "SmithLewickiDictionary":
        """Train on raw audio. Returns self."""
        seg_len = max(int(self.sr * seg_ms / 1000), self.kernel_len + 1)
        sig = signal.astype(np.float64)
        sig /= sig.std() + 1e-8

        n_segs = (len(sig) - self.kernel_len) // seg_len
        segs = np.stack([
            sig[i * seg_len : i * seg_len + seg_len] for i in range(n_segs)
        ])  # (n_segs, T_seg)

        L = self.kernel_len
        T = seg_len
        n = T + L - 1
        nfft = int(2 ** np.ceil(np.log2(n)))
        valid_len = T - L + 1

        print(f"  {n_segs} segments, {n_spikes} spikes/seg, nfft={nfft}, "
              f"chunk={chunk_size}, {n_epochs} epochs")

        for epoch in range(n_epochs):
            K_fft = rfft(
                self.kernels.astype(np.float64), n=nfft, axis=1, workers=-1
            )

            residuals = segs.copy()
            spike_records = [[] for _ in range(n_segs)]

            for _ in range(n_spikes):
                for start in range(0, n_segs, chunk_size):
                    end = min(start + chunk_size, n_segs)
                    chunk = residuals[start:end]

                    R = rfft(chunk, n=nfft, axis=1, workers=-1)

                    C = irfft(
                        R[:, np.newaxis, :] * np.conj(K_fft[np.newaxis, :, :]),
                        n=nfft, axis=2, workers=-1,
                    )
                    C_valid = C[:, :, :valid_len]

                    C_abs = np.abs(C_valid).reshape(end - start, -1)
                    flat = C_abs.argmax(axis=1)
                    k_best, t_best = np.unravel_index(flat, (self.n_kernels, valid_len))
                    a_best = C_valid[np.arange(end - start), k_best, t_best]

                    for i in range(end - start):
                        k, t, a = int(k_best[i]), int(t_best[i]), float(a_best[i])
                        spike_records[start + i].append((k, t, a))
                        residuals[start + i, t : t + L] -= (
                            a * self.kernels[k].astype(np.float64)
                        )

            self._update_kernels_batch(segs, spike_records, residuals, lr)

            recon_err = float(np.mean((segs - (segs - residuals)) ** 2))
            sig_pow = float(np.mean(segs ** 2))
            snr = 10.0 * np.log10(sig_pow / max(recon_err, 1e-12))
            print(f"  epoch {epoch + 1}/{n_epochs}  SNR={snr:.1f} dB")

        return self

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _update_kernels_batch(
        self,
        segs: np.ndarray,
        spike_records: list,
        residuals: np.ndarray,
        lr: float,
    ) -> None:
        L = self.kernel_len
        grad = np.zeros((self.n_kernels, L), dtype=np.float64)
        count = np.zeros(self.n_kernels, dtype=np.int32)

        for seg_idx, spikes in enumerate(spike_records):
            res = residuals[seg_idx]
            for k, t, a in spikes:
                r_local = res[t : t + L] + a * self.kernels[k].astype(np.float64)
                grad[k] += a * r_local
                count[k] += 1

        for k in range(self.n_kernels):
            if count[k] > 0:
                self.kernels[k] = (
                    self.kernels[k].astype(np.float64) + lr / count[k] * grad[k]
                ).astype(np.float32)

        norms = np.linalg.norm(self.kernels, axis=1, keepdims=True)
        self.kernels /= np.maximum(norms, 1e-8)


def sl_gram(
    signal: np.ndarray,
    kernels: np.ndarray,
    sr: int = 16000,
    frame_rate: int = 1000,
) -> np.ndarray:
    """Compute a Smith-Lewicki cochleagram from learned kernels.

    Drop-in replacement for cochleagram(): convolves the signal with each
    kernel (batched FFT, all CPU cores), takes the absolute value, applies
    sqrt compression, and decimates to frame_rate.

    Returns (n_kernels, n_frames) float32 with values proportional to
    sqrt(signal_amplitude). No per-signal normalization is applied; the
    caller is responsible for presenting audio at a consistent RMS level.
    See auditory/README.md for normalization conventions.
    """
    n_kernels, L = kernels.shape
    T = len(signal)
    n = T + L - 1
    nfft = int(2 ** np.ceil(np.log2(n)))

    sig = signal.astype(np.float64)
    R = rfft(sig, n=nfft, workers=-1)
    K = rfft(kernels.astype(np.float64), n=nfft, axis=1, workers=-1)
    C = irfft(R[np.newaxis, :] * np.conj(K), n=nfft, axis=1, workers=-1)

    gram = np.abs(C[:, : T - L + 1])
    gram = np.sqrt(gram)

    g = gcd(sr, frame_rate)
    gram = resample_poly(gram, frame_rate // g, sr // g, axis=1)
    gram = np.clip(gram, 0.0, None).astype(np.float32)

    return gram


def sl_gram_to_spikes(
    gram: np.ndarray,
    threshold: float = 3.5,
    scale: float = 1000.0,
    gain: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Convert a Smith-Lewicki gram to Poisson spike trains.

    Firing rate model:  rate_hz = scale * exp(gain * (gram - threshold))
    P(spike in 1 ms bin) = min(rate_hz * 1e-3, 1)

    Parameters
    ----------
    gram      : (n_kernels, T) float, values in [0, 1]
    threshold : log-rate shift; raise to increase sparsity.
                To target a specific mean rate r_hz on a gram with mean g:
                  threshold = g + log(scale / r_hz) / gain
    scale     : rate in Hz when (gram - threshold) = 0
    gain      : steepness of the exponential; higher gain → sharper selectivity
    rng       : numpy Generator (created from seed if None)
    seed      : RNG seed used only when rng is None

    Returns
    -------
    spikes : (n_kernels, T) float32, binary 0/1
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    rate_hz = scale * np.exp(gain * (gram.astype(np.float64) - threshold))
    prob    = np.clip(rate_hz * 1e-3, 0.0, 1.0)
    return (rng.random(gram.shape) < prob).astype(np.float32)
