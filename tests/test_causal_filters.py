"""The causal minimum-phase filter bank.

The property that matters is causality -- an activation at time t must not depend on
anything at or after t -- and it is not visible in any summary statistic, so it is
pinned here by truncation and by perturbation: changing the signal from t on must leave
every earlier activation unchanged. A non-causal solver fails this while still producing
a plausible-looking code.
"""

import numpy as np
import pytest

from spiking_ven import CausalFilterEncoder
from spiking_ven.causal_filters import (
    _past_windows,
    causal_meansub,
    infer,
    learn,
    min_phase,
    prune_silent,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def toy(rng):
    K, C, L = 12, 8, 6
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    X = rng.standard_normal((C, 80)) * 0.3
    return V, X, K, C, L


def test_past_windows_exclude_the_present(toy):
    _, X, _, C, L = toy
    P = _past_windows(X, L).reshape(-1, C, L)
    for t in (0, 1, L, 40):
        expected = np.zeros((C, L))
        lo = max(t - L, 0)
        if t > 0:
            expected[:, L - (t - lo):] = X[:, lo:t]
        assert np.array_equal(P[t], expected)


def test_inference_is_causal_under_truncation(toy):
    """Cut the signal at t; every earlier activation must be unchanged.

    Compared to a tolerance, not bit-exactly: truncating the signal changes the shape of
    the drive matmul, and BLAS may round the same dot product differently. ISTA's
    soft-threshold is Lipschitz, so it cannot turn a rounding difference into a large
    one. 1e-9 sits many orders of magnitude above double rounding and far below a real
    leak.
    """
    V, X, _, _, _ = toy
    S_full = infer(X, V, lam=0.05)
    for t in (20, 45, 60):
        S_cut = infer(X[:, :t], V, lam=0.05)
        assert np.allclose(S_full[:, :t], S_cut, rtol=0.0, atol=1e-9), (
            f"future leaked into t<{t}: "
            f"max |diff| = {np.abs(S_full[:, :t] - S_cut).max():.3e}"
        )


@pytest.mark.parametrize("divisive_gain", [True, False])
@pytest.mark.parametrize("t0_ms", [150.0, 150.5, 151.0])
def test_encode_signal_is_causal_from_the_waveform(rng, divisive_gain, t0_ms):
    """Waveform in, activations out: changing the audio from t0 on must leave every
    activation before t0 unchanged. This covers the cochleagram's frame timing and the
    divisive gain, which ``infer`` alone does not see.

    ``rms_from`` is the unperturbed signal for both encodes. It sets one fixed input gain
    -- the sound level -- and the pipeline always passes it, so the gain is not allowed
    to depend on the perturbation. t0 is placed on, between and off frame boundaries
    (2 ms frames), since a lookahead shorter than a frame only shows at some offsets.
    """
    sr, C, L, K = 16000, 8, 6, 12
    V = rng.standard_normal((K, C, L))
    V /= np.linalg.norm(V.reshape(K, -1), axis=1)[:, None, None]
    cp = {"n_channels": C, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    enc = CausalFilterEncoder(V, np.zeros(K), 0.01, cp)

    x = rng.standard_normal(int(0.3 * sr)) * 0.05
    y = x.copy()
    s0 = int(t0_ms * sr / 1000)
    y[s0:] += rng.standard_normal(len(y) - s0) * 0.5
    kw = dict(sr=sr, upsample_to_ms=True, rms_from=x, divisive_gain=divisive_gain)
    a, b = enc.encode_signal(x, **kw), enc.encode_signal(y, **kw)

    t0 = int(np.ceil(t0_ms))
    assert np.abs(a[:, :t0]).max() > 0, "no activations before t0: the test is vacuous"
    assert np.abs(a[:, t0:] - b[:, t0:]).max() > 0, "the perturbation must register"
    leak = np.abs(a[:, :t0] - b[:, :t0]).max(axis=0)
    assert np.all(leak <= 1e-9), f"audio at {t0_ms} ms reached ms {np.flatnonzero(leak > 1e-9)[0]}"


def test_causal_meansub_is_causal(toy):
    _, X, _, _, L = toy
    full = causal_meansub(X, L)
    for t in (10, 50):
        assert np.allclose(full[:, :t], causal_meansub(X[:, :t], L))


def test_min_phase_preserves_magnitude_and_advances_energy(toy):
    """Minimum phase is a phase-only operation: same magnitude response, earlier energy."""
    V, _, _, _, L = toy
    M = min_phase(V)
    a = np.abs(np.fft.rfft(V[:, :, ::-1], 2048, axis=2))
    b = np.abs(np.fft.rfft(M[:, :, ::-1], 2048, axis=2))
    # Exact preservation is unreachable: the minimum-phase counterpart of a finite
    # kernel is IIR and is truncated back to L frames. 1% pins a sufficiently large
    # default n_fft.
    assert np.abs(a - b).max() / a.max() < 0.01

    def centroid(A):
        E = (A ** 2).sum(axis=1)
        return ((E / E.sum(axis=1, keepdims=True)) * np.arange(L)).sum(axis=1).mean()

    # Column L-1 is the most recent frame, so a later centroid means earlier energy.
    assert centroid(M) > centroid(V)


def test_min_phase_is_idempotent(toy):
    V, _, _, _, _ = toy
    M = min_phase(V)
    assert np.abs(M - min_phase(M)).max() / np.abs(M).max() < 0.01


def test_learn_and_prune(rng):
    """learn returns unit-norm kernels; prune_silent keeps exactly the filters that fire."""
    cochs = [causal_meansub(rng.standard_normal((8, 120)) * 0.3, 6) for _ in range(3)]
    V = learn(cochs, K=10, L=6, lam=0.05, n_epochs=8, n_iter=10, seed=1,
              log=lambda _m: None)
    assert V.shape == (10, 8, 6) and np.all(np.isfinite(V))
    np.testing.assert_allclose(np.linalg.norm(V.reshape(10, -1), axis=1), 1.0)
    V_kept, kept = prune_silent(V, cochs, 0.05, n_iter=10)
    fired = np.zeros(len(V), dtype=bool)
    for X in cochs:
        fired |= (infer(X, V, 0.05, n_iter=10) != 0).any(axis=1)
    assert np.array_equal(kept, np.flatnonzero(fired)) and len(V_kept) == len(kept)


def test_encoder_roundtrip_and_width(tmp_path, toy):
    V, _, K, C, L = toy
    cp = {"n_channels": C, "K_frames": L, "frame_rate": 500,
          "lo_hz": 200.0, "hi_hz": 8000.0, "rms_ref": 0.1}
    enc = CausalFilterEncoder(V, np.zeros(K), 0.05, cp)
    assert enc.n_channels == K and enc.L == L
    path = tmp_path / "enc.npz"
    enc.save(path)
    back = CausalFilterEncoder.load(path)
    assert np.array_equal(back.V, enc.V) and back.lam == enc.lam
    assert back.coch_params == cp
