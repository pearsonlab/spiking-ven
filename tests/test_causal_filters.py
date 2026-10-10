"""The causal minimum-phase filter bank.

The property that matters is causality -- an activation at time t must not depend on
anything at or after t -- and it is not visible in any summary statistic, so it is
pinned here by truncation: cutting the signal after t must leave S[:, :t] bit-identical.
A non-causal solver fails this while still producing a plausible-looking code, which is
how an earlier convolutional encoder reported a response peaking 2 ms BEFORE its
stimulus.
"""

import numpy as np
import pytest

from spiking_ven import CausalFilterEncoder
from spiking_ven.causal_filters import _past_windows, causal_meansub, infer, learn, min_phase


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

    Compared to a tolerance, not bit-exactly. ``infer`` builds its drive with
    ``_past_windows(X, L) @ W.T``, a matmul whose FIRST dimension is T, so truncating the
    signal changes the matrix shape; BLAS may then partition the same dot product
    differently and round it differently. Everything after that is a deterministic,
    continuous function of the drive -- ISTA's soft-threshold is Lipschitz, so it cannot
    turn a rounding difference into a large one -- which is why a tolerance is safe here.

    Bit-exactness held on macOS/Accelerate at every thread count but failed
    intermittently on Linux CI, on commits that did not touch this file.

    1e-9 is ~7 orders of magnitude above double rounding and ~8 below a real leak: when a
    genuinely non-causal solver was tested this way, truncation moved earlier
    coefficients by up to 0.549.
    """
    V, X, _, _, _ = toy
    S_full = infer(X, V, lam=0.05)
    for t in (20, 45, 60):
        S_cut = infer(X[:, :t], V, lam=0.05)
        assert np.allclose(S_full[:, :t], S_cut, rtol=0.0, atol=1e-9), (
            f"future leaked into t<{t}: "
            f"max |diff| = {np.abs(S_full[:, :t] - S_cut).max():.3e}"
        )


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
    # kernel is IIR and is truncated back to L frames. 1% pins the default n_fft, which
    # was 7.7% off at L=16 before it was raised.
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


def test_learn_keeps_every_filter_alive(rng):
    """Usage equalisation must reverse dropout, not merely discourage it."""
    cochs = [causal_meansub(rng.standard_normal((8, 120)) * 0.3, 6) for _ in range(3)]
    V, bias = learn(cochs, K=10, L=6, lam=0.05, n_epochs=8, n_iter=10, seed=1,
                    target_usage=0.05, log=lambda _m: None)
    assert V.shape == (10, 8, 6)
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(bias))
    assert bias.min() >= -0.05 - 1e-12          # threshold can reach 0, never below
    usage = (infer(cochs[0], V, 0.05, bias=bias) != 0).mean(axis=1)
    assert (usage > 0).all(), f"{int((usage == 0).sum())} filters never fire"


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
