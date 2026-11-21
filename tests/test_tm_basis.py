"""Takenaka-Malmquist temporal basis for the S&W encoder STRFs."""

import numpy as np

from spiking_ven.sw_encoder import learn_relu, tm_basis


def test_tm_basis_is_orthonormal_on_a_long_horizon():
    B = tm_basis([2, 4, 8, 16, 32], L=2000, frame_ms=2.0)
    assert np.allclose(B @ B.T, np.eye(5), atol=1e-8)


def test_equal_poles_give_the_discrete_laguerre_basis():
    # Discrete Laguerre: F_n(z) = sqrt(1-a^2) z^-1/(1-a z^-1) * ((z^-1 - a)/(1 - a z^-1))^n
    a, L = np.exp(-2.0 / 6.0), 400
    B = tm_basis([6.0] * 4, L=L, frame_ms=2.0)[:, ::-1]          # lag order, lag 1 first
    lag = np.arange(1, L + 1)
    assert np.allclose(B[0], np.sqrt(1 - a ** 2) * a ** (lag - 1))
    assert np.allclose(B @ B.T, np.eye(4), atol=1e-8)


def test_span_is_sums_of_exponentials_with_the_basis_time_constants():
    taus, L = [2.0, 5.0, 13.0], 32
    B = tm_basis(taus, L=L, frame_ms=2.0)[:, ::-1]
    lag = np.arange(1, L + 1)
    E = np.stack([np.exp(-2.0 * (lag - 1) / t) for t in taus])   # exp(-t/tau), t from lag 1
    coef, *_ = np.linalg.lstsq(B.T, E.T, rcond=None)
    assert np.allclose(B.T @ coef, E.T, atol=1e-10)


def test_training_keeps_strfs_in_the_basis_span():
    rng = np.random.default_rng(0)
    cochs = [np.abs(rng.standard_normal((6, 120))) * 0.1 for _ in range(3)]
    L = 16
    B = tm_basis([2, 6, 18], L=L, frame_ms=2.0)
    V, *_ = learn_relu(cochs, K=4, L=L, n_epochs=2, conv_frames=4, l1_V=1e-4,
                       smooth_V=1e-2, mp_strength=0.25, init_scale=0.0, init_bias=0.01,
                       basis=B, log=lambda s: None)
    coef, *_ = np.linalg.lstsq(B.T, V.reshape(-1, L).T, rcond=None)
    assert np.allclose(B.T @ coef, V.reshape(-1, L).T, atol=1e-10)
    assert np.abs(V).max() > 0


def test_activity_target_holds_every_unit_at_its_firing_fraction():
    from spiking_ven.causal_filters import _past_windows

    rng = np.random.default_rng(0)
    cochs = [np.abs(rng.standard_normal((10, 300))) * 0.1 for _ in range(6)]
    L = 16
    V, b, mu, _ = learn_relu(cochs, K=12, L=L, n_epochs=30, l1_V=0, smooth_V=0,
                             mp_strength=0, init_scale=0, init_bias=0.01,
                             basis=tm_basis([2, 6, 18], L, 2.0), act_target=0.3,
                             log=lambda s: None)
    frac = np.concatenate([(_past_windows(X, L) - mu) @ V.reshape(12, -1).T + b > 0
                           for X in cochs]).mean(axis=0)
    assert np.all(np.abs(frac - 0.3) < 0.05)
