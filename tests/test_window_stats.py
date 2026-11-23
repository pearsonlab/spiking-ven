"""Streaming window statistics used to centre and scale the S&W encoder's input."""

import numpy as np

from spiking_ven.causal_filters import _past_windows
from spiking_ven.sw_encoder import _window_mean_var, window_scale


def test_streaming_stats_match_the_concatenated_windows():
    rng = np.random.default_rng(0)
    cochs = [rng.gamma(2.0, 1.0, size=(5, T)) for T in (40, 7, 113)]
    L = 6
    P = np.concatenate([_past_windows(X, L) for X in cochs], axis=0)
    mu, var = _window_mean_var(cochs, L)
    assert np.allclose(mu, P.mean(axis=0))
    assert np.allclose(var, P.var(axis=0))
    assert np.isclose(window_scale(cochs, L), np.sqrt(np.mean(P.var(axis=0))))
