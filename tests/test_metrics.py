"""Known-answer tests for the DAF response measures.

Synthetic trials with a response of known onset and length are fed to
:func:`spiking_ven.metrics.daf_response`, so the latency and duration rules are checked
against the answer they should give rather than against a recorded run.

Latency comes back exact to a bin. Duration reads long by design of the rule: a response
ends at the first run of ten non-significant bins, and at p < 0.05 about one baseline bin
in twenty is significant by chance, which postpones that run. On these trials the excess
averages ~8 ms (1-18 ms over 20 draws). The paper's rule has the same property, so the
model-to-bird comparison is like for like.
"""

import numpy as np
import pytest

from spiking_ven import metrics
from spiking_ven.metrics import daf_response

#: Allowed excess of the measured duration over the true one (see the module docstring).
DURATION_EXCESS = (-12, 20)

ONSET = 300          # ms into the rendition
T = 900              # rendition length (ms); room for a 300 ms duration window
N_TRIALS = 40


class _Ven:
    n_e = 30


def _trials(latency_ms, length_ms, *, base_hz=5.0, resp_hz=150.0, n_resp=10, seed=0):
    """Bernoulli spikes at ``base_hz``; units < n_resp fire at ``resp_hz`` from
    ``ONSET + latency_ms`` for ``length_ms``."""
    rng = np.random.default_rng(seed)
    p = np.full((N_TRIALS, _Ven.n_e, T), base_hz / 1000)
    t0 = ONSET + latency_ms
    p[:, :n_resp, t0:t0 + length_ms] = resp_hz / 1000
    return (rng.random(p.shape) < p).astype(np.float32)


def _response(monkeypatch, trials, **kw):
    monkeypatch.setattr(metrics, "daf_trials", lambda ven, enc, st, *, n_trials, seed: trials)
    st = {"daf_window_s": (ONSET / 1000, (ONSET + 50) / 1000), "T_song": T}
    return daf_response(_Ven(), None, st, n_trials=N_TRIALS, **kw)


@pytest.mark.parametrize("latency, length", [(20, 40), (30, 90)])
def test_latency_and_duration_recover_a_known_response(monkeypatch, latency, length):
    r = _response(monkeypatch, _trials(latency, length))
    assert r["n_responders"] == 10 and r["n_timed"] == 10
    assert r["latency_ms"] == pytest.approx(latency, abs=2)
    lo, hi = DURATION_EXCESS
    assert length + lo <= r["duration_ms"] <= length + hi
    assert r["duration_censored"] == 0


def test_a_response_longer_than_the_responder_window_is_timed_not_dropped(monkeypatch):
    """200 ms outlasts the 150 ms responder window; its duration must still be measured."""
    r = _response(monkeypatch, _trials(20, 200))
    assert r["n_timed"] == 10
    lo, hi = DURATION_EXCESS
    assert 200 + lo <= r["duration_ms"] <= 200 + hi
    assert r["duration_censored"] == 0


def test_a_response_still_running_at_the_window_end_is_counted_as_censored(monkeypatch):
    r = _response(monkeypatch, _trials(20, 500), duration_window_ms=300)
    assert r["duration_window_ms"] == 300
    assert r["duration_censored"] == 10
    assert r["duration_ms"] == pytest.approx(300 - 20, abs=4)   # the length it reached


def test_the_duration_window_stops_at_the_end_of_the_motif(monkeypatch):
    trials = _trials(20, 40)
    monkeypatch.setattr(metrics, "daf_trials", lambda ven, enc, st, *, n_trials, seed: trials)
    st = {"daf_window_s": (ONSET / 1000, (ONSET + 50) / 1000), "T_song": ONSET + 200}
    r = daf_response(_Ven(), None, st, n_trials=N_TRIALS, duration_window_ms=300)
    assert r["duration_window_ms"] == 200
