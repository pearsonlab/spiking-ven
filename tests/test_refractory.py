"""Absolute refractory period, on every spiking population.

The E and I membranes enforce the period in their dynamics; the auditory and HVC input
populations are thinned after sampling (``apply_refractory``), which also collapses
Poisson counts to at most one spike per bin.
"""

import numpy as np
import pytest

from spiking_ven.common import apply_refractory, generate_hvc_spikes
from spiking_ven.common import acts_to_spikes
from spiking_ven.vocal_error_net import VocalErrorNetV2


def _min_isi(S):
    isis = [np.diff(np.flatnonzero(r)) for r in np.asarray(S) if (np.asarray(r) > 0).sum() > 1]
    return int(np.concatenate(isis).min()) if isis else None


@pytest.mark.parametrize("refr", [2.0, 3.0, 5.0])
def test_membrane_refractory_sets_the_isi_floor(refr):
    """Under saturating drive a cell fires exactly every ``refr`` ms -- not refr + 1."""
    v = VocalErrorNetV2(n_e=20, n_i=10, n_aud=4, n_hvc=4, seed=0, t_refr_ms=refr,
                        noise_e=0.0, noise_i=0.0)
    v.theta_e = np.full(v.n_e, 1e-6)                      # fire whenever permitted
    sE, _ = v.transform_all(np.zeros((4, 150), dtype=np.float32),
                            np.ones((4, 150), dtype=np.float32))
    assert _min_isi(sE) == int(refr)
    assert sE.max() <= 1.0


def test_membrane_refractory_off_is_unbounded():
    """t_refr_ms=0 disables the period, so models saved without one reload as trained."""
    v = VocalErrorNetV2(n_e=20, n_i=10, n_aud=4, n_hvc=4, seed=0, t_refr_ms=0.0,
                        noise_e=0.0, noise_i=0.0)
    v.theta_e = np.full(v.n_e, 1e-6)
    sE, _ = v.transform_all(np.zeros((4, 150), dtype=np.float32),
                            np.ones((4, 150), dtype=np.float32))
    assert _min_isi(sE) == 1


def test_apply_refractory_collapses_multi_spike_bins():
    counts = np.array([[0, 5, 0, 0, 3, 0, 0, 0]], dtype=float)
    out = apply_refractory(counts, 2.0)
    assert out.max() == 1.0
    assert out.tolist() == [[0, 1, 0, 0, 1, 0, 0, 0]]


def test_apply_refractory_drops_close_spikes_and_keeps_the_first():
    s = np.array([[1, 1, 1, 0, 1, 0, 1, 0]], dtype=float)
    # Spike at 0 kept. 1 is 1 ms later -> dropped. 2 is exactly 2 ms later -> KEPT: a 2 ms
    # refractory permits an ISI of 2, it does not require 3. 4 and 6 follow at 2 ms each.
    assert apply_refractory(s, 2.0).tolist() == [[1, 0, 1, 0, 1, 0, 1, 0]]
    assert apply_refractory(s, 1.0).tolist() == s.tolist()   # r<=1 only clips the bins


def test_input_populations_respect_the_floor():
    rng = np.random.default_rng(0)
    acts = np.abs(rng.standard_normal((12, 400))) ** 3      # bursty, like the real code
    spk = acts_to_spikes(acts, mean_rate_hz=40.0, frame_rate=1000, seed=1, t_refr_ms=2.0)
    assert spk.max() <= 1.0 and _min_isi(spk) >= 2

    hvc = generate_hvc_spikes(20, 600, 1, 400, 0, T_post=200, seed=3, t_refr_ms=2.0)
    assert np.asarray(hvc).max() <= 1.0 and _min_isi(hvc) >= 2
