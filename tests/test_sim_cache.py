import numpy as np

from spiking_ven.sim_cache import cached_arrays


def test_cached_arrays_hits_and_misses(tmp_path):
    model = tmp_path / "model.npz"
    np.savez(model, x=np.arange(3))
    calls = []

    def compute():
        calls.append(1)
        return {"spikes": (np.random.default_rng(len(calls)).random((4, 5)) > 0.5).astype(float),
                "rate": np.array([1.5, 2.5])}

    a = cached_arrays(model, "t", {"k": 1}, compute, verbose=False)
    b = cached_arrays(model, "t", {"k": 1}, compute, verbose=False)
    assert len(calls) == 1                                  # second call is a hit
    assert a["spikes"].dtype == np.uint8                    # 0/1 arrays stored compactly
    assert a["rate"].dtype == np.float64
    np.testing.assert_array_equal(a["spikes"], b["spikes"])

    cached_arrays(model, "t", {"k": 2}, compute, verbose=False)
    assert len(calls) == 2                                  # different key: recompute

    np.savez(model, x=np.arange(4))                         # model retrained in place
    cached_arrays(model, "t", {"k": 2}, compute, verbose=False)
    assert len(calls) == 3
