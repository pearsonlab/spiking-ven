"""The seed must govern the simulation, not just weight initialisation.

The vocal error network injects Gaussian membrane noise every timestep. That noise was
originally drawn from the global ``np.random``, so ``seed=`` fixed only the initial
weights and two runs with identical arguments diverged (~1% in the E rate). These tests
pin the fix: the seed now controls the simulation too.
"""

import numpy as np

from spiking_ven import VocalErrorNetV2, generate_hvc_spikes


def _tiny_net(seed: int) -> VocalErrorNetV2:
    return VocalErrorNetV2(n_e=30, n_i=8, n_hvc=6, n_aud=10, seed=seed)


def _tiny_inputs(n_aud: int = 10, n_hvc: int = 6, T_song: int = 120, T_post: int = 30):
    T = T_song + T_post
    rng = np.random.default_rng(0)
    aud = (rng.random((n_aud, T)) < 0.02).astype(np.float32)
    hvc = generate_hvc_spikes(n_hvc=n_hvc, T=T, n_renditions=1, T_song=T_song,
                              T_burn=0, T_post=T_post, seed=1)
    return hvc, aud, T_song, T_post


def test_noise_is_nonzero_by_default():
    """Guard the premise: if noise_e were 0 these tests would pass trivially."""
    assert _tiny_net(0).noise_e > 0.0


def test_same_seed_gives_identical_training():
    hvc, aud, T_song, T_post = _tiny_inputs()
    out = []
    for _ in range(2):
        ven = _tiny_net(42)
        ven.fit(hvc, aud, n_renditions=1, T_song=T_song, T_burn=0, T_post=T_post,
                verbose=False)
        out.append(ven.JIE.copy())
    np.testing.assert_array_equal(out[0], out[1])


def test_same_seed_gives_identical_inference():
    hvc, aud, _, _ = _tiny_inputs()
    rates = []
    for _ in range(2):
        ven = _tiny_net(7)
        rates.append(float(ven.transform(hvc, aud).mean()))
    assert rates[0] == rates[1]


def test_different_seeds_diverge():
    """Sanity: the seed must actually reach the noise, not be ignored entirely."""
    hvc, aud, _, _ = _tiny_inputs()
    a = float(_tiny_net(1).transform(hvc, aud).mean())
    b = float(_tiny_net(2).transform(hvc, aud).mean())
    assert a != b


def test_loaded_model_can_be_pinned(tmp_path):
    """A saved model reloaded with a seed must also evaluate reproducibly."""
    hvc, aud, _, _ = _tiny_inputs()
    path = tmp_path / "ven.npz"
    _tiny_net(3).save(str(path))
    r1 = float(VocalErrorNetV2.load(str(path), seed=99).transform(hvc, aud).mean())
    r2 = float(VocalErrorNetV2.load(str(path), seed=99).transform(hvc, aud).mean())
    assert r1 == r2
