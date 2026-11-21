import numpy as np

from spiking_ven.metrics import first_peak


def _bump(t, centre, width, height):
    return height * np.exp(-0.5 * ((t - centre) / width) ** 2)


def test_first_peak_prefers_the_first_prominent_hump():
    t = np.arange(150.0)
    x = _bump(t, 20, 4, 10) + _bump(t, 80, 6, 24)          # smaller first, larger second
    assert abs(first_peak(x) - 20) <= 1


def test_first_peak_ignores_small_early_bumps():
    t = np.arange(150.0)
    x = _bump(t, 10, 2, 2) + _bump(t, 60, 6, 20)           # 10% bump, then the response
    assert abs(first_peak(x) - 60) <= 1


def test_first_peak_single_hump_is_its_maximum():
    t = np.arange(150.0)
    x = _bump(t, 45, 8, 15)
    assert first_peak(x) == int(np.argmax(x))


def test_first_peak_no_positive_response():
    x = -np.ones(20)
    assert first_peak(x) == 0
