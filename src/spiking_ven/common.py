"""Shared helpers for the vocal error network.

Spike/rate conversion, synthetic HVC premotor input, and sparse log-normal weight
initialisation. These were factored out of the original (v1) network module; the v1
``VocalErrorNet`` class itself is not part of this package -- see the note in
:mod:`spiking_ven.vocal_error_net`.
"""

import numpy as np
from numpy.random import default_rng
from scipy.signal import lfilter
from scipy.sparse import random as sparse_random


def spike_to_rate(
    spikes: np.ndarray,
    tau_s: float = 10.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Causal exponential filter applied row-by-row.

    ``h[t] = decay * h[t-1] + spikes[t]``, which is a one-pole IIR filter, so it is
    evaluated with ``lfilter`` rather than a Python loop over timesteps. This runs once
    per population per simulation call -- a few thousand times over a training run --
    and the float32 recursion is bit-identical to the explicit loop it replaces.

    Parameters
    ----------
    spikes : (n_units, T) bool or float array
    tau_s  : filter time constant in ms
    dt     : time step in ms

    Returns
    -------
    h : (n_units, T) float32
    """
    decay = np.float32(np.exp(-dt / tau_s))
    b = np.array([1.0], dtype=np.float32)
    a = np.array([1.0, -decay], dtype=np.float32)
    return lfilter(b, a, spikes.astype(np.float32), axis=1).astype(np.float32)


def generate_hvc_spikes(
    n_hvc: int,
    T: int,
    n_renditions: int,
    T_song: int,
    T_burn: int,
    T_post: int = 200,
    peak_rate: float = 150.0,
    kernel_width: float = 20.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Poisson spike trains for a parametric HVC population.

    Each of the n_hvc neurons bursts once per rendition at a linearly-spaced
    time within the motif window, with a Gaussian rate envelope.  Mirrors
    generate_HVC() + generate_syl_time() in the paper's train_funcs.py.

    Parameters
    ----------
    n_hvc       : number of HVC neurons
    T           : total time steps (must equal T_burn + n_renditions*(T_song+T_post))
    n_renditions: number of song renditions
    T_song      : duration of one motif in ms
    T_burn      : silent burn-in period in ms
    T_post      : silence after each motif in ms
    peak_rate   : peak firing rate in Hz
    kernel_width: Gaussian σ in ms

    The ``peak_rate``/``kernel_width`` defaults here are the function's original values,
    not the calibrated operating point. Pass ``constants.PEAK_RATE_HZ`` and
    ``constants.KERNEL_WIDTH_MS`` (as ``evaluate.build_stimuli`` and the figures do) to
    get the drive the shipped model was trained with.

    Returns
    -------
    spikes : (n_hvc, T) bool array
    """
    rng = default_rng(seed)
    T_rend = T_song + T_post
    # Bursts are placed at absolute times up to T_burn + n_renditions * T_rend. If T is
    # shorter, the tail renditions land outside the array and are silently dropped --
    # the population simply stops driving, with nothing to indicate why.
    T_needed = T_burn + n_renditions * T_rend
    if T < T_needed:
        raise ValueError(
            f"T={T} is too short for {n_renditions} rendition(s) of {T_song}+{T_post} ms "
            f"after a {T_burn} ms burn-in; need at least {T_needed}."
        )
    ts = np.arange(T, dtype=np.float64)
    rates = np.zeros((n_hvc, T), dtype=np.float64)

    for r in range(n_renditions):
        t_start = T_burn + r * T_rend
        burst_centers = np.linspace(t_start, t_start + T_song, num=n_hvc, endpoint=False)
        for i in range(n_hvc):
            rates[i] += peak_rate * np.exp(
                -(ts - burst_centers[i]) ** 2 / (2 * kernel_width ** 2)
            )

    prob = rates * 1e-3
    spikes = rng.random((n_hvc, T)) < prob
    return spikes


def _lognormal_sparse(shape, c, mean, std, rng):
    """Sparse log-normal weight matrix with connection probability c."""
    sigma2 = np.log(1 + (std / mean) ** 2)
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)

    n_rows, n_cols = shape
    data_rvs = lambda n: rng.lognormal(mu, sigma, size=n)  # noqa: E731
    W = sparse_random(n_rows, n_cols, density=c, format="csr",
                      data_rvs=data_rvs, random_state=rng.integers(2**31))
    return W.toarray().astype(np.float32)
