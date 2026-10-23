"""Shared helpers for the vocal error network.

Spike/rate conversion, the activation-to-spike stage, synthetic HVC premotor input, and
sparse log-normal weight initialisation.
"""

import numpy as np
from numpy.random import default_rng
from scipy.signal import lfilter
from scipy.sparse import random as sparse_random

from .constants import AUD_MEAN_RATE_HZ


def spike_to_rate(
    spikes: np.ndarray,
    tau_s: float = 10.0,
    dt: float = 1.0,
) -> np.ndarray:
    """Causal exponential filter applied row-by-row.

    ``h[t] = decay * h[t-1] + spikes[t]``, a one-pole IIR filter, evaluated with
    ``lfilter`` rather than a Python loop over timesteps.

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


def apply_refractory(spikes, t_refr_ms: float = 2.0):
    """Enforce an absolute refractory period on a generated spike train.

    For the Poisson/Bernoulli input populations, which are sampled per 1 ms bin with no
    history and so can emit spikes a neuron could not produce. Two separate violations
    are fixed here: a bin carrying more than one spike (``acts_to_spikes`` returns Poisson
    counts), and spikes closer together than ``t_refr_ms``.

    The E and I populations do NOT use this: their refractory belongs in the membrane
    dynamics, where it also gates integration, not in a post-hoc filter.

    Thinning lowers the realised rate slightly, by construction -- it removes spikes and
    adds none. The rate calibration is deliberately not re-run afterwards, which would put
    the removed spikes back elsewhere.
    """
    s = (np.asarray(spikes) > 0).astype(np.float32)     # at most one spike per bin
    r = int(round(t_refr_ms))
    if r <= 1:
        return s
    for i in range(s.shape[0]):
        idx = np.flatnonzero(s[i])
        last = -(10 ** 9)
        for j in idx:
            if j - last < r:
                s[i, j] = 0.0
            else:
                last = j
    return s


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
    t_refr_ms: float = 2.0,
) -> np.ndarray:
    """Generate Poisson spike trains for a parametric HVC population.

    Each of the n_hvc neurons bursts once per rendition at a linearly-spaced
    time within the motif window, with a Gaussian rate envelope, following
    ``generate_HVC`` and ``generate_syl_time`` in ``train_funcs.py`` of the rate-model
    repository (pearsonlab/vocal-error-network).

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

    The ``peak_rate``/``kernel_width`` defaults here are NOT the calibrated operating
    point. Pass ``constants.PEAK_RATE_HZ`` and
    ``constants.KERNEL_WIDTH_MS`` (as ``stimuli.build_stimuli`` and the figures do) to
    get the drive the shipped model was trained with.

    Returns
    -------
    spikes : (n_hvc, T) bool array
    """
    rng = default_rng(seed)
    T_rend = T_song + T_post
    # Bursts are placed at absolute times up to T_burn + n_renditions * T_rend; a shorter
    # T would silently drop the tail renditions.
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
    return apply_refractory(spikes, t_refr_ms)


def generate_hvc_intro(
    n_hvc: int,
    T_song: int,
    n_notes: int,
    note_ms: int,
    gap_ms: int,
    peak_rate: float = 150.0,
    kernel_width: float = 20.0,
    seed: int | None = None,
    t_refr_ms: float = 2.0,
) -> np.ndarray:
    """HVC spikes for ``n_notes`` introductory notes: ``(n_hvc, n_notes * (note_ms + gap_ms))``.

    Each note is a sparse HVC sequence with the motif's burst density: ``round(n_hvc *
    note_ms / T_song)`` neurons, spread evenly over the population, burst at evenly spaced
    times within the note (Gaussian envelopes, as in :func:`generate_hvc_spikes`). The same
    neurons and times repeat every note; each note is followed by ``gap_ms`` of silence.
    Pass ``constants.PEAK_RATE_HZ`` and ``KERNEL_WIDTH_MS`` for the operating point.
    """
    period = note_ms + gap_ms
    out = np.zeros((n_hvc, n_notes * period), dtype=bool)
    if n_notes <= 0 or note_ms <= 0:
        return out
    k = max(1, int(round(n_hvc * note_ms / T_song)))
    rows = np.unique(np.linspace(0, n_hvc - 1, k).round().astype(int))
    note = generate_hvc_spikes(n_hvc=len(rows), T=period, n_renditions=1, T_song=note_ms,
                               T_burn=0, T_post=gap_ms, peak_rate=peak_rate,
                               kernel_width=kernel_width, seed=seed, t_refr_ms=t_refr_ms)
    for r in range(n_notes):
        out[rows, r * period:(r + 1) * period] = note > 0
    return out


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


def acts_to_spikes(
    activations: np.ndarray,
    mean_rate_hz: float = AUD_MEAN_RATE_HZ,
    frame_rate: int = 1000,
    seed: int | None = None,
    calibrate_on: np.ndarray | None = None,
    t_refr_ms: float = 2.0,
) -> np.ndarray:
    """Convert encoder activations to Poisson spike trains.

    The activations are rectified and scaled by one population-wide factor so the mean
    rate is ``mean_rate_hz``.

    Parameters
    ----------
    activations  : (n_channels, T) float, signed ISTA coefficients
    mean_rate_hz : target mean firing rate (Hz) over the whole population
    frame_rate   : temporal resolution of activations in Hz
    calibrate_on : activations to derive the rate scale from, instead of
                   ``activations`` itself. Use this when ``activations`` is a
                   sparse timeline -- e.g. one song tiled into motif windows with
                   silence between them. Calibrating on the whole timeline would
                   divide by a mean diluted by the silence and inflate the rate
                   during song; pass the song-window activations here so the
                   target rate means what it says.
    Returns
    -------
    spikes : (n_channels, T) float32, at most one spike per bin when ``t_refr_ms > 0``.
    """
    rng = np.random.default_rng(seed)
    act = np.maximum(activations.astype(np.float64), 0.0)
    ref = act if calibrate_on is None else np.maximum(calibrate_on.astype(np.float64), 0.0)

    rate_scale = mean_rate_hz / max(float(ref.mean()), 1e-12)
    lam = act * rate_scale * (1.0 / frame_rate)
    counts = rng.poisson(lam)
    if t_refr_ms <= 0.0:
        # Raw Poisson counts, multi-spike bins included. Not routed through
        # apply_refractory, which would still collapse each bin to one spike.
        return counts.astype(np.float32)
    # A bin can carry several spikes; apply_refractory collapses that to at most one and
    # enforces the refractory floor.
    return apply_refractory(counts, t_refr_ms)
