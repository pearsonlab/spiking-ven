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


def generate_hvc_bursts(
    n_hvc: int,
    T: int,
    n_renditions: int,
    T_song: int,
    T_burn: int,
    T_post: int = 200,
    spikes_per_burst: int = 4,
    intra_burst_hz: float = 500.0,
    seed: int | None = None,
) -> np.ndarray:
    """Stereotyped HVC_RA-style bursts: one fixed burst per neuron per motif.

    Ported from finchsim's ``ra.build_hvcra_spike_trains``. The motif window is cut into
    ``n_hvc`` equal slots; neuron ``i`` starts its burst at a uniformly jittered time in
    slot ``i`` and fires ``spikes_per_burst`` spikes at ``intra_burst_hz``. Nothing is
    random within a burst, and the onsets are drawn once and repeated every rendition:
    HVC_RA bursts are stereotyped to under 1 ms across motifs (Hahnloser et al. 2002).
    Onsets and spikes are rounded down to the 1 ms grid.

    Parameters
    ----------
    n_hvc       : number of HVC neurons
    T           : total time steps (at least T_burn + n_renditions*(T_song+T_post))
    n_renditions: number of song renditions
    T_song      : duration of one motif in ms
    T_burn      : silent burn-in period in ms
    T_post      : silence after each motif in ms

    Returns an ``(n_hvc, T)`` float32 array of 0/1 spikes. It needs no refractory
    thinning as long as the intra-burst interval is at least 2 ms.
    """
    rng = default_rng(seed)
    T_rend = T_song + T_post
    T_needed = T_burn + n_renditions * T_rend
    if T < T_needed:
        raise ValueError(
            f"T={T} is too short for {n_renditions} rendition(s) of {T_song}+{T_post} ms "
            f"after a {T_burn} ms burn-in; need at least {T_needed}."
        )
    if intra_burst_hz <= 0 or spikes_per_burst < 1:
        raise ValueError("need spikes_per_burst >= 1 and intra_burst_hz > 0")
    slot = T_song / n_hvc
    onsets = np.arange(n_hvc) * slot + rng.uniform(0.0, slot, n_hvc)
    onsets = np.clip(onsets, 0.0, np.nextafter(T_song, 0.0))
    offsets = np.floor(np.arange(spikes_per_burst) * 1000.0 / intra_burst_hz)
    rel = np.floor(onsets[:, None] + offsets[None, :]).astype(int)   # (n_hvc, spikes)

    spikes = np.zeros((n_hvc, T), dtype=np.float32)
    rows = np.repeat(np.arange(n_hvc), spikes_per_burst)
    for r in range(n_renditions):
        t = (T_burn + r * T_rend + rel).ravel()
        ok = t < T
        spikes[rows[ok], t[ok]] = 1.0
    return spikes


def generate_hvc_intro(
    n_hvc: int,
    T_song: int,
    n_notes: int,
    note_ms: int,
    gap_ms: int,
    seed: int | None = None,
    **burst_kw,
) -> np.ndarray:
    """HVC spikes for ``n_notes`` introductory notes: ``(n_hvc, n_notes * (note_ms + gap_ms))``.

    Each note is a sparse HVC sequence with the motif's burst density: ``round(n_hvc *
    note_ms / T_song)`` neurons, spread evenly over the population, burst at evenly spaced
    times within the note (:func:`generate_hvc_bursts`; ``burst_kw`` sets the burst shape).
    The same neurons and times repeat every note; each note is followed by ``gap_ms`` of
    silence.
    """
    period = note_ms + gap_ms
    out = np.zeros((n_hvc, n_notes * period), dtype=bool)
    if n_notes <= 0 or note_ms <= 0:
        return out
    k = max(1, int(round(n_hvc * note_ms / T_song)))
    rows = np.unique(np.linspace(0, n_hvc - 1, k).round().astype(int))
    note = generate_hvc_bursts(n_hvc=len(rows), T=period, n_renditions=1, T_song=note_ms,
                               T_burn=0, T_post=gap_ms, seed=seed, **burst_kw)
    for r in range(n_notes):
        out[rows, r * period:(r + 1) * period] = note > 0
    return out


def uniform_scalar(x, name: str = "value") -> float:
    """``x`` as a float, accepting an array only if every element is the same.

    Models saved before ``theta_e`` became a scalar stored it once per E cell, always with
    one value; this reads those back. A non-uniform array raises ``ValueError``.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 or np.any(a != a.flat[0]):
        raise ValueError(f"{name} must be a scalar (or a uniform array), got {a!r}")
    return float(a.flat[0])


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
    norm_sigma: float | None = None,
    norm_power: float | None = None,
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
    norm_sigma   : divisive normalisation across the population: each channel's rectified
                   activation is divided by ``1 + (sum_k a_k(t) / norm_sigma)**norm_power``
                   at each moment,
                   before the rate calibration (applied to ``calibrate_on`` too). Sparse song
                   activity (small population sum) passes almost unchanged; broadband
                   activity that recruits many channels at once is scaled down. ``None``
                   disables it.
    Returns
    -------
    spikes : (n_channels, T) float32, at most one spike per bin when ``t_refr_ms > 0``.
    """
    rng = np.random.default_rng(seed)
    act = np.maximum(activations.astype(np.float64), 0.0)
    ref = act if calibrate_on is None else np.maximum(calibrate_on.astype(np.float64), 0.0)
    if norm_sigma is not None:
        n = 1.0 if norm_power is None else float(norm_power)
        act = act / (1.0 + (act.sum(axis=0, keepdims=True) / norm_sigma) ** n)
        ref = ref / (1.0 + (ref.sum(axis=0, keepdims=True) / norm_sigma) ** n)

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
