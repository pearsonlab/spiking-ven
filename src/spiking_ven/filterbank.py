"""ERB-spaced gammatone filterbank (vendored).

Turns a waveform into a cochleagram-style time-frequency energy array using
Patterson-Holdsworth gammatone filters on an ERB frequency scale. This is stage 0 of
the encoder: :mod:`spiking_ven.cochleagram` calls :func:`gammatone_spectrogram`, whose
output is then patch-extracted and sparse-coded by
:mod:`spiking_ven.olshausen_field`.

Provenance and licensing
------------------------
The filterbank maths below is **vendored verbatim** from the ``Gammatone`` package,
version 1.0.3 as published on PyPI (project home
<https://github.com/Lightning-Sandbox/gammatone>, a fork of
<https://github.com/detly/gammatone>, which was archived 2024-09-10). It is a Python port
of MATLAB implementations by Malcolm Slaney and Dan Ellis.

It is vendored rather than taken as a runtime dependency so that this package installs
from numpy and scipy alone: upstream is unmaintained, and only four of its functions are
used. The code is copied byte-for-byte, not retyped, so the numerics are identical to the
released package the published results were produced with. Note that 1.0.3 adds the
``f_max`` argument to :func:`centre_freqs` / :func:`gtgram`, which earlier upstream
revisions lack and which this project relies on.

Vendored under the 3-clause BSD license:

    Copyright (c) 1998, Malcolm Slaney <malcolm@interval.com>
    Copyright (c) 2009, Dan Ellis <dpwe@ee.columbia.edu>
    Copyright (c) 2014, Jason Heeris <jason.heeris@gmail.com>

The full license text is reproduced in ``licenses/gammatone-COPYING.txt`` and
``THIRD_PARTY_NOTICES.md``. Neither the names of the copyright holders nor of their
contributors are used to endorse or promote this package.

Modifications made by spiking-ven
---------------------------------
* ``filters.py`` and ``gtgram.py`` are merged into this single module, and the module is
  named ``filterbank`` rather than ``gammatone`` so it cannot shadow the upstream
  distribution name.
* ``gtgram.py``'s ``from .filters import ...`` is dropped (same module now).
* Upstream's unused entry points -- ``fftweight``, ``plot``, ``__main__`` -- are omitted.
* :func:`centre_frequencies` and :func:`gammatone_spectrogram` at the end of this file are
  **not** upstream code; they are this project's wrappers.
"""

from __future__ import division

import numpy as np
from scipy import signal as sgn

# ---------------------------------------------------------------------------
# Vendored verbatim from Gammatone 1.0.3 -- gammatone/filters.py
# ---------------------------------------------------------------------------

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 / 4


def erb_point(low_freq, high_freq, fraction):
    """
    Calculates a single point on an ERB scale between ``low_freq`` and
    ``high_freq``, determined by ``fraction``. When ``fraction`` is ``1``,
    ``low_freq`` will be returned. When ``fraction`` is ``0``, ``high_freq``
    will be returned.

    ``fraction`` can actually be outside the range ``[0, 1]``, which in general
    isn't very meaningful, but might be useful when ``fraction`` is rounded a
    little above or below ``[0, 1]`` (eg. for plot axis labels).
    """
    # Change the following three parameters if you wish to use a different ERB
    # scale. Must change in MakeERBCoeffs too.
    # TODO: Factor these parameters out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = -ear_q * min_bw + np.exp(
        fraction * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
    ) * (high_freq + ear_q * min_bw)

    return erb_point


def erb_space(low_freq=DEFAULT_LOW_FREQ, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM):
    """
    This function computes an array of ``num`` frequencies uniformly spaced
    between ``high_freq`` and ``low_freq`` on an ERB scale.

    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
    """
    return erb_point(low_freq, high_freq, np.arange(1, num + 1) / num)


def centre_freqs(fs, num_freqs, cutoff, f_max=None):
    """
    Calculates an array of centre frequencies (for :func:`make_erb_filters`)
    from a sampling frequency, lower cutoff frequency and the desired number of
    filters.

    :param fs: sampling rate
    :param num_freqs: number of centre frequencies to calculate
    :type num_freqs: int
    :param cutoff: lower cutoff frequency
    :param f_max: upper cutoff frequency (default ``fs / 2``)
    :return: same as :func:`erb_space`
    """
    if f_max is None:
        f_max = fs / 2
    return erb_space(cutoff, f_max, num_freqs)


def make_erb_filters(fs, centre_freqs, width=1.0):
    """
    This function computes the filter coefficients for a bank of
    Gammatone filters. These filters were defined by Patterson and Holdworth for
    simulating the cochlea.

    The result is returned as a :class:`ERBCoeffArray`. Each row of the
    filter arrays contains the coefficients for four second order filters. The
    transfer function for these four filters share the same denominator (poles)
    but have different numerators (zeros). All of these coefficients are
    assembled into one vector that the ERBFilterBank can take apart to implement
    the filter.

    The filter bank contains "numChannels" channels that extend from
    half the sampling rate (fs) to "lowFreq". Alternatively, if the numChannels
    input argument is a vector, then the values of this vector are taken to be
    the center frequency of each desired filter. (The lowFreq argument is
    ignored in this case.)

    Note this implementation fixes a problem in the original code by
    computing four separate second order filters. This avoids a big problem with
    round off errors in cases of very small cfs (100Hz) and large sample rates
    (44kHz). The problem is caused by roundoff error when a number of poles are
    combined, all very close to the unit circle. Small errors in the eigth order
    coefficient, are multiplied when the eigth root is taken to give the pole
    location. These small errors lead to poles outside the unit circle and
    instability. Thanks to Julius Smith for leading me to the proper
    explanation.

    Execute the following code to evaluate the frequency response of a 10
    channel filterbank::

        fcoefs = MakeERBFilters(16000,10,100);
        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
        resp = 20*log10(abs(fft(y')));
        freqScale = (0:511)/512*16000;
        semilogx(freqScale(1:255),resp(1:255,:));
        axis([100 16000 -60 0])
        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');

    | Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    | (c) 1998 Interval Research Corporation
    |
    | (c) 2012 Jason Heeris (Python implementation)
    """
    T = 1 / fs
    # Change the followFreqing three parameters if you wish to use a different
    # ERB scale. Must change in ERBSpace too.
    # TODO: factor these out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erb = width * ((centre_freqs / ear_q) ** order + min_bw**order) ** (1 / order)
    B = 1.019 * 2 * np.pi * erb

    arg = 2 * centre_freqs * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    rt_pos = np.sqrt(3 + 2**1.5)
    rt_neg = np.sqrt(3 - 2**1.5)

    common = -T * np.exp(-(B * T))

    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = np.exp(1j * arg - B * T)

    gain = np.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * np.exp(B * T) / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))) ** 4
    )

    allfilts = np.ones_like(centre_freqs)

    fcoefs = np.column_stack([A0 * allfilts, A11, A12, A13, A14, A2 * allfilts, B0 * allfilts, B1, B2, gain])

    return fcoefs


def erb_filterbank(wave, coefs):
    """
    :param wave: input data (one dimensional sequence)
    :param coefs: gammatone filter coefficients

    Process an input waveform with a gammatone filter bank. This function takes
    a single sound vector, and returns an array of filter outputs, one channel
    per row.

    The fcoefs parameter, which completely specifies the Gammatone filterbank,
    should be designed with the :func:`make_erb_filters` function.

    | Malcolm Slaney @ Interval, June 11, 1998.
    | (c) 1998 Interval Research Corporation
    | Thanks to Alain de Cheveigne' for his suggestions and improvements.
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    output = np.zeros((coefs[:, 9].shape[0], wave.shape[0]))

    gain = coefs[:, 9]
    # A0, A11, A2
    As1 = coefs[:, (0, 1, 5)]
    # A0, A12, A2
    As2 = coefs[:, (0, 2, 5)]
    # A0, A13, A2
    As3 = coefs[:, (0, 3, 5)]
    # A0, A14, A2
    As4 = coefs[:, (0, 4, 5)]
    # B0, B1, B2
    Bs = coefs[:, 6:9]

    # Loop over channels
    for idx in range(0, coefs.shape[0]):
        # These seem to be reversed (in the sense of A/B order), but that's what
        # the original code did...
        # Replacing these with polynomial multiplications reduces both accuracy
        # and speed.
        y1 = sgn.lfilter(As1[idx], Bs[idx], wave)
        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
        output[idx, :] = y4 / gain[idx]

    return output

# ---------------------------------------------------------------------------
# Vendored verbatim from Gammatone 1.0.3 -- gammatone/gtgram.py
# (the upstream `from .filters import ...` is unnecessary here)
# ---------------------------------------------------------------------------

def round_half_away_from_zero(num):
    """Implement the round-half-away-from-zero rule, where fractional parts of
    0.5 result in rounding up to the nearest positive integer for positive
    numbers, and down to the nearest negative number for negative integers.
    """
    return np.sign(num) * np.floor(np.abs(num) + 0.5)


def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    """
    Calculates the window size for a gammatonegram.

    @return a tuple of (window_size, hop_samples, output_columns)
    """
    nwin = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns = 1 + int(np.floor((filterbank_cols - nwin) / hop_samples))

    return (nwin, hop_samples, columns)


def gtgram_xe(wave, fs, channels, f_min, f_max):
    """Calculate the intermediate ERB filterbank processed matrix"""
    cfs = centre_freqs(fs, channels, f_min, f_max)
    fcoefs = np.flipud(make_erb_filters(fs, cfs))
    xf = erb_filterbank(wave, fcoefs)
    xe = np.power(xf, 2)
    return xe


def gtgram(wave, fs, window_time, hop_time, channels, f_min, f_max=None):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters. The waveform ``wave`` (at sample rate ``fs``) is
    passed through a multi-channel gammatone auditory model filterbank, with
    lowest frequency ``f_min`` and highest frequency ``f_max``. The outputs of
    each band then have their energy integrated over windows of ``window_time``
    seconds, advancing by ``hop_time`` secs for successive columns. These
    magnitudes are returned as a nonnegative real matrix with ``channels`` rows.

    | 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    xe = gtgram_xe(wave, fs, channels, f_min, f_max)

    nwin, hop_samples, ncols = gtgram_strides(fs, window_time, hop_time, xe.shape[1])
    channels, ncols = int(channels), int(ncols)  # typing for some compatibility reasons
    y = np.zeros((channels, ncols))

    for cnum in range(ncols):
        segment = xe[:, cnum * hop_samples + np.arange(nwin)]
        y[:, cnum] = np.sqrt(segment.mean(1))

    return y


# ---------------------------------------------------------------------------
# spiking-ven wrappers (not upstream code)
# ---------------------------------------------------------------------------


def centre_frequencies(sr: int, n_channels: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Return n_channels ERB-spaced centre frequencies between lo_hz and hi_hz."""
    return centre_freqs(sr, n_channels, lo_hz, f_max=hi_hz)


def gammatone_spectrogram(signal: np.ndarray, sr: int, n_channels: int = 64,
                          frame_rate: int = 1000, lo_hz: float = 100.0,
                          hi_hz: float = 8000.0) -> np.ndarray:
    """Gammatone spectrogram via energy integration.

    Returns (n_channels, n_frames) array of per-band energy, non-negative.
    window_time = 2 / frame_rate gives a mild overlap; hop_time = 1 / frame_rate.
    """
    hop_time = 1.0 / frame_rate
    window_time = 2.0 * hop_time
    gram = gtgram(signal, sr, window_time, hop_time, n_channels, lo_hz, hi_hz)
    return gram  # shape: (n_channels, n_frames)
