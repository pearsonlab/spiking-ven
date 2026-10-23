"""Stimulus construction: song, reversed song and DAF, from waveform to spike trains.

Every stimulus is built by :func:`build_stimuli`, normalised against one reference (the
song) at both the waveform and the spike-rate stage, so relative level -- the DAF noise is
5.6x the song's RMS -- reaches the network. The trainer, the CLIs, the tests and the
figures all use it, so they cannot drift apart.

A trained model records the settings its stimuli were built with (``ven.stimulus``, from
:func:`stimulus_record`), and :func:`resolve_stimulus_settings` reads them back so later
evaluation rebuilds the same stimuli.
"""

from __future__ import annotations

import sys

import numpy as np

from .common import acts_to_spikes, generate_hvc_spikes
from .constants import (
    AUD_MEAN_RATE_HZ,
    DAF_WINDOW_S,
    DAF_WN_AMPLITUDE,
    KERNEL_WIDTH_MS,
    N_HVC,
    N_ISTA,
    PEAK_RATE_HZ,
    T_BURN,
    T_POST,
)
from .corpus import load_corpus

__all__ = [
    "daf_waveform",
    "encode_acts",
    "encode_stimulus",
    "build_stimuli",
    "daf_window_ms",
    "stimulus_record",
    "resolve_stimulus_settings",
    "STIMULUS_DEFAULTS",
    "DAF_WN_AMPLITUDE",
]

#: Fallbacks for the settings :func:`resolve_stimulus_settings` resolves, used when the
#: model records none. ``sr=None`` means "the corpus's own rate".
STIMULUS_DEFAULTS = {
    "sr": None,
    "t_post": T_POST,
    "t_burn": T_BURN,
    "mean_rate_hz": AUD_MEAN_RATE_HZ,
    "n_ista": N_ISTA,
    "daf_window_s": tuple(DAF_WINDOW_S),
}


def daf_window_ms(window_s) -> tuple[int, float]:
    """``(onset_ms, burst_ms)`` of a DAF window given in seconds.

    The onset is an index into 1 ms spike arrays, so it is rounded to an integer.
    """
    w0, w1 = (float(x) for x in window_s)
    return int(round(w0 * 1000)), float((w1 - w0) * 1000)


def daf_waveform(sig_train, sr: int, *, seed: int = 42,
                 amplitude: float = DAF_WN_AMPLITUDE,
                 window_s: tuple[float, float] = DAF_WINDOW_S):
    """The DAF stimulus: the song, with white noise mixed into one window.

    This is the paradigm the biological numbers come from: the bird sings, and noise is
    played back during singing. Returns the waveform; the caller encodes it.
    """
    sig = np.asarray(sig_train, dtype=np.float64)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    noise = np.random.default_rng(seed).standard_normal(len(sig)) * rms * amplitude
    s0, s1 = int(window_s[0] * sr), int(window_s[1] * sr)
    if s0 >= len(sig):
        raise ValueError(
            f"the DAF window {window_s} s starts past the end of a "
            f"{len(sig) / sr:.3f} s motif, so the DAF stimulus would be identical to "
            "the song. Pass a window inside the motif (daf_window_s)."
        )
    out = sig.copy()
    out[s0:s1] += noise[s0:s1]
    return out


def encode_acts(sig, encoder, *, sr, n_ista, T_out_ms, rms_from):
    """Waveform -> (n_channels, T) activations at 1 ms resolution.

    Every activation goes through here, so the metrics, the responder analysis and the
    figures cannot normalise differently from one another.
    """
    sig = np.asarray(sig, dtype=np.float64)
    return encoder.encode_signal(sig, sr=sr, n_ista=n_ista, upsample_to_ms=True,
                                 T_out_ms=T_out_ms, rms_from=rms_from)


def encode_stimulus(sig, encoder, *, sr, T_out_ms, acts_train, rms_from,
                    mean_rate_hz: float = AUD_MEAN_RATE_HZ,
                    n_ista: int = N_ISTA, seed: int = 0):
    """Waveform to auditory spike train, on the shared reference.

    Both normalisation stages take their reference from the song: the waveform scaling
    from ``rms_from``, the spike rate from ``acts_train``.
    """
    acts = encode_acts(sig, encoder, sr=sr, n_ista=n_ista, T_out_ms=T_out_ms,
                       rms_from=rms_from)
    return acts_to_spikes(acts, mean_rate_hz=mean_rate_hz, frame_rate=1000, seed=seed,
                          calibrate_on=acts_train).astype(np.float32)


def build_stimuli(
    encoder,
    motifs_path,
    *,
    sr: int | None = None,
    seed: int = 42,
    t_post: int = T_POST,
    t_burn: int = T_BURN,
    n_hvc: int = N_HVC,
    mean_rate_hz: float = AUD_MEAN_RATE_HZ,
    n_ista: int = N_ISTA,
    daf_window_s: tuple[float, float] = DAF_WINDOW_S,
    verbose: bool = True,
) -> dict:
    """Build every stimulus the trainer and the metrics need.

    ``sr`` defaults to the corpus's sample rate; a different explicit value is an error,
    since the waveforms would be read at the wrong rate.

    Returns a dict with the waveforms (``sig_train``, ``sig_reversed``, ``sig_daf``),
    their auditory spike arrays (``aud_correct``, ``aud_reversed``, ``aud_daf``, plus the
    full ``correct_pool``), the HVC inputs (``hvc_on``, ``hvc_off``, ``hvc_burn`` and its
    padded ``aud_burn``), the bookkeeping the caller needs (``T_song``, ``T_rend``,
    ``train_idx``, ``fwd_rev_corr``, ``n_kernels``), and every setting it was built with
    (``sr``, ``seed``, ``t_post``, ``t_burn``, ``n_hvc``, ``mean_rate_hz``, ``n_ista``,
    ``daf_window_s``) so later encodes and analyses match.
    """
    corpus = load_corpus(motifs_path)
    if sr is None:
        sr = int(corpus.sr)
    elif int(sr) != int(corpus.sr):
        raise ValueError(
            f"sr={sr} Hz does not match the corpus's {corpus.sr} Hz ({motifs_path}); "
            "omit sr to use the corpus rate"
        )
    daf_window_s = (float(daf_window_s[0]), float(daf_window_s[1]))
    n_motifs = corpus.n_motifs
    T_song = corpus.T_song
    T_rend = T_song + t_post
    n_kernels = encoder.n_channels

    sig_train, train_idx = corpus.template()
    rms_train = float(np.sqrt(np.mean(sig_train**2)))

    # The encoder path normalises TWICE, and both have to share one reference (the song)
    # or the level difference between stimuli does not reach the network:
    #   encode_signal scales the waveform    -> rms_from=sig_train
    #   acts_to_spikes scales the spike rate -> calibrate_on=acts_train
    acts_train = encode_acts(sig_train, encoder, sr=sr, n_ista=n_ista,
                             T_out_ms=T_rend, rms_from=sig_train)

    def sig_to_aud(sig, seed_offset: int):
        return encode_stimulus(sig, encoder, sr=sr, T_out_ms=T_rend, acts_train=acts_train,
                               rms_from=sig_train, mean_rate_hz=mean_rate_hz,
                               n_ista=n_ista, seed=seed + seed_offset)

    if verbose:
        print(f"T_song={T_song} ms  T_rend={T_rend} ms  train_motif={train_idx}  "
              f"rms={rms_train:.4f}")
        print(f"Building correct training pool ({n_motifs} motifs)...")

    aud_correct = sig_to_aud(sig_train, 10)
    correct_pool = [aud_correct]                 # index 0 is the template motif
    for ci in range(n_motifs):
        if ci == train_idx:
            continue
        correct_pool.append(sig_to_aud(corpus.signal(ci), 100 + ci))

    sig_rev = sig_train[::-1].astype(np.float64)
    aud_reversed = sig_to_aud(sig_rev, 200)

    # How separable forward and reversed song are in the encoder bounds how much more
    # the reversed song can drive E than the trained one. acts_train is the forward
    # encoding already, so only the reversed one is new.
    acts_fwd = acts_train
    acts_rev = encode_acts(sig_rev, encoder, sr=sr, n_ista=n_ista,
                           T_out_ms=T_rend, rms_from=sig_train)
    fv, rv = acts_fwd.ravel(), acts_rev.ravel()
    fwd_rev_corr = (float(np.corrcoef(fv, rv)[0, 1])
                    if fv.std() > 0 and rv.std() > 0 else float("nan"))
    del acts_rev, fv, rv

    # Song with noise mixed into one window. The figure draws these same arrays.
    sig_daf = daf_waveform(sig_train, sr, seed=seed + 2, window_s=daf_window_s)
    aud_daf = sig_to_aud(sig_daf, 20)

    if verbose:
        for label, arr in (("training motif", aud_correct), ("reversed motif", aud_reversed),
                           (f"DAF WN ({DAF_WN_AMPLITUDE}x RMS)", aud_daf)):
            print(f"  {label}: mean rate={arr.mean() * 1000:.1f} Hz  "
                  f"active fraction={float((arr > 0).mean()) * 100:.1f}%")
        print(f"  forward vs reversed activation correlation: {fwd_rev_corr:.4f}  "
              f"(lower -> more room for the reversed song to stand out)")
        print("Building HVC inputs...")

    T_total_burn = t_burn + T_rend
    hvc_burn = generate_hvc_spikes(n_hvc=n_hvc, T=T_total_burn, n_renditions=1,
                                   T_song=T_song, T_burn=t_burn, T_post=t_post,
                                   peak_rate=PEAK_RATE_HZ, kernel_width=KERNEL_WIDTH_MS,
                                   seed=seed)
    aud_burn = np.zeros((n_kernels, T_total_burn), dtype=np.float32)
    aud_burn[:, t_burn: t_burn + T_rend] = aud_correct

    hvc_on = generate_hvc_spikes(n_hvc=n_hvc, T=T_rend, n_renditions=1, T_song=T_song,
                                 T_burn=0, T_post=t_post, peak_rate=PEAK_RATE_HZ,
                                 kernel_width=KERNEL_WIDTH_MS, seed=seed)
    hvc_off = np.zeros((n_hvc, T_rend), dtype=np.float32)

    return {
        "aud_correct": aud_correct,
        "aud_reversed": aud_reversed,
        "aud_daf": aud_daf,
        "correct_pool": correct_pool,
        "hvc_on": hvc_on,
        "hvc_off": hvc_off,
        "hvc_burn": hvc_burn,
        "aud_burn": aud_burn,
        "T_song": T_song,
        "T_rend": T_rend,
        "train_idx": train_idx,
        "acts_train": acts_train,
        "sig_train": sig_train,
        "sig_reversed": sig_rev,
        "sig_daf": sig_daf,
        "sr": sr,
        "seed": seed,
        "t_post": t_post,
        "t_burn": t_burn,
        "n_hvc": n_hvc,
        "mean_rate_hz": mean_rate_hz,
        "n_ista": n_ista,
        "daf_window_s": daf_window_s,
        "fwd_rev_corr": fwd_rev_corr,
        "n_kernels": n_kernels,
        "n_motifs": n_motifs,
    }


# ---------------------------------------------------------------------------
# Provenance: what a model records about its stimuli, and reading it back
# ---------------------------------------------------------------------------

def stimulus_record(st: dict, *, encoder_path=None, motifs_path=None) -> dict:
    """The settings ``st`` was built with, plus checksums of its inputs.

    JSON-serialisable, for ``ven.stimulus``. The checksums let a later evaluation notice
    that it was given a different encoder or corpus than the model was trained on.
    """
    from .manifest import sha256_file

    rec = {
        "sr": int(st["sr"]),
        "seed": int(st["seed"]),
        "t_post": int(st["t_post"]),
        "t_burn": int(st["t_burn"]),
        "n_hvc": int(st["n_hvc"]),
        "mean_rate_hz": float(st["mean_rate_hz"]),
        "n_ista": int(st["n_ista"]),
        "daf_window_s": [float(x) for x in st["daf_window_s"]],
    }
    if encoder_path is not None:
        rec["encoder_sha256"] = sha256_file(encoder_path)
    if motifs_path is not None:
        rec["motifs_sha256"] = sha256_file(motifs_path)
    return rec


def _normalise(key, value):
    """Recorded values come back from JSON (a window as a list); compare like with like."""
    if value is None:
        return None
    if key == "daf_window_s":
        return (float(value[0]), float(value[1]))
    return value


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def resolve_stimulus_settings(recorded: dict | None, given: dict, *,
                              encoder_path=None, motifs_path=None) -> dict:
    """Settings to rebuild a model's stimuli with: the flag, else the record, else the default.

    ``recorded`` is ``ven.stimulus`` (possibly empty); ``given`` maps setting names to
    command-line values, ``None`` meaning "not given". An explicit value that disagrees
    with the record is used but warned about, naming both. A recorded checksum that does
    not match ``encoder_path`` or ``motifs_path`` is warned about too: the model was then
    trained on different inputs than it is being evaluated on.

    Names not in :data:`STIMULUS_DEFAULTS` (``seed``, say) fall back to ``None`` when
    neither the flag nor the record has them, so the caller supplies that default.
    """
    from .manifest import sha256_file

    recorded = recorded or {}
    out = {}
    for key, value in given.items():
        rec = _normalise(key, recorded.get(key))
        value = _normalise(key, value)
        if value is None:
            out[key] = rec if rec is not None else STIMULUS_DEFAULTS.get(key)
        else:
            if rec is not None and value != rec:
                _warn(f"{key}={value!r} was given, but the model records {key}={rec!r} "
                      "for its training stimuli; using the given value")
            out[key] = value
    for key, path in (("encoder_sha256", encoder_path), ("motifs_sha256", motifs_path)):
        if path is not None and key in recorded and sha256_file(path) != recorded[key]:
            what = key.split("_")[0]
            _warn(f"{path} is not the {what} file the model was trained with "
                  f"(sha256 differs from the recorded {recorded[key][:12]}...)")
    return out
