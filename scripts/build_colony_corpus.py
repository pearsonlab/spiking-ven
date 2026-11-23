"""Build a calibrated, level-preserving corpus of zebra finch nestbox audio for training
the auditory encoder.

    python scripts/build_colony_corpus.py \\
        --src ~/code/finchsim/data/moseley_meliza_data \\
        --out outputs/colony_corpus_cr.npz [--setup colony|soundbox] [--ref-spl-db 80]

Source: Moseley & Meliza, "Acoustical statistics of zebra finch breeding environments"
(figshare doi:10.6084/m9.figshare.27961518, CC BY 4.0; preprint doi:10.1101/2024.06.25.600670).
``--setup colony`` (default) takes the colony-reared (CR) nestbox of sire cobalt 57 in the
indoor breeding colony; ``--setup soundbox`` takes the pair-reared (PR) nestbox of purple
166 in an acoustic isolation box. Each setup is a set of unscaled 30 s excerpts (48 kHz)
plus one longer ``example_`` excerpt, with a CSV of calibrated A-weighted dB SPL quantiles
for 10-min segments of the full recording.

Nothing is cut into bouts or selected for song: the corpus is everything the nestbox
microphone heard (song, calls, begging, cage and room noise).

**Level.** Every file gets ONE gain, so relative levels within and across excerpts are
kept (quiet stretches stay quiet; per-segment RMS normalisation would raise room noise to
song level). The excerpts are unscaled cuts from one recording chain, so a single offset
maps their dBFS to dB SPL. It is fitted here: for each excerpt the median A-weighted 10 ms
frame level (dBFS) is compared with the CSV median (dBA SPL) of the 10-min segment(s)
that contain it; the offset is the median difference across excerpts. (A-weighting is
0 dB at 1 kHz, the calibration-tone frequency, so the offset is a plain gain.) The audio
is then scaled so that a sound at ``--ref-spl-db`` (unweighted RMS) has RMS ``--rms-ref``,
the level the motifs are normalised to before the cochleagram (``CAUSAL_RMS_REF``); i.e.
``--ref-spl-db`` is the SPL the network's motifs are taken to have. The default, 80 dB SPL,
puts the loudest colony events (99.9th-percentile 10 ms frame) about at the motifs' loud
(90th-percentile) frames.

**No splicing.** Each source file is its own segment: it is DC-removed and resampled on
its own, and nothing is joined to anything else. (The ``audio`` array below is storage
only; the trainer slices each segment back out by ``starts``/``lengths`` and builds its
cochleagram separately, so no window spans two files.) The excerpts were cut from the
long recordings at arbitrary samples, so each one starts and ends mid-waveform -- a click
to the gammatone filters -- so the ends get a ``--fade-ms`` raised-cosine fade. Files longer
than ``--max-segment-sec`` (30 s: only the 300 s example) are split into pieces faded the
same way, to bound the trainer's per-segment memory. Whole source files are held out for
validation (``--val-frac``).

Writes the ``build_song_corpus.py`` keys -- ``audio`` (float32), ``starts``/``lengths``
(samples), ``bird``, ``birds``, ``conditions``, ``val``, ``sr`` -- plus ``prescaled``
(True: ``train_sw_encoder.py --train-corpus`` uses the level as is, without per-segment
RMS normalisation), ``gain_db``, ``dbfs_to_spl_db``, ``ref_spl_db``, ``rms_ref``,
``source`` (file per segment) and ``setup``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import bilinear_zpk, resample_poly, sosfilt, zpk2sos

from spiking_ven.constants import CAUSAL_RMS_REF

SETUPS = {
    # setup: (file stem, amplitude CSV, bird label, condition)
    "colony": ("cobalt_57_family-colony-nestbox", "amplitude_cobalt-57_colony.csv",
               "cobalt_57", "CR (indoor colony)"),
    "soundbox": ("purple_166_family-pair-nestbox", "amplitude_purple-166_soundbox.csv",
                 "purple_166", "PR (acoustic isolation box)"),
}
CSV_SEGMENT_S = 600.0   # each CSV row summarises a 10-min segment starting at its timestamp
FRAME_MS = 10.0         # level frames, as in the dataset's amplitude statistics


def a_weighting_sos(sr: int) -> np.ndarray:
    """IEC 61672 A-weighting filter (bilinear transform), 0 dB at 1 kHz."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    z = [0.0] * 4
    p = [-2 * np.pi * f for f in (f1, f1, f2, f3, f4, f4)]
    k = (2 * np.pi * f4) ** 2 * 10 ** (1.9997 / 20)
    zd, pd, kd = bilinear_zpk(z, p, k, sr)
    return zpk2sos(zd, pd, kd)


def frame_levels_db(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> np.ndarray:
    """Mean-square level (dB re full scale) of Hann-windowed frames overlapping by 50%."""
    n = int(round(sr * frame_ms / 1000))
    hop = n // 2
    if len(x) < n:
        raise ValueError("signal shorter than one frame")
    w = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    ms = (frames ** 2 * w ** 2).sum(axis=1) / (w ** 2).sum()
    return 10 * np.log10(ms + 1e-20)


def fade(x: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Copy of ``x`` with raised-cosine ramps of ``fade_ms`` at both ends."""
    n = min(int(round(sr * fade_ms / 1000)), len(x) // 2)
    y = np.array(x, dtype=np.float64)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return y


def timestamp(path: Path) -> dt.datetime:
    m = re.search(r"(\d{8}-\d{6})\.wav$", path.name)
    if m is None:
        raise ValueError(f"no timestamp in {path.name}")
    return dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")


def load_amplitude_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(segment start times as POSIX seconds, median dBA SPL) from an amplitude CSV."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([dt.datetime.strptime(f"{r['date']} {r['time'][:8]}",
                                       "%Y-%m-%d %H:%M:%S").timestamp() for r in rows])
    return t, np.array([float(r["ampl_q50"]) for r in rows])


def fit_dbfs_to_spl(files, sigs, sr: int, csv_t, csv_q50) -> tuple[float, int]:
    """Median over excerpts of (CSV median dBA SPL - excerpt median A-weighted dBFS).

    Only excerpts inside a CSV segment are used. Returns ``(offset_db, n_matched)``.
    """
    sos = a_weighting_sos(sr)
    diffs = []
    for f, x in zip(files, sigs):
        t0 = timestamp(f).timestamp()
        d = t0 - csv_t
        sel = (d >= 0) & (d < CSV_SEGMENT_S)
        if not sel.any():
            continue
        diffs.append(csv_q50[sel].mean() - np.median(frame_levels_db(sosfilt(sos, x), sr)))
    if not diffs:
        raise SystemExit("no excerpt falls inside the amplitude CSV's segments; "
                         "pass --dbfs-to-spl-db explicitly")
    return float(np.median(diffs)), len(diffs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="moseley_meliza_data directory")
    p.add_argument("--out", required=True)
    p.add_argument("--setup", choices=sorted(SETUPS), default="colony",
                   help="colony = CR nestbox (cobalt 57); soundbox = PR nestbox (purple 166)")
    p.add_argument("--ref-spl-db", type=float, default=80.0,
                   help="SPL (dB, unweighted RMS) mapped to RMS --rms-ref, the motifs' level")
    p.add_argument("--rms-ref", type=float, default=CAUSAL_RMS_REF)
    p.add_argument("--dbfs-to-spl-db", type=float, default=None,
                   help="skip the fit and use this dBFS -> dB SPL offset")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--max-segment-sec", type=float, default=30.0,
                   help="split longer files into faded pieces of this length (0: never). "
                        "30 s matches the excerpts and splits only the 300 s example; one "
                        "300 s segment took the trainer to 11 GB peak RSS vs 2.9 GB")
    p.add_argument("--fade-ms", type=float, default=10.0,
                   help="raised-cosine fade at each segment's start and end")
    p.add_argument("--no-example", action="store_true",
                   help="leave out the long example_ excerpt")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of source files held out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    src = Path(a.src).expanduser()
    stem, csv_name, bird, condition = SETUPS[a.setup]
    files = sorted(src.glob(f"{stem}_*.wav"))
    if not a.no_example:
        files += sorted(src.glob(f"example_{stem}_*.wav"))
    if not files:
        raise SystemExit(f"no {stem} WAVs in {src}")

    sigs, sr_in = [], None
    for f in files:
        x, sr = sf.read(f, always_2d=True, dtype="float64")
        if sr_in is None:
            sr_in = sr
        elif sr != sr_in:
            raise ValueError(f"{f.name}: {sr} Hz, others {sr_in} Hz")
        x = x.mean(axis=1)
        sigs.append(x - x.mean())
    assert sr_in is not None
    total_s = sum(len(x) for x in sigs) / sr_in
    print(f"{a.setup}: {len(files)} files, {total_s / 60:.1f} min at {sr_in} Hz")

    if a.dbfs_to_spl_db is None:
        csv_t, csv_q50 = load_amplitude_csv(src / csv_name)
        offset, n_match = fit_dbfs_to_spl(files, sigs, sr_in, csv_t, csv_q50)
        print(f"dBFS -> dB SPL offset {offset:.1f} dB (fitted on {n_match} excerpts)")
    else:
        offset = a.dbfs_to_spl_db
        print(f"dBFS -> dB SPL offset {offset:.1f} dB (given)")
    gain_db = offset - a.ref_spl_db + 20 * np.log10(a.rms_ref)
    gain = 10 ** (gain_db / 20)
    print(f"gain {gain_db:+.1f} dB: {a.ref_spl_db:.0f} dB SPL -> RMS {a.rms_ref}")

    rng = np.random.default_rng(a.seed)
    n_val = max(1, int(round(a.val_frac * len(files)))) if a.val_frac > 0 else 0
    val_files = set(rng.choice(len(files), n_val, replace=False).tolist())

    g = gcd(a.sr, sr_in)
    min_n = a.sr  # drop split tails shorter than 1 s
    audio, starts, lengths, val, source = [], [], [], [], []
    pos = 0
    for fi, (f, x) in enumerate(zip(files, sigs)):
        y = resample_poly(x, a.sr // g, sr_in // g) * gain
        seg_n = len(y) if not a.max_segment_sec else int(round(a.max_segment_sec * a.sr))
        for s0 in range(0, len(y), seg_n):
            seg = fade(y[s0:s0 + seg_n], a.sr, a.fade_ms).astype(np.float32)
            if len(seg) < min_n and s0 > 0:
                continue
            audio.append(seg)
            starts.append(pos)
            lengths.append(len(seg))
            val.append(fi in val_files)
            source.append(f.name)
            pos += len(seg)

    out_audio = np.concatenate(audio)  # storage only: segments are read back separately
    lv = np.concatenate([frame_levels_db(s.astype(np.float64), a.sr) for s in audio])
    ref_db = 20 * np.log10(a.rms_ref)
    pct = np.percentile(lv, [10, 50, 90, 99, 99.9])
    print("10 ms frame level re the reference (dB), pct 10/50/90/99/99.9: "
          + " ".join(f"{v - ref_db:+.1f}" for v in pct))
    print(f"peak |x| {np.abs(out_audio).max():.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, audio=out_audio, starts=np.array(starts), lengths=np.array(lengths),
             bird=np.zeros(len(lengths), dtype=int), birds=np.array([bird]),
             conditions=np.array([condition]), val=np.array(val), sr=a.sr,
             prescaled=True, gain_db=gain_db, dbfs_to_spl_db=offset,
             ref_spl_db=a.ref_spl_db, rms_ref=a.rms_ref, source=np.array(source),
             setup=a.setup)
    print(f"{len(lengths)} segments ({sum(val)} held out, from {n_val} files), "
          f"{pos / a.sr / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
