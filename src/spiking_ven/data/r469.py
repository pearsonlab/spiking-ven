"""Extract per-motif audio from the Koch 2024 R469 recordings.

Replicates the logic of ``segment_song.py`` from pearsonlab/vocal-error-network: find
every complete ``iabcde`` syllable run in the Evsonganaly ``.not.mat`` annotations and
concatenate the corresponding audio clips.

Produces ``R469_concat.npy`` (concatenated motif audio at the native sample rate) and
``R469_ann.npz`` (per-motif durations and syllable onsets/offsets). Note the spiking
pipeline's ``motifs.npz`` does **not** come from these files -- :mod:`spiking_ven.data.motifs`
works straight from the WAV + ``.not.mat`` pairs and does its own resampling and DTW
alignment. These outputs exist for parity with the paper's preprocessing; nothing in this
package reads them, so building them is opt-in via ``--with-concat``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio

# Full motif: introductory note plus five syllables.
CODE = "iabcde"


def main(argv: list[str] | None = None) -> None:
    # soundfile lives in the `data` extra, so import it where it is used rather than at
    # module import time. That keeps this module importable in a core (numpy+scipy)
    # install and matches the convention in spiking_ven.cochleagram.load_wav.
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", default="data/song_wavs",
                        help="directory of WAV + .not.mat pairs")
    parser.add_argument("--out", default="data/R469_concat.npy",
                        help="output .npy of concatenated motif audio")
    parser.add_argument("--ann-out", default="data/R469_ann.npz",
                        help="output .npz of per-motif annotations")
    args = parser.parse_args(argv)

    not_files = sorted(glob.glob(os.path.join(args.wav_dir, "*.not.mat")))
    if not not_files:
        raise FileNotFoundError(
            f"No .not.mat files found in {args.wav_dir}. "
            "Run download_data.sh first to fetch the Koch 2024 R469 data."
        )

    print(f"Found {len(not_files)} annotation files in {args.wav_dir}")

    clips     = []   # raw audio arrays at native SR
    song_Ts   = []   # motif duration in ms
    syl_on    = []   # (6,) syllable onsets relative to motif start, ms
    syl_off   = []   # (6,) syllable offsets relative to motif start, ms
    sr_check  = None

    for ann_path in not_files:
        wav_path = ann_path.replace(".not.mat", "")
        if not os.path.exists(wav_path):
            print(f"  SKIP (no WAV): {os.path.basename(ann_path)}")
            continue

        ann = sio.loadmat(ann_path)
        labels  = str(ann["labels"][0])
        onsets  = ann["onsets"].flatten()    # ms
        offsets = ann["offsets"].flatten()   # ms

        starts = [m.start() for m in re.finditer(CODE, labels)]
        if not starts:
            continue

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono
        if sr_check is None:
            sr_check = sr
        elif sr != sr_check:
            raise ValueError(f"Sample rate mismatch: {wav_path} is {sr} Hz, expected {sr_check} Hz")

        n = len(CODE)
        for i in starts:
            t0_ms = onsets[i : i + n]
            t1_ms = offsets[i : i + n]

            # Start at onset of first song syllable ('a'), not intro note ('i').
            # Intro notes are ~60 ms long with ~138 ms lead before 'a'; they are
            # variable and not correlated with HVC premotor timing.
            song_start = t0_ms[1]   # onset of 'a'
            song_end   = t1_ms[-1]  # offset of 'e'

            i0 = int(round(song_start / 1000 * sr))
            i1 = int(round(song_end   / 1000 * sr))
            clip = audio[i0:i1]

            clips.append(clip)
            song_Ts.append(float(song_end - song_start))
            syl_on.append(t0_ms[1:] - song_start)   # syllables a–e, relative to 'a' onset
            syl_off.append(t1_ms[1:] - song_start)

        print(f"  {os.path.basename(wav_path)}: {len(starts)} motif(s)")

    if not clips:
        raise RuntimeError("No motifs found. Check that .not.mat files are present and labelled correctly.")

    print(f"\nExtracted {len(clips)} motifs")
    durations = np.array(song_Ts)
    print(f"Durations: min={durations.min():.0f}  max={durations.max():.0f}  mean={durations.mean():.0f} ms")
    print(f"Sample rate: {sr_check} Hz")

    concat = np.concatenate(clips).astype(np.float32)
    print(f"Concatenated array: {len(concat)} samples = {len(concat)/sr_check:.2f} s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, concat)
    print(f"Saved → {args.out}")

    np.savez(args.ann_out,
        song_Ts=np.array(song_Ts),
        syl_on=np.array(syl_on),
        syl_off=np.array(syl_off),
        fs=sr_check,
    )
    print(f"Saved → {args.ann_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
