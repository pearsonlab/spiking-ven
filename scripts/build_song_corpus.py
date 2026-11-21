"""Build an unwarped multi-bird song corpus for training the auditory encoder.

    python scripts/build_song_corpus.py --src DIR --out outputs/song_corpus_koch.npz \\
        [--exclude-condition FP1_KD] [--max-sec-per-bird 20]

Reads each bird's ``<bird>.zip`` in ``--src`` (Koch et al. 2024 layout: WAVs with
Evsonganaly ``.wav.not.mat`` annotations, plus ``Bird_list.csv``) directly from the
archives, without extracting them. Each recording is cut into bouts: labeled syllables
less than ``--gap-ms`` apart, padded by ``--pad-ms``. Bouts are resampled to ``--sr`` and
kept unwarped. Each bird contributes at most ``--max-sec-per-bird`` seconds (bouts in a
seeded random order), so no bird dominates. ``--val-frac`` of the bouts are marked
held-out, for monitoring training.

Writes ``audio`` (all bouts concatenated, float32), ``starts``/``lengths`` (samples),
``bird`` (index into ``birds``), ``val`` (held-out mask) and ``sr``.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.wavfile
from scipy.signal import resample_poly


def bird_conditions(src: Path) -> dict[str, str]:
    with open(src / "Bird_list.csv", encoding="utf-8-sig") as fh:
        return {r["Bird_ID"].strip(): r["Condition"].strip() for r in csv.DictReader(fh)}


def bouts_from_zip(path: Path, *, gap_ms: float, pad_ms: float, sr_out: int):
    """Yield resampled bouts (float32) from every annotated WAV in the archive."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for wav in sorted(n for n in names if n.endswith(".wav")):
            ann = wav + ".not.mat"
            if ann not in names:
                continue
            mat = scipy.io.loadmat(io.BytesIO(zf.read(ann)))
            on = np.asarray(mat["onsets"], dtype=float).ravel()
            off = np.asarray(mat["offsets"], dtype=float).ravel()
            if on.size == 0:
                continue
            sr, x = scipy.io.wavfile.read(io.BytesIO(zf.read(wav)))
            x = x.astype(np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            order = np.argsort(on)
            on, off = on[order], off[order]
            # Group syllables into bouts separated by gaps of at least gap_ms.
            starts = [0] + [i for i in range(1, len(on)) if on[i] - off[i - 1] >= gap_ms]
            ends = starts[1:] + [len(on)]
            g = gcd(int(sr), int(sr_out))
            for s, e in zip(starts, ends):
                a = max(int((on[s] - pad_ms) * sr / 1000), 0)
                b = min(int((off[e - 1] + pad_ms) * sr / 1000), len(x))
                if b - a < sr * 0.05:
                    continue
                y = resample_poly(x[a:b], sr_out // g, int(sr) // g)
                yield y.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-condition", nargs="*", default=["FP1_KD"],
                   help="drop birds whose Condition contains any of these")
    p.add_argument("--max-sec-per-bird", type=float, default=20.0)
    p.add_argument("--gap-ms", type=float, default=200.0)
    p.add_argument("--pad-ms", type=float, default=20.0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    src = Path(a.src)
    cond = bird_conditions(src)
    birds = sorted(b for b, c in cond.items()
                   if not any(x in c for x in a.exclude_condition)
                   and (src / f"{b}.zip").exists())
    rng = np.random.default_rng(a.seed)
    audio, starts, lengths, bird_idx = [], [], [], []
    pos = 0
    for bi, b in enumerate(birds):
        bouts = list(bouts_from_zip(src / f"{b}.zip", gap_ms=a.gap_ms, pad_ms=a.pad_ms,
                                    sr_out=a.sr))
        kept, total = 0, 0.0
        for k in rng.permutation(len(bouts)):
            y = bouts[k]
            if total + len(y) / a.sr > a.max_sec_per_bird:
                continue
            audio.append(y)
            starts.append(pos)
            lengths.append(len(y))
            bird_idx.append(bi)
            pos += len(y)
            total += len(y) / a.sr
            kept += 1
        print(f"{b:6s} {cond[b]:10s} bouts {len(bouts):4d}, kept {kept:3d} "
              f"({total:5.1f} s)", flush=True)
    n = len(lengths)
    val = np.zeros(n, dtype=bool)
    val[rng.choice(n, max(1, int(round(a.val_frac * n))), replace=False)] = True
    np.savez(a.out, audio=np.concatenate(audio), starts=np.array(starts),
             lengths=np.array(lengths), bird=np.array(bird_idx), birds=np.array(birds),
             conditions=np.array([cond[b] for b in birds]), val=val, sr=a.sr)
    print(f"{len(birds)} birds, {n} bouts ({val.sum()} held out), "
          f"{pos / a.sr / 60:.1f} min of audio -> {a.out}")


if __name__ == "__main__":
    main()
