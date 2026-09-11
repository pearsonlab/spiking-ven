"""Segment and DTW-align R469 song motifs into ``motifs.npz``.

Scans every WAV + Evsonganaly ``.not.mat`` pair in ``--wav_dir``, finds each complete
``abcded`` motif run, resamples to 16 kHz, DTW-aligns all renditions to a common template
(the highest-RMS rendition), and writes the aligned corpus.

This is the single upstream artifact for everything else: the sparse encoder trains on it
and the vocal error network is trained and evaluated against it.

Output keys: ``audio``, ``lengths``, ``sr``, ``syl_on``, ``syl_off``, ``song_Ts``,
``raw_song_Ts``.
"""

from __future__ import annotations

import argparse
import glob
import os
from math import gcd

import numpy as np
import scipy.io
import soundfile as sf
from scipy.signal import resample_poly
from scipy.signal import spectrogram as sp_spectrogram
from scipy.spatial.distance import cdist

from ..paths import ensure_parent

MOTIF_PATTERN = "abcded"
TARGET_SR = 16000
HOP_MS = 10  # DTW feature frame hop in ms



# ---------------------------------------------------------------------------
# Helper: extract spectral features for DTW (log mel-filterbank energies)
# ---------------------------------------------------------------------------

def _mel_filterbank(sr, n_fft, n_mel=40, f_min=300.0, f_max=8000.0):
    """Triangular mel filterbank matrix, shape (n_mel, n_fft//2+1)."""
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freq  = n_fft // 2 + 1
    fb = np.zeros((n_mel, n_freq))
    for m in range(1, n_mel + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            if mid > lo:
                fb[m - 1, k] = (k - lo) / (mid - lo)
        for k in range(mid, hi):
            if hi > mid:
                fb[m - 1, k] = (hi - k) / (hi - mid)
    return fb


def compute_features(audio, sr, hop_ms):
    """Log mel-filterbank spectrogram for DTW, shape (n_mel, T_frames)."""
    hop    = max(1, int(sr * hop_ms / 1000))
    win    = max(512, 4 * hop)
    f, _t, Sxx = sp_spectrogram(audio, fs=sr, nperseg=win,
                                 noverlap=win - hop, window="hann")
    fb  = _mel_filterbank(sr, win, n_mel=40)
    fb  = fb[:, : Sxx.shape[0]]  # trim to actual freq bins
    mel = fb @ Sxx               # (n_mel, T)
    return np.log(mel + 1e-10)   # (n_mel, T)


# ---------------------------------------------------------------------------
# Helper: DTW alignment
# ---------------------------------------------------------------------------

def dtw_warp(ref_feat, query_feat):
    """
    DTW-align query_feat to ref_feat.

    Returns warp: 1-D array of length T_ref, where warp[i] is the query frame
    index that aligns with reference frame i.
    """
    T_r = ref_feat.shape[1]
    T_q = query_feat.shape[1]

    C = cdist(ref_feat.T, query_feat.T, metric="cosine")  # (T_r, T_q)

    # Accumulated cost with sentinels
    D = np.full((T_r + 1, T_q + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(T_r):
        for j in range(T_q):
            D[i + 1, j + 1] = C[i, j] + min(D[i, j], D[i, j + 1], D[i + 1, j])

    # Backtrack
    i, j = T_r, T_q
    path_i, path_j = [], []
    while i > 0 or j > 0:
        path_i.append(i - 1)
        path_j.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    path_i = np.array(path_i[::-1])  # ref frames  (monotone non-decreasing)
    path_j = np.array(path_j[::-1])  # query frames

    # Build warp[i] = mean query frame aligned to ref frame i
    warp = np.empty(T_r)
    for ir in range(T_r):
        mask = path_i == ir
        warp[ir] = path_j[mask].mean() if mask.any() else (
            warp[ir - 1] if ir > 0 else 0.0)
    return warp


def apply_audio_warp(query_audio, ref_n_samples, warp, sr, hop_ms):
    """
    Time-stretch query_audio to ref_n_samples using the DTW warp function.

    warp[i] = query frame corresponding to reference frame i.
    """
    hop  = max(1, int(sr * hop_ms / 1000))
    T_r  = len(warp)

    # For each output sample, compute the corresponding query sample position
    out_idx  = np.arange(ref_n_samples, dtype=np.float64)
    ref_frame = np.clip(out_idx / hop, 0, T_r - 1)
    ref_frames_int = np.arange(T_r, dtype=np.float64)
    query_frame = np.interp(ref_frame, ref_frames_int, warp)
    query_sample = np.clip(query_frame * hop, 0, len(query_audio) - 1)

    warped = np.interp(query_sample,
                       np.arange(len(query_audio), dtype=np.float64),
                       query_audio.astype(np.float64))
    return warped.astype(np.float32)


def warp_time_ms(t_ms_query, warp, hop_ms):
    """
    Convert a query-relative time (ms) to the corresponding reference time (ms)
    using the inverse DTW warp.

    warp[i_ref] = j_query  →  inverse: j_query → i_ref
    """
    T_r = len(warp)
    ref_frames = np.arange(T_r, dtype=np.float64)
    # warp is non-decreasing; unique points for safe interpolation
    _, uidx = np.unique(warp, return_index=True)
    j_query = t_ms_query / hop_ms
    i_ref   = np.interp(j_query, warp[uidx], ref_frames[uidx])
    return float(np.clip(i_ref * hop_ms, 0, (T_r - 1) * hop_ms))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav_dir", default="data/song_wavs")
    parser.add_argument("--out", default="outputs/motifs.npz")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="target sample rate")
    parser.add_argument("--hop_ms", type=float, default=HOP_MS,
                        help="DTW feature frame hop (ms)")
    args = parser.parse_args(argv)
    # The original script did os.makedirs("outputs") at import time, which silently
    # depended on the process working directory. Create the real output parent instead.
    ensure_parent(args.out)

    # ---------------------------------------------------------------------------
    # Step 1: parse WAV + .not.mat files, extract raw motifs at target SR
    # ---------------------------------------------------------------------------

    wav_paths = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    print(f"Found {len(wav_paths)} WAV files in {args.wav_dir}")

    raw_motifs = []   # list of dicts: audio, syl_on_rel, syl_off_rel, src

    for wav_path in wav_paths:
        mat_path = wav_path + ".not.mat"
        if not os.path.exists(mat_path):
            continue

        mat = scipy.io.loadmat(mat_path)
        labels  = str(mat["labels"][0])            # e.g. 'iiiiabcdedfiabcded'
        onsets  = mat["onsets"].ravel()            # ms, absolute from WAV start
        offsets = mat["offsets"].ravel()           # ms
        sr_wav  = int(mat["Fs"].ravel()[0])        # 44100

        audio_wav, sr_check = sf.read(wav_path)
        assert sr_check == sr_wav, f"SR mismatch in {wav_path}"
        if audio_wav.ndim > 1:
            audio_wav = audio_wav[:, 0]            # mono
        audio_wav = audio_wav.astype(np.float64)

        # Find all 'abcded' runs
        pat = MOTIF_PATTERN
        n_pat = len(pat)
        for pos in range(len(labels) - n_pat + 1):
            if labels[pos : pos + n_pat] != pat:
                continue

            syl_on_abs  = onsets[pos : pos + n_pat]   # ms, wav-absolute
            syl_off_abs = offsets[pos : pos + n_pat]

            motif_on_ms  = syl_on_abs[0]
            motif_off_ms = syl_off_abs[-1]

            # Extract samples
            on_samp  = int(round(motif_on_ms  * sr_wav / 1000))
            off_samp = int(round(motif_off_ms * sr_wav / 1000))
            clip = audio_wav[on_samp:off_samp]

            # Resample to target SR
            g  = gcd(args.sr, sr_wav)
            up, dn = args.sr // g, sr_wav // g
            clip_r = resample_poly(clip, up, dn).astype(np.float32)

            # Syllable times relative to motif onset, in ms
            syl_on_rel  = syl_on_abs  - motif_on_ms
            syl_off_rel = syl_off_abs - motif_on_ms

            raw_motifs.append(dict(
                audio=clip_r,
                syl_on=syl_on_rel,
                syl_off=syl_off_rel,
                src=os.path.basename(wav_path),
                raw_dur_ms=motif_off_ms - motif_on_ms,
            ))

    n_motifs = len(raw_motifs)
    print(f"Extracted {n_motifs} complete '{MOTIF_PATTERN}' renditions")
    raw_durs = np.array([m["raw_dur_ms"] for m in raw_motifs])
    print(f"  raw duration: mean={raw_durs.mean():.1f}  std={raw_durs.std():.1f}  "
          f"min={raw_durs.min():.1f}  max={raw_durs.max():.1f} ms")

    # ---------------------------------------------------------------------------
    # Step 2: choose template (highest RMS)
    # ---------------------------------------------------------------------------

    rms_all   = np.array([float(np.sqrt(np.mean(m["audio"] ** 2))) for m in raw_motifs])
    train_idx = int(np.argmax(rms_all))
    tpl_audio = raw_motifs[train_idx]["audio"]
    tpl_n     = len(tpl_audio)
    tpl_ms    = tpl_n / args.sr * 1000

    print(f"Template: rendition {train_idx}  ({raw_motifs[train_idx]['src']})"
          f"  dur={tpl_ms:.1f} ms  RMS={rms_all[train_idx]:.4f}")

    tpl_feat = compute_features(tpl_audio.astype(np.float64), args.sr, args.hop_ms)

    # ---------------------------------------------------------------------------
    # Step 3: DTW-align each motif to the template
    # ---------------------------------------------------------------------------

    audio_out  = np.zeros((n_motifs, tpl_n), dtype=np.float32)
    syl_on_out = np.zeros((n_motifs, len(MOTIF_PATTERN)), dtype=np.float64)
    syl_off_out= np.zeros((n_motifs, len(MOTIF_PATTERN)), dtype=np.float64)
    song_Ts_out= np.full(n_motifs, tpl_ms)

    for idx, mot in enumerate(raw_motifs):
        q_audio = mot["audio"].astype(np.float64)

        if idx == train_idx:
            # Template maps to itself
            audio_out[idx] = tpl_audio
            syl_on_out[idx]  = mot["syl_on"]
            syl_off_out[idx] = mot["syl_off"]
        else:
            q_feat = compute_features(q_audio, args.sr, args.hop_ms)
            warp   = dtw_warp(tpl_feat, q_feat)
            audio_out[idx] = apply_audio_warp(q_audio, tpl_n, warp, args.sr, args.hop_ms)
            syl_on_out[idx]  = [warp_time_ms(t, warp, args.hop_ms)
                                for t in mot["syl_on"]]
            syl_off_out[idx] = [warp_time_ms(t, warp, args.hop_ms)
                                for t in mot["syl_off"]]

        if (idx + 1) % 5 == 0 or idx == n_motifs - 1:
            print(f"  DTW aligned {idx+1}/{n_motifs}", flush=True)

    # ---------------------------------------------------------------------------
    # Step 4: Save
    # ---------------------------------------------------------------------------

    lengths = np.full(n_motifs, tpl_n, dtype=np.int32)

    np.savez_compressed(
        args.out,
        audio=audio_out,
        lengths=lengths,
        sr=args.sr,
        syl_on=syl_on_out,
        syl_off=syl_off_out,
        song_Ts=song_Ts_out,
        raw_song_Ts=raw_durs,
    )
    print(f"\nSaved {args.out}")
    print(f"  shape: {audio_out.shape}  SR: {args.sr} Hz  T_template: {tpl_n} samples ({tpl_ms:.1f} ms)")
    print(f"  template index: {train_idx}")
    print(f"  syl_on  mean per syllable: {syl_on_out.mean(axis=0).round(1)}")
    print(f"  syl_off mean per syllable: {syl_off_out.mean(axis=0).round(1)}")


if __name__ == "__main__":  # pragma: no cover
    main()
