"""``sven-prep-data`` -- acquire the song corpus and build ``motifs.npz``.

Three stages, each skipped if its output already exists:

1. download the R469 WAV + ``.not.mat`` pairs (Koch 2024)
2. optionally build ``R469_concat.npy`` / ``R469_ann.npz`` (concatenated corpus)
3. resample + DTW-align into ``outputs/motifs.npz``

If the download is blocked (a TLS-inspecting proxy, or no network), place the WAV +
``.not.mat`` pairs in ``<data-dir>/song_wavs`` yourself and pass ``--skip-download``;
nothing downstream touches the network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import data_dir, motifs_npz


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="raw song data directory (default: $SVEN_DATA_DIR or ./data)")
    parser.add_argument("--out", default=None,
                        help="output motifs .npz (default: $SVEN_OUTPUT_DIR/motifs.npz)")
    parser.add_argument("--skip-download", action="store_true",
                        help="assume <data-dir>/song_wavs is already populated")
    parser.add_argument("--with-concat", action="store_true",
                        help="also build R469_concat.npy / R469_ann.npz "
                             "(concatenated corpus; nothing in this package reads it)")
    parser.add_argument("--with-duke-features", action="store_true",
                        help="also fetch the optional Duke feature archive (rate-model only)")
    args = parser.parse_args(argv)

    ddir = Path(args.data_dir) if args.data_dir else data_dir()
    out = Path(args.out) if args.out else motifs_npz()
    wav_dir = ddir / "song_wavs"

    print(f"data dir: {ddir}")
    print(f"output:   {out}")

    # --- 1. corpus ---------------------------------------------------------
    print("\nStep 1/3 - song corpus")
    if args.skip_download:
        if not (wav_dir.is_dir() and any(wav_dir.glob("*.wav"))):
            raise SystemExit(f"--skip-download given but no WAVs found in {wav_dir}")
        print(f"  skipping download; using {wav_dir}")
    else:
        from ..data.download import fetch_optional_duke_features, fetch_r469

        fetch_r469(ddir)
        if args.with_duke_features:
            fetch_optional_duke_features(ddir)

    # --- 2. optional concatenated corpus ----------------------------------
    print("\nStep 2/3 - concatenated corpus (optional)")
    if args.with_concat:
        from ..data.r469 import main as build_concat

        build_concat([
            "--wav-dir", str(wav_dir),
            "--out", str(ddir / "R469_concat.npy"),
            "--ann-out", str(ddir / "R469_ann.npz"),
        ])
    else:
        print("  skipped (pass --with-concat for the concatenated corpus)")

    # --- 3. aligned motifs -------------------------------------------------
    print("\nStep 3/3 - resample + DTW align")
    if out.exists():
        print(f"  {out} already present, skipping")
    else:
        from ..data.motifs import main as build_motifs

        build_motifs(["--wav_dir", str(wav_dir), "--out", str(out)])

    print(f"\nDone. {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
