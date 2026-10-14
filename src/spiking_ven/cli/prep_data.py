"""``sven-prep-data`` -- acquire the song corpus and build ``motifs.npz``.

Two stages, each skipped if its output already exists:

1. download the R469 WAV + ``.not.mat`` pairs (Koch 2024)
2. resample + DTW-align into ``outputs/motifs.npz``

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
    args = parser.parse_args(argv)

    ddir = Path(args.data_dir) if args.data_dir else data_dir()
    out = Path(args.out) if args.out else motifs_npz()
    wav_dir = ddir / "song_wavs"

    print(f"data dir: {ddir}")
    print(f"output:   {out}")

    # --- 1. corpus ---------------------------------------------------------
    print("\nStep 1/2 - song corpus")
    if args.skip_download:
        if not (wav_dir.is_dir() and any(wav_dir.glob("*.wav"))):
            raise SystemExit(f"--skip-download given but no WAVs found in {wav_dir}")
        print(f"  skipping download; using {wav_dir}")
    else:
        from ..data.download import fetch_r469

        fetch_r469(ddir)

    # --- 2. aligned motifs -------------------------------------------------
    print("\nStep 2/2 - resample + DTW align")
    if out.exists():
        print(f"  {out} already present, skipping")
    else:
        from ..data.motifs import main as build_motifs

        build_motifs(["--wav-dir", str(wav_dir), "--out", str(out)])

    print(f"\nDone. {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
