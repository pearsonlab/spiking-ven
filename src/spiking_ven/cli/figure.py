"""``sven-figure`` -- render the auditory cancellation figure.

Six rows (waveform / spectrogram / auditory neurons / inhibitory interneurons /
excitatory projection neurons / E population rate) by three stimulus columns (training
song / time-reversed motif / distorted auditory feedback).

The result is in the excitatory rows: near-silent on the trained song, active on the
other two. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

``--view psth`` renders the DAF response time course against the Fig. 8E reading instead.

Requires the ``plots`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..causal_filters import CausalFilterEncoder
from ..constants import AUD_MEAN_RATE_HZ, SR, T_POST
from ..evaluate import build_stimuli
from ..paths import encoder_npz, motifs_npz, output_dir, ven_model_npz
from ..vocal_error_net import VocalErrorNetV2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="trained VEN .npz")
    p.add_argument("--encoder", default=None,
                   help="trained causal filter bank .npz (default: outputs/causal_encoder.npz)")
    p.add_argument("--motifs", default=None, help="motifs .npz")
    p.add_argument("--out-dir", default=None, help="directory for the rendered figure")
    p.add_argument("--view", default="raster", choices=["raster", "psth"],
                   help="raster (the publication figure) or "
                        "psth (DAF response time course against the Fig. 8E reading)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=SR)
    p.add_argument("--t-post", type=int, default=T_POST)
    p.add_argument("--mean-rate-hz", type=float, default=AUD_MEAN_RATE_HZ,
                   help="auditory spike-rate calibration; MUST match what the model was "
                        "trained with, or the figure encodes a different stimulus than "
                        "the metrics beside it")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_dir = args.out_dir or str(output_dir())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    enc = CausalFilterEncoder.load(args.encoder or str(encoder_npz()))
    # Pin the seed so the rendered figure is reproducible, not just the training.
    ven = VocalErrorNetV2.load(model_path, seed=args.seed)
    st = build_stimuli(enc, motifs_path, sr=args.sr, seed=args.seed, t_post=args.t_post,
                       mean_rate_hz=args.mean_rate_hz, verbose=False)
    print(f"T_song={st['T_song']} ms  train_motif={st['train_idx']}  "
          f"delays: aud={ven.aud_delay_ms} ms  hvc={ven.hvc_delay_ms} ms")

    # Tag the output after the model, so figures and models stay paired.
    _m = re.search(r"ven_model(.*?)\.npz$", model_path)
    tag = _m.group(1) if _m else f"_{Path(model_path).stem}"

    if args.view == "psth":
        from ..figures.daf_psth import make_psth_figure

        for path in make_psth_figure(ven, enc, st, out_dir, tag=tag, seed=args.seed):
            print(f"  wrote {path}")
    else:
        from ..figures.cancellation import make_figure

        make_figure(ven, st, tag, out_dir=out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
