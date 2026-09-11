"""``sven-figure`` -- render the auditory cancellation figure.

Five rows (waveform / spectrogram / auditory neurons / inhibitory interneurons /
excitatory projection neurons) by four stimulus columns (training song / time-reversed
motif / white noise / distorted auditory feedback).

The result is in the excitatory row: near-silent on the trained song, active on the other
three. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

``--view rates`` renders the supplementary population-rate view instead (3 columns,
Auditory over Error, both on fixed 0-90 Hz axes). It does not supersede the raster figure.

The stage-1 encoder is the Olshausen-Field sparse coder. The Smith-Lewicki alternative was
removed from this entry point because nothing in this package produces its dictionary
cache -- that training code stayed in finchsim.

Requires the ``plots`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ..constants import AUD_DELAY_MS, HVC_DELAY_MS, KERNEL_WIDTH_MS, PEAK_RATE_HZ, SR
from ..corpus import load_corpus
from ..olshausen_field import OlshausenFieldEncoder
from ..paths import motifs_npz, of_encoder_npz, output_dir, ven_model_npz
from ..vocal_error_net import VocalErrorNetV2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="trained VEN .npz")
    p.add_argument("--of-encoder", default=None, help="OF encoder .npz")
    p.add_argument("--motifs", default=None, help="motifs .npz")
    p.add_argument("--out-dir", default=None, help="directory for the rendered figure")
    p.add_argument("--view", default="raster", choices=["raster", "rates"],
                   help="raster (the publication figure) or rates (supplementary)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=SR)
    p.add_argument("--t-post", type=int, default=200)
    p.add_argument("--aud-delay-ms", type=int, default=AUD_DELAY_MS)
    p.add_argument("--hvc-delay-ms", type=int, default=HVC_DELAY_MS)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_dir = args.out_dir or str(output_dir())
    # The original script assumed outputs/ already existed, having been created at
    # import time by whichever training script ran first. Do not rely on that.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading model and data ...")
    print(f"  model: {model_path}")
    enc = OlshausenFieldEncoder.load(args.of_encoder or str(of_encoder_npz()))
    # Pin the seed so the rendered figure is reproducible, not just the training.
    ven = VocalErrorNetV2.load(model_path, seed=args.seed)

    # Template selection lives in corpus.py, so the figure draws the same rendition
    # the network trained on.
    corpus = load_corpus(motifs_path)
    sig_train, train_idx = corpus.template()
    T_song = corpus.T_song
    print(f"T_song={T_song} ms  T_rend={T_song + args.t_post} ms  "
          f"train_motif={train_idx}  rms={float(np.sqrt(np.mean(sig_train**2))):.4f}")

    # Tag the output after the model, so figures and models stay paired.
    tag = re.sub(r".*(?:ven_model|of_ven_model)(.*?)\.npz$", r"\1", model_path) or ""

    common = {
        "out_dir": out_dir,
        "T_post": args.t_post,
        "sr": args.sr,
        "seed": args.seed,
        "n_kernels": enc.n_channels,
        "kernel_width": KERNEL_WIDTH_MS,
        "peak_rate": PEAK_RATE_HZ,
    }
    if args.view == "raster":
        from ..figures.encoding_comparison import make_figure

        make_figure(ven, enc, sig_train, T_song, tag,
                    aud_delay_ms=args.aud_delay_ms,
                    hvc_delay_ms=args.hvc_delay_ms, **common)
    else:
        from ..figures.encoding_rates import make_rate_figure

        make_rate_figure(ven, enc, sig_train, T_song, tag, **common)


if __name__ == "__main__":  # pragma: no cover
    main()
