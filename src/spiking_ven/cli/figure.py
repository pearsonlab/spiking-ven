"""``sven-figure`` -- render the auditory cancellation figure.

Five rows (waveform / spectrogram / auditory neurons / inhibitory interneurons /
excitatory projection neurons) by four stimulus columns (training song / time-reversed
motif / white noise / distorted auditory feedback).

The result is in the excitatory row: near-silent on the trained song, active on the other
three. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

``--view rates`` renders the supplementary population-rate view instead (3 columns,
Auditory over Error, both on fixed 0-90 Hz axes). It does not supersede the raster figure.

Requires the ``plots`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ..constants import AUD_MEAN_RATE_HZ, KERNEL_WIDTH_MS, PEAK_RATE_HZ, SR
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
    p.add_argument("--view", default="raster", choices=["raster", "rates", "filters", "psth"],
                   help="raster (the publication figure), rates (supplementary), "
                        "filters (causal filter bank diagnostics; needs no VEN model), or "
                        "psth (DAF response time course against the Fig. 8E reading)")
    p.add_argument("--causal-encoder", default=None,
                   help="causal filter bank .npz (--view filters)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=SR)
    p.add_argument("--t-post", type=int, default=200)
    p.add_argument("--mean-rate-hz", type=float, default=AUD_MEAN_RATE_HZ,
                   help="auditory spike-rate calibration; MUST match what the model was "
                        "trained with, or the figure encodes a different stimulus than "
                        "the metrics beside it")
    # No delay flags: the conduction delays are baked into the trained model and are
    # applied by its own inference pass. This entry point used to accept
    # --aud-delay-ms/--hvc-delay-ms and pass them to a function that ignored them.
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_dir = args.out_dir or str(output_dir())
    # The original script assumed outputs/ already existed, having been created at
    # import time by whichever training script ran first. Do not rely on that.
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.view == "filters":
        # The filter diagnostics describe the encoder alone, so this branch loads no VEN
        # model and no corpus -- asking for one would make the figure undrawable before
        # stage 2 has ever run.
        from ..causal_filters import CausalFilterEncoder
        from ..figures.causal_filters import make_filter_figures

        enc_path = args.causal_encoder or str(output_dir() / "causal_encoder.npz")
        print(f"Loading causal encoder: {enc_path}")
        cenc = CausalFilterEncoder.load(enc_path)
        patch = None
        try:
            patch = OlshausenFieldEncoder.load(args.of_encoder or str(of_encoder_npz()))
        except (OSError, KeyError):
            print("  (no comparable patch encoder; drawing the causal bank alone)")
        for path in make_filter_figures(cenc, out_dir, sr=args.sr, patch_encoder=patch):
            print(f"  wrote {path}")
        return

    print("Loading model and data ...")
    print(f"  model: {model_path}")
    if args.causal_encoder:
        from ..causal_filters import CausalFilterEncoder
        enc = CausalFilterEncoder.load(args.causal_encoder)
    else:
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
    print(f"  model delays: aud={ven.aud_delay_ms} ms  hvc={ven.hvc_delay_ms} ms")

    # Tag the output after the model, so figures and models stay paired. re.sub returns
    # its input unchanged when nothing matches, which turned any other model filename
    # into a tag containing the whole path -- and a figure path with a directory in the
    # middle of the basename. Match explicitly and fall back to the stem.
    _m = re.search(r"(?:of_)?ven_model(.*?)\.npz$", model_path)
    tag = _m.group(1) if _m else f"_{Path(model_path).stem}"

    common = {
        "out_dir": out_dir,
        "T_post": args.t_post,
        "sr": args.sr,
        "seed": args.seed,
        "kernel_width": KERNEL_WIDTH_MS,
        "peak_rate": PEAK_RATE_HZ,
        "of_mean_rate_hz": args.mean_rate_hz,
    }
    if args.view == "psth":
        from ..evaluate import build_stimuli
        from ..figures.daf_psth import make_psth_figure

        st = build_stimuli(enc, motifs_path, sr=args.sr, seed=args.seed,
                           t_post=args.t_post, mean_rate_hz=args.mean_rate_hz,
                           verbose=False)
        for path in make_psth_figure(ven, enc, st, out_dir, tag=tag, seed=args.seed,
                                     mean_rate_hz=args.mean_rate_hz):
            print(f"  wrote {path}")
        return

    if args.view == "raster":
        from ..figures.encoding_comparison import make_figure

        # n_kernels sizes the raster's auditory row; the rate view has no raster.
        make_figure(ven, enc, sig_train, T_song, tag,
                    n_kernels=enc.n_channels, **common)
    else:
        from ..figures.encoding_rates import make_rate_figure

        make_rate_figure(ven, enc, sig_train, T_song, tag, **common)


if __name__ == "__main__":  # pragma: no cover
    main()
