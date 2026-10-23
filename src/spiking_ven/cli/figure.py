"""``sven-figure`` -- render the auditory cancellation figure.

Six rows (waveform / spectrogram / auditory neurons / inhibitory interneurons /
excitatory projection neurons / E population rate) by three stimulus columns (training
song / time-reversed motif / distorted auditory feedback).

The result is in the excitatory rows: near-silent on the trained song, active on the
other two. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

``--view psth`` renders the DAF response time course against the Fig. 8E reading instead,
and ``--view filters`` the encoder's kernels as spectrotemporal images (its STRFs) plus a
summary of their timing, frequency tuning and redundancy. ``--view units`` puts a few
example units' kernels over their spikes to every motif in the corpus, and
``--view gammatone`` follows the training motif through the encoder's front end:
spectrogram, gammatone filters, cochleagram and spike output. ``filters``, ``units`` and
``gammatone`` need no VEN model, so they can be drawn before a network has been trained.

The ``raster`` and ``psth`` views rebuild the stimuli with the settings the model recorded
at training time; a stimulus flag given explicitly overrides that, with a warning if it
disagrees. The other views have no model, so the flags fall back to the constants.

Requires the ``plots`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..causal_filters import CausalFilterEncoder
from ..constants import AUD_MEAN_RATE_HZ, N_ISTA, SR, T_POST
from ..paths import encoder_npz, motifs_npz, output_dir, ven_model_npz
from ..vocal_error_net import VocalErrorNet
from ._stimulus_args import add_stimulus_flags, stimuli_for_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="trained VEN .npz")
    p.add_argument("--encoder", default=None,
                   help="trained causal filter bank .npz (default: outputs/causal_encoder.npz)")
    p.add_argument("--motifs", default=None, help="motifs .npz")
    p.add_argument("--out-dir", default=None, help="directory for the rendered figure")
    p.add_argument("--view", default="raster",
                   choices=["raster", "psth", "filters", "units", "gammatone"],
                   help="raster (the publication figure), "
                        "psth (DAF response time course against the Fig. 8E reading), "
                        "filters (the encoder's kernels; needs no VEN model), "
                        "units (example kernels over multi-motif rasters; no VEN model), or "
                        "gammatone (the encoder's front end on the training motif; "
                        "no VEN model)")
    p.add_argument("--units", type=int, nargs="+", default=None,
                   help="unit indices for --view units (default: the most reliable unit "
                        "in each third of the peak-frequency range)")
    p.add_argument("--seed", type=int, default=42)
    add_stimulus_flags(p)
    return p


def _or(value, default):
    return default if value is None else value


def _corpus_sr(motifs_path, sr):
    """The corpus's sample rate, checked against an explicit ``sr``."""
    from ..corpus import load_corpus

    corpus_sr = int(load_corpus(motifs_path).sr)
    if sr is not None and int(sr) != corpus_sr:
        raise ValueError(f"--sr {sr} does not match the corpus's {corpus_sr} Hz "
                         f"({motifs_path})")
    return corpus_sr


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_dir = args.out_dir or str(output_dir())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    enc_path = args.encoder or str(encoder_npz())
    enc = CausalFilterEncoder.load(enc_path)

    if args.view == "filters":
        from ..figures.causal_filters import make_filter_figures

        for path in make_filter_figures(enc, out_dir, sr=_or(args.sr, SR)):
            print(f"  wrote {path}")
        return

    if args.view == "units":
        from ..figures.unit_strfs import make_unit_figure

        paths, units = make_unit_figure(
            enc, motifs_path, out_dir, units=args.units,
            sr=_corpus_sr(motifs_path, args.sr), seed=_or(args.stimulus_seed, args.seed),
            t_post=_or(args.t_post, T_POST),
            mean_rate_hz=_or(args.mean_rate_hz, AUD_MEAN_RATE_HZ),
            n_ista=_or(args.n_ista, N_ISTA))
        print(f"  units {units}")
        for path in paths:
            print(f"  wrote {path}")
        return

    if args.view == "gammatone":
        from ..corpus import load_corpus
        from ..figures.gammatone import make_gammatone_figure

        _corpus_sr(motifs_path, args.sr)
        for path in make_gammatone_figure(
                enc, load_corpus(motifs_path), out_dir,
                seed=_or(args.stimulus_seed, args.seed),
                mean_rate_hz=_or(args.mean_rate_hz, AUD_MEAN_RATE_HZ),
                n_ista=_or(args.n_ista, N_ISTA)):
            print(f"  wrote {path}")
        return

    print(f"Loading model: {model_path}")
    # Pin the seed so the rendered figure is reproducible, not just the training.
    ven = VocalErrorNet.load(model_path, seed=args.seed)
    st = stimuli_for_model(ven, enc, args, encoder_path=enc_path, motifs_path=motifs_path,
                           verbose=False)
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
