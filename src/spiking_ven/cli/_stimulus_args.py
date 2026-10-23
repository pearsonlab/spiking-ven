"""Stimulus flags shared by the CLIs that rebuild a trained model's stimuli.

Every flag defaults to None, meaning "what the model recorded at training time, else the
constant": a model evaluated on different stimuli than it was trained on describes a
different experiment. See :func:`spiking_ven.stimuli.resolve_stimulus_settings`.
"""

from __future__ import annotations

import argparse

from ..stimuli import build_stimuli, resolve_stimulus_settings

_FROM_MODEL = "default: the model's recorded value, else the constant"


def add_stimulus_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("stimulus", "how the stimuli are rebuilt (" + _FROM_MODEL + ")")
    g.add_argument("--sr", type=int, default=None,
                   help="corpus sample rate (Hz); must match the corpus. "
                        "Default: the corpus's own rate")
    g.add_argument("--stimulus-seed", type=int, default=None,
                   help="seed the HVC drive and auditory spikes were drawn with "
                        "(default: the model's recorded value, else --seed)")
    g.add_argument("--t-post", type=int, default=None, help="silence after the motif (ms)")
    g.add_argument("--t-burn", type=int, default=None, help="silent burn-in (ms)")
    g.add_argument("--mean-rate-hz", type=float, default=None,
                   help="auditory spike-rate calibration (Hz)")
    g.add_argument("--n-ista", type=int, default=None, help="ISTA iterations per frame")
    g.add_argument("--daf-window-s", type=float, nargs=2, default=None,
                   metavar=("START", "END"), help="DAF noise window within the motif (s)")


def stimuli_for_model(ven, encoder, args, *, encoder_path, motifs_path,
                      verbose: bool = True) -> dict:
    """:func:`~spiking_ven.stimuli.build_stimuli` with the settings ``ven`` was trained on.

    ``n_hvc`` is the model's own, since the HVC drive has to fit its weight matrix.
    """
    if int(encoder.n_channels) != int(ven.n_aud):
        raise SystemExit(
            f"the encoder has {encoder.n_channels} channels but the model expects "
            f"{ven.n_aud}: {encoder_path} is not the encoder the model was trained with"
        )
    settings = resolve_stimulus_settings(
        getattr(ven, "stimulus", {}),
        {"sr": args.sr, "seed": args.stimulus_seed, "t_post": args.t_post,
         "t_burn": args.t_burn, "mean_rate_hz": args.mean_rate_hz,
         "n_ista": args.n_ista, "daf_window_s": args.daf_window_s},
        encoder_path=encoder_path, motifs_path=motifs_path,
    )
    seed = settings.pop("seed")
    return build_stimuli(encoder, motifs_path, seed=args.seed if seed is None else seed,
                         n_hvc=int(ven.n_hvc), verbose=verbose, **settings)
