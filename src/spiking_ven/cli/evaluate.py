"""``sven-evaluate`` -- recompute the error metrics for a saved model.

Loads a trained network and re-derives the Mandelblat-Cerf metrics (the E rate per
rendition for the trained and the reversed song, and the DAF response) without
retraining. Fully seeded: the same model file and seed give the same numbers every time.

The stimuli are rebuilt with the settings the model recorded at training time; a
stimulus flag given explicitly overrides that, with a warning if it disagrees.

Expect them to differ slightly from the metrics printed at the end of training: those are
computed with the simulation noise stream already advanced by hundreds of renditions,
whereas a freshly loaded model starts that stream at zero. Same distribution, different
draw.
"""

from __future__ import annotations

import argparse
import json

from ..causal_filters import CausalFilterEncoder
from ..metrics import (
    daf_response,
    format_rendition_rates,
    format_responders,
    rendition_rates,
)
from ..paths import encoder_npz, motifs_npz, ven_model_npz
from ..vocal_error_net import VocalErrorNet
from ._stimulus_args import add_stimulus_flags, stimuli_for_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="trained VEN .npz")
    p.add_argument("--encoder", default=None, help="trained causal filter bank .npz")
    p.add_argument("--motifs", default=None, help="motifs .npz")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the simulation noise and the DAF trials")
    add_stimulus_flags(p)
    p.add_argument("--n-trials", type=int, default=20,
                   help="renditions with independent noise draws for the DAF response")
    p.add_argument("--json", action="store_true", help="emit the metrics as JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    enc_path = args.encoder or str(encoder_npz())
    motifs_path = args.motifs or str(motifs_npz())

    if not args.json:
        print(f"model:   {model_path}")
        print(f"encoder: {enc_path}")

    encoder = CausalFilterEncoder.load(enc_path)
    ven = VocalErrorNet.load(model_path, seed=args.seed)
    st = stimuli_for_model(ven, encoder, args, encoder_path=enc_path,
                           motifs_path=motifs_path, verbose=not args.json)

    m = rendition_rates(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                        aud_trained=st["aud_correct"], aud_daf=st["aud_daf"],
                        aud_reversed=st["aud_reversed"], hvc_prelude=st["hvc_prelude"])

    # The DAF number that means something: a windowed, per-neuron responder measurement
    # rather than an excitatory rate averaged over a whole rendition, which cannot see a
    # brief burst.
    r = daf_response(ven, encoder, st, n_trials=args.n_trials, seed=args.seed)

    if args.json:
        print(json.dumps({**m, "daf_response": r, "fwd_rev_corr": st["fwd_rev_corr"],
                          "train_idx": st["train_idx"], "model": model_path}, indent=2))
    else:
        print()
        print(format_rendition_rates(m))
        print()
        print(format_responders(r))
        print(f"\n  encoder forward/reversed correlation: {st['fwd_rev_corr']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
