"""``sven-evaluate`` -- recompute the K1-K4 error metrics for a saved model.

Loads a trained network and re-derives the Mandelblat-Cerf metrics without retraining,
which is what you want when checking a model you were handed, or after changing the
stimulus construction.

Because the network is fully seeded, this is reproducible: the same model file and seed
give the same numbers every time.

Expect these to differ slightly (well inside the test tolerances) from the metrics printed
at the end of training. Nothing has changed about the model: training-time metrics are
computed with the simulation noise stream already advanced by hundreds of renditions,
whereas a freshly loaded model starts that stream at position zero. Same distribution,
different draw.
"""

from __future__ import annotations

import argparse
import json

from ..constants import SR
from ..evaluate import build_stimuli, daf_metrics, format_metrics
from ..olshausen_field import OlshausenFieldEncoder
from ..paths import motifs_npz, of_encoder_npz, ven_model_npz
from ..vocal_error_net import VocalErrorNetV2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="trained VEN .npz")
    p.add_argument("--encoder", default=None, help="trained OF encoder .npz")
    p.add_argument("--motifs", default=None, help="motifs .npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=SR)
    p.add_argument("--n-hvc", type=int, default=60)
    p.add_argument("--t-post", type=int, default=200)
    p.add_argument("--t-burn", type=int, default=500)
    p.add_argument("--mean-rate-hz", type=float, default=15.0)
    p.add_argument("--n-ista", type=int, default=50)
    p.add_argument("--r-e-target", type=float, default=16.0,
                   help="only used to annotate the no-HVC sanity line")
    p.add_argument("--json", action="store_true", help="emit the metrics as JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model or str(ven_model_npz())
    enc_path = args.encoder or str(of_encoder_npz())
    motifs_path = args.motifs or str(motifs_npz())

    if not args.json:
        print(f"model:   {model_path}")
        print(f"encoder: {enc_path}")

    encoder = OlshausenFieldEncoder.load(enc_path)
    ven = VocalErrorNetV2.load(model_path, seed=args.seed)
    st = build_stimuli(encoder, motifs_path, sr=args.sr, seed=args.seed,
                       t_post=args.t_post,
                       t_burn=args.t_burn, n_hvc=args.n_hvc,
                       mean_rate_hz=args.mean_rate_hz, n_ista=args.n_ista,
                       verbose=not args.json)

    m = daf_metrics(ven, hvc_on=st["hvc_on"], hvc_off=st["hvc_off"],
                    aud_correct=st["aud_correct"], aud_daf=st["aud_daf"],
                    aud_reversed=st["aud_reversed"])

    if args.json:
        print(json.dumps({**m, "fwd_rev_corr": st["fwd_rev_corr"],
                          "train_idx": st["train_idx"], "model": model_path}, indent=2))
    else:
        print()
        print(format_metrics(m, r_e_target=args.r_e_target))
        print(f"\n  encoder forward/reversed correlation: {st['fwd_rev_corr']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
