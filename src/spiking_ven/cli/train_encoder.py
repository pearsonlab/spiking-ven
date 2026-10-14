"""``sven-train-encoder`` -- train the sparse auditory encoder.

Stage 1 of the pipeline. Two encoders are available via ``--encoder``:

``patch`` (default)
    Olshausen-Field. Cochleagrams of every motif are cut into causal spectrotemporal
    patches (``n_channels x k_frames``, mean-subtracted) and a sparse dictionary is
    learned over them by ISTA with lifetime-sparsity bias adaptation. The learned atoms
    are STRFs.

``causal``
    A minimum-phase filter bank (:mod:`spiking_ven.causal_filters`). Same cochleagram
    front end and the same sparse competition, but inference at time t reads only frames
    before t and the kernels are projected onto minimum phase after every update. The
    patch encoder's atoms drift to an energy centroid of ~L/2 -- near-linear phase --
    which costs ~30 ms of encoder latency; minimum phase places the energy as early as
    the magnitude response allows, without constraining what a filter is tuned to. This
    is what brings the VEN's DAF response latency from 66 ms to 23 ms, against the bird's
    23 +/- 12 ms.

Mean subtraction matters: it removes the common energy envelope so activations encode
spectrotemporal *contrast*. Without it every basis responds to the shared broadband
signal, which is what makes the non-negative and sign-split encoder variants fail.

Diagnostic plots are not produced here; see the plotting entry points (they need the
``plots`` extra). This keeps matplotlib out of the training path.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..cochleagram import cochleagram
from ..constants import (
    AUD_MEAN_RATE_HZ,
    CAUSAL_EPOCHS,
    CAUSAL_L1,
    CAUSAL_LAMBDA,
    CAUSAL_LR,
    CAUSAL_MP_STRENGTH,
    CAUSAL_N_ISTA,
    CAUSAL_SMOOTH,
    SR,
)
from ..corpus import load_corpus
from ..causal_filters import CausalFilterEncoder, causal_meansub, prune_silent
from ..causal_filters import learn as causal_learn
from ..common import acts_to_spikes
from ..paths import encoder_npz, ensure_parent, motifs_npz


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motifs", default=None, help="input motifs .npz")
    p.add_argument("--out", default=None, help="output encoder .npz")
    p.add_argument("--seed", type=int, default=42)
    # cochleagram front-end
    p.add_argument("--sr", type=int, default=SR)
    p.add_argument("--n-coch-ch", type=int, default=40, help="cochleagram frequency channels")
    p.add_argument("--frame-rate", type=int, default=500, help="Hz (500 -> 2 ms per frame)")
    p.add_argument("--lo-hz", type=float, default=200.0)
    p.add_argument("--hi-hz", type=float, default=8000.0)
    p.add_argument("--k-frames", type=int, default=16, help="context window (16 x 2 ms = 32 ms)")
    p.add_argument("--rms-ref", type=float, default=0.1, help="RMS normalisation before cochleagram")
    # dictionary learning
    p.add_argument("--n-bases", type=int, default=64)
    p.add_argument("--causal-epochs", type=int, default=CAUSAL_EPOCHS)
    p.add_argument("--causal-lambda", type=float, default=CAUSAL_LAMBDA,
                   help="code sparsity; also sets how many filters survive")
    p.add_argument("--causal-lr", type=float, default=CAUSAL_LR)
    p.add_argument("--causal-l1", type=float, default=CAUSAL_L1,
                   help="L1 on kernel entries; scale against the gradient step")
    p.add_argument("--causal-smooth", type=float, default=CAUSAL_SMOOTH,
                   help="roughness penalty over channel and lag")
    p.add_argument("--causal-no-prune", action="store_true",
                   help="keep filters that never fire (they are stubs, not filters)")
    p.add_argument("--causal-n-ista", type=int, default=CAUSAL_N_ISTA)
    p.add_argument("--causal-mp-strength", type=float, default=CAUSAL_MP_STRENGTH,
                   help="pull toward minimum phase per update; 1.0 is the strict "
                        "projection, which collapses the kernels to impulses")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    motifs_path = args.motifs or str(motifs_npz())
    out_path = str(ensure_parent(args.out or encoder_npz()))

    coch_params = {
        "n_channels": args.n_coch_ch,
        "K_frames": args.k_frames,
        "frame_rate": args.frame_rate,
        "lo_hz": args.lo_hz,
        "hi_hz": args.hi_hz,
        "rms_ref": args.rms_ref,
    }

    print("Loading motifs...")
    # Via load_corpus, not a hand-rolled np.load: trimming to each rendition's valid
    # length and rounding T_song are corpus.py's job, in one place.
    corpus = load_corpus(motifs_path)
    print(f"  {corpus.n_motifs} motifs  T_song={corpus.T_song} ms")

    print("\nComputing cochleagrams...")
    all_cochs = []
    for sig in corpus.signals():
        sig = sig / max(float(np.sqrt(np.mean(sig**2))), 1e-12) * args.rms_ref
        all_cochs.append(cochleagram(sig, args.sr, n_channels=args.n_coch_ch,
                                     frame_rate=args.frame_rate,
                                     lo_hz=args.lo_hz, hi_hz=args.hi_hz))

    print(f"\nTraining causal minimum-phase filter bank "
          f"({args.n_bases} filters x {args.n_coch_ch} ch x {args.k_frames} frames)...")
    cochs = [causal_meansub(cg.astype(np.float64), args.k_frames) for cg in all_cochs]
    V = causal_learn(cochs, K=args.n_bases, L=args.k_frames,
                     lam=args.causal_lambda, n_epochs=args.causal_epochs,
                     n_iter=args.causal_n_ista, lr=args.causal_lr, seed=args.seed,
                     mp_strength=args.causal_mp_strength,
                     l1_V=args.causal_l1, smooth_V=args.causal_smooth)
    if not args.causal_no_prune:
        V_kept, kept = prune_silent(V, cochs, args.causal_lambda,
                                    n_iter=args.causal_n_ista)
        if len(kept) == 0:
            # Degenerate, and only reachable on a corpus too small or a run too short for
            # any filter to win a competition. Writing an encoder with no filters would
            # fail far downstream, where the cause is no longer visible.
            print(f"  WARNING: no filter fired; keeping all {args.n_bases} unpruned")
        else:
            V = V_kept
            print(f"  kept {len(kept)}/{args.n_bases} filters that ever fire")
    enc = CausalFilterEncoder(V, np.zeros(len(V)), args.causal_lambda, coch_params)
    enc.save(out_path)
    print(f"Saved {out_path}")
    E = (V ** 2).sum(axis=1)
    E = E / E.sum(axis=1, keepdims=True)
    cen = (E * np.arange(args.k_frames)).sum(axis=1)
    print(f"  kernel energy centroid {cen.mean():.1f}/{args.k_frames} "
          f"(sd {cen.std():.1f}); newest 4 frames hold "
          f"{E[:, -4:].sum(axis=1).mean():.3f} of the energy")
    act0 = enc.encode_signal(corpus.signal(0), sr=args.sr, n_ista=args.causal_n_ista)
    spk = acts_to_spikes(act0, mean_rate_hz=AUD_MEAN_RATE_HZ,
                       frame_rate=args.frame_rate, seed=args.seed + 200)
    print(f"  active fraction (motif 0): {(act0 != 0).mean() * 100:.1f}%"
          f"  mean rate: {spk.mean() * args.frame_rate:.1f} Hz")

if __name__ == "__main__":  # pragma: no cover
    main()
