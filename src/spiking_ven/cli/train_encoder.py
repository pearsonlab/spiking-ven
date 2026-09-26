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
from ..corpus import load_corpus
from ..causal_filters import CausalFilterEncoder, causal_meansub
from ..causal_filters import learn as causal_learn
from ..olshausen_field import OlshausenFieldEncoder, coch_extract_patches, of_to_spikes
from ..paths import ensure_parent, motifs_npz, of_encoder_npz


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motifs", default=None, help="input motifs .npz")
    p.add_argument("--out", default=None, help="output encoder .npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder", choices=("patch", "causal"), default="patch",
                   help="patch: Olshausen-Field STRF atoms; causal: minimum-phase filter bank")
    # cochleagram front-end
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-coch-ch", type=int, default=40, help="cochleagram frequency channels")
    p.add_argument("--frame-rate", type=int, default=500, help="Hz (500 -> 2 ms per frame)")
    p.add_argument("--lo-hz", type=float, default=200.0)
    p.add_argument("--hi-hz", type=float, default=8000.0)
    p.add_argument("--k-frames", type=int, default=16, help="context window (16 x 2 ms = 32 ms)")
    p.add_argument("--rms-ref", type=float, default=0.1, help="RMS normalisation before cochleagram")
    # dictionary learning
    p.add_argument("--n-bases", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda-sparse", type=float, default=0.05)
    p.add_argument("--n-ista-train", type=int, default=30)
    p.add_argument("--n-ista-encode", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--target-usage", type=float, default=0.10)
    p.add_argument("--lr-bias", type=float, default=0.005)
    p.add_argument("--mean-rate-hz", type=float, default=15.0)
    # --encoder causal only. Defaults are the converged operating point: 80 epochs with
    # 3% target usage keeps all 64 filters alive (see causal_filters.learn on why weak
    # equalisation makes dropout one-way).
    p.add_argument("--causal-epochs", type=int, default=80)
    p.add_argument("--causal-lambda", type=float, default=0.1)
    p.add_argument("--causal-target-usage", type=float, default=0.03)
    p.add_argument("--causal-lr", type=float, default=0.1)
    p.add_argument("--causal-n-ista", type=int, default=20)
    p.add_argument("--causal-mp-strength", type=float, default=0.25,
                   help="pull toward minimum phase per update; 1.0 is the strict "
                        "projection, which collapses the kernels to impulses")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    motifs_path = args.motifs or str(motifs_npz())
    out_path = str(ensure_parent(args.out or of_encoder_npz()))

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

    if args.encoder == "causal":
        print(f"\nTraining causal minimum-phase filter bank "
              f"({args.n_bases} filters x {args.n_coch_ch} ch x {args.k_frames} frames)...")
        cochs = [causal_meansub(cg.astype(np.float64), args.k_frames) for cg in all_cochs]
        V, bias = causal_learn(cochs, K=args.n_bases, L=args.k_frames,
                               lam=args.causal_lambda, n_epochs=args.causal_epochs,
                               n_iter=args.causal_n_ista, lr=args.causal_lr,
                               seed=args.seed, target_usage=args.causal_target_usage,
                               mp_strength=args.causal_mp_strength)
        enc = CausalFilterEncoder(V, bias, args.causal_lambda, coch_params)
        enc.save(out_path)
        print(f"Saved {out_path}")
        E = (V ** 2).sum(axis=1)
        E = E / E.sum(axis=1, keepdims=True)
        cen = (E * np.arange(args.k_frames)).sum(axis=1)
        print(f"  bias [{bias.min():.4f}, {bias.max():.4f}]  std={bias.std():.4f}")
        print(f"  kernel energy centroid {cen.mean():.1f}/{args.k_frames} "
              f"(sd {cen.std():.1f}); newest 4 frames hold "
              f"{E[:, -4:].sum(axis=1).mean():.3f} of the energy")
        act0 = enc.encode_signal(corpus.signal(0), sr=args.sr,
                                 n_ista=args.n_ista_encode)
        spk = of_to_spikes(act0, mean_rate_hz=args.mean_rate_hz,
                           frame_rate=args.frame_rate, seed=args.seed + 200)
        print(f"  active fraction (motif 0): {(act0 != 0).mean() * 100:.1f}%"
              f"  mean rate: {spk.mean() * args.frame_rate:.1f} Hz")
        return

    print("Extracting training patches (stride=1 frame)...")
    patches = np.concatenate(
        [coch_extract_patches(cg, args.k_frames, stride=1, mean_subtract=True)
         for cg in all_cochs],
        axis=0,
    )
    print(f"  {len(patches)} patches  patch_dim={patches.shape[1]}")

    # Report inner-product scale against a random dictionary, so the sparsity
    # threshold (lambda * step) can be sanity-checked against the data.
    tmp = OlshausenFieldEncoder(n_bases=args.n_bases, patch_len=patches.shape[1],
                                seed=args.seed)
    step = tmp._lipschitz_step()
    ip = np.abs(patches[:500].astype(np.float64) @ tmp.A.T)
    print(f"  Inner product stats (random dict): mean={ip.mean():.4f}  "
          f"p90={np.percentile(ip, 90):.4f}  step={step:.4f}  "
          f"thresh(lambda={args.lambda_sparse})={args.lambda_sparse * step:.4f}")

    print("\nTraining O&F encoder...")
    encoder = OlshausenFieldEncoder(n_bases=args.n_bases, patch_len=patches.shape[1],
                                    seed=args.seed)
    encoder.coch_params = coch_params
    encoder.fit(
        patches=patches,
        n_epochs=args.n_epochs,
        lr=args.lr,
        lambda_sparse=args.lambda_sparse,
        n_ista=args.n_ista_train,
        batch_size=args.batch_size,
        lifetime_sparsity=True,
        target_usage=args.target_usage,
        lr_bias=args.lr_bias,
        seed=args.seed,
    )

    encoder.save(out_path)
    print(f"Saved {out_path}")
    bias = encoder._bias
    print(f"  bias [{bias.min():.4f}, {bias.max():.4f}]  std={bias.std():.4f}")

    # Sparsity/rate diagnostic on the first motif, at cochleagram resolution. Reuses the
    # cochleagram computed above rather than recomputing an identical one.
    act0 = encoder.encode_patches(
        coch_extract_patches(all_cochs[0], args.k_frames, stride=1, mean_subtract=True),
        n_ista=args.n_ista_encode)
    spk = of_to_spikes(act0, mean_rate_hz=args.mean_rate_hz,
                       frame_rate=args.frame_rate, seed=args.seed + 200)
    print(f"  active fraction (motif 0): {(act0 != 0).mean() * 100:.1f}%"
          f"  mean rate: {spk.mean() * args.frame_rate:.1f} Hz")


if __name__ == "__main__":  # pragma: no cover
    main()
