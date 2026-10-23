"""``sven-train-ven`` -- train the vocal error network on the song corpus.

Stage 3. Each rendition presents one motif from the corpus together with HVC premotor
input, and the plastic I->E weights (JIE) learn by causal STDP. Over renditions the
excitatory response to the trained song is cancelled while responses to novel or
perturbed sound survive.

Defaults are the tuned operating point in :mod:`spiking_ven.constants`. Only JIE is
plastic; B (aud->E), B_hvc (HVC->I) and JEI (E->I) are fixed.

The saved model records how its stimuli were built (``ven.stimulus``: sample rate, seed,
protocol timing, rate calibration, DAF window, and checksums of the encoder and motifs
files), so ``sven-evaluate`` and ``sven-figure`` rebuild the same stimuli.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..constants import (
    A_SFA,
    AUD_DELAY_MS,
    AUD_MEAN_RATE_HZ,
    B_HVC_SCALE,
    B_SCALE,
    C_B,
    C_HVC,
    DRIVE_E,
    DRIVE_I,
    G_NMDA,
    HVC_DELAY_MS,
    HVC_INTRO,
    J_MAX_IE,
    N_E,
    N_HVC,
    N_I,
    N_ISTA,
    N_RENDITIONS,
    R_I_TH,
    T_BURN,
    T_POST,
    TAU_E,
    TAU_S,
    TAU_SFA,
    TAU_STDP,
    a_jie_for,
)
from ..metrics import (
    daf_response,
    format_rendition_rates,
    format_responders,
    rendition_rates,
)
from ..paths import encoder_npz, ensure_parent, motifs_npz, ven_model_npz
from ..stimuli import build_stimuli, stimulus_record
from ..vocal_error_net import VocalErrorNet

# Renditions at which progress is logged.
LOG_AT = {1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default=None,
                   help="trained causal filter bank .npz (default: outputs/causal_encoder.npz)")
    p.add_argument("--motifs", default=None, help="input motifs .npz")
    p.add_argument("--out", default=None, help="output model .npz")
    p.add_argument("--traces", default=None, help="output traces .npz (default: alongside --out)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=None,
                   help="corpus sample rate (Hz); must match the corpus. "
                        "Default: the corpus's own rate")
    # tuned operating point
    p.add_argument("--drive-e", type=float, default=DRIVE_E)
    p.add_argument("--drive-i", type=float, default=DRIVE_I)
    p.add_argument("--b-scale", type=float, default=B_SCALE)
    p.add_argument("--tau-s", type=float, default=TAU_S)
    p.add_argument("--tau-e", type=float, default=TAU_E,
                   help="E membrane time constant (ms). theta_e is derived from it, so "
                        "the tonic rate stays at target_rate as this changes.")
    p.add_argument("--n-rend", type=int, default=N_RENDITIONS)
    # network
    p.add_argument("--n-e", type=int, default=N_E)
    p.add_argument("--n-i", type=int, default=N_I)
    p.add_argument("--n-hvc", type=int, default=N_HVC)
    p.add_argument("--c-b", type=float, default=C_B)
    p.add_argument("--c-hvc", type=float, default=C_HVC)
    p.add_argument("--b-hvc-scale", type=float, default=B_HVC_SCALE)
    p.add_argument("--r-i-th", type=float, default=R_I_TH)
    p.add_argument("--j-max-ie", type=float, default=J_MAX_IE)
    p.add_argument("--theta-e-init", type=float, default=0.0)
    p.add_argument("--a-jie", type=float, default=None,
                   help="STDP rate; default auto-scales as 3e-3 * (10 / tau_stdp_eff)")
    p.add_argument("--tau-stdp", type=float, default=TAU_STDP,
                   help="STDP coincidence window (ms), independent of the synaptic "
                        "tau_s; 0 falls back to tau_s")
    # protocol
    p.add_argument("--t-post", type=int, default=T_POST)
    p.add_argument("--t-burn", type=int, default=T_BURN)
    p.add_argument("--aud-delay-ms", type=int, default=AUD_DELAY_MS,
                   help="cochlea->AIV-E conduction delay (estimated)")
    p.add_argument("--hvc-intro", type=int, nargs=3, default=list(HVC_INTRO),
                   metavar=("NOTES", "NOTE_MS", "GAP_MS"),
                   help="HVC introductory notes before each motif (no sound); 0 0 0 for none")
    p.add_argument("--hvc-delay-ms", type=int, default=HVC_DELAY_MS,
                   help="HVC->AIV-I (estimated)")
    # encoding
    p.add_argument("--mean-rate-hz", type=float, default=AUD_MEAN_RATE_HZ,
                   help="auditory spike-rate calibration (Schneider & Woolley 2013)")
    p.add_argument("--n-ista", type=int, default=N_ISTA)
    # Spike-frequency adaptation on E; the sweep behind A_SFA is in docs/design-notes.md.
    p.add_argument("--a-sfa", type=float, default=A_SFA,
                   help="adaptation increment per E spike (units of theta_e)")
    p.add_argument("--tau-sfa", type=float, default=TAU_SFA,
                   help="adaptation decay (ms); keep it shorter than the 50 ms DAF burst")
    p.add_argument("--g-nmda", type=float, default=G_NMDA,
                   help="NMDA conductance of the aud->E synapse, relative to the fast "
                        "weights (0 disables it); kinetics and Mg2+ block in constants")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    enc_path = args.encoder or str(encoder_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_path = str(ensure_parent(args.out or ven_model_npz()))
    traces_path = str(ensure_parent(args.traces)) if args.traces else out_path.replace(
        ".npz", "_traces.npz")

    # Derived rather than a constant: the per-spike STDP integral goes as
    # A_jie * tau, so the amplitude scales inversely with the kernel width. See
    # constants.a_jie_for for why that formula lives with its inputs.
    a_jie = (args.a_jie if args.a_jie is not None
             else a_jie_for(args.tau_stdp, args.tau_s))

    from ..causal_filters import CausalFilterEncoder

    print("Loading causal filter bank and motifs...")
    encoder = CausalFilterEncoder.load(enc_path)
    n_kernels = encoder.n_channels
    print(f"  n_bases={n_kernels}  coch_params={encoder.coch_params}")

    # Stimuli are built by stimuli.build_stimuli, which the reproduction tests also
    # call -- one definition, so training and the metrics cannot drift apart.
    st = build_stimuli(encoder, motifs_path, sr=args.sr, seed=args.seed,
                       t_post=args.t_post, t_burn=args.t_burn, n_hvc=args.n_hvc,
                       mean_rate_hz=args.mean_rate_hz, n_ista=args.n_ista,
                       hvc_intro=tuple(args.hvc_intro))
    aud_correct, aud_reversed = st["aud_correct"], st["aud_reversed"]
    hvc_prelude = st["hvc_prelude"]
    correct_pool = st["correct_pool"]
    hvc_on, hvc_off, hvc_burn, aud_burn = (st["hvc_on"], st["hvc_off"], st["hvc_burn"],
                                           st["aud_burn"])
    corr, train_idx, n_motifs = st["fwd_rev_corr"], st["train_idx"], st["n_motifs"]

    # --- network ----------------------------------------------------------
    ven = VocalErrorNet(
        n_e=args.n_e, n_i=args.n_i, n_hvc=args.n_hvc, n_aud=n_kernels,
        B_scale=args.b_scale, c_B=args.c_b, c_hvc=args.c_hvc,
        B_hvc_scale=args.b_hvc_scale, tau_s=args.tau_s,
        tau_e=args.tau_e,
        r_i_th=args.r_i_th, A_jie=a_jie, J_max_ie=args.j_max_ie,
        drive_e=args.drive_e, drive_i=args.drive_i,
        aud_delay_ms=args.aud_delay_ms, hvc_delay_ms=args.hvc_delay_ms,
        tau_stdp=args.tau_stdp, seed=args.seed,
        a_sfa=args.a_sfa, tau_sfa=args.tau_sfa, g_nmda=args.g_nmda,
    )
    if args.theta_e_init > 0.0:
        ven.theta_e[:] = args.theta_e_init

    print(f"\n{'=' * 65}")
    print(f"  VEN  A_jie={a_jie:.2e}  "
          f"tau_s={args.tau_s} ms" + (f"  tau_stdp={args.tau_stdp:.0f} ms"
                                      if args.tau_stdp > 0 else ""))
    print(f"  drive_e={args.drive_e}  B_scale={args.b_scale}  "
          f"theta_e_init={ven.theta_e[0]:.4f}")
    print(f"  n_hvc={args.n_hvc}  r_i_th={args.r_i_th} Hz  "
          f"correct_pool={n_motifs} motifs  {args.n_rend} renditions")
    print(f"{'=' * 65}")

    # --- training loop ----------------------------------------------------
    # Monitoring draws its membrane noise from its own generator, so logging does not
    # advance the noise stream training sees: a run is the same with or without it.
    monitor_rng = np.random.default_rng(args.seed + 20_000)

    def snapshot() -> tuple[float, float]:
        trained = float(ven.transform(hvc_on, aud_correct, rng=monitor_rng,
                                      hvc_prelude=hvc_prelude).mean() * 1000)
        reversed_ = float(ven.transform(hvc_on, aud_reversed, rng=monitor_rng,
                                        hvc_prelude=hvc_prelude).mean() * 1000)
        return trained, reversed_

    trained_trace: list[float] = []
    reversed_trace: list[float] = []
    jie_trace: list[float] = []

    def log(r: int) -> None:
        ratio = (reversed_trace[-1] / trained_trace[-1] if trained_trace[-1] > 0
                 else float("nan"))
        print(f"  r{r:<5} trained={trained_trace[-1]:.2f}  reversed={reversed_trace[-1]:.2f}  "
              f"ratio={ratio:.3f}  JIE={jie_trace[-1]:.5f}", flush=True)

    def record() -> None:
        trained, reversed_ = snapshot()
        trained_trace.append(trained)
        reversed_trace.append(reversed_)
        jie_trace.append(float(ven.JIE.mean()))

    # Rendition 1 carries the burn-in, which ends with the intro notes; every later
    # rendition runs them as a prelude.
    ven.fit(hvc_burn, aud_burn)
    record()
    if 1 in LOG_AT:
        log(1)

    for r in range(1, args.n_rend):
        ven.fit(hvc_on, correct_pool[r % len(correct_pool)], hvc_prelude=hvc_prelude)
        record()
        if (r + 1) in LOG_AT or r == args.n_rend - 1:
            log(r + 1)

    # --- save + evaluate --------------------------------------------------
    ven.stimulus = stimulus_record(st, encoder_path=enc_path, motifs_path=motifs_path)
    ven.save(out_path)
    print(f"\nSaved {out_path}")

    m = rendition_rates(ven, hvc_on=hvc_on, hvc_off=hvc_off, aud_trained=aud_correct,
                        aud_daf=st["aud_daf"], aud_reversed=aud_reversed,
                        hvc_prelude=hvc_prelude)
    print()
    print(format_rendition_rates(m))
    r = daf_response(ven, encoder, st, n_trials=20, seed=args.seed)
    print()
    print(format_responders(r))

    # Per-rendition traces, then the rendition-level rates of the final network.
    np.savez(traces_path,
             trained_hz=np.array(trained_trace), reversed_hz=np.array(reversed_trace),
             jie_mean=np.array(jie_trace),
             fwd_rev_corr=corr, train_idx=train_idx,
             final_trained_hz=m["trained_hz"], final_reversed_hz=m["reversed_hz"],
             final_reversed_over_trained=m["reversed_over_trained"],
             final_daf_not_singing_hz=m["daf_not_singing_hz"],
             B_scale=args.b_scale, drive_e=args.drive_e,
             tau_s=args.tau_s, n_rend=args.n_rend, a_jie=a_jie)
    print(f"Saved {traces_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
