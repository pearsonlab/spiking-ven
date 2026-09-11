"""``sven-train-ven`` -- train the vocal error network on the song corpus.

Stage 2. Each rendition presents one motif from the corpus together with HVC premotor
input and lets the plastic I->E weights (JIE) learn; interleaved "novel" presentations of
the time-reversed motif drive threshold homeostasis without driving STDP. Over renditions
the excitatory response to the trained song is cancelled while responses to novel or
perturbed sound survive.

Defaults are the tuned operating point (drive_e=0.06, B_scale=16, tau_s=20,
alpha_theta=0, 600 renditions), which reaches K1 ~7.8 Hz and K4 ~2.1x. Note these differ
from the bare defaults of the original experiment script: at B_scale=1.0 the auditory
drive is negligible against the tonic drive, JIE fills in non-selectively, and the
network cancels nothing in particular.

Only JIE is plastic. B (aud->E) and W_hvc (HVC->I) are fixed.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..constants import AUD_DELAY_MS, HVC_DELAY_MS, SR
from ..evaluate import build_stimuli, daf_metrics, format_metrics
from ..olshausen_field import OlshausenFieldEncoder
from ..paths import ensure_parent, motifs_npz, of_encoder_npz, ven_model_npz
from ..vocal_error_net import VocalErrorNetV2

# Renditions at which progress is logged.
LOG_AT = {1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default=None, help="trained OF encoder .npz")
    p.add_argument("--motifs", default=None, help="input motifs .npz")
    p.add_argument("--out", default=None, help="output model .npz")
    p.add_argument("--traces", default=None, help="output traces .npz (default: alongside --out)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sr", type=int, default=SR)
    # tuned operating point
    p.add_argument("--drive-e", type=float, default=0.06)
    p.add_argument("--drive-i", type=float, default=0.058)
    p.add_argument("--b-scale", type=float, default=16.0)
    p.add_argument("--tau-s", type=float, default=20.0)
    p.add_argument("--alpha-theta", type=float, default=0.0)
    p.add_argument("--n-rend", type=int, default=600)
    # network
    p.add_argument("--n-e", type=int, default=600)
    p.add_argument("--n-i", type=int, default=150)
    p.add_argument("--n-hvc", type=int, default=60)
    p.add_argument("--c-b", type=float, default=0.05)
    p.add_argument("--c-hvc", type=float, default=0.1)
    p.add_argument("--b-hvc-scale", type=float, default=0.333)
    p.add_argument("--c-jie", type=float, default=0.1)
    p.add_argument("--r-i-th", type=float, default=6.0)
    p.add_argument("--j-max-ie", type=float, default=1.0)
    p.add_argument("--theta-e-init", type=float, default=0.0)
    p.add_argument("--a-jie", type=float, default=None,
                   help="STDP rate; default auto-scales as 3e-3 * (10 / tau_stdp_eff)")
    p.add_argument("--tau-stdp", type=float, default=0.0,
                   help="STDP kernel width (0 = use tau_s)")
    # protocol
    p.add_argument("--novel-ratio", type=int, default=1)
    p.add_argument("--r-e-target", type=float, default=16.0)
    p.add_argument("--t-post", type=int, default=200)
    p.add_argument("--t-burn", type=int, default=500)
    p.add_argument("--aud-delay-ms", type=int, default=AUD_DELAY_MS,
                   help="cochlea->AIV-E (measured)")
    p.add_argument("--hvc-delay-ms", type=int, default=HVC_DELAY_MS,
                   help="HVC->AIV-I (estimated)")
    # encoding
    p.add_argument("--mean-rate-hz", type=float, default=15.0)
    p.add_argument("--n-ista", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    enc_path = args.encoder or str(of_encoder_npz())
    motifs_path = args.motifs or str(motifs_npz())
    out_path = str(ensure_parent(args.out or ven_model_npz()))
    traces_path = str(ensure_parent(args.traces)) if args.traces else out_path.replace(
        ".npz", "_traces.npz")

    # A_jie auto-scales inversely with the effective STDP kernel width so the per-spike
    # STDP integral (~ A_jie * tau_stdp) is constant across widths.
    tau_stdp_eff = args.tau_stdp if args.tau_stdp > 0.0 else args.tau_s
    a_jie = args.a_jie if args.a_jie is not None else 3e-3 * (10.0 / tau_stdp_eff)

    print("Loading OF encoder and motifs...")
    encoder = OlshausenFieldEncoder.load(enc_path)
    n_kernels = encoder.n_bases
    print(f"  n_bases={n_kernels}  coch_params={encoder.coch_params}")

    # Stimuli are built by evaluate.build_stimuli, which the reproduction tests also
    # call -- one definition, so training and the metrics cannot drift apart.
    st = build_stimuli(encoder, motifs_path, sr=args.sr, seed=args.seed,
                       t_post=args.t_post, t_burn=args.t_burn, n_hvc=args.n_hvc,
                       mean_rate_hz=args.mean_rate_hz, n_ista=args.n_ista)
    T_song = st["T_song"]
    aud_correct, aud_reversed, aud_daf = (st["aud_correct"], st["aud_reversed"],
                                          st["aud_daf"])
    correct_pool = st["correct_pool"]
    hvc_on, hvc_off, hvc_burn, aud_burn = (st["hvc_on"], st["hvc_off"], st["hvc_burn"],
                                           st["aud_burn"])
    corr, train_idx, n_motifs = st["fwd_rev_corr"], st["train_idx"], st["n_motifs"]

    # --- network ----------------------------------------------------------
    ven = VocalErrorNetV2(
        n_e=args.n_e, n_i=args.n_i, n_hvc=args.n_hvc, n_aud=n_kernels,
        B_scale=args.b_scale, c_B=args.c_b, c_hvc=args.c_hvc,
        B_hvc_scale=args.b_hvc_scale, c_JIE=args.c_jie, tau_s=args.tau_s,
        r_i_th=args.r_i_th, A_jie=a_jie, J_max_ie=args.j_max_ie,
        drive_e=args.drive_e, drive_i=args.drive_i, alpha_theta=args.alpha_theta,
        r_e_target=args.r_e_target, r_e_song_target=0.0,
        A_jei_plastic=0.0, use_mask=False,
        aud_delay_ms=args.aud_delay_ms, hvc_delay_ms=args.hvc_delay_ms,
        tau_stdp=args.tau_stdp, seed=args.seed,
    )
    if args.theta_e_init > 0.0:
        ven.theta_e[:] = args.theta_e_init

    print(f"\n{'=' * 65}")
    print(f"  OF-VEN  A_jie={a_jie:.2e}  alpha_theta={args.alpha_theta}  "
          f"tau_s={args.tau_s} ms" + (f"  tau_stdp={args.tau_stdp:.0f} ms"
                                      if args.tau_stdp > 0 else ""))
    print(f"  drive_e={args.drive_e}  B_scale={args.b_scale}  "
          f"theta_e_init={ven.theta_e[0]:.4f}")
    print(f"  n_hvc={args.n_hvc}  c_JIE={args.c_jie}  r_i_th={args.r_i_th} Hz  "
          f"r_e_target={args.r_e_target} Hz")
    print(f"  novel_ratio={args.novel_ratio}  correct_pool={n_motifs} motifs  "
          f"{args.n_rend} renditions")
    print(f"{'=' * 65}")

    # --- training loop ----------------------------------------------------
    def snapshot() -> tuple[float, float]:
        rc = float(ven.transform(hvc_on, aud_correct).mean() * 1000)
        rn = float(ven.transform(hvc_on, aud_reversed).mean() * 1000)
        return rc, rn

    rc_trace: list[float] = []
    rn_trace: list[float] = []
    jie_trace: list[float] = []
    theta_trace: list[float] = []

    def log(r: int) -> None:
        ratio = rn_trace[-1] / rc_trace[-1] if rc_trace[-1] > 0 else float("nan")
        print(f"  r{r:<5} correct={rc_trace[-1]:.2f}  novel={rn_trace[-1]:.2f}  "
              f"ratio={ratio:.3f}  JIE={jie_trace[-1]:.5f}  "
              f"theta={theta_trace[-1]:.4f}", flush=True)

    def record() -> None:
        rc, rn = snapshot()
        rc_trace.append(rc)
        rn_trace.append(rn)
        jie_trace.append(float(ven.JIE.mean()))
        theta_trace.append(float(ven.theta_e.mean()))

    # Interleaved "novel" passes. At the shipped operating point (--alpha-theta 0, and
    # A_jei_plastic fixed at 0) adapt() has no homeostasis to run and updates nothing --
    # but it still simulates, and so still advances the network's noise stream. The
    # published K-numbers were produced with these passes in place, so they are kept
    # rather than skipped as dead work; --novel-ratio 0 drops them and will land on
    # slightly different numbers. Turning homeostasis on makes them load-bearing again.
    #
    # Rendition 1 carries the burn-in.
    ven.fit(hvc_burn, aud_burn, n_renditions=1, T_song=T_song,
            T_burn=args.t_burn, T_post=args.t_post, verbose=False)
    for _ in range(args.novel_ratio):
        ven.adapt(hvc_on, aud_reversed)
    record()
    if 1 in LOG_AT:
        log(1)

    for r in range(1, args.n_rend):
        ven.fit(hvc_on, correct_pool[r % len(correct_pool)], n_renditions=1,
                T_song=T_song, T_burn=0, T_post=args.t_post, verbose=False)
        for _ in range(args.novel_ratio):
            ven.adapt(hvc_on, aud_reversed)
        record()
        if (r + 1) in LOG_AT or r == args.n_rend - 1:
            log(r + 1)

    # --- save + evaluate --------------------------------------------------
    ven.save(out_path)
    print(f"\nSaved {out_path}")

    m = daf_metrics(ven, hvc_on=hvc_on, hvc_off=hvc_off, aud_correct=aud_correct,
                    aud_daf=aud_daf, aud_reversed=aud_reversed)
    print()
    print(format_metrics(m, r_e_target=args.r_e_target))

    np.savez(traces_path,
             rc=np.array(rc_trace), rn=np.array(rn_trace),
             jie=np.array(jie_trace), theta=np.array(theta_trace),
             fwd_rev_corr=corr, train_idx=train_idx,
             rc_hvc=m["k1"], rn_hvc=m["k2"], rn_noh=m["k2_no_hvc"],
             rn_rev_hvc=m["k4_rate"], ratio_daf=m["k3"], ratio_k4=m["k4"],
             B_scale=args.b_scale, drive_e=args.drive_e, alpha_theta=args.alpha_theta,
             tau_s=args.tau_s, c_jie=args.c_jie, n_rend=args.n_rend, a_jie=a_jie)
    print(f"Saved {traces_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
