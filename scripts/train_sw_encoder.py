"""Train the Schneider & Woolley two-stage auditory front end (``spiking_ven.sw_encoder``).

    python scripts/train_sw_encoder.py --out outputs/encoders/sw_encoder.npz

Stage 1: the causal encoder's learning rule with ISTA replaced by a ReLU (a tied rectified
autoencoder with roughness, minimum-phase and L1 steps and no kernel renormalisation;
``sw_encoder.learn_relu``) on the raw cochleagrams of the corpus motifs (same cochleagram
settings as the causal encoder). Units that never activate are dropped; the
primary gain is set so the song's mean primary rate is 32 Hz. Stage 2: the
difference-of-Gaussians threshold is set per unit, at a percentile of that unit's DoG output
on the song (``--global-threshold`` for one population-wide percentile), so that higher-AC
activity is sparse and every unit responds to something; its spike rate is calibrated later, by the stimulus
pipeline.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from spiking_ven.cochleagram import cochleagram
from spiking_ven.constants import (
    CAUSAL_EPOCHS, CAUSAL_FRAME_RATE, CAUSAL_HI_HZ, CAUSAL_L1, CAUSAL_LO_HZ,
    CAUSAL_MP_STRENGTH, CAUSAL_N_COCH_CH, CAUSAL_RMS_REF, CAUSAL_SMOOTH,
)
from spiking_ven.corpus import load_corpus
from spiking_ven.sw_encoder import (
    PRIMARY_RATE_HZ, SWEncoder, learn_relu, tm_basis, unfold, window_scale,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=128, help="primary AC units")
    p.add_argument("--l-frames", type=int, default=32, help="STRF length (2 ms frames)")
    p.add_argument("--epochs", type=int, default=CAUSAL_EPOCHS)
    p.add_argument("--lr", type=float, default=0.1,
                   help="gradient step (0.2 starts killing units)")
    p.add_argument("--reg-lr", type=float, default=None,
                   help="step for the L1 and roughness penalties (default: --lr)")
    p.add_argument("--l1", type=float, default=CAUSAL_L1, help="L1 on kernel entries")
    p.add_argument("--l2", type=float, default=0.0, help="L2 weight decay on kernel entries")
    p.add_argument("--smooth", type=float, default=CAUSAL_SMOOTH,
                   help="roughness penalty (first differences over channel and lag)")
    p.add_argument("--mp-strength", type=float, default=CAUSAL_MP_STRENGTH,
                   help="pull toward minimum phase per update")
    p.add_argument("--tied", action="store_true",
                   help="tied decoder (default: linear decoder with unit-norm rows)")
    p.add_argument("--threshold-pct", type=float, default=90.0,
                   help="higher-AC threshold: this percentile of the DoG output on song")
    p.add_argument("--global-threshold", action="store_true",
                   help="one threshold for all units (default: one per unit)")
    p.add_argument("--primary-spikes", action="store_true",
                   help="feed the DoG Poisson primary spikes (default: primary rates, as in "
                        "Schneider & Woolley's model)")
    p.add_argument("--train-corpus", default=None,
                   help="learn the STRFs on this unwarped multi-bird corpus "
                        "(scripts/build_song_corpus.py) instead of the motifs; the gain and "
                        "thresholds are still set on the motifs, which the network hears")
    p.add_argument("--patience", type=int, default=0,
                   help="stop when held-out error has not improved for this many epochs "
                        "(needs --train-corpus; 0 = run all epochs)")
    p.add_argument("--min-delta", type=float, default=0.002)
    p.add_argument("--smooth-freq-only", action="store_true",
                   help="roughness penalty across frequency only, not along lag")
    p.add_argument("--recon-tau-ms", type=float, default=None,
                   help="weight each lag's reconstruction error by exp(-lag/tau) "
                        "(default: uniform over the window)")
    p.add_argument("--predict-frames", type=int, default=0,
                   help="objective: predict the next N cochleagram frames from the code of "
                        "the past window (instead of reconstructing the window); 0 = off")
    p.add_argument("--checkpoint", default=None,
                   help="save the STRFs after every epoch here (default: <out>.ckpt.npz), "
                        "so the run can be stopped and plotted (scripts/plot_strfs.py)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the checkpoint (same arguments as the original run); "
                        "lower --epochs to the checkpoint's epoch to just finalize the encoder "
                        "from the best weights so far")
    p.add_argument("--init-scale", type=float, default=1.0,
                   help="scale of the random initial STRFs (0 = start from zero; needs "
                        "--init-bias > 0)")
    p.add_argument("--init-bias", type=float, default=0.0,
                   help="initial bias of every primary unit")
    p.add_argument("--conv-frames", type=int, default=0,
                   help="convolutional (overlap-add) reconstruction: each frame rebuilt once "
                        "from the codes of the next M time steps (decoder kernel M frames)")
    p.add_argument("--code-l1", type=float, default=0.0,
                   help="L1 penalty on the code (sparsity; used with --conv-frames)")
    p.add_argument("--tm-basis", default=None, metavar="LO,HI,N",
                   help="constrain each channel's temporal kernel to a Takenaka-Malmquist "
                        "basis with N real poles, time constants log-spaced LO..HI ms")
    p.add_argument("--act-target", type=float, default=None,
                   help="per-unit activity constraint: fraction of frames each unit is active "
                        "(each unit's threshold tracks the (1 - target) quantile of its input)")
    p.add_argument("--act-lr", type=float, default=0.1,
                   help="running-average rate of each unit's threshold toward the "
                        "(1 - target) quantile of its input, per bout")
    p.add_argument("--act-mode", choices=("threshold", "penalty", "auglag", "rate"),
                   default="threshold",
                   help="threshold: each unit's threshold tracks a quantile (no effect on STRF "
                        "shape); penalty: activity cost in the objective with per-unit "
                        "multipliers (dual ascent), which also reshapes the STRFs; auglag: "
                        "augmented Lagrangian (multipliers + quadratic term, growing mu) on the "
                        "fraction active; rate: same on each unit's mean output (target = "
                        "--act-target in rate units, e.g. 1)")
    p.add_argument("--act-mu", type=float, default=0.2, help="initial augmented-Lagrangian mu")
    p.add_argument("--unit-norm-strf", action="store_true",
                   help="each STRF = unit-norm shape x learned per-unit gain (decoder rows stay "
                        "unit-norm); penalties act on the shape only")
    p.add_argument("--act-warmup", type=int, default=3,
                   help="auglag: unconstrained epochs before the constraint switches on")
    p.add_argument("--act-mu-max", type=float, default=2.0)
    p.add_argument("--softplus-beta0", type=float, default=None,
                   help="start with softplus(beta z)/beta at this beta, sharpened each epoch "
                        "(--softplus-growth) until --softplus-beta-max, then ReLU")
    p.add_argument("--softplus-growth", type=float, default=1.25)
    p.add_argument("--softplus-beta-max", type=float, default=3000.0)
    p.add_argument("--softplus-hold", type=float, default=None,
                   help="ramp beta up to this value, hold until held-out error plateaus, then "
                        "anneal slowly to the ReLU")
    p.add_argument("--softplus-hold-patience", type=int, default=5)
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="std of Gaussian noise added to each unit's input before the "
                        "nonlinearity during training (sqrt(0.1) on normalised input = SNR 10)")
    p.add_argument("--output-noise", type=float, default=0.0,
                   help="std of Gaussian noise on each unit's rate before the decoder during "
                        "training (a gain floor; e.g. sqrt(0.1) * target for SNR 10)")
    p.add_argument("--al-exact", action="store_true",
                   help="faithful augmented Lagrangian: inner loop to plateau with lambda, mu "
                        "fixed, then lambda += mu * c, mu grown if the violation fell < 75%%")
    p.add_argument("--al-inner-max", type=int, default=8)
    p.add_argument("--al-plain", action="store_true",
                   help="rate mode: plain first-order dual ascent (lambda += act-dual-lr * "
                        "violation every --act-update-every bouts; no quadratic term)")
    p.add_argument("--scale-free-shape", action="store_true",
                   help="unit-norm STRFs: shape step uses the median gain, not each unit's own "
                        "(a collapsed unit can recover)")
    p.add_argument("--al-mu-growth", type=float, default=4.0)
    p.add_argument("--act-update-every", type=int, default=0,
                   help="update the activity multipliers every N bouts (0 = once per epoch)")
    p.add_argument("--softplus-anneal", type=float, default=1.15,
                   help="per-epoch beta growth after the plateau")
    p.add_argument("--act-dual-lr", type=float, default=1e-3,
                   help="dual-ascent step for the activity multipliers (penalty mode)")
    p.add_argument("--act-beta", type=float, default=0.01,
                   help="auglag: width of the smooth active-frame indicator, as a multiple of "
                        "each unit's input std (try ~0.3)")
    p.add_argument("--normalize-input", action="store_true",
                   help="divide centred windows by the corpus rms per-element std (global "
                        "scalar), so a random unit-norm STRF's output has variance ~1")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    corpus = load_corpus(a.motifs)
    sr = int(corpus.sr)
    cp = {"n_channels": CAUSAL_N_COCH_CH, "K_frames": a.l_frames,
          "frame_rate": CAUSAL_FRAME_RATE, "lo_hz": CAUSAL_LO_HZ, "hi_hz": CAUSAL_HI_HZ,
          "rms_ref": CAUSAL_RMS_REF}
    cochs = []
    for sig in corpus.signals():
        sig = sig / max(float(np.sqrt(np.mean(sig ** 2))), 1e-12) * CAUSAL_RMS_REF
        cochs.append(cochleagram(sig, sr, n_channels=cp["n_channels"],
                                 frame_rate=cp["frame_rate"], lo_hz=cp["lo_hz"],
                                 hi_hz=cp["hi_hz"]).astype(np.float64))
    train, val = cochs, None
    if a.train_corpus:
        d = np.load(a.train_corpus, allow_pickle=False)
        if int(d["sr"]) != sr:
            raise ValueError(f"{a.train_corpus}: sr {int(d['sr'])} != motifs' {sr}")
        train, val = [], []
        for s0, n0, held in zip(d["starts"], d["lengths"], d["val"]):
            x = d["audio"][s0:s0 + n0].astype(np.float64)
            x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12) * CAUSAL_RMS_REF
            c = cochleagram(x, sr, n_channels=cp["n_channels"], frame_rate=cp["frame_rate"],
                            lo_hz=cp["lo_hz"], hi_hz=cp["hi_hz"]).astype(np.float64)
            if c.shape[1] > a.l_frames:
                (val if held else train).append(c)
        print(f"training corpus {a.train_corpus}: {len(d['birds'])} birds, {len(train)} "
              f"bouts ({sum(c.shape[1] for c in train)} frames), {len(val)} held out")
    print(f"{len(cochs)} motifs, {sum(c.shape[1] for c in cochs)} frames (calibration); "
          f"training {a.k} STRFs x {cp['n_channels']} ch x {a.l_frames} frames")
    ckpt_path = a.checkpoint or a.out.replace(".npz", ".ckpt.npz")

    def save_ckpt(state):
        # Write then rename, so killing the run mid-write never leaves a broken checkpoint.
        tmp = ckpt_path.replace(".npz", ".tmp.npz")
        np.savez(tmp, frame_ms=1000.0 / cp["frame_rate"], **state)
        os.replace(tmp, ckpt_path)

    resume = None
    if a.resume and os.path.exists(ckpt_path):
        resume = dict(np.load(ckpt_path, allow_pickle=False))
        print(f"resuming from {ckpt_path} (epoch {int(resume['epoch'])})")
    elif a.resume:
        print(f"no checkpoint at {ckpt_path}: starting fresh")

    basis = None
    if a.tm_basis:
        lo, hi, nb = a.tm_basis.split(",")
        taus = np.geomspace(float(lo), float(hi), int(nb))
        basis = tm_basis(taus, a.l_frames, 1000.0 / cp["frame_rate"])
        print("TM basis time constants (ms):", np.round(taus, 2).tolist())

    in_scale = window_scale(train, a.l_frames) if a.normalize_input else 1.0
    if a.normalize_input:
        print(f"input normalisation: windows / {in_scale:.4g}")

    V, b, mu, Dk = learn_relu(train, K=a.k, L=a.l_frames, n_epochs=a.epochs, lr=a.lr,
                              reg_lr=a.reg_lr, seed=a.seed, mp_strength=a.mp_strength,
                              l1_V=a.l1, smooth_V=a.smooth, tied=a.tied, val_cochs=val,
                              patience=a.patience, min_delta=a.min_delta,
                              log_every=a.log_every,
                              smooth_axes=(1,) if a.smooth_freq_only else (1, 2),
                              recon_tau_ms=a.recon_tau_ms, predict_frames=a.predict_frames,
                              checkpoint=save_ckpt, resume=resume,
                              init_scale=a.init_scale, init_bias=a.init_bias,
                              conv_frames=a.conv_frames, code_l1=a.code_l1, l2_V=a.l2,
                              basis=basis, act_target=a.act_target, act_lr=a.act_lr,
                              act_beta=a.act_beta, act_mode=a.act_mode,
                              act_dual_lr=a.act_dual_lr, act_mu=a.act_mu,
                              act_mu_max=a.act_mu_max, softplus_beta0=a.softplus_beta0,
                              softplus_growth=a.softplus_growth,
                              softplus_beta_max=a.softplus_beta_max, act_warmup=a.act_warmup,
                              unit_norm_strf=a.unit_norm_strf, input_scale=in_scale,
                              softplus_hold=a.softplus_hold,
                              softplus_hold_patience=a.softplus_hold_patience,
                              softplus_anneal=a.softplus_anneal, input_noise=a.input_noise,
                              act_update_every=a.act_update_every,
                              output_noise=a.output_noise, al_exact=a.al_exact,
                              al_inner_max=a.al_inner_max, al_mu_growth=a.al_mu_growth,
                              scale_free_shape=a.scale_free_shape, al_plain=a.al_plain)
    W = V.reshape(a.k, -1)
    b = b.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    D, d0, scale = Dk.T.copy(), np.zeros_like(mu), in_scale
    if a.predict_frames or a.conv_frames:
        # The decoder predicts future frames, not the window: nothing to reconstruct with.
        D, d0 = None, None

    # The trained primary nonlinearity: a softplus kept to the end (no anneal) is the
    # encoder's nonlinearity too; otherwise ReLU.
    beta = (a.softplus_hold if a.softplus_beta0 is not None and a.softplus_hold is not None
            and a.softplus_anneal <= 1.0 else None)

    def nl(Z):
        return np.maximum(Z, 0.0) if beta is None else np.logaddexp(0.0, beta * Z) / beta

    # Units that never activate on the corpus are dropped (silent channels).
    R = np.concatenate([nl(W @ ((unfold(X, a.l_frames) - mu) / scale) + b)
                        for X in cochs], axis=1)
    keep = np.flatnonzero((R > 0).any(axis=1))
    gain = PRIMARY_RATE_HZ / float(R[keep].mean())
    enc = SWEncoder(W, b, scale, cp, primary_gain=gain, keep=keep, D=D, d0=d0, mu=mu,
                    primary_noise=a.primary_spikes, primary_beta=beta)
    print(f"kept {keep.size}/{a.k} units; primary gain {gain:.2f} Hz/unit; "
          f"active fraction {(R[keep] > 0).mean():.1%}")

    # Higher-AC thresholds from the DoG output on the song.
    drive = []
    for i, sig in enumerate(corpus.signals()):
        X = enc.cochleagram(sig, sr=sr)
        drive.append(enc.drive(X, seed=a.seed + i))
    drive = np.concatenate(drive, axis=1)
    enc.threshold = (float(np.percentile(drive, a.threshold_pct)) if a.global_threshold
                     else np.percentile(drive, a.threshold_pct, axis=1)[:, None])
    act = np.maximum(drive - enc.threshold, 0.0)
    rate = act.mean(axis=1) / act.mean() * 2.4
    print(f"higher-AC threshold {np.median(enc.threshold):.4f} (median; "
          f"{a.threshold_pct:g}th pct); unit rates at a 2.4 Hz mean: median "
          f"{np.median(rate):.2f}, 10-90% {np.percentile(rate, 10):.2f}-"
          f"{np.percentile(rate, 90):.2f} Hz; "
          f"active fraction {(act > 0).mean():.1%}; units ever active "
          f"{(act > 0).any(axis=1).sum()}/{keep.size}")
    enc.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
