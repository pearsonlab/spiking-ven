"""How much of the song-evoked E response does the HVC pathway cancel, and is it time-locked?

    python scripts/cancellation_check.py --out fig.png \\
        --model A.npz --cache stimuli_cache_A.pkl --label A [--model B.npz ...]

For each model, trial-averaged E PSTHs (cells with auditory input, ``--trials`` noise draws)
over the motif, for: the trained song with HVC aligned (singing), with HVC shifted by
``--shift-ms`` (singing, wrong timing), with no HVC (hearing the song, not singing), the
time-reversed motif with HVC, and HVC alone (singing, no sound). Scores, each on the excess
over the HVC-only baseline (the no-HVC case over the silent baseline):

    cancelled  = 1 - excess(aligned) / excess(no HVC)     fraction of the song response removed
    timed      = 1 - excess(aligned) / excess(shifted)    how much of that needs the timing
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402

#: ms excluded from the scores at motif start (simulation-onset transient without prelude).
SKIP = 20

COLORS = {"aligned": "#2a78d6", "shifted": "#eb6834", "no HVC": "#5f5e58",
          "reversed": "#1baf7a", "HVC only": "#b8b6ae"}


def psths(ven, st, *, n_trials, shift_ms, seed=7):
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(np.asarray(ven.A).any(axis=1))
    T = int(st["T_song"])
    pre = st.get("hvc_prelude")
    hvc, hoff = st["hvc_on"], st["hvc_off"]
    hvc_shift = np.roll(hvc, shift_ms, axis=1)
    silent = np.zeros_like(st["aud_correct"])
    conds = {"aligned": (hvc, st["aud_correct"], pre),
             "shifted": (hvc_shift, st["aud_correct"], pre),
             "no HVC": (hoff, st["aud_correct"], None),
             "reversed": (hvc, st["aud_reversed"], pre),
             "HVC only": (hvc, silent, pre),
             "silent": (hoff, silent, None)}
    out = {}
    for name, (h, a, p) in conds.items():
        r = np.mean([ven.transform(h, a, rng=rng, hvc_prelude=p)[cells, :T].mean(axis=0)
                     for _ in range(n_trials)], axis=0)
        out[name] = r * 1000.0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--cache", action="append", required=True)
    p.add_argument("--label", action="append", required=True)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--shift-ms", type=int, default=150)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = len(a.model)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, squeeze=False)
    for ax, m, c, lab in zip(axes[:, 0], a.model, a.cache, a.label):
        with open(c, "rb") as fh:
            st = pickle.load(fh)
        ven = VocalErrorNet.load(m, seed=42)
        r = psths(ven, st, n_trials=a.trials, shift_ms=a.shift_ms)
        # Scores skip the first SKIP ms: runs without the HVC prelude start with a
        # simulation-onset transient that is not a response to the song.
        sk = slice(SKIP, None)
        base_s, base_h = r["silent"][sk].mean(), r["HVC only"][sk].mean()
        ex = {k: r[k][sk].mean() - base_h for k in ("aligned", "shifted", "reversed")}
        ex["no HVC"] = r["no HVC"][sk].mean() - base_s
        canc = 1 - ex["aligned"] / ex["no HVC"]
        timed = 1 - ex["aligned"] / ex["shifted"]
        line = (f"{lab}: excess Hz aligned {ex['aligned']:.2f}, shifted {ex['shifted']:.2f}, "
                f"no HVC {ex['no HVC']:.2f}, reversed {ex['reversed']:.2f}; "
                f"cancelled {canc:.0%}, timed {timed:.0%}")
        print(line, flush=True)
        for k in ("no HVC", "shifted", "reversed", "aligned", "HVC only"):
            ax.plot(gaussian_filter1d(r[k], 3), color=COLORS[k], lw=1.6 if k == "aligned"
                    else 1.0, label=k)
        ax.set_title(f"{lab}: {canc:.0%} of the song response cancelled, "
                     f"{timed:.0%} of it time-locked", loc="left", fontsize=9)
        ax.set_ylabel("E rate (Hz)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(fontsize=8, ncols=5, loc="upper right", frameon=False)
    axes[-1, 0].set_xlabel("time in motif (ms)")
    fig.suptitle(f"Trial-averaged E PSTH, cells with auditory input ({a.trials} trials; "
                 f"HVC shifted by {a.shift_ms} ms)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
