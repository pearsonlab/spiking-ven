"""Cancellation figure with a training-pool rendition in the "Training song" column.

    python scripts/cancellation_rendition.py --model M.npz --cache stimuli.pkl \\
        --out-dir DIR [--rendition K | --median]

The template is an outlier for some auditory units (it is the loudest rendition), so this
draws the first column from another rendition of the training pool: its auditory spikes
and its waveform. ``--median`` picks the rendition whose mean BS rate is the median of the
pool (excluding the template). The reversed-motif and DAF columns stay the template's test
stimuli. The bottom row also shows the E population rate averaged over every rendition
of the pool (one noise draw each, ``--pool-trials`` to change). A first column shows the
same song through the network before training (``--no-before`` to omit): the learned
I_HVC weights reset to their start (no I_HVC -> E inhibition, uniform HVC -> I_HVC). The
pool-averaged rates are cached beside the model (``spiking_ven.sim_cache``).
"""

from __future__ import annotations

import argparse
import copy
import pickle

import numpy as np

from spiking_ven.corpus import load_corpus
from spiking_ven.figures.cancellation import make_figure
from spiking_ven.sim_cache import cached_arrays
from spiking_ven.vocal_error_net import VocalErrorNet


def untrained(ven: VocalErrorNet) -> VocalErrorNet:
    """``ven`` with its learned I_HVC weights reset to their starting values.

    Only the I_HVC pathway is reset: in the baseline the I -> E STDP rule is off
    (``rho_ns`` 0), so ``J_IE`` is still its initial draw.
    """
    built, ven._built = ven._built, None
    try:
        v = copy.deepcopy(ven)
    finally:
        ven._built = built
    v.J_HE = np.zeros_like(v.J_HE)
    if v.n_ih > 0:
        w0 = float(v.hvc_h_mean_ns)
        M = np.asarray(v.M_HH, dtype=bool)
        v.B_H = (M * w0).astype(np.float32)
        v.S_H = (M.sum(axis=1) * w0).astype(np.float32)
        v.I_H = np.zeros(v.n_ih, dtype=np.float32)
    return v


def pool_rate(ven, st, n_trials: int, seed: int) -> np.ndarray:
    """E population rate (Hz per ms) averaged over every rendition of the pool."""
    rng = np.random.default_rng(seed)
    return np.mean([ven.transform(st["hvc_on"], aud, rng=rng,
                                  hvc_prelude=st.get("hvc_prelude")).mean(axis=0)
                    for aud in st["correct_pool"] for _ in range(n_trials)], axis=0) * 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--motifs", default="outputs/motifs.npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rendition", type=int, default=None,
                   help="index into the training pool (0 is the template)")
    p.add_argument("--median", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-trials", type=int, default=1)
    p.add_argument("--no-before", action="store_true",
                   help="omit the before-training column")
    a = p.parse_args()
    with open(a.cache, "rb") as fh:
        st = pickle.load(fh)
    corpus = load_corpus(a.motifs)
    ti = int(st["train_idx"])
    order = [ti] + [i for i in range(corpus.n_motifs) if i != ti]   # pool index -> corpus
    T = int(st["T_song"])
    rates = np.array([np.asarray(x)[:, :T].mean() * 1000 for x in st["correct_pool"]])
    if a.median:
        k = 1 + int(np.argsort(rates[1:])[len(rates[1:]) // 2])
    else:
        k = int(a.rendition if a.rendition is not None else 1)
    print(f"rendition {k} (corpus motif {order[k]}): BS rate {rates[k]:.2f} Hz "
          f"(template {rates[0]:.2f}, pool median {np.median(rates[1:]):.2f})")
    st2 = dict(st)
    st2["aud_correct"] = st["correct_pool"][k]
    st2["sig_train"] = np.asarray(corpus.signal(order[k]), dtype=np.float64)
    ven = VocalErrorNet.load(a.model, seed=a.seed)
    before = None if a.no_before else untrained(ven)
    key = {"stimuli": str(a.cache), "pool_trials": a.pool_trials, "seed": a.seed}

    def simulate():
        out = {"trained": pool_rate(ven, st, a.pool_trials, a.seed + 1)}
        if before is not None:
            out["before"] = pool_rate(before, st, a.pool_trials, a.seed + 1)
        return out

    rates = cached_arrays(a.model, "pool_rates", {**key, "before": before is not None},
                          simulate)
    n = len(st["correct_pool"])
    lab = f"all renditions (n={n})"
    extra = {0: (rates["trained"], lab)}
    if before is not None:
        extra = {0: (rates["before"], lab), 1: (rates["trained"], lab)}
    make_figure(ven, st2, a.tag or f"_rendition{k}", out_dir=a.out_dir,
                extra_rates=extra, shown_label="shown rendition", ven_before=before)


if __name__ == "__main__":
    main()
