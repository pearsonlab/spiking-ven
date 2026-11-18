"""The training protocol: renditions of the tutor song with HVC input, STDP on I -> E.

Rendition 1 carries a silent burn-in that ends with the introductory notes; every later
rendition runs the intro notes as a prelude, then a motif from the training pool with
HVC input, then silence. Each rendition starts the network from rest with the weights
learned so far.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LOG_AT", "train"]

#: Renditions at which progress is measured (and logged).
LOG_AT = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600)


def train(ven, st: dict, n_rend: int, *, seed: int = 42, log_at=LOG_AT,
          verbose: bool = True) -> dict:
    """Train ``ven`` in place for ``n_rend`` renditions of ``st`` (``build_stimuli``).

    At each rendition in ``log_at`` (and the last), the E rate to the trained and the
    reversed song is measured with a separate noise stream, so monitoring does not change
    what training sees. Returns those measurements: ``rendition``, ``trained_hz``,
    ``reversed_hz`` and ``j_ie_mean`` (nS), one entry per measurement.
    """
    monitor_rng = np.random.default_rng(seed + 20_000)
    pre = st["hvc_prelude"]
    out: dict[str, list] = {"rendition": [], "trained_hz": [], "reversed_hz": [],
                            "j_ie_mean": []}

    def measure(r: int) -> None:
        trained = float(ven.transform(st["hvc_on"], st["aud_correct"], rng=monitor_rng,
                                      hvc_prelude=pre).mean() * 1000)
        reversed_ = float(ven.transform(st["hvc_on"], st["aud_reversed"], rng=monitor_rng,
                                        hvc_prelude=pre).mean() * 1000)
        out["rendition"].append(r)
        out["trained_hz"].append(trained)
        out["reversed_hz"].append(reversed_)
        out["j_ie_mean"].append(float(np.mean(ven.J_IE)))
        if verbose:
            ratio = reversed_ / trained if trained > 0 else float("nan")
            print(f"  r{r:<5} trained={trained:.2f}  reversed={reversed_:.2f}  "
                  f"ratio={ratio:.3f}  J_IE={out['j_ie_mean'][-1]:.5f}", flush=True)

    pool = st["correct_pool"]
    for r in range(n_rend):
        if r == 0:
            ven.fit(st["hvc_burn"], st["aud_burn"])
        else:
            ven.fit(st["hvc_on"], pool[r % len(pool)], hvc_prelude=pre)
        if (r + 1) in log_at or r == n_rend - 1:
            measure(r + 1)
    return {k: np.asarray(v) for k, v in out.items()}
