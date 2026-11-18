"""Staged calibration of an E/I-balanced vocal error network, on untrained networks.

    python scripts/staged_balance.py balance --grid '{"i_drive_e_pa": [453, 600], ...}'
    python scripts/staged_balance.py aud     --base '{...}' --grid '{"aud_mean_ns": [...]}'
    python scripts/staged_balance.py hvc     --base '{...}' --grid '{"c_hvc": [...], ...}'

Stage ``balance``: no HVC, no sound. E and I rates, and E's rate with I -> E removed (how much
work the inhibition does). Stage ``aud``: E's spike-triggered rate change after an input
spike on one of its auditory channels (song, no HVC). Stage ``hvc``: HVC on, no sound --
mean I rate against silence, and how time-modulated single I cells are (CV over the motif in
20 ms bins, 10 trials). Each grid point is ``VocalErrorNet`` keyword overrides on ``--base``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # type: ignore  # noqa: E402

from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402


def _net(st, kw):
    return VocalErrorNet(n_aud=st["n_channels"], seed=42, **kw)


def balance(st, kw, n=3):
    z = np.zeros_like(st["aud_correct"])
    v = _net(st, kw)
    rng = np.random.default_rng(1)
    e, i = zip(*[v.transform_all(st["hvc_off"], z, rng=rng) for _ in range(n)])
    re, ri = np.mean(e) * 1000, np.mean(i) * 1000
    v.J_IE = np.zeros_like(v.J_IE)
    e0 = np.mean([v.transform(st["hvc_off"], z, rng=rng) for _ in range(n)]) * 1000
    return f"E {re:5.2f} Hz  I {ri:5.1f} Hz  | E without I->E {e0:5.1f} Hz"


def aud(st, kw, n=5, win=30):
    """Mean E rate in the ``win`` ms after a spike on one of the cell's channels, minus its
    rate in silence (song, no HVC)."""
    T, z = st["T_song"], np.zeros_like(st["aud_correct"])
    v = _net(st, kw)
    rng = np.random.default_rng(2)
    base = np.mean([v.transform(st["hvc_off"], z, rng=rng)[:, :T].mean() for _ in range(n)])
    song = np.mean([v.transform(st["hvc_off"], st["aud_correct"], rng=rng)
                    for _ in range(n)], axis=0)            # (n_e, T) trial-mean
    aud_sp = st["aud_correct"][:, :T] > 0
    A = np.asarray(v.A) > 0
    vals = []
    for e in range(v.n_e):
        t = np.flatnonzero(aud_sp[A[e]].any(axis=0))
        t = t[t + win < T]
        if t.size:
            vals.append(np.mean([song[e, s:s + win].mean() for s in t]))
    sta = np.mean(vals) - base
    return (f"silent {base * 1000:5.2f} Hz  song-driven mean {song[:, :T].mean() * 1000:5.2f} Hz"
            f"  | after an input spike +{sta * 1000:5.2f} Hz ({win} ms)")


def hvc(st, kw, n=10, bin_ms=20):
    T, z = st["T_song"], np.zeros_like(st["aud_correct"])
    v = _net(st, kw)
    rng = np.random.default_rng(3)
    i_off = np.mean([v.transform_all(st["hvc_off"], z, rng=rng)[1][:, :T]
                     for _ in range(n)], axis=0)
    e_on, i_on = zip(*[v.transform_all(st["hvc_on"], z, rng=rng, hvc_prelude=st["hvc_prelude"])
                       for _ in range(n)])
    i_on = np.mean(i_on, axis=0)[:, :T]
    nb = T // bin_ms
    r = i_on[:, :nb * bin_ms].reshape(i_on.shape[0], nb, bin_ms).mean(2) * 1000
    cv = r.std(1) / np.maximum(r.mean(1), 1e-9)
    return (f"I silent {i_off.mean() * 1000:5.1f} Hz  singing {i_on.mean() * 1000:5.1f} Hz "
            f"({i_on.mean() / max(i_off.mean(), 1e-12):4.2f}x) | single-I CV {np.median(cv):4.2f}"
            f"  peak/mean {np.median(r.max(1) / np.maximum(r.mean(1), 1e-9)):4.1f} | E singing "
            f"{np.mean([e[:, :T].mean() for e in e_on]) * 1000:5.2f} Hz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["balance", "aud", "hvc"])
    p.add_argument("--base", default="{}")
    p.add_argument("--grid", default="{}")
    a = p.parse_args()
    st = sweep._stimuli()
    base, grid = json.loads(a.base), json.loads(a.grid)
    fn = {"balance": balance, "aud": aud, "hvc": hvc}[a.stage]
    for vals in itertools.product(*grid.values()) if grid else [()]:
        kw = {**base, **dict(zip(grid, vals))}
        # SDs follow their means (half the mean, as the shipped HVC and auditory weights)
        for m, s in (("hvc_mean_ns", "hvc_sd_ns"), ("aud_mean_ns", "aud_sd_ns"),
                     ("ei_mean_ns", "ei_sd_ns")):
            if m in kw and s not in kw:
                kw[s] = kw[m] * (0.1 if m == "ei_mean_ns" else 0.5)
        print(f"{json.dumps(dict(zip(grid, vals)))}\n   {fn(st, kw)}", flush=True)


if __name__ == "__main__":
    main()
