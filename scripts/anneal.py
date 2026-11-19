"""Continue training a model at a new I_HVC -> E learning rate (annealing), then report the
spread of the per-neuron singing / not-singing ratio.

    python scripts/anneal.py --model M.npz --eta 0.3 --n-rend 600 --out M2.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sweep  # type: ignore  # noqa: E402

from spiking_ven.training import train  # noqa: E402
from spiking_ven.vocal_error_net import VocalErrorNet  # noqa: E402


def spread(v, st, n=20, seed=6):
    """SD of log10(singing / not singing) over cells firing > 0.5 Hz, and the median ratio."""
    T, z = st["T_song"], np.zeros_like(st["aud_correct"])
    rng = np.random.default_rng(seed)
    q = np.mean([v.transform(st["hvc_off"], z, rng=rng)[:, :T] for _ in range(n)],
                axis=(0, 2)) * 1000
    s = np.mean([v.transform(st["hvc_on"], st["aud_correct"], rng=rng,
                             hvc_prelude=st["hvc_prelude"])[:, :T] for _ in range(n)],
                axis=(0, 2)) * 1000
    m = (q > 0.5) & (s > 0.0)
    d = np.log10(s[m] / q[m])
    return {"sd_log10": float(d.std()), "fold": float(10 ** d.std()),
            "median_ratio": float(10 ** np.median(d)),
            "silenced": int(((q > 0.5) & (s == 0)).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--n-rend", type=int, default=600)
    p.add_argument("--report-every", type=int, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    st = sweep._stimuli()
    v = VocalErrorNet.load(a.model, seed=123)
    v.eta_h_ns = a.eta
    out = {"start": spread(v, st)}
    print("start", json.dumps(out["start"]), flush=True)
    done = 0
    while done < a.n_rend:
        k = min(a.report_every, a.n_rend - done)
        train(v, st, k, seed=123 + done, verbose=False, log_at=())
        done += k
        out[str(done)] = spread(v, st)
        print(done, json.dumps(out[str(done)]), flush=True)
    v.save(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
