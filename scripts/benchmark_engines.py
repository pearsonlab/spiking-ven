"""Wall-clock cost of one rendition, per simulation engine.

Times ``fit`` (training, STDP on) and ``transform`` (inference) on one rendition of the
real stimuli -- 200 ms of intro notes plus a 1048 ms motif-and-silence window -- and
writes the medians to a JSON file. The same code times the numpy engine of spiking-ven
<= 0.6 (run with that version on ``PYTHONPATH``; its ``VocalErrorNet`` has the same
``fit``/``transform``) and the Brian2 engine.

    python scripts/benchmark_engines.py --engine brian2 --dt-ms 0.1 --out bench.json
    python scripts/benchmark_engines.py --engine brian2-standalone --dt-ms 0.1 ...

``brian2-standalone`` builds the same equations for Brian2's C++ standalone device and
runs ``--n-rend`` inference renditions back to back in one simulation (standalone code
cannot restart between renditions), reporting the per-rendition share of the run time
and, separately, the one-off compile time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _stimuli():
    from spiking_ven.causal_filters import CausalFilterEncoder
    from spiking_ven.stimuli import build_stimuli
    enc = CausalFilterEncoder.load("outputs/causal_encoder.npz")
    st = build_stimuli(enc, "outputs/motifs.npz", sr=16000, seed=42, n_hvc=600,
                       n_ista=50, verbose=False)
    return enc, st


def _time(fn, n_rep: int) -> list[float]:
    out = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def bench_runtime(engine: str, dt_ms: float, n_rep: int) -> dict:
    """numpy engine, or Brian2 runtime mode (Cython, or Brian2's numpy codegen)."""
    from spiking_ven.vocal_error_net import VocalErrorNet
    enc, st = _stimuli()
    kw = {} if engine == "numpy" else {"dt_ms": dt_ms}
    if engine == "brian2-numpy-codegen":
        from brian2 import prefs
        prefs.codegen.target = "numpy"
    ven = VocalErrorNet(n_aud=enc.n_channels, seed=42, **kw)
    hvc, aud, pre = st["hvc_on"], st["aud_correct"], st["hvc_prelude"]
    t0 = time.perf_counter()
    ven.fit(hvc, aud, hvc_prelude=pre)                 # first call: build (+ compile)
    first = time.perf_counter() - t0
    fit = _time(lambda: ven.fit(hvc, aud, hvc_prelude=pre), n_rep)
    tr = _time(lambda: ven.transform(hvc, aud, hvc_prelude=pre), n_rep)
    return {"first_call_s": first, "fit_s": float(np.median(fit)),
            "transform_s": float(np.median(tr)), "fit_all": fit, "transform_all": tr}


def bench_standalone(dt_ms: float, n_rend: int, build_dir: str) -> dict:
    """The same equations on Brian2's C++ standalone device, ``n_rend`` renditions in one run."""
    from brian2 import device, ms, set_device
    from spiking_ven.network import build_network, spike_events
    from spiking_ven.vocal_error_net import VocalErrorNet
    enc, st = _stimuli()
    set_device("cpp_standalone", build_on_run=False, directory=build_dir)
    ven = VocalErrorNet(n_aud=enc.n_channels, seed=42, dt_ms=dt_ms)
    one_hvc = np.concatenate([st["hvc_prelude"], st["hvc_on"]], axis=1).astype(np.float32)
    one_aud = np.concatenate([np.zeros((enc.n_channels, st["hvc_prelude"].shape[1])),
                              st["aud_correct"]], axis=1).astype(np.float32)
    hvc, aud = np.tile(one_hvc, n_rend), np.tile(one_aud, n_rend)
    o = build_network(ven.namespace(), ven.weights(), n_aud=ven.n_aud, n_hvc=ven.n_hvc,
                      aud_delay=ven.aud_delay_ms * ms, hvc_delay=ven.hvc_delay_ms * ms,
                      dt=dt_ms * ms)
    o["aud"].set_spikes(*spike_events(aud))
    o["hvc"].set_spikes(*spike_events(hvc))
    o["net"].run(hvc.shape[1] * ms)
    t0 = time.perf_counter()
    device.build(directory=build_dir, compile=True, run=False)
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    device.run(directory=build_dir)
    run_s = time.perf_counter() - t0
    return {"compile_s": compile_s, "run_s": run_s, "n_rend": n_rend,
            "transform_s": run_s / n_rend,
            "e_rate_hz": float(o["mon_e"].num_spikes / ven.n_e / (hvc.shape[1] * 1e-3))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True,
                   choices=["numpy", "brian2", "brian2-numpy-codegen", "brian2-standalone"])
    p.add_argument("--dt-ms", type=float, default=0.1)
    p.add_argument("--n-rep", type=int, default=5)
    p.add_argument("--n-rend", type=int, default=50, help="standalone: renditions per run")
    p.add_argument("--build-dir", default="outputs/benchmarks/standalone_build")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.engine == "brian2-standalone":
        r = bench_standalone(a.dt_ms, a.n_rend, a.build_dir)
    else:
        r = bench_runtime(a.engine, a.dt_ms, a.n_rep)
    r.update(engine=a.engine, dt_ms=None if a.engine == "numpy" else a.dt_ms)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_all")}, indent=2))


if __name__ == "__main__":
    main()
