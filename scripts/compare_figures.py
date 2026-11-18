"""Figures comparing the numpy engine with the Brian2 port: metrics, DAF PSTHs, runtimes.

    python scripts/compare_figures.py --models outputs/compare/numpy.npz ... \
        --bench outputs/benchmarks --out-dir outputs/figures

``--models`` are ``collect_metrics.py`` outputs, in legend order; ``--bench`` holds the
``benchmark_engines.py`` JSON files (and ``training_runs.json``, if present).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spiking_ven.targets import BIOLOGICAL_TARGETS as T  # noqa: E402

# Categorical slots 1-4 of the validated default palette (fixed order), and text inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, BAND = "#1f1f1e", "#5f5e58", "#e4e3df", "#ecebe7"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200,
})

#: (metric key, panel title, unit, target band (lo, hi), scale)
METRICS = [
    ("baseline_hz_all", "Rate while singing", "Hz", T["singing_rate_hz"]),
    ("reversed_over_trained", "Reversed / trained song", "ratio", (1.0, None)),
    ("responder_fraction", "DAF responders", "fraction",
     (T["responder_fraction"] - 0.1, T["responder_fraction"] + 0.1)),
    ("latency_ms", "Response latency", "ms",
     (T["response_latency_ms"][0] - T["response_latency_ms"][1],
      T["response_latency_ms"][0] + T["response_latency_ms"][1])),
    ("duration_ms", "Response duration", "ms",
     (T["response_duration_ms"][0] - T["response_duration_ms"][1],
      T["response_duration_ms"][0] + T["response_duration_ms"][1])),
    ("peak_latency_ms", "Peak time after onset", "ms", T["responder_peak_latency_ms"]),
    ("late_over_peak", "Late / peak (100-150 ms)", "ratio",
     (None, T["responder_late_over_peak_max"])),
    ("extra_spikes_responders", "Extra spikes per burst", "spikes",
     T["extra_spikes_per_burst"]),
]


def load_models(paths):
    out = []
    for p in paths:
        d = np.load(p)
        out.append({"label": str(d["label"]), "m": json.loads(str(d["metrics"])),
                    "psth": d["psth_responders"], "t": d["t_ms"]})
    return out


def fig_metrics(models, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
    n = len(models)
    for ax, (key, title, unit, (lo, hi)) in zip(axes.flat, METRICS):
        vals = [mod["m"][key] for mod in models]
        lo_ax = min([v for v in vals] + [lo if lo is not None else np.inf])
        hi_ax = max([v for v in vals] + [hi if hi is not None else -np.inf])
        pad = 0.15 * (hi_ax - lo_ax or 1.0)
        band_lo = lo if lo is not None else lo_ax - pad
        band_hi = hi if hi is not None else hi_ax + pad
        ax.axhspan(band_lo, band_hi, color=BAND, zorder=0, lw=0)
        for i, (mod, v) in enumerate(zip(models, vals)):
            ax.plot([i], [v], "o", ms=8, color=SERIES[i], mec="white", mew=2, zorder=3)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(lo_ax - pad, hi_ax + pad)
        ax.set_xticks([])
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    handles: list[Any] = [plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[i], mec="white",
                          mew=2, label=m["label"]) for i, m in enumerate(models)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="biological target"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5)
    fig.suptitle("Model metrics against the biological targets (Mandelblat-Cerf 2014)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def fig_psth(models, path: Path, bin_ms: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axvspan(0, 50, color=BAND, lw=0, zorder=0)
    ax.text(25, 1.0, "noise burst", transform=ax.get_xaxis_transform(), ha="center",
            va="top", color=INK2, fontsize=8)
    for i, mod in enumerate(models):
        t, y = mod["t"], mod["psth"]
        nb = len(t) // bin_ms
        tb = t[: nb * bin_ms].reshape(nb, bin_ms).mean(axis=1)
        yb = y[: nb * bin_ms].reshape(nb, bin_ms).mean(axis=1)
        yb = yb - yb[tb < 0].mean()
        ax.plot(tb, yb, color=SERIES[i], lw=2, label=mod["label"])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("time from noise onset (ms)")
    ax.set_ylabel("rate change, responders (Hz)")
    ax.set_title(f"DAF response, responders' mean PSTH ({bin_ms} ms bins)", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_runtime(bench_dir: Path, path: Path) -> None:
    rows = []
    order = [("numpy.json", "numpy engine, 1 ms"),
             ("brian2_dt1.json", "Brian2 (Cython), 1 ms"),
             ("brian2_dt0.1.json", "Brian2 (Cython), 0.1 ms"),
             ("brian2_numpy_codegen_dt0.1.json", "Brian2 (numpy codegen), 0.1 ms"),
             ("brian2_standalone_dt0.1.json", "Brian2 (C++ standalone), 0.1 ms")]
    for fname, label in order:
        f = bench_dir / fname
        if f.exists():
            rows.append((label, json.loads(f.read_text())))
    runs_f = bench_dir / "training_runs.json"
    runs = json.loads(runs_f.read_text()) if runs_f.exists() else []

    fig, axes = plt.subplots(1, 2 if runs else 1, figsize=(10 if runs else 6, 3.2),
                             squeeze=False)
    ax = axes[0, 0]
    y = np.arange(len(rows))[::-1]
    vals = [r["transform_s"] for _, r in rows]
    ax.barh(y, vals, height=0.6, color=SERIES[0])
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:.2f} s", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlabel("wall time per rendition (s), inference")
    ax.set_title("One 1.25 s rendition", loc="left")
    ax.set_xlim(0, max(vals) * 1.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if runs:
        ax = axes[0, 1]
        y = np.arange(len(runs))[::-1]
        vals = [r["minutes"] for r in runs]
        ax.barh(y, vals, height=0.6, color=SERIES[0])
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:.1f} min", va="center", color=INK, fontsize=8)
        ax.set_yticks(y, [r["label"] for r in runs])
        ax.set_xlabel("wall time (min)")
        ax.set_title("Full run: train 600 + score 100 DAF trials", loc="left")
        ax.set_xlim(0, max(vals) * 1.3)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--bench", default="outputs/benchmarks")
    p.add_argument("--out-dir", default="outputs/figures")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(a.models)
    fig_metrics(models, out / "compare_metrics.png")
    fig_psth(models, out / "compare_daf_psth.png")
    fig_runtime(Path(a.bench), out / "compare_runtime.png")
    print("wrote", *sorted(str(f) for f in out.glob("compare_*.png")))


if __name__ == "__main__":
    main()
