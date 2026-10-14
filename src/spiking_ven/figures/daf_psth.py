"""Responder-averaged DAF PSTH, against what Figure 8E of the paper shows.

The raster figure shows that a DAF response exists; it does not show its time course,
which is what the shape assertions in ``evaluate.daf_response`` are about. This draws it.

Mandelblat-Cerf et al. 2014 Fig. 8E is NOT reproduced here. Their data is not in this
repository, so the panel would have to be invented. What is drawn instead are the three
things that can be read off the published panel -- peak ~19-20 Hz, occurring ~40-60 ms
after noise onset, back to ~0 by ~120-150 ms -- as annotated reference marks, labelled as
a figure reading rather than a measurement.

Public API
----------
make_psth_figure(ven, encoder, st, out_dir, *, tag="", ...)
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..constants import DAF_WINDOW_S  # noqa: E402
from ..evaluate import (  # noqa: E402
    BIOLOGICAL_TARGETS,
    PEAK_BIN_MS,
    daf_responders,
    daf_trials,
)

__all__ = ["make_psth_figure"]


def make_psth_figure(ven, encoder, st, out_dir, *, tag="", pre_ms=300, post_ms=400,
                     n_trials=20, seed=42):
    # Stay inside the motif. The burst is at DAF_WINDOW_S[0]; running past T_song reaches
    # song offset, where HVC drive stops and E is released from inhibition -- a large
    # rise that has nothing to do with DAF and only invites misreading.
    onset = int(DAF_WINDOW_S[0] * 1000)
    post_ms = min(post_ms, int(st["T_song"]) - onset - PEAK_BIN_MS)
    pre_ms = min(pre_ms, onset)

    # The same trials and responder criterion as evaluate.daf_response.
    trials = daf_trials(ven, encoder, st, n_trials=n_trials, seed=seed)
    resp, _, _ = daf_responders(trials)
    n_resp = int(resp.sum())

    # Per-neuron change from its own pre-onset mean, binned at PEAK_BIN_MS, averaged over
    # responders.
    spk = trials[:, :, onset - pre_ms:onset + post_ms]
    nb = spk.shape[2] // PEAK_BIN_MS
    binned = spk[:, :, :nb * PEAK_BIN_MS].reshape(n_trials, ven.n_e, nb, PEAK_BIN_MS)
    per_neuron = binned.mean(axis=(0, 3)) * 1000.0            # (n_e, nb) Hz
    base = per_neuron[:, :pre_ms // PEAK_BIN_MS].mean(axis=1, keepdims=True)
    d = (per_neuron - base)[resp]
    t_ms = (np.arange(nb) * PEAK_BIN_MS) - pre_ms
    mean, sem = d.mean(axis=0), d.std(axis=0) / np.sqrt(max(d.shape[0], 1))

    pk_lo, pk_hi = BIOLOGICAL_TARGETS["responder_peak_latency_ms"]
    paper_peak = BIOLOGICAL_TARGETS["responder_peak_change_hz"]
    post = t_ms >= 0
    peak_hz = float(mean[post].max())
    peak_at = int(t_ms[post][int(np.argmax(mean[post]))])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, zoom in zip(axes, (False, True)):
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.axvspan(0, 50, color="0.9", zorder=0)                    # the 50 ms burst
        ax.fill_between(t_ms, mean - sem, mean + sem, color="C0", alpha=0.25, lw=0)
        ax.plot(t_ms, mean, color="C0", lw=1.4,
                label=f"model, responder average (n = {n_resp})")
        # Reference marks READ OFF Fig. 8E -- not their data, which is not in this repo.
        ax.axvspan(pk_lo, pk_hi, color="C3", alpha=0.10, lw=0)
        ax.plot([(pk_lo + pk_hi) / 2], [paper_peak], marker="v", ms=9, color="C3",
                linestyle="none",
                label=f"Fig. 8E reading: ~{paper_peak:.0f} Hz at ~{pk_lo:.0f}–{pk_hi:.0f} ms")
        ax.set_xlabel("time from noise onset (ms)")
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.axvline(0.0, color="0.4", lw=0.8, ls=":")
        if zoom:
            ax.set_ylim(-3, 25)
            ax.set_title("same trace on the paper's y-scale", fontsize=10)
        else:
            ax.set_ylabel("change in firing rate (Hz)")
            ax.set_title(f"model peak {peak_hz:.0f} Hz at +{peak_at} ms", fontsize=10)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(
        "DAF response time course — shaded band is the 50 ms noise burst, "
        f"{PEAK_BIN_MS} ms bins\n"
        "Fig. 8E is not redrawn: only the values readable from the published panel are "
        "marked, and the paper states no bin width for it.\n"
        "Fluctuations away from the burst are song-driven: these are responses during "
        "singing, and song cancellation is imperfect.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"daf_psth{tag}.png", out / f"daf_psth{tag}.pdf"]
    for pth in paths:
        fig.savefig(pth, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return paths
