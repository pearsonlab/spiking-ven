"""Diagnostic figures for the causal minimum-phase filter bank.

Two figures, because they answer different questions:

``causal_filters_kernels``
    Every kernel as a spectrotemporal image -- frequency against lag, most recent frame
    at the right edge. What each filter is tuned to.

``causal_filters_summary``
    Where the energy sits in time, how tuning is spread over frequency, and how evenly
    the bank is used. The first panel is the one the minimum-phase constraint exists
    for: an unconstrained kernel puts its energy mid-window (~L/2, near-linear phase),
    which is where ~30 ms of encoder latency came from.

Public API
----------
make_filter_figures(encoder, out_dir, *, tag="", sr=16000, patch_encoder=None)
    Render both. ``patch_encoder`` is optional; when given, the summary overlays the
    Olshausen-Field atoms' lag profile for comparison.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..filterbank import centre_frequencies  # noqa: E402

__all__ = ["make_filter_figures"]


def _lag_energy(A):
    """(K, L) per-lag energy, each kernel normalised to sum 1. ``A`` is (K, C, L)."""
    E = (np.asarray(A, dtype=np.float64) ** 2).sum(axis=1)
    return E / np.maximum(E.sum(axis=1, keepdims=True), 1e-30)


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return paths


def _kernel_figure(V, cf_hz, dt_ms, out_dir, tag):
    K, C, L = V.shape
    # Sort by peak channel so the bank reads as a tonotopic sheet rather than a scramble;
    # ties break by lag centroid, which groups fast and slow filters. Peak, not the
    # energy-weighted centroid: these kernels are broadband, so every centroid lands
    # mid-band and the labels all come out the same.
    w = (V ** 2).sum(axis=2)
    bf = w.argmax(axis=1)
    lagc = (_lag_energy(V) * np.arange(L)).sum(axis=1)
    order = np.lexsort((lagc, bf))

    ncol = int(np.ceil(np.sqrt(K)))
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.35 * ncol, 1.25 * nrow))
    axes = np.atleast_1d(axes).ravel()
    lags = np.arange(-L, 0) * dt_ms          # -32 .. -2 ms, present at the right edge
    # Robust colour limit. Scaling each panel by its own max hides everything else when
    # one frame dominates -- which it does here: the minimum-phase pull concentrates
    # energy at the newest frame by construction.
    for ax, k in zip(axes, order):
        A = V[k]
        m = float(np.percentile(np.abs(A), 99)) or 1.0
        ax.imshow(A, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-m, vmax=m,
                  extent=[lags[0], 0, 0, C])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{cf_hz[bf[k]]:.0f} Hz", fontsize=5.5, pad=1.5)
    for ax in axes[K:]:
        ax.axis("off")
    newest = _lag_energy(V).mean(axis=0)[-1]
    fig.suptitle(f"Causal minimum-phase filter bank — {K} kernels, "
                 f"{C} channels x {L} frames ({L * dt_ms:.0f} ms of past)\n"
                 "red +, blue −; time runs left to right, the present is the right edge. "
                 f"Colour clipped at the 99th percentile; the newest frame holds "
                 f"{newest:.0%} of the mean kernel energy.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_dir, f"causal_filters_kernels{tag}")


def _summary_figure(V, cf_hz, dt_ms, out_dir, tag, patch_A=None):
    K, C, L = V.shape
    E = _lag_energy(V)
    lags = (np.arange(L) - L) * dt_ms + dt_ms      # most recent frame at 0

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.plot(lags, E.T, color="0.75", lw=0.5, alpha=0.6)
    ax.plot(lags, E.mean(axis=0), color="C0", lw=2.2, label="causal (min-phase)")
    if patch_A is not None:
        Ep = _lag_energy(patch_A)
        ax.plot(lags, Ep.mean(axis=0), color="C3", lw=2.2, ls="--",
                label="Olshausen-Field patches")
        ax.axvline(lags[int(round((Ep * np.arange(L)).sum(axis=1).mean()))],
                   color="C3", lw=0.8, ls=":")
    ax.axvline(lags[int(round((E * np.arange(L)).sum(axis=1).mean()))],
               color="C0", lw=0.8, ls=":")
    ax.set_xlabel("lag from present (ms)")
    ax.set_ylabel("fraction of kernel energy")
    ax.set_title("Where the energy sits in time\n(dotted = mean centroid)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    # Peak channel, the same statistic the kernel grid sorts and labels by. The
    # energy-weighted centroid disagrees with it here and is the wrong summary: these
    # kernels are broadband, so every centroid lands mid-band whatever the peak does.
    peak_hz = cf_hz[(V ** 2).sum(axis=2).argmax(axis=1)]
    ax.hist(peak_hz, bins=np.geomspace(cf_hz[0], cf_hz[-1], 17), color="C0",
            edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("peak channel frequency (Hz)")
    ax.set_ylabel("filters")
    ax.set_title(f"Frequency tuning of the {K} filters\n(peak channel, log axis)",
                 fontsize=9)

    ax = axes[2]
    F = V.reshape(K, -1)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-30)
    G = np.abs(F @ F.T)
    np.fill_diagonal(G, 0.0)
    ax.hist(G[np.triu_indices(K, 1)], bins=40, color="0.5", edgecolor="white")
    s = np.linalg.svd(F, compute_uv=False) ** 2
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    ax.set_xlabel("|correlation| between filter pairs")
    ax.set_ylabel("pairs")
    ax.set_title(f"Redundancy: worst pair {G.max():.2f}, "
                 f"effective rank {eff:.0f}/{K}", fontsize=9)

    fig.tight_layout()
    return _save(fig, out_dir, f"causal_filters_summary{tag}")


def make_filter_figures(encoder, out_dir, *, tag="", sr=16000, patch_encoder=None):
    """Render both diagnostic figures for a :class:`CausalFilterEncoder`."""
    V = np.asarray(encoder.V, dtype=np.float64)
    K, C, L = V.shape
    cp = encoder.coch_params
    dt_ms = 1000.0 / float(cp["frame_rate"])
    cf_hz = centre_frequencies(sr, C, float(cp["lo_hz"]), float(cp["hi_hz"]))
    cf_hz = np.sort(np.asarray(cf_hz, dtype=float))

    patch_A = None
    if patch_encoder is not None:
        # The patch dictionary is stored flat; it is only comparable at the same shape.
        A = np.asarray(patch_encoder.A, dtype=np.float64)
        if A.shape[1] == C * L:
            patch_A = A.reshape(A.shape[0], C, L)

    paths = _kernel_figure(V, cf_hz, dt_ms, out_dir, tag)
    paths += _summary_figure(V, cf_hz, dt_ms, out_dir, tag, patch_A=patch_A)
    return paths
