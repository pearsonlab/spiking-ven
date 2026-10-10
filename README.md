# spiking-ven

A **spiking** sparse auditory encoder and vocal error network (VEN) for the zebra finch song
system: song audio → sparse spike code → an excitatory/inhibitory network that learns to
*cancel* the predictable auditory response to the bird's own song, leaving an error signal.

This is the spiking counterpart to the rate-based Wilson-Cowan model in
[`pearsonlab/vocal-error-network`](https://github.com/pearsonlab/vocal-error-network)
(Gong et al. 2026, [eLife reviewed preprint](https://elifesciences.org/reviewed-preprints/111771),
[doi:10.7554/eLife.111771.1](https://doi.org/10.7554/eLife.111771.1)).
It is used standalone for the analyses here, and imported by the full song-circuit model
(finchsim) to drive the AIV → VTA/VP error pathway.

## Install

```bash
uv sync                       # core: numpy + scipy only
uv sync --extra brian2        # + Brian2 runtime glue
uv sync --all-extras          # + data prep (soundfile) and plotting (matplotlib)
```

Dependencies are deliberately minimal. The ERB/gammatone filterbank is **vendored**, not
depended on (see `THIRD_PARTY_NOTICES.md`), so the model and training core install from
numpy and scipy alone. No PyTorch, numba, seaborn or scikit-learn.

## Reproduce the cancellation figure

```bash
make figure      # data -> encoder -> VEN -> figure   (~10 min, CPU)
make verify      # check artifact checksums against MANIFEST.sha256
make test        # fast unit tests (no artifacts needed)
make test-repro  # the reproduction assertions below, against outputs/
make rates       # the supplementary population-rate view
```

`make figure` runs four stages, each cached and skipped if its output exists:

| Stage | Command | Output |
|---|---|---|
| 1. Song corpus | `sven-prep-data` | `outputs/motifs.npz` (38 motifs @ 16 kHz) |
| 2. Sparse encoder | `sven-train-encoder` | `outputs/of_encoder.npz` (64 bases, 300 epochs) |
| 3. Train the VEN | `sven-train-ven` | `outputs/of_ven_model_k4max.npz` (600 renditions) |
| 4. Figure | `sven-figure` | `outputs/encoding_comparison_k4max.pdf` |

`sven-evaluate` recomputes the metrics below for an existing model without retraining
(`--json` for machine-readable output) -- useful for checking a model you were handed.

Raw song data is downloaded, never vendored: "Labeled Zebra Finch Songs" (Koch, Therese),
adult zebra finch R469, [DOI 10.18738/T8/SAWMUN](https://doi.org/10.18738/T8/SAWMUN).
The dataset is released under **CC0 1.0** (public domain dedication, no stated terms of
use or access restrictions), which is why the derived `motifs.npz` can be attached to
releases. Please still cite it.

> **The download only works from a normal network.** Both dataset hosts sit behind
> CloudFront, which returns `403 Forbidden` to datacenter IP ranges — so it fails on CI
> runners and some cloud hosts regardless of the User-Agent the downloader sends. If you
> are blocked, place the WAV + `.not.mat` pairs in `<data-dir>/song_wavs` yourself and
> pass `--skip-download`; nothing downstream touches the network. For a TLS-inspecting
> proxy, point `SVEN_CA_BUNDLE` at a PEM bundle that includes the public roots.

## What the figure shows

Five rows (waveform / spectrogram / auditory neurons / inhibitory interneurons / excitatory
projection neurons) by four stimulus columns (training song / time-reversed motif /
white noise / distorted auditory feedback).

The result is in the excitatory row: **~7 Hz on the trained song versus ~18, ~21 and ~19 Hz**
on the other three. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

## Expected numbers (asserted in the test suite)

Definitions follow Mandelblat-Cerf et al. 2014: a 50 ms noise burst during singing, spike
counts in 150 ms windows either side of onset, compared per neuron across 20 renditions by
paired t-test, responder at p < 0.02. Latency is an onset -- the first 2 ms bin followed by
five significant ones -- not a PSTH peak. `tests/test_reproduction_causal.py` pins these
against the shipped model; `make test-repro` runs it.

| Metric | Model | Bird |
|---|---|---|
| auditory input rate during song | 2.15 Hz | 2.4 ± 2.7 Hz (Schneider & Woolley 2013, broad-spiking higher AC) |
| baseline, all units | 3.20 Hz | 1–10 Hz (Fig. 7C) |
| responders | 54% | 41% = 7/17, 95% CI 21–64% |
| response onset latency | 29.3 ms | 23 ± 12 ms |
| response duration | 55.0 ms | 90 ± 43 ms |
| **extra spikes per responder per burst** | **3.34** | **0.47–2.66** |
| peak change, responders | 96.0 Hz | ~20 Hz (Fig. 8E, n = 7) — see caveat |

**The open problem is response magnitude.** Everything else is inside the bird's range;
the response is ~1.3–7x too strong. Stated in spikes rather than Hz because that needs no
assumption on our side: the bird's responder adds one or two spikes on hearing the
distortion, on a baseline under one spike in the same window. The 0.47–2.66 range comes
from a ~20 Hz peak over a 90 ± 43 ms duration, taking the response as triangular to
rectangular -- the paper reports no shape, and no intra-burst or instantaneous rate.

Responder fraction is deliberately not treated as a mismatch: 7/17 has a 95% confidence
interval of 21–64%, which 54% sits inside.

**The peak is a diagnostic, not a target.** It is bin-width dependent -- measured here it
runs 96.9 / 96.0 / 78.6 / 77.7 / 64.2 Hz at 1 / 2 / 5 / 10 / 20 ms bins -- and the paper
never states how its plotted PSTHs were binned. 2 ms is the only binning it gives (for the
latency and duration z-tests) and is what we use; no smoothing is described anywhere in the
paper, though the published trace looks smoother than 2 ms binning alone would give. So any
ratio against ~20 Hz carries an unknown mismatch. The spike count does not: it is invariant
to binning, and is the quantity the paper's own bootstrap used ("the maximal spike count
over 150 ms").

A previous version of this table reported K1–K4 against a "DAF" column that was pure white
noise at 5.6x RMS, with each stimulus normalised to itself, and quoted targets of 7.7 ± 8.7
Hz for singing and ~28 Hz for responders. None of those targets are numbers the paper
states -- Fig. 8E plots a *change* in rate peaking near 20 Hz, not an absolute rate -- and
the 2.1x/3.6x ratios were quotients computed from them rather than reported. The DAF
stimulus is now song with a burst mixed into a window of it, both normalisation stages
share the song as their reference, and the superseded assertions have been retired.

## Citing

Cite the paper, not this software:

> Gong Z, Duarte F, Mooney R, Pearson J (2026). Correctness is its own reward:
> bootstrapping error signals in self-guided reinforcement learning. *eLife*
> [doi:10.7554/eLife.111771.1](https://doi.org/10.7554/eLife.111771.1)

`CITATION.cff` names that paper as the repository's `preferred-citation`, so
GitHub's "Cite this repository" button and any CFF-aware tool will hand you the
paper rather than a separate software citation.
