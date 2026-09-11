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

Metric definitions follow Mandelblat-Cerf et al. 2014, Fig. 7/8.

| Metric | Value | Biological target |
|---|---|---|
| encoder forward/reversed activation correlation | 0.0036 | near zero (direction selectivity is possible) |
| K1 — correct song + HVC | 7.78 Hz | mean 7.7 Hz, SD 8.7 (Fig. 7C) |
| K2 — white noise (DAF) + HVC | 22.19 Hz | pop. avg ~16 Hz; responders ~28 Hz |
| K3 — K2/K1 | 2.85x | pop. avg ~2.1x; responders ~3.6x |
| K4 — reversed motif / correct | 2.13x | > 1x |

These come from the seeded 600-rendition reference run, and `make test-repro` asserts them
against the shipped model. The network is fully seeded, so re-running the pipeline
reproduces them rather than landing nearby.

The model is a **responder-only** population (every excitatory unit receives identical
auditory drive), so the responder-only targets are the relevant ones.

**What K2/K3 compare.** Every stimulus goes through the same encoder path, which
normalises twice: to the encoder's reference RMS, and then to a target spike rate. The
white-noise input therefore carries the *same* mean drive as the song input by
construction, and the DAF amplitude cancels out. K2 and K3 measure the response to a
spectrotemporal mismatch at equal input rate -- a stronger test than a loudness
difference, since the extra response cannot come from extra drive, but not the amplitude
manipulation the "DAF" name implies. The figure's DAF column is a genuine amplitude
manipulation: there the noise is mixed into a window of the song.

## Citing

Cite the paper, not this software:

> Gong Z, Duarte F, Mooney R, Pearson J (2026). Correctness is its own reward:
> bootstrapping error signals in self-guided reinforcement learning. *eLife*
> [doi:10.7554/eLife.111771.1](https://doi.org/10.7554/eLife.111771.1)

`CITATION.cff` names that paper as the repository's `preferred-citation`, so
GitHub's "Cite this repository" button and any CFF-aware tool will hand you the
paper rather than a separate software citation.
