# spiking-ven

A **spiking** sparse auditory encoder and vocal error network (VEN) for the zebra finch song
system: song audio → sparse spike code → an excitatory/inhibitory network that learns to
*cancel* the predictable auditory response to the bird's own song, leaving an error signal.

This is the spiking counterpart to the rate-based Wilson-Cowan model in
[`pearsonlab/vocal-error-network`](https://github.com/pearsonlab/vocal-error-network)
(Duarte Ortiz et al. 2025, [bioRxiv 2025.07.18.665446](https://doi.org/10.1101/2025.07.18.665446)).
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
make figure     # data -> encoder -> VEN -> figure   (~10 min, CPU)
make verify     # check artifact checksums against MANIFEST.sha256
uv run pytest   # includes the regression assertions below
```

`make figure` runs four stages, each cached and skipped if its output exists:

| Stage | Command | Output |
|---|---|---|
| 1. Song corpus | `sven-prep-data` | `outputs/motifs.npz` (38 motifs @ 16 kHz) |
| 2. Sparse encoder | `sven-train-encoder` | `outputs/of_encoder.npz` (64 bases, 300 epochs) |
| 3. Train the VEN | `sven-train-ven` | `outputs/of_ven_model_k4max.npz` (600 renditions) |
| 4. Figure | `sven-figure` | `outputs/encoding_comparison_k4max.pdf` |

Raw song data is downloaded, never vendored: Koch 2024, adult zebra finch R469
([DOI 10.18738/T8/SAWMUN](https://doi.org/10.18738/T8/SAWMUN)).

## What the figure shows

Five rows (waveform / spectrogram / auditory neurons / inhibitory interneurons / excitatory
projection neurons) by four stimulus columns (training song / time-reversed motif /
white noise / distorted auditory feedback).

The result is in the excitatory row: **~6 Hz on the trained song versus ~18, ~21 and ~17 Hz**
on the other three. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal.

## Expected numbers (asserted in the test suite)

Metric definitions follow Mandelblat-Cerf et al. 2014, Fig. 7/8.

| Metric | Value | Biological target |
|---|---|---|
| encoder forward/reversed activation correlation | 0.0036 | near zero (direction selectivity is possible) |
| K1 — correct song + HVC | 7.84 Hz | mean 7.7 Hz, SD 8.7 (Fig. 7C) |
| K2 — white noise (DAF) + HVC | 22.25 Hz | pop. avg ~16 Hz; responders ~28 Hz |
| K3 — K2/K1 | 2.84x | pop. avg ~2.1x; responders ~3.6x |
| K4 — reversed motif / correct | 2.07x | > 1x |

The model is a **responder-only** population (every excitatory unit receives identical
auditory drive), so the responder-only targets are the relevant ones.

## Citing

See `CITATION.cff`. Please also cite the paper above.
