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

New to the model? [`docs/MODEL.md`](docs/MODEL.md) gives the equations, units and a
glossary; [`docs/design-notes.md`](docs/design-notes.md) records why the parameters and
choices are what they are.

## Install

```bash
uv sync                       # core: numpy, scipy, Brian2 (+ Cython)
uv sync --all-extras          # + data prep (soundfile) and plotting (matplotlib)
```

The network is written as Brian2 equations (`src/spiking_ven/network.py`) and simulated
by Brian2, which compiles it with Cython when a C compiler is available (the first run
in a fresh environment spends ~1 min compiling; later runs reuse the cache). The
ERB/gammatone filterbank is **vendored**, not depended on (see `THIRD_PARTY_NOTICES.md`).
No PyTorch, numba, seaborn or scikit-learn.

## Reproduce the cancellation figure

```bash
make figure      # data -> encoder -> VEN -> figures   (~15 min, CPU)
make verify      # check artifact checksums against MANIFEST.sha256
make test        # fast unit tests (no artifacts needed)
make test-repro  # the reproduction assertions below, against outputs/
sven-figure --view filters  # the encoder's kernels (STRFs) and a timing/tuning summary
sven-figure --view units    # example units: kernel over spikes to all 38 motifs (--units 36 53 45)
sven-figure --view gammatone  # spectrogram, gammatone filters, cochleagram, encoder spikes
```

`make figure` runs four stages, each cached and skipped if its output exists:

| Stage | Command | Output |
|---|---|---|
| 1. Song corpus | `sven-prep-data` | `outputs/motifs.npz` (38 motifs @ 16 kHz) |
| 2. Causal encoder | `sven-train-encoder` | `outputs/causal_encoder.npz` (64 filters, 80 epochs, pruned to those that fire) |
| 3. Train the VEN | `sven-train-ven` | `outputs/ven_model.npz` (600 renditions) |
| 4. Figures | `sven-figure` | `outputs/cancellation.pdf`, `outputs/daf_psth.pdf` (`--view psth`) |

`sven-evaluate` recomputes the metrics below for an existing model without retraining
(`--json` for machine-readable output) -- useful for checking a model you were handed.
A trained model records how its stimuli were built (sample rate, seed, protocol timing,
rate calibration, DAF window, and checksums of the encoder and corpus it was trained on),
so `sven-evaluate` and `sven-figure` rebuild the same stimuli by default. Their stimulus
flags (`--mean-rate-hz`, `--n-ista`, `--t-post`, `--t-burn`, `--daf-window-s`,
`--stimulus-seed`) override that, with a warning when they disagree with the model's
record.

Raw song data is downloaded, never vendored: "Labeled Zebra Finch Songs" (Koch, Therese),
adult zebra finch R469, [DOI 10.18738/T8/SAWMUN](https://doi.org/10.18738/T8/SAWMUN).
The dataset is released under **CC0 1.0** (public domain dedication, no stated terms of
use or access restrictions), which is why the derived `motifs.npz` can be attached to
releases. Please still cite it.

> **The download only works from a normal network.** The dataset host sits behind
> CloudFront, which returns `403 Forbidden` to datacenter IP ranges — so it fails on CI
> runners and some cloud hosts regardless of the User-Agent the downloader sends. If you
> are blocked, place the WAV + `.not.mat` pairs in `<data-dir>/song_wavs` yourself and
> pass `--skip-download`; nothing downstream touches the network. For a TLS-inspecting
> proxy, point `SVEN_CA_BUNDLE` at a PEM bundle that includes the public roots.

## What the figure shows

Six rows (waveform / spectrogram / auditory neurons / inhibitory interneurons / excitatory
projection neurons / E population rate) by three stimulus columns (training song /
time-reversed motif / distorted auditory feedback, i.e. the song with a 50 ms noise
burst).

The result is in the excitatory rows: the trained song is cancelled (~2.6 Hz, flat), while
the time-reversed motif and the DAF burst still drive the E population -- the burst as a
sharp transient. The network has learned to cancel the response to the song it hears every
rendition, while responses to novel or perturbed sound survive as an error signal. The
figure draws exactly the stimuli the metrics below are computed on.

Each motif is preceded by two introductory notes of HVC activity with no sound
(`HVC_INTRO`), in training, evaluation and the figure, which shows only the motif. They
engage the HVC-driven inhibition before the song starts; without them E fires through the
first ~50 ms of every motif while that inhibition switches on.

## Expected numbers (asserted in the test suite)

Definitions follow Mandelblat-Cerf et al. 2014: a 50 ms noise burst during singing, spike
counts in 150 ms windows either side of onset, compared per neuron across 100 renditions by
paired t-test, responder at p < 0.02. 100 matches the paper's ~100 bursts per neuron (the
Fig. 7G and 7J rasters show 100 and 126 trials); the tests are per neuron, so every DAF
metric depends on this count. Latency is an onset -- the first of five consecutive
significant 2 ms bins -- not a PSTH peak. `tests/test_reproduction_causal.py` pins these
against the shipped model; `make test-repro` runs it.

| Metric | Model | Bird |
|---|---|---|
| auditory input rate, per rendition (2.66 Hz during the song itself) | 2.16 Hz | 2.4 ± 2.7 Hz (Schneider & Woolley 2013, broad-spiking higher AC) |
| baseline, all units | 2.66 Hz | 1–10 Hz (Fig. 7C) |
| responders | 55% | 41% = 7/17, 95% CI 21–64% |
| response onset latency | 25.1 ms | 23 ± 12 ms |
| response duration | 50.3 ms | 90 ± 43 ms |
| extra spikes per responder per burst | 0.93 | ~0.8–1.0 |
| peak change, responders | 21.5 Hz | ~20 Hz (Fig. 8E, n = 7) — diagnostic, see caveat |
| time to peak | 38 ms | ~40–60 ms (Fig. 8E, read off the panel) |
| late (100–150 ms) / peak | 0.073 | < 0.15 — a transient, not a step (Fig. 8E) |
| reversed / trained song rate | 1.09 | > 1 |

**Every metric is inside the bird's range at this seed.** Training seeds 43 and 44 give extra
spikes 1.07 and 0.82, latency 21.4 and 26.4 ms, duration 54.0 and 49.3 ms and reversed/trained
1.04 and 1.10; seed 43 is just above the magnitude range, and seed 44's response sags mid-burst,
so its 2 ms-bin peak falls at 22 ms. Magnitude is stated in spikes Magnitude is stated in spikes
rather than Hz because a count needs no assumption about binning: the bird's responder adds
about one spike on hearing the distortion, on a baseline under one spike in the same
window. (Converting the paper's ~20 Hz peak over a 90 ± 43 ms duration into a count
needs an assumed shape, since none is reported; triangular and rise-then-exponential shapes
both give ~0.8–1.0 spikes, a rectangle 1.8.)

Two things brought magnitude into range, after it had sat at ~1.7–2.1x: the paper's trial
count, and an HVC input like HVC_RA -- 600 neurons, each firing one stereotyped 4-spike burst
per motif. With 20 trials the per-neuron tests admit only the strongest responders, which
reads as a larger, later, shorter response; and a dense HVC code keeps inhibition on through
the whole motif, which weakens the response (`docs/design-notes.md`).

Duration is 50.3 ms at the test seed (51.2 ± 2.2 ms over ten evaluation seeds, lowest 48.2,
against the 47 ms bound). With AMPA-fast synapses it comes from the NMDA component of the
auditory synapses (below). Latency is 25.1 ms at the test seed, 22.9 ± 1.0 ms over ten seeds. Duration follows the paper's rule -- the
response ends at the first run of ten non-significant 2 ms bins -- searched up to 300 ms
after onset, so long responses are timed rather than dropped.

`baseline, responders` is **not** comparable to the 1–10 Hz range. Responders are selected
by a paired t-test for post > pre, which preferentially picks neurons quiet in the
pre-window, so their baseline is biased low by the selection (~1.5 Hz against the
population's ~2.7 Hz). The paper applies the same criterion, so responder-vs-responder
comparisons stay fair — but only `baseline, all units` estimates a firing rate.

Responder fraction is deliberately not treated as a mismatch: 7/17 has a 95% confidence
interval of 21–64%, which 55% sits inside.

**The peak is a diagnostic, not a target, and is not asserted.** It is bin-width dependent
and the paper never states how its plotted PSTHs were binned; 2 ms is the only binning it
gives (for the latency and duration z-tests) and is what we use. So any ratio against
~20 Hz carries an unknown mismatch (the model's peak falls from 23 to 18 Hz, and its time
from 38 to 30 ms, between 1 and 10 ms bins). The spike count does not: it is invariant to
binning, and is the quantity the paper's own bootstrap used ("the maximal spike count over
150 ms").

## Units and synapses: AMPA, GABA_A and NMDA conductances

The model is in physiological units: membranes rest and reset at -70 mV and fire at -50 mV
(500 pF and 16.7 nS for E, 200 pF and 20 nS for I), and every synapse is a **conductance**
acting through its reversal potential -- AMPA and NMDA at 0 mV, GABA_A at -70 mV (Wang 2002).
Each synaptic weight is the peak conductance (nS) one spike opens, drawn log-normally with a
stated mean and SD -- e.g. auditory -> E 38.1 +/- 19.0 nS, HVC -> I 6.4 +/- 3.2 nS. Tonic drives,
adaptation and noise are currents (pA). `docs/MODEL.md` has the equations and every value.

With `E_GABA` at rest, I -> E inhibition is **shunting**: it passes no current at rest and pulls a
depolarised cell back toward rest, dividing the effect of the auditory excitation rather than
subtracting from it, so the learned inhibition cancels the song by division.

Every synapse has a realistic decay. The auditory -> E and HVC -> I inputs and E -> I are AMPA
(`TAU_S = TAU_EXC = 3 ms`); I -> E is GABA_A (`TAU_INH = 15 ms`, inside the 5-20 ms
juvenile range of Luo & Perkel 2002). The auditory synapses onto E
also carry a standard NMDA conductance (Wang 2002: saturating gating with 2 ms rise and
100 ms decay, Mg2+ block, reversal 0 mV), on the same connections as the fast weights,
with a mean peak conductance of `G_NMDA_MEAN_NS = 2500` nS per connection at full gating
before the Mg2+ block (which lets ~4% through at rest and ~14% at threshold). With fast
synapses NMDA is the only slow excitation, and it is what holds the DAF response up: without
enough of it the response peaks ~30 ms after onset and fades while the noise is still on.
Lasting ~73 ms per spike against AMPA's ~3.5, it carries
~190x the AMPA charge per spike near threshold (61x at rest), though its peak current is
~6x AMPA's.
Both runtimes implement it; see `docs/MODEL.md` for the equations and
`docs/design-notes.md` for the sweeps that chose it.

Learning does not depend on `TAU_S`. STDP on I -> E has one window, `TAU_STDP = 20 ms`: the
I-spike trace decays with it, and an E spike counts only if its cell received an auditory
input spike within that long before it (the audio gate).

## Citing

Cite the paper, not this software:

> Gong Z, Duarte F, Mooney R, Pearson J (2026). Correctness is its own reward:
> bootstrapping error signals in self-guided reinforcement learning. *eLife*
> [doi:10.7554/eLife.111771.1](https://doi.org/10.7554/eLife.111771.1)

`CITATION.cff` names that paper as the repository's `preferred-citation`, so
GitHub's "Cite this repository" button and any CFF-aware tool will hand you the
paper rather than a separate software citation.
