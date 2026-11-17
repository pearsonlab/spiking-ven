# Design notes

Why the model and pipeline are the way they are: the experiments, measurements and dead ends behind
choices that the code comments state only briefly. **Numbers here are historical.** Each was
measured on the model version current at the time, which may differ from the shipped one; the
README and `tests/test_reproduction_causal.py` hold the current numbers.

## Operating point and reproducibility

- **The operating point lives in `constants.py`, and the CLIs read it as their defaults.** A
  parameter sweep once built the network from the library's constructor defaults instead, because
  the real values existed only in a scratch script; four grid points came back identical and broken.
  For the same reason `tests/test_cli.py` checks that each flag *reads* its constant rather than
  repeating the value.
- **The template rendition** (highest RMS) was once chosen separately in four places. It now lives in
  `corpus.py`, so training, scoring and the figure cannot pick different motifs.
- **Stimuli are built once, by `stimuli.build_stimuli`**, for the trainer, the CLIs, the tests and
  the figures alike. A trained model also records the settings and input checksums its stimuli
  were built with (`ven.stimulus`), because a figure rendered at a different auditory rate than
  its model was trained at encodes a different stimulus, with nothing to show it.
- **`VocalErrorNet` defaults are the operating point; `VocalErrorNetV2` keeps the old ones.** The
  class's constructor defaults once differed from `constants.py` in 12 places (no adaptation, no
  delay, `n_hvc = 15`, ...). finchsim builds networks from those old defaults in its tests, so the
  old name stays as a subclass with them.
- **Monitoring draws its own noise.** `sven-train-ven`'s per-rendition snapshots once used the
  training noise stream, so the trained weights depended on how often progress was logged.
- **The template is the DTW reference.** `motifs.npz` stores `template_index`, the rendition the
  others were aligned to. Recomputing highest RMS after alignment could pick a different one (the
  top two aligned renditions differ by <1% in RMS).
- **Simulation noise is seeded.** It was originally drawn from the global `np.random`, so `seed=`
  fixed only the initial weights, and two identical runs differed by ~1% in E rate.
- **Version.** v0.5.0 shipped with `__version__` and `pyproject.toml` disagreeing (0.4.0 vs 0.5.0);
  the version is now read from `__version__` only.
- **Packaging.** An unanchored `data/` rule in `.gitignore` once kept `src/spiking_ven/data/` out of
  the repository; `test_every_subpackage_is_importable` guards against a repeat. The figure modules
  once created `outputs/` in the working directory at import time.

## Auditory input

- **`AUD_MEAN_RATE_HZ`** was 15.0, an unsourced default repeated across seven call sites, which made
  the auditory input ~4x faster than the broad-spiking higher-auditory-cortex population it models
  (Schneider & Woolley 2013: 2.4 +/- 2.7 Hz).
- **Two normalisation stages share one reference (the song).** If only the waveform scaling shares
  it, the level difference reaches the activations and is then flattened again at the spike stage.
  Measured on an earlier encoder: DAF at 5.6x song RMS gave 2.81x the activation energy but still
  15.6 Hz/channel, against 43.4 Hz/channel with both stages shared.
- **Refractory thinning of the input populations.** `acts_to_spikes` returns Poisson counts, so one
  1 ms bin could carry several spikes (measured up to 5 for song and 8 for DAF on an earlier
  encoder), and ~40% of auditory and ~27% of HVC inter-spike intervals were 1 ms. `apply_refractory`
  removes both. `t_refr_ms=0` keeps the raw counts, so models trained before thinning existed still
  reproduce.

## The causal encoder

- **Minimum phase.** An unconstrained kernel drifts toward near-linear phase, with its energy
  centroid around L/2 -- 7.5 of 16 frames -- costing ~30 ms of encoder latency. Pulling toward
  minimum phase moved the centroid to 10.9/16, raised the energy in the newest 4 frames from 0.26 to
  0.50, and cut encoder latency from ~30 ms to ~2 ms, with no loss of forward/reversed selectivity.
  An earlier convolutional (non-causal) encoder reported a response peaking 2 ms *before* its
  stimulus, which is why causality is tested by truncation rather than by a summary statistic.
- **Soft pull, not projection (`CAUSAL_MP_STRENGTH`).** The strict projection (`a = 1`) has a
  degenerate fixed point. Measured there: lag-axis spectral flatness 0.08 -> 0.57, 58% of every
  kernel's energy in the newest frame, song/reversed contrast 2.29 -> 1.83, DAF responder fraction
  51% -> 74% (bird 41%). Over `a` = 0.10 / 0.25 / 0.50 the bank landed in the same place (centroid
  10.1/16, sd 0.5; newest four frames 0.49 / 0.49 / 0.48 of the energy). The softness was once
  accidental: `min_phase` defaulted to an FFT of `8 * L` points, which aliases the cepstrum (7.7%
  magnitude error at L=16, against 1.2% at 512 and 0.5% at 1024).
- **L1 on kernel entries (`CAUSAL_L1`).** Without it the kernels did not converge: at 3% activation
  a filter fires too rarely to accumulate gradient, and after 80 epochs the kernels still correlated
  +0.67 with their random initialisation (+0.81 over the 15 older frames), with reconstruction error
  improved only 5%. The visible "speckle" was untouched initialisation. At `lr = 0.1` an `l1_V` of
  0.03 put the soft threshold 1300x above the median gradient step and zeroed 99.8% of entries.
- **No usage equalisation.** Forcing all filters to fire equally bought nothing on this stereotyped
  corpus (effective rank 29/64 with it, 32/64 without) and kept the code too sparse to train.
- **Pruning.** At `lam = 0.6` only 19 of 64 filters ever fired; the rest decayed into stubs that
  never won a competition. Left in, they are permanently silent auditory channels that dilute each
  E cell's sampled inputs.
- **Code sparsity (`CAUSAL_LAMBDA`).** Measured on an earlier encoder: 0.3 -> 10.7% activation and
  56/64 filters; 0.6 -> 3.2% and 23; 1.2 -> 0.15% and 1. 0.3 was chosen for song cancellation
  (song/reversed 2.47 vs 1.72 at 0.6), at a cost in responder fraction (52% vs 43%, bird 41%) and
  peak change (316 vs 262 Hz, on the binning of the time). At 0.3, white noise drives the
  dictionary 17x harder than at 0.6; with no measurement of continuous noise during singing to
  compare against, that was not used to set the value.
- **Mean subtraction.** Without `causal_meansub` the code is dominated by level and cannot tell
  forward from reversed song (forward/reversed correlation 0.998).
- **Frame timing (PR #12).** `gtgram` labels each window by its first frame, so the code at frame t
  read audio up to ~2 ms past t, and the divisive gain (since removed) up to ~4 ms (measured
  lookahead 2-3 ms).
  Removing that added ~4 ms of response latency, compensated by lowering `AUD_DELAY_MS` from 5 to 2.

## Network dynamics

- **Conduction delay (`AUD_DELAY_MS`).** It was once set to the bird's 23 ms response *latency*,
  which double-counts: the network then integrates for a further ~43 ms (modelled latency ~66 ms)
  and the comparison with 23 ms becomes circular. finchsim's driver scripts carried the same 23 ms
  as their default until finchsim PR #27; they now use the trained model's delay.
- **`tau_s` and the STDP window** were once the same constant, which tied the coincidence window to
  the synaptic time course; `tau_stdp` now sets it separately. A coupling survived in the audio
  gate (`B @ h_aud > 0`) until the AMPA recalibration below, which also moved `tau_s` to 3 ms.
- **Membrane refractory period.** Without it the ceiling was 1000 Hz and was reached: during a DAF
  burst the median E inter-spike interval was 1 ms, with 73% of burst spikes in the very next bin,
  while the baseline median ISI was 31 ms. Decrementing the countdown after the spike test, rather
  than before, would silently make the period `t_refr_ms + 1`.
- **Spike-frequency adaptation rather than a weaker drive.** Extra spikes per burst depended only on
  the product of auditory rate and `B_scale`, roughly as drive^0.36 (two runs at drive 149 and 161
  gave 9.04 and 9.36 spikes despite 4x and 4.4x differences in rate and weight). Reaching the bird's
  ~1 spike that way needed an auditory rate near 0.3 Hz, an order of magnitude below Schneider &
  Woolley. A gain change also has the wrong shape: per responder the model spanned 1.1-19.9 extra
  spikes, the weak end already inside the bird's range, and a gain change moves both ends.
- **`TAU_SFA` shorter than the burst.** At 100 ms adaptation integrated over the motif and acted as
  a tonic gain change (baseline -59%, peak barely moved); at 10 ms it builds and recovers within the
  50 ms burst.
- **`A_SFA` sweep** at `tau_sfa = 10 ms` (model version before PR #12):

  | a_sfa | extra spikes | peak (Hz) | latency (ms) | duration (ms) | spikes / peak (ms) |
  |---|---|---|---|---|---|
  | 0.3 | 2.89 | 87.6 | 34.5 | 57.1 | 33.0 |
  | 0.6 | 2.11 | 61.5 | 35.0 | 55.3 | 34.3 |
  | 1.0 | 1.61 | 46.9 | 33.9 | 52.3 | 34.3 |
  | 2.0 | 1.18 | 31.5 | 38.0 | 43.1 | 37.5 |
  | bird | 0.78-1.01 | ~20 | 23 +/- 12 | 90 +/- 43 | ~45 |

  1.0 was the strongest setting that kept latency and duration inside the bird's ranges; 2.0 nearly
  matched the spike count but pushed latency past 35 ms and duration below 47 ms. Spikes per peak
  rose with adaptation, against the expectation that adaptation narrows the response: it suppresses
  the peak more than the tail. Part of the fall in duration is the measure, not the response:
  duration is a z-test against pre-onset variability, so a weaker response stays detectable for less
  time.

## Lengthening the DAF response

The bird's response is long and weak (~1 extra spike over ~90 ms); the model's is short and strong
(~1.8 spikes over ~46 ms). Numbers below are 5-seed means on the model after PR #14.

- **Adaptation trades one for the other.** At `tau_sfa = 10 ms`, `a_sfa` 0 / 0.5 / 1.0 / 1.5 gave
  durations of 68 / 51 / 46 / 43 ms and 6.1 / 2.6 / 1.8 / 1.4 extra spikes. No single value fixes
  duration without making magnitude worse.
- **E -> I feedback is too weak to delay usefully.** During the burst, I rises only ~1-2 Hz (22.7 ->
  23.7 Hz), and the extra E -> I input is under 10% of `theta_i`: I is driven by HVC. A delay on
  that loop would only postpone inhibition, adding spikes.
- **A linear slow synaptic trace works, but is not a mechanism.** Moving a fraction `f` of each
  auditory spike's drive to an 80 ms exponential, total charge unchanged, gave 68 ms and 1.8 spikes
  at `f = 0.5`, with every other metric inside the bird's range. It was rejected as unprincipled:
  nothing in a real synapse redistributes charge that way.
- **Standard NMDA (`G_NMDA`), adopted.** Wang (2002) kinetics and Mg2+ block on the
  auditory -> E synapses. Duration grows with `g_nmda` (46 -> 59 ms at 40, an NMDA:AMPA charge
  ratio of ~2.2 at threshold), but alone it adds spikes. Shifting drive from the fast weights to
  NMDA helped on every axis but latency: at `g_nmda = 40`, `B_SCALE = 10`, duration 60 ms, extra
  spikes 1.57, peak 35 Hz, latency 34 ms (bound 35), reversed/trained 1.15. The effect is weaker
  than the linear trace's because the Mg2+ block self-limits: once the burst ends and the membrane
  falls toward reset, the NMDA current shuts off.
- **The adopted point: `G_NMDA = 60`, `B_SCALE = 10`** (with intro notes, 5 seeds). Over
  `g_nmda` 30 / 40 / 60 at `B_SCALE` 10: durations 56 / 57 / 63 ms, extra spikes 1.56 / 1.59 /
  1.66, latency 33.9 / 33.8 / 33.2 ms. At `B_SCALE` 12, `g_nmda` 40 / 60 / 80 gave 59 / 64 /
  66 ms but 1.76 / 1.82 / 1.89 spikes, more than without NMDA. 60 / 10 is the setting that
  lengthens the response and weakens it. Its NMDA:AMPA charge ratio at threshold, ~3.3, is
  on the high side; 40 / 10 (~2.2) is the conservative alternative at 57 ms. Ten-seed means
  at the adopted point: duration 60.6 ms, extra spikes 1.66, peak 35 Hz, latency 33.9 +/- 1.8
  ms (bound 35), reversed/trained 1.16.

## Onset of each motif

- **Intro notes, not a burn-in.** E fired at ~5 Hz (peak ~12 Hz in 5 ms bins) through the first
  60 ms of every motif against 2.5 Hz later. A silent burn-in of 0-500 ms did not change it: the
  bump appeared with HVC alone and no sound, and without HVC E sits at ~10 Hz on tonic drive. It
  is the HVC-driven inhibition switching on: I fired at ~17 Hz in the first 30 ms against 33 Hz
  later, while the filtered HVC input builds up. Two 80 ms intro notes of HVC activity before the
  motif (`HVC_INTRO`) cut it to 3.6 Hz (peak 7.1 Hz) and left every DAF metric within its noise.
  The 20 ms gap before the motif lets inhibition sag again; with no gap the onset rate fell to
  2.9 Hz, on a model trained without intro notes.

## The encoder's loudness gain (removed)

Each frame's code was once multiplied by ``level / ||reconstruction||``: divided by the norm of
the code's reconstruction (so size stopped reporting match quality) and multiplied by the
running level that mean subtraction removed. Variants tried, each with the network retrained
(10-seed means, NMDA and intro notes on):

| gain | `B_SCALE` | noise-burst input ÷ song | extra spikes | latency (ms) | duration (ms) | reversed/trained |
|---|---|---|---|---|---|---|
| level / reconstruction (removed) | 10 | 18x | 1.66 | 33.9 | 60.6 | 1.156 |
| level only | 10 | 31x | 2.37 | 25.5 | 79.2 | 1.176 |
| level only | 6 | 31x | 1.75 | 27.6 | 74.0 | 1.136 |
| level^0.25 | 10 | 24.5x | 1.68 | 28.2 | 63.9 | 1.172 |
| none (adopted) | 13 | 8.5x | 1.69 | 31.5 | 59.7 | 1.155 |

- Responder magnitude barely follows the input: the burst's input drive ranged 3.6x across
  these, extra spikes only ~1.5-2.4. The remaining ~2x excess over the bird is set by the
  network's own dynamics, not by how loud the encoder makes the burst.
- No gain was adopted for simplicity: at `B_SCALE` 13 it matches the divisive version on
  magnitude and song selectivity with slightly faster onset. Level-only gains lengthen the
  response and cut latency but raise magnitude; weaker weights trade it back at the cost of
  selectivity.
- Sparser aud -> E connectivity (`C_B` 0.03, level-only gain) brought responders to the bird's
  41% but raised baseline and trained-song rates: E cells left without auditory input never
  acquire inhibition (learning is audio-gated) and fire tonically through the song.

## HVC input model: first survey

Measured with the encoder's former divisive gain, `B_SCALE` 10 and 20 DAF trials, when the
60-unit Gaussian-envelope input (a Poisson burst under a 10 ms Gaussian, peak 300 Hz, ~6
spikes per motif) was the default and the burst generator an option. Both are superseded by
the next section.

- **finchsim's HVC_RA bursts.** The burst generator ports finchsim's `ra.build_hvcra_spike_trains`: one burst per neuron per motif, 4
  spikes at 500 Hz, onsets on a jittered grid, identical every rendition -- the means of the
  HVC_RA statistics finchsim cites (Hahnloser et al. 2002; Picardo et al. 2016; Long et al.
  2010), not their spread. It delivers 4 spikes per neuron per motif against the Gaussian
  model's 6.2 +/- 1.6, so 64% of the HVC -> I drive. Retrained at seed 42 (Gaussian / burst at
  `B_HVC_SCALE` 0.333 / burst at 0.518, which matches the drive):

  | metric | Gaussian | burst 0.333 | burst 0.518 |
  |---|---|---|---|
  | baseline, all units (Hz) | 2.63 | 2.93 | 2.62 |
  | responders | 53% | 54% | 50% |
  | peak change (Hz) | 34.8 | 48.6 | 45.8 |
  | latency (ms) | 36.9 | 32.8 | 39.5 |
  | duration (ms) | 61.2 | 53.3 | 43.0 |
  | extra spikes / neuron / burst | 1.62 | 1.45 | 1.40 |
  | reversed / trained | 1.16 | 1.17 | 1.15 |

  Matching the drive restores the baseline, so the rest is the burst structure itself: a
  sharper, shorter DAF response (peak +9 SD, duration -6 SD of the seed-to-seed noise), with
  duration below the 47 ms lower bound of the bird's 90 +/- 43 ms. Adopting it needs the
  response lengthened again, as `A_SFA`/NMDA did for the Gaussian model.
- **60 HVC units tile the motif thinly; 600 tile it densely.** At 60, one burst starts every
  14 ms and about 0.5 neurons burst at any instant (HVC_RA: ~20,000 cells, so roughly
  130-220). The 20 ms synaptic filter keeps the summed drive above half its mean, but each I
  neuron has ~6 HVC inputs (`c_hvc` 0.1) and spends about half the motif below 10% of its own
  mean drive (longest such stretch ~240 ms). At 600, with `c_hvc` unchanged (~60 inputs per I),
  that falls to 1% (longest ~46 ms) and ~120 of 150 I neurons are driven at any ms, against
  50-58. `B_HVC_SCALE` is per connection, so it has to shrink with `n_hvc`: 10x smaller keeps
  the mean drive to I, but the baseline rose (Gaussian 3.04 Hz, burst 2.94 Hz), so each was
  recalibrated to the 2.63 Hz baseline of the shipped model. Seed 42, all at that baseline:

  | metric | Gaussian 60 (0.333) | Gaussian 600 (0.045) | burst 60 (0.518) | burst 600 (0.065) |
  |---|---|---|---|---|
  | baseline, all units (Hz) | 2.63 | 2.64 | 2.62 | 2.64 |
  | responders | 53% | 39% | 50% | 43% |
  | peak change (Hz) | 34.8 | 29.2 | 45.8 | 36.9 |
  | latency (ms) | 36.9 | 38.3 | 39.5 | 35.2 |
  | duration (ms) | 61.2 | 45.9 | 43.0 | 49.8 |
  | peak latency (ms) | +36 | +62 | +48 | +50 |
  | extra spikes / neuron / burst | 1.62 | 1.19 | 1.40 | 1.26 |
  | reversed / trained | 1.16 | 1.14 | 1.15 | 1.15 |
  | responders with a detectable onset | 82 | 36 | 74 | 48 |

  A dense HVC code weakens the DAF response toward the bird (extra spikes 1.6 -> 1.2) and, for
  the burst generator, undoes most of what it did to the response at 60 units. It also
  shortens the Gaussian model's response below the 47 ms bound. One seed only, and latency
  and duration rest on 36-82 units here; multi-seed evaluation is needed before adopting any of
  these. Untuned: `c_hvc` (inputs per I) at 600.

## HVC input and DAF trial count (adopted)

Redone on the no-gain encoder (`B_SCALE` 13), with `B_HVC_SCALE` carried over from the
survey above (it still matched the 2.7 Hz singing baseline). Seed-42 networks, 5 evaluation
seeds each.

At 20 DAF trials no 600-unit configuration reached the 35 ms latency bound:

| 20 trials | Gaussian 60 (was shipped) | Gaussian 600 | burst 60 | burst 600 |
|---|---|---|---|---|
| responders | 54% | 37% | 52% | 41% |
| extra spikes | 1.69 | 1.15 | 1.43 | 1.21 |
| latency (ms) | 31.5 | 46.8 | 34.2 | 41.4 |
| duration (ms) | 59.7 | 45.6 | 39.8 | 47.2 |
| reversed/trained | 1.155 | 1.106 | 1.133 | 1.124 |

On burst 600, none of these moved latency below ~36 ms: `B_SCALE` up to 20 (adds magnitude),
`G_NMDA` 90, `A_SFA` 0.5-0.7 (raises the baseline), `c_hvc` 0.015-0.1, `TAU_E` 15-20 (theta
is derived from it, so the gain mostly cancels), and a higher tonic drive (`TARGET_RATE_E`
up to 50 Hz, `DRIVE_E` up to 0.10: STDP inhibition absorbs it, raising the baseline only to
~3.6 Hz). Retraining with another seed moved latency by up to 10 ms.

The input chain is not the delay. Encoder activations change 2 ms after noise onset (the
frame quantization) and reach ~70% of their plateau by 12 ms; `AUD_DELAY_MS` adds 2. The
responders' population PSTH rises at the same time in every model (10% of peak at 10-13 ms,
50% at 23-27 ms). The difference is in detection: onset is found per neuron, by five
consecutive 2 ms bins above that neuron's pre-burst distribution, and 20 trials of a sparse
neuron detect a weak response late. Mandelblat-Cerf had ~100 bursts per neuron (the Fig. 7G
and 7J rasters show 100 and 126 trials). At 100 trials:

| 100 trials | Gaussian 60 | burst 60 | burst 600 (adopted) | burst 600, `c_hvc` 0.03 | bird |
|---|---|---|---|---|---|
| responders | 71% | 69% | 58% | 64% | 41% (CI 21-64%) |
| extra spikes | 1.36 | 1.15 | 0.94 | 1.00 | ~0.8-1.0 |
| latency (ms) | 21.6 | 22.5 | 27.7 | 27.5 | 23 +/- 12 |
| duration (ms) | 76.4 | 56.5 | 60.7 | 62.1 | 90 +/- 43 |
| peak latency (ms) | 36 | 48 | 50 | 38 | ~35-75 |
| reversed/trained | 1.155 | 1.133 | 1.124 | 1.185 | > 1 |

- **Trial count moves every DAF metric**, not just latency: more power admits weaker
  responders (lower mean magnitude, higher fraction) and detects onsets earlier and offsets
  later. `DAF_N_TRIALS = 100` is now the evaluation default.
- **Burst 600 adopted, `c_hvc` 0.1, `B_HVC_SCALE` 0.065**: every target inside the bird's
  range. Training seeds 43 and 44 agree (responders 55-56%, extra spikes 0.86-0.94, latency
  32 ms, duration 55-60 ms, reversed/trained 1.17-1.18). The Gaussian input was removed.
- **`c_hvc` 0.03** has the best selectivity but sits at the edge on responders and
  magnitude. `TAU_E` 20 and more tonic drive buy ~2 ms of latency and push responders past
  the CI.

## STDP audio gate and AMPA-fast input synapses (adopted)

Two changes, then a recalibration. Seed-42 networks, 100 DAF trials, one evaluation seed unless
stated.

- **The audio gate was not a window.** It was `B @ h_aud > 0`. `h_aud` is an exponential that
  never returns to zero (it underflows ~2 s later in float32), so after a cell's first input spike
  the gate stayed open for the rest of the run: 50.8% of E-cell time in the motif, the same at
  `tau_s` 20, 5 and 2 ms. It is now `stdp_audio_gate`: an auditory input spike within the last
  `tau_stdp` ms. Open 7.8% of the time; the same 69% of E cells can ever learn.
- **The gate alone weakened the response** (`tau_s` 20): extra spikes 0.95 -> 0.57, latency
  26.7 -> 43.6 ms, duration 60 -> 41 ms, reversed/trained 1.12 -> 1.05. Inhibition rose (mean
  `JIE` 0.042 -> 0.063): the late E spikes the old gate let through were mostly *depression*
  events. `A_jie` at 0.5-4x barely moved anything; it sets how fast the weights get there, not
  where.
- **`r_i_th` was far below the I rate.** The rule is zero-mean for unrelated I activity only if
  `r_i_th` is the I rate during song. It was 6 Hz against a measured 32 Hz (`tau_s` 20) or 41 Hz
  (`tau_s` 3), so nearly every gated E spike potentiated whatever I had been doing, and selectivity
  suffered. At `tau_s` 20, `r_i_th` 30 restored nearly everything (extra spikes 0.81, latency
  30 ms, duration 58 ms, reversed/trained 1.11).
- **`tau_s` 20 -> 3 ms (AMPA).** A spike's charge is the filter's sum, 20.5 -> 3.5, so matched
  mean drive needs `B_SCALE` and `B_HVC_SCALE` x5.81 and, since NMDA uses `B`, `G_NMDA` /5.81.
  At matched drive 2, 3 and 5 ms were indistinguishable (duration 21-24 ms, reversed/trained
  1.01-1.02), so 3 ms was taken as the middle of the AMPA range.
- **Fast synapses shorten the response.** The responder PSTH (10 ms bins) at 20 ms holds a plateau
  from 30 to 60 ms after onset; at 3 ms it peaks at 30 ms and fades while the noise is still on.
  The 20 ms filter averaged the sparse Poisson input into a sustained drive; with AMPA kinetics
  only NMDA is slow. More NMDA restores the plateau, with `B_SCALE` lowered to hold magnitude;
  longer `TAU_E` (45, 60 ms) and weaker adaptation (`A_SFA` 0.5, 0.7) did not lengthen it.

| `tau_s` 3, `r_i_th` 40 | NMDA/fast charge | extra spikes | latency (ms) | duration (ms) | reversed/trained |
|---|---|---|---|---|---|
| `G_NMDA` 10.3, `B_SCALE` 75.5 (matched) | 3.3x | 0.76 | 26.7 | 33.1 | 1.08 |
| 60, 55 | 19x | 0.85 | 25.6 | 46.6 | 1.10 |
| 90, 45 | 28x | 0.88 | 25.7 | 49.4 | 1.10 |
| 120, 40 | 38x | 0.93 | 23.8 | 49.3 | 1.11 |
| **150, 35 (adopted)** | 48x | 0.96 | 23.9 | 54.5 | 1.12 |

- **Across training seeds 42-44**, duration was 41.5-49.4 ms at `G_NMDA` 90, 45.8-49.3 at 120 and
  48.2-54.5 at 150, so 150 is the lowest that keeps every seed above the 47 ms bound. At 150:
  extra spikes 0.92-1.04, latency 24-28 ms, responders 57-59%, reversed/trained 1.11-1.18,
  baseline 2.1-2.6 Hz. Over evaluation seeds 42-51, duration 52.5 +/- 1.9 ms (47.8-54.6).
- **Adopted then:** `TAU_S` 3, `B_SCALE` 35, `B_HVC_SCALE` 0.378, `G_NMDA` 150, `R_I_TH` 40.
  The recurrent synapses were still one-step pulses; superseded by the next section.

## Synaptic recurrent connections and the NMDA duration sweep (adopted)

- **I -> E and E -> I were one-step pulses** (`JIE @ sI[t-1]`, `JEI @ sE[t-1]`; the Brian2 port
  approximated them with 1 ms decays). They are now traces like the inputs': GABA_A at
  `TAU_INH` 5 ms (IPSC decay ~5-10 ms; 5 as in Brunel & Wang 2001) and AMPA at `TAU_EXC` 3 ms.
  0 keeps the one-step pulse, which is how older model files and `VocalErrorNetV2` load.
- **The GABA charge compensates itself; E -> I does not.** A 5 ms trace gives each I spike 5.5x
  the charge, and learning settled at ~5x smaller `JIE` (row sum 5.9 -> 1.2): matching
  `J_MAX_IE` and `A_jie` to it changed little. `JEI` is fixed, so the 3 ms trace made E -> I 3.5x
  stronger, and that fast feedback inhibition cut the DAF response (extra spikes 0.96 -> 0.75,
  latency 24 -> 39 ms, duration 54 -> 43 ms, reversed/trained 1.12 -> 1.06). Its scale, once a
  hard-coded 0.17, is now `J_EI_SCALE`; 0.048 = 0.17 / 3.53 keeps the charge per spike and
  recovers most of it (0.87, 32 ms, 48 ms, 1.09). 0.034-0.10 behave alike. `TAU_INH` 10 ms was
  worse on every axis (0.78, 34 ms, 44 ms, 1.07).
- **NMDA then lengthens the response to ~70 ms**, with `B_SCALE` lowered to hold magnitude
  (seed 42): `G_NMDA` 150/35 -> 48 ms; 200/30 -> 50; 250/25 -> 55; 300/20 -> 61; 400/15 -> 67;
  beyond that it saturates at 68-70 ms while magnitude falls (300/15, 400/10, 500/10: extra
  spikes 0.62-0.72). The response stays a transient (late/peak 0.07).
- **Latency is the binding constraint**, near the 35 ms bound: at `R_I_TH` 40 training seed 44
  gave 35.7 ms (400/15) and 35.9 ms (300/20). Raising `R_I_TH` slightly above the I rate
  (40.6 Hz) shortens it: 46 gave 26.7-32.5 ms but responders 62-63% (CI to 64%) and extra spikes
  1.07 on one seed. 43 sits between:

| `G_NMDA` 400, `B_SCALE` 15, `R_I_TH` 43, training seed | 42 | 43 | 44 |
|---|---|---|---|
| responders | 60% | 59% | 60% |
| extra spikes | 0.92 | 1.02 | 0.95 |
| latency (ms) | 28.7 | 26.6 | 33.7 |
| duration (ms) | 70.6 | 68.5 | 70.2 |
| late / peak | 0.07 | 0.10 | 0.11 |
| reversed / trained | 1.105 | 1.103 | 1.157 |

- **Adopted:** `TAU_INH` 5, `TAU_EXC` 3, `J_EI_SCALE` 0.048, `G_NMDA` 400, `B_SCALE` 15,
  `R_I_TH` 43. Over evaluation seeds 42-51: duration 67.6 +/- 2.5 ms, latency 29.0 +/- 1.1 ms,
  extra spikes 0.92, responders 60%, peak ~21 Hz at 40 ms (the bird's ~20 Hz).
- **Open: the NMDA share.** Near threshold NMDA carries ~130x the fast charge per spike (3.3x
  at the original 20 ms operating point), and duration rests on it; it saturates near 70 ms,
  short of the bird's 90. A slower auditory input code or recurrent E -> E excitation are the
  alternatives not yet tried.

## Physiological units (adopted)

- **The model was dimensionless**: the membrane relaxed toward its input from a reset of 0 and
  fired at a threshold `theta` derived from the tonic drive, and drives, weights, noise and
  adaptation were in those input units (E: `theta_e` 0.044; I: `theta_i` 0.43). Only NMDA used mV,
  through a map of reset to -70 and threshold to -50 mV.
- **It is now in mV and pA** (`units.py`): `C dV/dt = -g_L (V - E_L) + I`, rest and reset -70 mV,
  threshold -50 mV, Wang's (2002) capacitances (500 pF E, 200 pF I). One old input unit is
  `(V_th - E_L) / theta` mV: 453 mV on E, 46 mV on I. Weights are peak PSPs (mV) with a log-normal
  mean and SD, replacing `B_SCALE / n_aud`, `B_HVC_SCALE` and `J_EI_SCALE / sqrt(n_e c_JEI)` with
  fixed "half the mean" or "10%" spreads. E's tonic drive is derived from its 25 Hz target rate,
  as `theta_e` was. NMDA's driving force is normalised at rest, so `G_NMDA` 400 became 61.9.
- **A change of units, not of model.** The shipped model loaded from its old file and run in the
  new code reproduced every metric to the last printed digit. Retrained from the rounded
  constants (3-4 significant figures), the ten-evaluation-seed means moved by under one seed SD
  (responders 60.3 -> 61.1%, extra spikes 0.916 -> 0.915, latency 29.0 -> 29.1 ms, duration
  67.6 -> 66.1 ms, reversed/trained 1.115 -> 1.116). Values in the sections above are in the
  earlier dimensionless units.

## DAF metrics

- **Responder-windowed, not rendition-averaged.** A whole-rendition E rate dilutes a 50 ms burst
  ~20:1 and lands on top of the reversed-song rate, so the DAF number is computed per neuron in
  150 ms windows (`metrics.daf_response`).
- **Peak over responders only**, matching Fig. 8E (n = 7). Averaging over all units mixed
  non-responders' flat PSTHs into a number compared against a responders-only one.
- **Peak bin width.** It was computed at 1 ms, finer than any binning the paper states, which
  inflated it ~7%. 2 ms is now part of the metric's definition (`PEAK_BIN_MS`).
- **Onset, not argmax.** The 23 +/- 12 ms latency is an onset; the PSTH argmax (~50 ms) was once
  printed against it.
- **Duration is timed to +300 ms.** Responses still running at the end of the 150 ms responder
  window were once dropped from the mean, biasing it short; that changed the mean by ~1 ms. On
  synthetic responses of known length, the paper's rule reads ~8 ms long (1-18 ms over 20 draws):
  at p < 0.05 about one baseline bin in twenty is significant by chance and postpones the ten-bin
  end run. The bird's number was measured with the same rule.
- **Not every responder is timed.** Latency and duration come from responders whose
  trial-averaged PSTH has five consecutive significant 2 ms bins: ~84 of ~312 at 20 trials
  with the former Gaussian input, 199 of 349 at 100 trials with the current one.

## Testing

- **Truncation causality is compared to a tolerance.** Bit-exactness held on macOS/Accelerate at
  every thread count but failed intermittently on Linux CI, because truncating the signal changes the
  shape of the drive matmul and BLAS may round it differently. A genuinely non-causal solver tested
  the same way moved earlier coefficients by up to 0.549.
- **Pinned tolerances are ~3 SD of the noise.** Each DAF metric in
  `test_reproduction_causal.py` allows about three standard deviations of its spread across ten
  evaluation seeds. Evaluating the same model after other simulations have advanced the noise
  stream moved duration by up to ~6 ms, more than the earlier fixed tolerances allowed.
- **Brian2 port gaps that tests now pin:** `noise_i` was accepted and dropped (no noise term in the
  I equation), and the port once lacked the refractory period and adaptation entirely.
