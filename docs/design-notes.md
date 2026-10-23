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
  read audio up to ~2 ms past t, and the divisive gain up to ~4 ms (measured lookahead 2-3 ms).
  Removing that added ~4 ms of response latency, compensated by lowering `AUD_DELAY_MS` from 5 to 2.

## Network dynamics

- **Conduction delay (`AUD_DELAY_MS`).** It was once set to the bird's 23 ms response *latency*,
  which double-counts: the network then integrates for a further ~43 ms (modelled latency ~66 ms)
  and the comparison with 23 ms becomes circular. finchsim's driver scripts carried the same 23 ms
  as their default until finchsim PR #27; they now use the trained model's delay.
- **`tau_s` and the STDP window** were once the same constant, which tied the coincidence window to
  the synaptic time course; `tau_stdp` now sets it separately.
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
- **Only ~1/4 of responders are timed.** Latency and duration come from responders whose
  trial-averaged PSTH has five consecutive significant 2 ms bins: ~84 of ~312 at the test seed.

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
