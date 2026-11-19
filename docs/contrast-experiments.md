# Forward/reversed contrast experiments (for review)

Goal: raise the contrast between the E response to the trained song and to the
time-reversed motif, keeping every other target in range -- above all the DAF magnitude
(extra spikes per responder per burst, 0.8-1.0). Each entry gives the parameters, how far
they are from the shipped model, a note on biological plausibility, and the result, so
any can be vetoed.

Baseline: the retuned Brian2 model (PR #35): dt 0.1 ms, `C_HVC` 0.1, `HVC_MEAN_NS` 7.48,
`R_I_HZ` 36, `BETA_ADAPT_PA` 10180, `TAU_GABA` 15, `TAU_STDP` 20, 600 renditions.

**Measures.** `rev/tr`: whole-rendition mean E rate, reversed / trained (the README
metric). `diff`: the song-driven contrast (`metrics.song_contrast`): over the motif, for
the 566 E cells with auditory input, reversed minus trained rate (Hz), each as the excess
over a singing-without-sound run (10 noise draws). `score`: summed miss over all targets,
in units of each target's tolerance; 0 = all in range (`scripts/sweep.py`). Each row is one
training seed (42 unless stated), 600 renditions, 100 DAF trials. Results and models:
`outputs/sweeps/<round>/<id>.json|npz`.

| baseline (seed 42) | score | xs | resp | lat | dur | pk | rev/tr | diff (Hz) |
|---|---|---|---|---|---|---|---|---|
| shipped retune | 0.00 | 0.99 | 0.45 | 30.3 | 53.8 | 38 | 1.070 | +0.57 |

## C1. A cleaner contrast measure (no model change)

The whole-rendition ratio is dominated by E's background (~1.9 Hz from tonic drive and
noise, which no learning removes). On the song-driven measure the trained song is almost
fully cancelled -- it adds ~0.06 Hz over singing without sound, the reversed motif ~0.6 Hz
-- so the excess *ratio* is huge and unstable (11-50); the excess *difference* is the
stable number. Numpy model +0.71 Hz, Brian2 1 ms +0.99, Brian2 0.1 ms untuned +1.50,
retune +0.57. **This changes the measure, not the model.**

## C2. Sparser HVC -> I (`C_HVC`), mean HVC weight scaled to keep I's total input

Biology: the number of HVC inputs per auditory-forebrain interneuron is unknown (an
HVC -> Av route exists, Roberts et al. 2017); 0.1 x 600 = 60 inputs per I cell is
already a guess, and 30 (0.05) or 18 (0.03) are no less plausible. Mean weights rise to
15-37 nS per connection to keep I's drive: 15 nS is within the range of strong
cortical/interneuron AMPA inputs; 25-37 nS per synapse is large (flag).

| `C_HVC` | `HVC_MEAN_NS` | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 7.48 (shipped) | 0.00 | 0.99 | 0.45 | 30.3 | 53.8 | 38 | 1.070 | +0.57 |
| 0.1 | 5.98 (x0.8) | 1.59 | 1.29 | 0.58 | 26.8 | 54.4 | 34 | 1.189 | +1.34 |
| **0.05** | **14.96** | **0.03** | **0.98** | **0.45** | 27.3 | 51.1 | 34 | **1.117** | **+0.83** |
| 0.05 | 11.97 (x0.8) | 0.91 | 1.17 | 0.57 | 23.6 | 50.1 | 54 | 1.193 | +1.33 |
| 0.03 | 24.93 | 1.18 | 1.22 | 0.54 | 27.2 | 47.3 | 32 | 1.149 | +1.01 |
| 0.03 | 19.95 (x0.8) | 2.49 | 1.45 | 0.60 | 24.4 | 53.5 | 36 | 1.260 | +1.71 |
| 0.02 | 37.4 | 2.54 | 1.48 | 0.59 | 24.8 | 62.8 | 36 | 1.206 | +1.34 |
| 0.02 | 29.92 (x0.8) | 4.87 | 1.88 | 0.64 | 21.3 | 68.4 | 32 | 1.293 | +1.91 |

Verdict: `C_HVC` 0.05 raises contrast ~45% (diff +0.83 vs +0.57, rev/tr 1.12 vs 1.07)
with every target in range. Sparser, or weaker HVC weights, raise contrast further but
re-couple it to the DAF magnitude.

## C3. Around `C_HVC` 0.05 (`C05` = `C_HVC` 0.05, `HVC_MEAN_NS` 14.96, SD 7.48)

| change (on C05 unless stated) | seed | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|
| C05 | 42 | 0.03 | 0.98 | 0.45 | 27.3 | 51.1 | 34 | 1.117 | +0.83 |
| C05 | 1 | 1.04 | **1.21** | 0.52 | 26.9 | 51.5 | 38 | 1.107 | +0.64 |
| C05 | 2 | 0.54 | **1.11** | 0.43 | 32.4 | 67.5 | 44 | 1.036 | +0.47 |
| shipped (`C_HVC` 0.1), for comparison | 1 | 0.03 | 0.89 | 0.41 | 30.4 | 49.8 | 34 | 1.114 | +0.65 |
| 1200 renditions | 42 | 0.04 | 0.93 | 0.45 | 26.3 | **43.8** | 36 | 1.120 | +0.81 |
| `C_HVC` 0.03, HVC 29.9 nS (x1.2) | 42 | 0.70 | 1.13 | 0.46 | 28.2 | 56.8 | 34 | 1.120 | +0.84 |
| `C_HVC` 0.03, HVC 34.9 nS (x1.4) | 42 | 0.20 | 1.04 | 0.41 | 26.1 | 50.2 | 34 | 1.068 | +0.55 |
| HVC 17.95 nS (x1.2) | 42 | 0.03 | 0.82 | 0.39 | 35.1 | 48.8 | 34 | 1.055 | +0.48 |
| `TAU_STDP` 10 ms | 42 | 0.31 | 0.75 | 0.46 | 24.6 | 41.9 | 36 | 1.047 | +0.48 |
| presyn. depression 0.38 nS | 42 | 6.91 | 2.33 | 0.61 | 18.6 | 62.0 | 34 | 1.144 | +1.44 |
| presyn. depression 1.13 nS | 42 | 9.65 | 2.90 | 0.58 | 17.6 | 63.6 | 34 | 1.077 | +1.02 |
| presyn. depression 1.13 nS, `R_I_HZ` 32 | 42 | 9.66 | 2.92 | 0.55 | 18.2 | 67.0 | 34 | 1.064 | +0.98 |
| **`TAU_GABA` 8 ms** | 42 | **0.00** | 0.96 | 0.48 | 28.5 | 47.7 | 52 | **1.122** | **+0.86** |
| `TAU_GABA` 8 ms, `NOISE_I` 150 pA sqrt(ms) | 42 | 0.00 | 0.91 | 0.47 | 28.7 | 46.8 | 54 | 1.102 | +0.73 |
| `NOISE_I` 150 pA sqrt(ms) | 42 | 0.03 | 0.93 | 0.46 | 23.3 | 51.5 | 34 | 1.106 | +0.74 |

Verdicts:
- **The C05 gain does not survive other seeds** (+0.64 and +0.47 Hz at seeds 1 and 2, against
  +0.65 for the shipped model at seed 1), and seeds 1-2 push extra spikes above range.
  Seed-to-seed spread (~0.15-0.2 Hz in `diff`, ~0.03-0.04 in `rev/tr`) is as large as most
  effects here, so single-seed wins are not trustworthy; from here on, candidates are
  compared over 4 seeds.
- **Presynaptic depression** (`RHO_PRE_NS`, an option added for this) erodes the learned
  inhibition: E rate 5-8 Hz, DAF response 2-3x too large. Rejected as implemented.
- **Shorter STDP window** (10 ms) and **longer training** (1200 renditions) do not help; the
  latter shortens the response below range.
- **`TAU_GABA` 8 ms** was the best single-seed result with every target in range; peak
  time moves later (52 ms, inside 35-75). Biology: GABA_A IPSC decay 5-10 ms is typical of
  adult cortex (the shipped 15 ms was chosen from the juvenile 5-20 ms range, Luo & Perkel
  2002), so 8 ms is, if anything, more conventional. Noise on I (1.7 mV free-membrane SD)
  did not help.

## C4. Multi-seed comparison (training seeds 42, 1, 2, 3)

`python scripts/summarize_sweeps.py --min-seeds 3` prints the per-seed rows. Means over the
four seeds (the shipped row's seed 42 is the retuned model itself):

| configuration | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|
| shipped | 0.07 | 0.90 | 0.44 | 31.5 | 52.3 | 36.5 | 1.080 | +0.64 |
| `TAU_GABA` 8 ms | 0.23 | 0.83 | 0.47 | 30.4 | 47.9 | 33.0 | 1.071 | +0.61 |
| C05 | 0.41 | **1.07** | 0.47 | 29.1 | 54.0 | 41.5 | 1.083 | +0.66 |
| C05 + `TAU_GABA` 8 ms | 0.37 | **1.02** | 0.53 | 32.9 | 43.9 | 46.0 | 1.086 | +0.69 |

Verdict: **no change so far moves contrast beyond seed-to-seed spread** (per-seed SD
~0.03 in rev/tr, ~0.1 Hz in diff), and the `C_HVC` 0.05 variants enlarge the DAF response
on average (seed 1: 1.21-1.22 extra spikes). The single-seed wins in C2-C3 were seed 42's
draw. This supports the brainstorm's diagnosis: the reversed motif drives the same E
cells with the same total input, and the learned inhibition is a per-cell gain on a
nearly flat I drive (I ~42 Hz throughout the motif), so network-side timing knobs have
little to work with. The remaining lever that changes *which* cells the reversed motif
drives is the encoder (C5).

## C5. Encoder context and NMDA decay

**Longer encoder context** (`CAUSAL_K_FRAMES` 16 -> 24, i.e. 32 -> 48 ms of past; encoder
retrained, `outputs/encoders/enc_k24.npz`, 59 filters, 2.2 Hz). Idea: longer
spectro-temporal kernels can prefer one sweep direction, so the reversed motif drives
partly different channels. Biology: 48 ms integration windows are well within auditory
forebrain STRF durations. Shipped network otherwise.

| seed | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.05 | 0.87 | 0.49 | 32.0 | 42.5 | 40 | 1.005 | +0.62 |
| 1 | 0.03 | 0.88 | 0.47 | 35.4 | 64.3 | 50 | 1.044 | +0.65 |
| 2 | 0.53 | 0.88 | 0.49 | 40.3 | 43.9 | 26 | 0.998 | +0.42 |
| 3 | 0.71 | 0.85 | 0.45 | 47.5 | 53.1 | 54 | 0.976 | +0.45 |
| mean | 0.33 | 0.87 | 0.48 | 38.8 | 50.9 | 42.5 | **1.006** | +0.53 |

Verdict: worse -- whole-rendition contrast falls to ~1.0, latency rises. Rejected.

**Faster NMDA decay** (`TAU_NMDA_DECAY` 100 -> 70 ms). Biology: NMDA decay shortens with
development in songbird forebrain (Livingston & Mooney 1997); 70 ms is plausible.

| seed | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.09 | 0.86 | 0.45 | 27.2 | **39.6** | 38 | 1.081 | +0.52 |
| 1 | 0.24 | 0.79 | 0.42 | 33.3 | **34.7** | 34 | 1.126 | +0.60 |

Verdict: duration drops below the 47 ms bound and contrast does not improve. Rejected.

## C6. Two more encoders (shipped network)

`enc_k32`: `CAUSAL_K_FRAMES` 32 (64 ms of past), 59 filters, 2.0 Hz. `enc_k24_lam05`:
24 frames and `CAUSAL_LAMBDA` 0.5 (sparser code), only 34 filters, 2.3 Hz.

| encoder | seed | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|
| k32 | 42 | 0.08 | 0.97 | 0.57 | 25.2 | 54.9 | 42 | 0.976 | +0.68 |
| k32 | 1 | 0.12 | 0.93 | 0.56 | 22.5 | 63.6 | 50 | 0.965 | +0.75 |
| k24, lambda 0.5 | 42 | 0.33 | 1.06 | 0.35 | 35.6 | 47.4 | 40 | 0.940 | +0.52 |
| k24, lambda 0.5 | 1 | 0.04 | 1.00 | 0.40 | 29.4 | 49.3 | 52 | 0.950 | +0.56 |

Verdict: longer encoder kernels make the whole-rendition contrast *worse* (below 1: the
reversed motif now drives E slightly less than the trained song does over the whole
rendition), although k32 raises the song-driven `diff` a little (+0.68-0.75 vs +0.64 mean
shipped) -- the two measures disagree, so this is not a clear win. Encoders retrained;
the network was not retuned for them.

## C7. Cap on each E cell's total inhibition (heterosynaptic normalisation)

`J_IE_ROW_MAX_NS` (an option added for this, off by default): after every training
rendition, each E row of `J_IE` whose sum exceeds the cap is scaled down, preserving its
pattern -- a slow homeostatic scaling, which is biologically plausible in kind. Caps at the
shipped model's 75th (2300 nS) and 50th (1600 nS) percentile row sums.

| configuration | seed | score | xs | resp | lat | dur | pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|
| cap 2300 | 42 | 2.27 | 1.45 | 0.40 | 27.6 | 58.3 | 34 | 1.057 | +0.55 |
| cap 2300 | 1 | 1.68 | 1.33 | 0.37 | 30.6 | 55.9 | 34 | 1.109 | +0.65 |
| cap 1600 | 42 | 6.04 | 2.21 | 0.38 | 23.0 | 65.7 | 36 | 1.040 | +0.52 |
| cap 1600 | 1 | 4.97 | 1.99 | 0.40 | 23.4 | 64.9 | 34 | 1.073 | +0.55 |
| C05 + cap 2300 | 42 | 2.19 | 1.43 | 0.44 | 24.8 | 63.0 | 34 | 1.096 | +0.77 |
| C05 + cap 2300 | 1 | 3.48 | 1.70 | 0.49 | 31.2 | 63.1 | 38 | 1.102 | +0.64 |
| C05 + cap 1600 | 42 | 5.85 | 2.17 | 0.43 | 20.9 | 64.1 | 34 | 1.069 | +0.74 |
| C05 + cap 1600 | 1 | 7.36 | 2.47 | 0.48 | 23.5 | 70.9 | 38 | 1.082 | +0.65 |

Verdict: the cells the cap binds are the strongly driven ones whose inhibition keeps the
DAF response small, so capping them enlarges it (1.3-2.5 extra spikes) without improving
contrast. Rejected.

## Summary (as of the end of C7)

Tried, against 4 (or 2) training seeds unless noted: sparser HVC -> I (`C_HVC` 0.05, 0.03,
0.02, with matched or scaled weights), `TAU_GABA` 8 ms, noise on I, both together,
`TAU_STDP` 10 ms, 1200 renditions, presynaptic depression (new option), a cap on total
inhibition per E cell (new option), faster NMDA decay (70 ms), and three retrained encoders
(24 and 32 frames; 24 frames with lambda 0.5).

- **Nothing robustly raised contrast while keeping the DAF metrics in range.** The best
  multi-seed means (`C_HVC` 0.05, with or without `TAU_GABA` 8) gain ~+0.02-0.05 Hz in
  `diff` and ~+0.005 in rev/tr over the shipped model, within seed-to-seed spread, and push
  extra spikes to ~1.02-1.07 (seed 1: 1.2).
- **What does raise contrast** -- weaker or sparser HVC drive to I, a lower STDP baseline,
  less inhibition -- does so by weakening the learned inhibition overall, which enlarges the
  DAF response in proportion: contrast and DAF magnitude are coupled through the strength
  of the learned inhibition, as the brainstorm predicted.
- **On the song-driven measure (C1) the trained song is already almost fully cancelled**
  (~0.06 Hz above singing without sound); the reversed motif is cancelled almost as well
  because it drives the same cells, and the learned inhibition is a per-cell gain on a
  nearly flat I drive (I ~42 Hz throughout). Raising contrast likely needs a structural
  change that makes I activity time-specific or E responses direction-selective, not a
  parameter change -- e.g. HVC -> I plasticity, or I cells with their own auditory tuning
  (not tried; outside the brainstormed list).
- **Biological flags for review**: HVC -> I weights of 15 nS (`C_HVC` 0.05) are strong but
  plausible; 25-37 nS per synapse (`C_HVC` 0.03-0.02) are large. Presynaptic depression and
  the row cap are new mechanisms (both off by default); neither helped. Retrained encoders
  were not followed by a network retune.
- **The shipped model is unchanged** by these experiments.

# Part 2: singing vs not singing (new target, Fig 7D)

New target `singing_over_not_singing` (0.5-2.0): E rate over the motif while singing the
trained song, over the rate in silence without singing. Shipped model: 0.24 (E ~2.3 Hz
singing vs ~9.6 Hz silent). An HVC-shift test showed ~96% of the cancellation is not
time-locked: HVC-driven inhibition (I ~40 Hz throughout singing) suppresses E whatever the
sound.

## T1-T2. Lower tonic E drive (seed 42)

`TARGET_RATE_E_HZ` 25 -> 16/12/8/5 barely lowers the silent rate (9.6 -> 7.4 Hz; noise
dominates). Setting the drive directly (`i_drive_e_pa` 150-300 pA, from 453) gives a silent
rate of 2.4-5.7 Hz, but singing still pulls E to ~1/4 of it: ratio 0.25-0.31 in every case.
Contrast improves with the quieter background (drive 150 pA, `R_I_HZ` 46: rev/tr 1.35, diff
+1.16 Hz, but extra spikes 1.21). Verdict: the blanket inhibition is structural (HVC drives
I hard), not a matter of E's background rate.

## B1. E/I-balanced start (new options `C_IE`, `IE_INIT_MEAN_NS`; seed 42)

Sparse plastic I -> E (`C_IE` 0.2) with log-normal initial weights (5-10 nS), E -> I x5-10
(1.1-2.2 nS) so E recruits I, and HVC -> I weakened (x0.2 / x0.35) so HVC re-times I rather
than adding to its mean. Untrained screen: at HVC x0.2-0.35 the mean I rate barely changes
with singing (13.5 -> 11.0-12.6 Hz) and E singing/silent ~1.0; song while not singing drives
E to 8-15 Hz from a 2-4 Hz background. Biology: E -> I of ~1 nS and I -> E of 5-10 nS per
synapse are conventional cortical values; HVC -> I of 1.5-2.6 nS is weaker than shipped.

| HVC x | aud x | `R_I_HZ` | E sing/silent | I singing/silent | xs | resp | lat | dur | late/pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 1 | 12 | 0.95 | 1.12 | 1.25 | 0.27 | 42.0 | 79.0 | 0.54 | 1.070 | +0.09 |
| 0.2 | 1 | 20 | 1.07 | 1.13 | 1.38 | 0.29 | 36.8 | 106.5 | 0.59 | 1.013 | +0.06 |
| 0.2 | 1.5 | 12 | 0.97 | 1.09 | 1.23 | 0.28 | 40.5 | 75.7 | 0.56 | 1.011 | +0.11 |
| 0.2 | 2 | 12 | 0.97 | 1.19 | 1.25 | 0.27 | 44.4 | 77.8 | 0.55 | 1.057 | +0.11 |
| 0.2 (I->E 10 nS) | 1 | 12 | 1.03 | 1.09 | 1.24 | 0.28 | 37.4 | 95.9 | 0.58 | 1.009 | +0.09 |
| 0.2 (E->I x10) | 1 | 15 | 1.04 | 1.18 | 0.87 | 0.23 | 43.1 | 118.6 | 0.59 | 1.042 | +0.04 |
| 0.35 | 1 | 12 | 0.64 | 2.44 | 1.01 | 0.29 | 44.5 | 69.7 | 0.42 | 1.113 | +0.26 |
| 0.35 | 1 | 20 | 0.79 | 2.00 | 1.27 | 0.28 | 35.8 | 92.5 | 0.47 | 1.051 | +0.23 |
| 0.35 | 1.5 | 12 | 0.71 | 2.31 | 1.05 | 0.28 | 40.2 | 77.6 | 0.47 | 1.131 | +0.25 |
| 0.35 (I->E 10 nS) | 1.5 | 20 | 0.82 | 1.49 | 1.26 | 0.29 | 44.8 | 63.3 | 0.50 | 1.064 | +0.22 |

Verdict: **the new target is met (0.64-1.07) and HVC x0.2 leaves the mean I rate within
~10-20%**, but song-specific learning is lost (diff ~+0.1 Hz) -- with weak HVC drive, I's
timing carries too little information -- and the DAF response becomes sustained (late/peak
~0.55 vs <= 0.15), longer, and reaches fewer cells (23-29%). I rates fell to 3-6 Hz after
training (13 Hz untrained).

## B2. Balanced start + sparse, strong HVC -> I (seed 42)

Base as B1 (`C_IE` 0.2, I -> E 5 nS initial, E -> I 1.1 nS, `R_I_HZ` 12), with fewer HVC
inputs per I cell (`C_HVC` 0.02 = 12 inputs, or 0.05 = 30) at the same total drive as HVC
x0.2 / x0.35 (per-synapse 7.5 / 13.1 nS at 0.02; 3.0 / 5.2 nS at 0.05).

| `C_HVC` | HVC nS | other | E sing/silent | I singing/silent | xs | resp | lat | dur | late/pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 13.1 | | 0.35 | 4.82 | 1.01 | 0.37 | 35.6 | 61.4 | 0.25 | 1.322 | +0.75 |
| 0.02 | 13.1 | rho x3 | 0.28 | 4.67 | 0.96 | 0.38 | 38.8 | 60.4 | 0.22 | **1.410** | **+0.79** |
| 0.02 | 13.1 | `R_I_HZ` 20 | 0.66 | 3.98 | 1.14 | 0.37 | 39.0 | 70.2 | 0.40 | 1.121 | +0.60 |
| 0.02 | 13.1 | `C_IE` 0.1, 10 nS | 0.66 | 3.86 | 1.15 | 0.36 | 39.4 | 65.0 | 0.41 | 1.067 | +0.45 |
| 0.02 | 7.5 | | 0.80 | 2.51 | 1.10 | 0.35 | 37.2 | 56.5 | 0.44 | 1.077 | +0.42 |
| 0.02 | 7.5 | rho x3 | 0.76 | 2.50 | 1.10 | 0.34 | 41.4 | 51.0 | 0.41 | 1.138 | +0.45 |
| 0.02 | 7.5 | beta x1.5 | 0.79 | 2.64 | 1.03 | 0.36 | 43.5 | 44.7 | 0.48 | 1.083 | +0.38 |
| 0.02 | 7.5 | `C_IE` 0.1, 10 nS | 0.95 | 2.27 | 1.25 | 0.32 | 43.7 | 66.4 | 0.48 | 1.079 | +0.17 |
| 0.05 | 5.2 | | 0.47 | 3.34 | 1.03 | 0.29 | 33.8 | 66.8 | 0.28 | 1.179 | +0.38 |
| 0.05 | 3.0 | | 0.91 | 1.52 | 1.30 | 0.28 | 38.3 | 83.7 | 0.55 | 1.079 | +0.20 |

Verdict: sparse, strong HVC inputs restore song-specific cancellation (rev/tr up to 1.41,
diff +0.79 Hz -- the best contrast so far, with extra spikes ~1.0) but multiply the I rate
4-5x while singing, which brings back the blanket suppression (E ratio 0.28-0.35). Weaker
HVC keeps E's ratio in range but loses contrast. The trade-off runs through I's trained
baseline (~3 Hz): any HVC input is a large relative change. Next: raise I's baseline
(tonic I drive near threshold, or noise on I) so HVC re-times I without multiplying it.
Biology flag: 12 HVC inputs of ~13 nS per I cell is sparse and strong but not implausible.

## B3. Raised I baseline (tonic I drive near threshold) (seed 42)

Base as B2 with `C_HVC` 0.02; `I_DRIVE_I_PA` 300 or 380 (threshold current 400 pA; shipped
53.7), `R_I_HZ` raised to 20-30 to sit near the new I rate.

| HVC nS | I drive | `R_I_HZ` | other | E sing/silent (Hz) | I singing/silent | xs | resp | lat | dur | late/pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.5 | 300 | 20 | | 1.03/1.25 = 0.83 | 17.3/10.7 = 1.61 | 0.71 | 0.25 | 33.2 | 97.3 | 0.35 | 1.210 | +0.27 |
| 7.5 | 300 | 30 | | 1.47/1.26 = 1.17 | 17.3/10.6 = 1.64 | 0.86 | 0.26 | 43.1 | 67.5 | 0.40 | 1.102 | +0.22 |
| 7.5 | 380 | 20 | | 0.19/0.36 = 0.54 | 23.3/16.6 = 1.40 | 0.59 | 0.16 | 32.6 | 97.0 | 0.21 | 1.707 | +0.21 |
| 7.5 | 380 | 30 | | 0.39/0.36 = 1.08 | 23.0/17.0 = 1.36 | 0.65 | 0.19 | 45.8 | 52.0 | 0.26 | 1.421 | +0.21 |
| 13.1 | 300 | 20 | | 0.49/1.25 = 0.39 | 24.1/10.7 = 2.25 | 0.68 | 0.26 | 38.3 | 93.0 | 0.31 | 1.441 | +0.46 |
| 13.1 | 300 | 30 | | 1.06/1.19 = 0.89 | 24.1/11.1 = 2.17 | 0.81 | 0.26 | 43.7 | 59.9 | 0.36 | 1.206 | +0.33 |
| 13.1 | 380 | 20 | | 0.10/0.36 = 0.29 | 29.4/16.5 = 1.78 | 0.59 | 0.15 | 36.3 | 64.9 | 0.19 | 2.220 | +0.25 |
| 13.1 | 380 | 30 | | 0.24/0.36 = 0.67 | 29.6/17.0 = 1.74 | 0.63 | 0.18 | 53.4 | 41.9 | 0.25 | 1.746 | +0.26 |
| 13.1 | 380 | 30 | rho x3 | 0.21/0.36 = 0.59 | 29.6/17.0 = 1.74 | 0.60 | 0.18 | 48.9 | 41.8 | 0.24 | 1.841 | +0.26 |
| 13.1 | 300 | 20 | noise_i 300 | 0.03/0.25 = 0.11 | 35.1/16.0 = 2.19 | 0.55 | 0.18 | 48.9 | 46.5 | 0.19 | 3.679 | +0.28 |

Verdict: a higher I baseline makes HVC's effect on the mean I rate smaller (1.4-1.6x at
7.5 nS) while keeping song-specific cancellation (diff +0.2-0.3 Hz), but the extra
inhibition silences E's background (0.25-1.3 Hz, below AIV's ~2-15 Hz) and shrinks the
DAF response (0.55-0.86 extra spikes, 15-26% responders). The large rev/tr values here
are inflated by the low E rates; `diff` is the fair comparison. Next: more E drive to
rebalance (B4).

## B4. Raised I baseline + more E drive (seed 42)

Base as B3 (`C_HVC` 0.02, `I_DRIVE_I_PA` 300 unless noted), with `i_drive_e_pa` 600-900
(shipped 453).

| HVC nS | E drive | `R_I_HZ` | other | E sing/silent (Hz) | I singing/silent | xs | resp | lat | dur | pk | late/pk | rev/tr | diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.5 | 600 | 30 | | 1.49/1.38 = 1.08 | 1.22 | 0.87 | 0.26 | 37.3 | 90.5 | 38 | 0.40 | 1.117 | +0.22 |
| 7.5 | 600 | 20 | | 1.09/1.45 = 0.75 | 1.20 | 0.69 | 0.25 | 39.2 | 97.0 | 12 | 0.36 | 1.158 | +0.28 |
| 7.5 | 750 | 30 | | 1.51/1.61 = 0.94 | 1.01 | 0.87 | 0.24 | 32.9 | 88.8 | 12 | 0.41 | 1.100 | +0.24 |
| 7.5 | 750 | 20 | | 1.21/1.61 = 0.75 | 1.06 | 0.66 | 0.24 | 43.8 | 73.6 | 12 | 0.38 | 1.138 | +0.26 |
| 7.5 | 900 | 30 | | 1.55/1.90 = 0.82 | 1.02 | 0.85 | 0.24 | 42.0 | 88.0 | 12 | 0.44 | 1.092 | +0.25 |
| 7.5 | 900 | 20 | | 1.35/1.90 = 0.71 | 1.02 | 0.61 | 0.23 | 43.5 | 84.0 | 36 | 0.39 | 1.105 | +0.24 |
| 7.5 | 750 | 30 | I drive 340 | 0.97/1.16 = 0.84 | 1.12 | 0.75 | 0.22 | 37.4 | 105.7 | 12 | 0.38 | 1.169 | +0.23 |
| 10.0 | 750 | 30 | | 1.36/1.61 = 0.85 | 1.08 | 0.85 | 0.24 | 40.5 | 84.7 | 12 | 0.39 | 1.119 | +0.27 |
| 13.1 | 600 | 30 | | 1.08/1.44 = 0.75 | 1.53 | 0.80 | 0.26 | 40.1 | 87.5 | 12 | 0.34 | 1.232 | +0.33 |
| 13.1 | 750 | 30 | | 1.14/1.61 = 0.71 | 1.29 | 0.80 | 0.26 | 36.3 | 91.1 | 12 | 0.34 | 1.185 | +0.34 |

Best (HVC 7.5 nS, E drive 600, `R_I_HZ` 30): singing/silent 1.08, I rate 1.22x with
singing, extra spikes 0.87, latency 37 ms, duration 90 ms, peak 38 ms -- but late/peak
0.40 (sustained), responders 26%, E background only ~1.4 Hz. HVC-shift test on it (song-
driven cells): HVC-only 0.51 Hz; song without HVC adds only +1.9 Hz (shipped: +14.4 -- the
balance quenches the song drive); trained song with HVC aligned +1.04, shifted 25-200 ms
+1.06-1.19, reversed +1.24-1.28. So cancellation is now partial (~46%) and still mostly not
time-locked (~0.15 Hz of it is).

## Summary of part 2 (as of B4)

- The singing ~ not-singing target and a near-constant mean I rate can both be met in an
  E/I-balanced start with weak or sparse HVC -> I and a raised I baseline (B3-B4).
- Every balanced configuration so far fails the DAF shape (late/peak 0.3-0.6, target
  <= 0.15) and has few responders (15-29%) and a low E background (~1-2 Hz).
- Song-specific cancellation is weak in the balanced regime: the balance quenches most of
  the song drive itself (+1.9 vs +14 Hz), and the learned cancellation is mostly not
  time-locked.

## S1-S4. Staged calibration (balance -> auditory -> HVC basis -> learning)

`scripts/staged_balance.py` (untrained screens), then training.

1. **Balance without inputs**: `C_IE` 0.2, I -> E initial 20 nS, E -> I 1.1 nS, E drive
   1500 pA, I drive 0: E 4.5 Hz, I 27 Hz (driven by E); without I -> E, E would fire 21 Hz
   (adaptation caps it there). Shallower variants: E 4-6 Hz at 700-1000 pA.
2. **Auditory -> E**, as E's rate in the 30 ms after an input spike on one of its channels
   (song, no HVC): NMDA dominates -- +35-94 Hz at 1250-5000 nS. "A few Hz" needs NMDA off:
   +2.6 / +5.2 / +9.2 Hz at 10 / 20 / 45 nS AMPA; any NMDA (150-600 nS) gives +12-25 Hz.
3. **HVC -> I basis** (`C_HVC` 0.02, 12 inputs per I cell): at 1-2 nS the mean I rate barely
   moves with singing (0.94-0.96x; single-I CV 0.6-0.8, peak/mean 1.9 -- modest); at 4-8 nS
   single I cells are strongly time-modulated (CV 1.3-1.9, peak/mean 5-7) but the mean I
   rate *halves* (0.51x) -- the paradoxical response of an inhibition-stabilised network.
4. **Learning** (600 renditions): HVC 2-3 nS, auditory 20-45 nS with NMDA 0-75, `R_I_HZ`
   20-32, rho x1/x3. E singing/silent 0.93-1.11 (target met), E silent 5.2-6.2 Hz, I 32-38 Hz
   changing 0.66-0.92x with singing. But the song drives song-driven E cells by only +0.1 to
   +0.5 Hz averaged over the motif (sparse inputs: +5 Hz per input spike averages to little),
   so there is nothing to cancel: trained, HVC-shifted and reversed song give the same
   excess (+0.1-0.7 Hz). DAF too small and short (0.45-0.67 extra spikes, 14-24%
   responders, 22-39 ms, latency 48-62 ms).

Verdict: balance and a weak HVC basis that leaves mean I unchanged both work; the auditory
calibration needs revisiting -- the song should raise E's *mean* rate by a few Hz (per input
spike ~+25-35 Hz: auditory ~45 nS with NMDA ~600-1250 nS on this base), not each input spike.

## S5. Balanced, asynchronous network with strong auditory drive (single I population)

Async balance needed sparse, heterogeneous E -> I (`C_EI` 0.05, 11 nS, SD = mean): dense
E -> I gave a ~30 Hz E/I rhythm (population-rate CV of I 3.6; 1.4 at `C_EI` 0.05; I noise
alone did not remove it). Auditory gain (extra E spikes per input spike): 2.0 / 3.9 / 6.5 /
10.7 at 90 / 150 / 250 / 400 nS (NMDA off). Trained with HVC -> I at 2 nS: STDP doubled the
mean I -> E weight but built generic feedback inhibition -- trained, HVC-shifted and
reversed song all +0.6-1.0 Hz -- because with weak HVC drive, I's timing comes from E.

## H1. Two inhibitory populations: I_bal (balance, fixed) + I_HVC (HVC basis, plastic)

New option `n_ih`: a second inhibitory population driven only by HVC (no E input), with
plastic I_HVC -> E (from zero); I_bal -> E fixed (`rho_ns` 0), HVC -> I_bal off. Base: S5
(`C_EI` 0.05 / 11 nS, auditory 400 nS, NMDA off, E drive 1500 pA, I -> E 20% at 20 nS).
I_HVC: 150 cells; untrained, 6 HVC inputs x 20 nS gives 9 Hz, each cell active ~7% of the
motif, the population tiling all of it, identical whatever the sound.

| HVC -> I_HVC | r_IH | rho_h | E sing/silent | xs | resp | lat | dur | pk | late/pk | song no HVC | trained | trained, HVC +100 ms | reversed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 x 20 nS | 5 | 3.8 | 0.75 | 0.83 | 0.46 | 35.0 | 22.2 | 34 | 0.04 | +2.71 | **+0.01** | +0.53 | +0.54 |
| 6 x 20 nS | 9 | 3.8 | 0.76 | 0.93 | 0.48 | 42.2 | 24.8 | 34 | 0.04 | +2.77 | **+0.03** | +0.74 | +0.71 |
| 6 x 20 nS | 5 | 11.3 | 0.76 | 0.78 | 0.44 | 34.6 | 21.9 | 34 | 0.03 | +2.75 | **+0.01** | +0.48 | +0.46 |
| 12 x 10 nS | 5 | 3.8 | 0.76 | 0.87 | 0.55 | 31.4 | 25.2 | 30 | 0.05 | +2.74 | **+0.04** | +0.65 | +0.79 |
| 12 x 10 nS | 9 | 3.8 | 0.77 | 1.08 | 0.55 | 31.5 | 27.3 | 30 | 0.06 | +2.69 | **+0.17** | +1.14 | +1.21 |
| 12 x 10 nS | 5 | 11.3 | 0.75 | 0.83 | 0.53 | 33.1 | 25.3 | 30 | 0.04 | +2.75 | **+0.02** | +0.51 | +0.70 |

(last four columns: song-driven E cells, excess over singing without sound, Hz.)

Verdict: **time-locked, song-specific cancellation** -- the trained song is cancelled almost
completely (+2.7 -> +0.01-0.17 Hz) only when HVC is aligned; shifting HVC by 100 ms or
reversing the song leaves +0.5-1.2 Hz. The singing/not-singing target is met (0.75), and the
DAF response is now transient (late/peak 0.03-0.06) with magnitude and responders in range.
Fails: DAF duration 22-27 ms (bound 47; NMDA is off) and peak 30-34 ms (bound 35). I_bal
falls with singing (0.59x) because cancelled E drives it less. Single seed (42).
Rasters: `rasters_ih_c01_w20_untrained.png`, `rasters_ih_c01_w20_trained.png`.

## V1. Rate-target iSTDP (Vogels et al. 2011) on I_HVC -> E, per-cell targets

Per-neuron scatter of the H1 model (`singing_vs_not_singing_scatter.png`) showed its
"singing ~ not singing" was an average of extremes: 440 E cells silenced while singing
(~6.4 uS of learned I_HVC inhibition each, silent even with HVC alone) and 128 cells with no
learned inhibition firing ~15 Hz (disinhibited as I_bal fell). The STDP rule's baseline is
set on the presynaptic rate, so nothing stops potentiation until E is silent.

New option `vogels_h`: I_HVC -> E learns by Vogels et al. 2011 (at each I spike
dw = eta (x_E - alpha_i); at each E spike dw = eta x_I), with alpha_i = 2 r_i tau_stdp and
r_i each E cell's untrained not-singing rate (`R_TARGET`, `set_rate_targets`). Otherwise as H1.

| HVC -> I_HVC | eta (nS) | per-cell median sing/not | mean sing/not | xs | resp | lat | dur | late/pk | song no HVC | trained | HVC +100 ms | reversed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 x 20 nS | 0.3 | | 0.82 | 1.29 | 0.43 | 39.7 | 26.1 | 0.12 | +2.75 | +0.88 | +2.02 | +1.99 |
| 6 x 20 nS | 1 | | 0.77 | 1.31 | 0.44 | 34.2 | 25.8 | 0.11 | +2.74 | +0.82 | +2.20 | +2.14 |
| 6 x 20 nS | 3 | | 0.70 | 1.37 | 0.43 | 31.4 | 24.6 | 0.11 | +2.77 | +0.89 | +2.34 | +2.32 |
| 12 x 10 nS | 0.3 | | 0.87 | 1.33 | 0.45 | 38.9 | 26.8 | 0.15 | +2.71 | +0.97 | +1.91 | +1.94 |
| 12 x 10 nS | 1 | 0.71 | 0.82 | 1.29 | 0.47 | 35.1 | 25.0 | 0.13 | +2.69 | +0.84 | +1.93 | +1.98 |
| 12 x 10 nS | 3 | 0.60 | 0.77 | 1.27 | 0.49 | 33.4 | 24.7 | 0.13 | +2.75 | +0.76 | +2.02 | +2.01 |

(H1 old rule: per-cell median 0.07.) Verdict: **E cells now lie along the diagonal of the
per-neuron singing scatter, like Fig 7D** (`singing_vs_not_singing_scatter_vogels.png`), and
cancellation is time-locked and song-specific with much better contrast: the trained song
keeps +0.8 Hz of +2.7, HVC-shifted or reversed song +1.9-2.3. Open: DAF too large (1.3 extra
spikes) and short (~25 ms; NMDA off), cells slightly below the diagonal (I_bal falls
0.47-0.55x while singing), one seed.

## V2. Higher, less heterogeneous background (Vogels rule)

Annealing the V1 model (600-1200 more renditions at eta 0.3 or 0.1, `scripts/anneal.py`)
did not tighten the per-neuron singing scatter (SD of log10 ratio 0.38-0.40 throughout): the
spread is the rule's equilibrium, not slow convergence. Measurement noise is far smaller
(x1.15-1.27 between two not-singing estimates). On Fig 7D's 1-100 Hz axes the V1 scatter
already follows the diagonal within ~2x, but 241/600 cells sit below 1 Hz in a condition.

Raising E drive or weakening inhibition did not reduce the low tail (21-28% of cells < 1 Hz:
I rebalances). It comes from heterogeneous inhibition; spreading I -> E over more I cells
does: connectivity 0.2 / 0.5 / 1.0 (same total) gives 24% / 10% / 0% below 1 Hz untrained
(10-90%: 0.4-17 / 0.9-13 / 2.9-9 Hz). Trained (drive 2500 pA, I_HVC 12 x 10 nS):

| I -> E | eta | E silent | per-cell median sing/not | cells < 1 Hz | xs | resp | lat | dur | song no HVC | trained | HVC +100 ms | reversed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50%, 6 nS | 1 | 6.2 | 0.75 | 136 | 1.57 | 0.45 | 30.6 | 27.7 | +2.44 | +0.90 | +1.82 | +1.93 |
| 50%, 6 nS | 3 | 6.2 | | | 1.48 | 0.48 | 31.3 | 25.9 | +2.42 | +0.78 | +1.81 | +1.84 |
| 100%, 3 nS | 1 | 6.0 | 0.76 | 49 | 1.57 | 0.47 | 33.1 | 27.6 | +2.52 | +0.99 | +2.26 | +2.27 |
| 100%, 3 nS | 3 | 6.0 | | | 1.56 | 0.48 | 35.6 | 26.8 | +2.60 | +1.01 | +2.30 | +2.33 |

Verdict: 50% looks most like Fig 7D (1-25 Hz along the diagonal); 100% removes the low
tail but compresses the not-singing range. Cancellation unchanged. DAF magnitude rose to
~1.5 extra spikes; duration still ~27 ms. (`singing_vs_not_singing_scatter_bg.png`)

## Why the Vogels rule does not equate singing and not-singing rates

The per-neuron scatter was measured on the template rendition only. Training cycles through
38 renditions whose timing varies, so each cell's cancellation fits the average rendition:
on any one rendition some cells are over- and some under-inhibited. Averaged over the
training renditions (`singing_scatter.py --pool`), the per-cell spread falls from x2.2 to
x1.5, with no cells silenced (I -> E 50%: 99 cells < 1 Hz in a condition; 100%: 8) -- the
fair comparison, since Fig 7D also averages over natural renditions
(`singing_vs_not_singing_scatter_pool.png`). Continuing training does not shrink the spread
on the template (weights still slowly grow), consistent with a rendition-average fixed point.
What remains is a uniform offset (median singing/not 0.78): Vogels' alpha = 2 r tau assumes
uncorrelated pre/post firing, which strong learned inhibition breaks. Candidate fix: scale
targets by ~1/0.78, or correct alpha for the correlation.
