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
