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

