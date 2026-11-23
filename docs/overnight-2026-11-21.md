# Overnight 2026-11-21: STRF fixes, conv decoder, DAF across encoders

## Morning summary

- **DAF shape is set downstream of the encoder.** All 11 encoders through the same baseline
  network: the random-onset two-hump response (early peak ~20 ms, larger one ~73-83 ms)
  appears with every encoder that drives the network well (baseline, conv A/B/C). Consistent
  with the earlier diagnosis that the second hump is the squared auditory normalisation.
  No STRF change tested fixes it.
- **Conv-decoder encoders from yesterday (A, B, C) are the best DAF drivers**: responders
  433-482 (baseline 444), fixed-onset peak +25-29 Hz at ~22 ms (baseline +32 at 23), raw
  encoder DAF peak ~40 ms with a 21-23 ms tail (baseline 47 / 29 ms). But longer network
  tails (0.11-0.20 of max vs 0.05).
- **Rate-constrained window encoders (D-I) are poor DAF drivers**: ~150-190 responders,
  +7-14 Hz, a sharp onset transient at 7-10 ms, then a flat plateau; random onsets only
  ~+5 Hz lasting ~35-125 ms. Raw encoder DAF peaks late (74-95 ms) and lingers ~75 ms after
  noise offset. Tighter targets (0.25, 0.1) are weaker still. Caveat: network and
  normalisation are tuned for the baseline encoder, not retuned per encoder.
- **Dead-unit restart and decorrelation (R1)**: no dead units arose, so restart never fired;
  decorr w=1 did not decorrelate (mean |corr| 0.12 -> 0.18) and w >= 10 destabilises. DAF
  unchanged vs the reference. **Recommend reverting both** (`0c43cef`, `7e06878`) unless you
  want to keep them as options.
- **Data init (`--init-data`)** cleans the window STRFs (no white-noise striations) but does
  not change DAF. Worth keeping.
- **Conv decoder + current setup (unit-norm shape x gain, rate target, noise) does not work**:
  R2/R3 (noise init) stay speckle; R5-R8 (data init, lr 1e-3/3e-3, with/without output noise)
  are weak/diffuse, and R7 degrades with more training (66% <=20 ms at epoch 100 -> 30% at
  300). Short conv STRFs so far only came from the free-norm, zero-init, L2 setup.
- **Softplus-at-inference fix** (`6171c46`): needed for correctness; keep.
- **Redundancy** (r > 0.9 partners): baseline 111/126 (!), conv A-C 12-13, window rate-0.5
  8-13, rate 0.25 36, rate 0.1 74.
- Figures: `outputs/overnight_2026-11-21/` (daf_*.png/json per encoder, ovn_encoder_daf_raw.png,
  STRF comparisons ovn_strfs_*.png, covariance ovn_unitcov_R1.png).

Branch `ac-tm-basis`. Every new idea is a separate flag (off by default) in its own commit,
so each can be reverted independently (`git revert <sha>`).

## Decisions log (in order)

1. **Softplus at inference** (`6171c46`): `SWEncoder.primary_beta`. Encoders trained with a
   held softplus (beta 9, no ReLU anneal) were being saved and run with a ReLU. Now saved
   with the trained nonlinearity; old encoders unchanged. The three TM16 window encoders
   (rate 0.5 / 0.25 / 0.1) were re-finalised from their epoch-150 checkpoints with it
   (ReLU versions kept as `*_relu.npz`).
2. **Dead-unit restart** (`0c43cef`, `--restart-dead 0.02`): after each epoch, units with
   activity std < 2% of the median get a new shape (random training window in the basis
   span), median gain, zero bias / lambda, fresh decoder row.
3. **Decorrelation penalty** (`7e06878`, `--decorr w`): w/2 sum_{j!=k} C_jk^2 / vbar^2 per
   bout (scale-free; gradient checked). Probe (60 bouts, 6 epochs): w = 1 slightly fewer
   near-duplicates (24 -> 19 of 128 with an r > 0.9 partner), w >= 10 destabilises
   (held-out 0.78, more duplicates). **Chose w = 1.**
4. **Target 0.5, 100 epochs** for the new runs (best STRFs so far; 0.5 converges by ~20).
5. **Conv decoder at lr 1e-3** (probe: 1e-3 and 3e-3 both stable; kept 1e-3 per your
   "lr too high" note).
6. **Conv runs R2 / R3 failed**: STRFs stayed unstructured speckle with many near-zero gains,
   and in R2 the rate constraint never converged (unit rate 10/90% 0.04/1.18). Cause: unit-
   norm shapes start as white noise and, with no penalty, the noise never decays (the earlier
   conv runs started from zero). **Added `--init-data`** (`aa932f8`): start each shape from a
   random training window (projected onto the basis). Reran as R4-R6.
7. **DAF analysis**: `scripts/run_daf_for_encoder.sh` trains the experiment-baseline V2
   network (configs/baseline.json: plastic HVC->I_HVC, aud norm 0.3 squared, per-rendition
   rms, 150 renditions) on each encoder and runs `daf_shape.py` (now also writes a JSON
   summary: first peak, max, dip, tail 50-150 ms after noise offset). **Same network and
   normalisation settings for every encoder** (not retuned per encoder), so differences are
   the encoder's.

8. **R1 (restart + decorr)**: no dead units arose, so the restart never fired; decorr w=1
   did not decorrelate (mean |corr| 0.12 -> 0.18). STRFs similar to the reference.
9. **Data init helps the window model** (R4: cleaner background, no striations). **Conv
   decoder still fails** with data init (R5 weak / diffuse; R6 with restart + decorr nearly
   blank: a few units dominate). Two diagnostic runs: R7 conv at lr 3e-3, R8 conv without
   output noise.
10. Runs take ~10 min per 100 epochs when not competing for CPU (the earlier slowness was
    contention).

## Runs

| id | decoder | init | restart / decorr | notes |
|---|---|---|---|---|
| (ref) TM16 win 0.5 | window | noise | - / - | from the evening (150 epochs) |
| R1 | window | noise | yes / 1 | |
| R2 | conv 16 ms | noise | - / - | failed (speckle) |
| R3 | conv 16 ms | noise | yes / 1 | failed (speckle) |
| R4 | window | data | - / - | |
| R5 | conv 16 ms | data | - / - | |
| R6 | conv 16 ms | data | yes / 1 | |

All: TM16 basis (2-64 ms), normalised input, lr 1e-3, rate target 0.5 (plain dual ascent,
step 0.1 every 20 bouts), input noise SNR 10, output noise SNR 3, unit-norm shape x gain with
the median-gain shape step, softplus beta 9, no early stopping.

| R7 | conv 16 ms | data | - / - | lr 3e-3; best conv at epoch 100 (66% <=20 ms), degraded by 300 |
| R8 | conv 16 ms | data | - / - | no output noise; like R5 |

## Results

Encoders (A-C from 2026-11-21 daytime: A conv 32 ms no penalty, B conv 16 ms + L2,
C TM8 conv 16 ms + L2; D TM8 window rate 0.5; E-G TM16 window rate 0.5 / 0.25 / 0.1;
H = R1; I = R4; J = R5).

### STRF / population statistics (held-out bouts, `scripts/encoder_stats.py`)

| encoder | STRF <=20 ms | mean lag | dead units | units with r>0.9 partner | mean abs corr |
|---|---|---|---|---|---|
| baseline | 51% | 25 ms | 7/126 | 111/126 | 0.54 |
| A_conv32_nopen | 66% | 16 ms | 1/128 | 12/128 | 0.40 |
| B_conv16_L2 | 96% | 9 ms | 0/127 | 13/127 | 0.39 |
| C_tm8_conv16_L2 | 96% | 9 ms | 0/127 | 13/127 | 0.39 |
| D_tm8_win_rate0.5 | 42% | 26 ms | 0/127 | 8/127 | 0.18 |
| E_tm16_win_rate0.5 | 31% | 30 ms | 9/128 | 11/128 | 0.12 |
| F_tm16_win_rate0.25 | 26% | 35 ms | 8/128 | 36/128 | 0.11 |
| G_tm16_win_rate0.1 | 23% | 39 ms | 4/128 | 74/128 | 0.11 |
| H_R1_restart_decorr | 40% | 28 ms | 0/128 | 19/128 | 0.18 |
| I_R4_win_datainit | 34% | 26 ms | 3/128 | 13/128 | 0.13 |
| J_R5_conv16_datainit | 49% | 25 ms | 35/128 | 85/128 | 0.49 |

### Network DAF (baseline V2 network per encoder, `scripts/run_daf_for_encoder.sh`)

Responder-mean change from pre-onset rate; tail = mean 50-150 ms after noise offset / max.

| encoder | fixed: resp | first peak | max | dip | tail | random: resp | first peak | max | tail |
|---|---|---|---|---|---|---|---|---|---|
| 0_baseline | 444 | +32 Hz @ 23 | +32 @ 23 | -2.6 @ 106 | 0.05 | 258 | +10 @ 20 | +24 @ 83 | -0.00 |
| A_conv32_nopen | 482 | +7 Hz @ 5 | +25 @ 22 | -2.1 @ 332 | 0.20 | 369 | +8 @ 19 | +18 @ 75 | 0.33 |
| B_conv16_L2 | 460 | +26 Hz @ 22 | +26 @ 22 | -3.0 @ 260 | 0.16 | 348 | +10 @ 20 | +20 @ 73 | 0.21 |
| C_tm8_conv16_L2 | 433 | +29 Hz @ 22 | +29 @ 22 | -3.3 @ 261 | 0.11 | 375 | +9 @ 20 | +20 @ 73 | 0.21 |
| D_tm8_win_rate0.5 | 191 | +14 Hz @ 7 | +14 @ 7 | -1.0 @ 225 | 0.29 | 72 | +2 @ 6 | +6 @ 133 | 0.66 |
| E_tm16_win_rate0.5 | 186 | +11 Hz @ 9 | +11 @ 9 | +0.3 @ 195 | 0.31 | 86 | +1 @ 15 | +5 @ 122 | 0.48 |
| F_tm16_win_rate0.25 | 164 | +10 Hz @ 10 | +10 @ 10 | +0.3 @ 220 | 0.46 | 77 | +2 @ 18 | +6 @ 83 | 0.39 |
| G_tm16_win_rate0.1 | 183 | +7 Hz @ 9 | +8 @ 266 | +0.2 @ 380 | 0.47 | 104 | +3 @ 15 | +5 @ 78 | 0.51 |
| H_R1_win_restart_decorr | 147 | +11 Hz @ 9 | +11 @ 9 | -0.6 @ 218 | 0.35 | 91 | +2 @ 22 | +4 @ 54 | 0.52 |
| I_R4_win_datainit | 168 | +12 Hz @ 10 | +12 @ 10 | -1.6 @ 210 | 0.38 | 71 | +2 @ 20 | +6 @ 52 | 0.51 |
| J_R5_conv16_datainit | 245 | +11 Hz @ 24 | +14 @ 119 | -1.3 @ 185 | 0.29 | 99 | +13 @ 114 | +13 @ 114 | 0.27 |

Raw encoder DAF (no network, `strf_compactness.py`): peak / fall to 10% after offset --
baseline 47 / 29 ms, A 40 / 23, B 39 / 21, D 81 / 74, E 79 / 75, G 43 / 70, R1 95 / 77,
R4 74 / 74 ms.


## 2026-11-22: rerun with linear divisive normalisation (now the baseline: sigma 0.05, power 1)

Same 11 encoders, same network settings, linear instead of squared normalisation
(`outputs/sweeps/daf_by_encoder_norm0.05_pow1/`, figures `outputs/daf_linear_2026-11-22/`).

| encoder | norm | fixed: resp | first peak | max | dip | tail | random: resp | first peak | max | tail |
|---|---|---|---|---|---|---|---|---|---|---|
| 0_baseline | lin | 416 | +30 @ 24 | +30 @ 24 | -2.8 @ 188 | 0.06 | 324 | +12 @ 24 | +18 @ 74 | -0.02 |
| 0_baseline | sq | 444 | +32 @ 23 | +32 @ 23 | -2.6 @ 106 | 0.05 | 258 | +10 @ 20 | +24 @ 83 | -0.00 |
| A_conv32_nopen | lin | 441 | +22 @ 24 | +22 @ 24 | -2.0 @ 266 | 0.18 | 372 | +11 @ 25 | +13 @ 68 | 0.28 |
| A_conv32_nopen | sq | 482 | +7 @ 5 | +25 @ 22 | -2.1 @ 332 | 0.20 | 369 | +8 @ 19 | +18 @ 75 | 0.33 |
| B_conv16_L2 | lin | 442 | +9 @ 6 | +25 @ 24 | -2.5 @ 263 | 0.12 | 380 | +11 @ 25 | +12 @ 59 | 0.26 |
| B_conv16_L2 | sq | 460 | +26 @ 22 | +26 @ 22 | -3.0 @ 260 | 0.16 | 348 | +10 @ 20 | +20 @ 73 | 0.21 |
| C_tm8_conv16_L2 | lin | 423 | +9 @ 5 | +23 @ 24 | -3.1 @ 260 | 0.15 | 380 | +11 @ 26 | +13 @ 64 | 0.23 |
| C_tm8_conv16_L2 | sq | 433 | +29 @ 22 | +29 @ 22 | -3.3 @ 261 | 0.11 | 375 | +9 @ 20 | +20 @ 73 | 0.21 |
| D_tm8_win_rate0.5 | lin | 273 | +13 @ 45 | +13 @ 45 | -0.7 @ 208 | 0.22 | 179 | +8 @ 30 | +8 @ 70 | 0.17 |
| D_tm8_win_rate0.5 | sq | 191 | +14 @ 7 | +14 @ 7 | -1.0 @ 225 | 0.29 | 72 | +2 @ 6 | +6 @ 133 | 0.66 |
| E_tm16_win_rate0.5 | lin | 266 | +8 @ 29 | +10 @ 56 | -0.0 @ 196 | 0.30 | 197 | +7 @ 35 | +8 @ 55 | 0.21 |
| E_tm16_win_rate0.5 | sq | 186 | +11 @ 9 | +11 @ 9 | +0.3 @ 195 | 0.31 | 86 | +1 @ 15 | +5 @ 122 | 0.48 |
| F_tm16_win_rate0.25 | lin | 265 | +8 @ 38 | +10 @ 52 | +0.7 @ 199 | 0.41 | 190 | +7 @ 32 | +9 @ 69 | 0.17 |
| F_tm16_win_rate0.25 | sq | 164 | +10 @ 10 | +10 @ 10 | +0.3 @ 220 | 0.46 | 77 | +2 @ 18 | +6 @ 83 | 0.39 |
| G_tm16_win_rate0.1 | lin | 282 | +10 @ 34 | +10 @ 59 | +0.5 @ 211 | 0.31 | 230 | +6 @ 29 | +6 @ 29 | 0.27 |
| G_tm16_win_rate0.1 | sq | 183 | +7 @ 9 | +8 @ 266 | +0.2 @ 380 | 0.47 | 104 | +3 @ 15 | +5 @ 78 | 0.51 |
| H_R1_win_restart_decorr | lin | 270 | +5 @ 8 | +10 @ 36 | +0.3 @ 180 | 0.25 | 202 | +9 @ 59 | +9 @ 59 | 0.18 |
| H_R1_win_restart_decorr | sq | 147 | +11 @ 9 | +11 @ 9 | -0.6 @ 218 | 0.35 | 91 | +2 @ 22 | +4 @ 54 | 0.52 |
| I_R4_win_datainit | lin | 279 | +12 @ 42 | +12 @ 42 | -0.3 @ 206 | 0.28 | 206 | +8 @ 47 | +9 @ 74 | 0.18 |
| I_R4_win_datainit | sq | 168 | +12 @ 10 | +12 @ 10 | -1.6 @ 210 | 0.38 | 71 | +2 @ 20 | +6 @ 52 | 0.51 |
| J_R5_conv16_datainit | lin | 350 | +16 @ 30 | +16 @ 30 | -2.8 @ 190 | 0.16 | 270 | +7 @ 35 | +10 @ 76 | 0.10 |
| J_R5_conv16_datainit | sq | 245 | +11 @ 24 | +14 @ 119 | -1.3 @ 185 | 0.29 | 99 | +13 @ 114 | +13 @ 114 | 0.27 |

- Linear normalisation removes the separate second hump: the baseline's random-onset
  response is one hump rising from ~15 ms to +18 Hz at 74 ms, then a sharp drop at ~95 ms
  (no rebound, small undershoot). Fixed onset: +30 Hz at 24 ms with a shoulder to ~80 ms.
- Short-STRF conv encoders (A-C): earlier and flatter (plateau ~20-75 ms, max +12-13 Hz at
  59-68 ms) but a slow tail to ~180 ms (tail 0.23-0.28 of max vs -0.02 for the baseline);
  not cleanly better than the baseline.
- Rate-constrained window encoders (D-I) improve a lot under linear normalisation (180-230
  responders vs 70-100 with squared) and peak at 30-60 ms, inside the paper's 35-75 ms
  window, but weakly (+6-10 Hz vs ~20 Hz).
- Baseline config now points at the linear-normalisation baseline network; its cancellation
  results have not been re-measured.

## 2026-11-23: where each STRF model succeeds / fails downstream (linear normalisation)

Cancellation: `cancellation_check.py`, single template, 20 trials (figures
`outputs/canc_lin_<NAME>.png`, grid `outputs/canc_lin_grid.png`). Baseline pooled over 38
renditions: flatness 55% (10-90% 50-60%), cancelled vs not singing 105% (squared network
was 53% / 107%), so linear normalisation does not cost cancellation.

Marks: DAF size ✓ >=15 Hz, ~ 10-15, ✗ <10 (fixed-onset max). Latency ✓ first peak in
35-75 ms. Tail ✓ <=0.15 of max, ~ <=0.25, ✗ >0.25 (worse of fixed/random). Cancelled ✓
>=55%, ~ 50-55, ✗ <50. Time-locked ✓ >=45%, ~ 40-45, ✗ <40.

| model | STRF (≤20 ms / dup / dead) | DAF size | DAF latency | DAF tail | cancelled | time-locked | flatness | cancellation figure |
|---|---|---|---|---|---|---|---|---|
| baseline | 51% / 111 / 7 ✗ | ✓ +30 | ✗ 24 ms (random 74) | ✓ .06 | ✓ 55% | ✓ 47% | 51% | big sparse song peaks; aligned keeps residual peaks (e.g. 530 ms) |
| A conv32 | 66% / 12 / 1 | ✓ +22 | ✗ 24 | ✗ .28 | ✗ 49% | ~ 40% | 49% | like baseline; residual aligned peaks |
| B conv16 L2 | 96% / 13 / 0 ✓ | ✓ +25 | ✗ 24 | ✗ .26 | ~ 51% | ~ 43% | 47% | like baseline |
| C TM8 conv16 L2 | 96% / 13 / 0 ✓ | ✓ +23 | ✗ 24 | ~ .23 | ✗ 48% | ✗ 39% | 45% | like baseline |
| D TM8 win r0.5 | 42% / 8 / 0 | ~ +13 | ✓ 45 | ~ .22 | ~ 53% | ✗ 38% | 52% | dense comb of small peaks; aligned flat-ish |
| E TM16 win r0.5 | 31% / 11 / 9 | ~ +10 | ✓ 56 | ✗ .30 | ~ 50% | ✗ 32% | 49% | dense comb; shifted ≈ aligned |
| F TM16 win r0.25 | 26% / 36 / 8 | ~ +10 | ✓ 52 | ✗ .41 | ✓ 59% | ~ 41% | 53% | dense comb; aligned low |
| G TM16 win r0.1 | 23% / 74 / 4 | ~ +10 | ✗ 34 (max 59) | ✗ .31 | ✓ 60% | ✓ 45% | 51% | dense comb; aligned low |
| H restart+decorr | 40% / 19 / 0 | ~ +10 | ✓ 36 | ~ .25 | ~ 53% | ✗ 34% | 50% | dense comb |
| I data init | 34% / 13 / 3 | ~ +12 | ✓ 42 | ✗ .28 | ~ 52% | ✗ 32% | 49% | dense comb |
| J conv16 data init | 49% / 85 / 35 ✗ | ✓ +16 | ✗ 30 | ~ .16 | ✓ 60% | ~ 42% | 54% | dense comb; aligned lowest |

- No model beats the baseline on everything. Baseline: strongest, cleanest single-hump DAF
  and best time-locking, but the fixed-onset peak is early (24 ms) and its STRFs are long
  and redundant.
- Conv-reconstruction encoders (A-C) have the best STRFs but are worse downstream than the
  baseline: same early peak, slow DAF tail, slightly less cancellation.
- Rate-constrained window encoders (D-I) put the DAF peak in the 35-75 ms window but at a
  third to half of the size, with long tails; cancellation similar, time-locking worse
  (shifted HVC cancels nearly as well as aligned). Their song response is a dense comb of
  small peaks rather than a few big ones, so there is less timing structure to learn.
- J (worst STRFs: 35 dead units, 85 duplicates) has the best cancellation; cancellation
  does not track STRF quality.
- Single-template, 20-trial numbers; differences of a few % are probably within noise.
