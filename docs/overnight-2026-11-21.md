# Overnight 2026-11-21: STRF fixes, conv decoder, DAF across encoders

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

## Results

(filled in below as runs finish)
