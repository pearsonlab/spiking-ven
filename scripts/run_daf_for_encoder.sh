#!/bin/bash
# Network-level DAF analysis for one encoder: train the experiment-baseline V2 network
# (configs/baseline.json: plastic HVC -> I_HVC, aud norm 0.3 squared, per-rendition rms,
# 150 renditions) on this encoder's stimuli, then the DAF time-course figure and summary.
#
#   bash scripts/run_daf_for_encoder.sh NAME ENCODER.npz [OUTDIR]
#
# Writes outputs/sweeps/daf_by_encoder/NAME.npz (+ .log) and OUTDIR/daf_NAME.{png,json}.
set -e
NAME=$1
ENC=$2
OUTDIR=${3:-outputs}
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export SWEEP_ENCODER=$ENC SWEEP_AUD_NORM=0.3 SWEEP_AUD_NORM_POW=2 SWEEP_RMS_SELF=1
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
D=outputs/sweeps/daf_by_encoder
mkdir -p "$D" "$OUTDIR"
if [ ! -f "$D/$NAME.npz" ]; then
  $PY -u -W ignore scripts/monitor_hvc_plastic.py --out "$D/$NAME.npz" > "$D/$NAME.log" 2>&1
fi
STEM=$(basename "$ENC" .npz)
CACHE=outputs/sweeps/stimuli_cache_${STEM}_norm0.3_pow2_rmsself.pkl
$PY -u -W ignore scripts/daf_shape.py --model "$D/$NAME.npz" --cache "$CACHE" --encoder "$ENC" \
  --label "$NAME" --out "$OUTDIR/daf_$NAME.png" >> "$D/$NAME.log" 2>&1
grep -E "^fixed|^random" "$D/$NAME.log" | sed "s/^/[$NAME] /"
