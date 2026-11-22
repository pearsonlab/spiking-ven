#!/bin/bash
# Network-level DAF analysis for one encoder: train the experiment-baseline V2 network
# (configs/baseline.json: plastic HVC -> I_HVC, its auditory normalisation and per-rendition
# rms, 150 renditions) on this encoder's stimuli, then the DAF time-course figure and summary.
#
#   bash scripts/run_daf_for_encoder.sh NAME ENCODER.npz [OUTDIR]
#
# The normalisation comes from the config (override with AUD_NORM / AUD_NORM_POW). Writes
# outputs/sweeps/daf_by_encoder_norm<s>_pow<p>/NAME.npz (+ .log) and
# OUTDIR/daf_<tag>_NAME.{png,json}.
set -e
NAME=$1
ENC=$2
OUTDIR=${3:-outputs}
cd "$(dirname "$0")/.."
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
read -r CFG_NORM CFG_POW < <($PY -c "import json; s = json.load(open('configs/baseline.json'))['stimulus']; print(s['aud_norm'], s['aud_norm_power'])")
NORM=${AUD_NORM:-$CFG_NORM}
POW=${AUD_NORM_POW:-$CFG_POW}
TAG=$($PY -c "print(f'norm{float($NORM):g}_pow{float($POW):g}')")
export PYTHONPATH=src OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export SWEEP_ENCODER=$ENC SWEEP_AUD_NORM=$NORM SWEEP_AUD_NORM_POW=$POW SWEEP_RMS_SELF=1
D=outputs/sweeps/daf_by_encoder_$TAG
mkdir -p "$D" "$OUTDIR"
if [ ! -f "$D/$NAME.npz" ]; then
  $PY -u -W ignore scripts/monitor_hvc_plastic.py --out "$D/$NAME.npz" > "$D/$NAME.log" 2>&1
fi
STEM=$(basename "$ENC" .npz)
CACHE=outputs/sweeps/stimuli_cache_${STEM}_${TAG}_rmsself.pkl
$PY -u -W ignore scripts/daf_shape.py --model "$D/$NAME.npz" --cache "$CACHE" --encoder "$ENC" \
  --label "$NAME ($TAG)" --out "$OUTDIR/daf_${TAG}_$NAME.png" >> "$D/$NAME.log" 2>&1
grep -E "^fixed|^random" "$D/$NAME.log" | sed "s/^/[$NAME $TAG] /"
