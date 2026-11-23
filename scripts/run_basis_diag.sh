#!/bin/bash
# Why do the Laguerre runs give low-rank checkerboard STRFs? Three song (Koch) runs that
# differ from the 11-21 TM16 rate-0.5 command (reproduced exactly as A) in one flag each:
#   A  exact 11-21 command (TM 2-64 ms x 16, softplus beta 1 -> held at 9, never annealed)
#   B  A with the single-pole Laguerre basis (--tm-basis 8,8,16)
#   C  A with softplus starting at beta 9
# Usage: scripts/run_basis_diag.sh OUT_DIR  (OUT_DIR holds song_corpus_koch.npz)
set -u
O=${1:?output dir}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
MOTIFS=${MOTIFS:-/Users/jmp33/code/spiking-ven/outputs/motifs.npz}
export PYTHONPATH=$W/src
BASE="--motifs $MOTIFS --train-corpus $O/song_corpus_koch.npz --normalize-input --lr 1e-3
  --unit-norm-strf --scale-free-shape --l1 0 --l2 0 --smooth 0 --mp-strength 0
  --act-target 0.5 --act-mode rate --al-plain --act-dual-lr 0.1 --act-update-every 20
  --input-noise 0.316 --output-noise 0.2887 --softplus-hold 9 --softplus-anneal 1
  --epochs 150 --patience 0 --log-every 20 --resume"
run() {  # name, extra flags
  # shellcheck disable=SC2086
  "$PY" -u -W ignore "$W/scripts/train_sw_encoder.py" $BASE $2 --out "$O/diag_$1.npz" \
    > "$O/diag_$1.log" 2>&1 &
}
run A_tm16_exact "--tm-basis 2,64,16 --softplus-beta0 1"
run B_laguerre "--tm-basis 8,8,16 --softplus-beta0 1"
run C_beta9 "--tm-basis 2,64,16 --softplus-beta0 9"
wait
