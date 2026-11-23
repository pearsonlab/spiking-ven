#!/bin/bash
# The most basic encoder, on three corpora in parallel:
#   untied ReLU autoencoder; STRFs on a Takenaka-Malmquist basis (16 distinct poles, 2-64 ms);
#   linear (non-convolutional) window decoder (unit-norm rows); uniform reconstruction
#   weighting; normalised input; ZERO initial STRFs (--init-scale 0 --init-bias 0.01), so no
#   random init noise survives; lr $LR (default 3e-4). No L1, L2, smoothness, min-phase,
#   activity/rate constraint, unit-norm STRFs, noise or softplus.
# Corpora (in OUT_DIR):
#   colony_rms   colony_corpus_cr_active40_rms.npz: frames >= -40 dB, each segment RMS-normed
#   colony_cal   colony_corpus_cr_active40.npz:     same segments, calibrated levels
#   song         song_corpus_koch.npz:              song bouts (RMS-normed per bout)
# Usage: scripts/run_basic_ae.sh OUT_DIR
set -u
O=${1:?output dir}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
MOTIFS=${MOTIFS:-/Users/jmp33/code/spiking-ven/outputs/motifs.npz}
LR=${LR:-3e-4}
export PYTHONPATH=$W/src
COMMON="--motifs $MOTIFS --normalize-input --lr $LR --tm-basis 2,64,16
  --init-scale 0 --init-bias 0.01 --l1 0 --l2 0 --smooth 0 --mp-strength 0
  --epochs 150 --patience 0 --log-every 1 --resume"
for c in colony_rms colony_cal song; do
  case $c in
    colony_rms) corpus=$O/colony_corpus_cr_active40_rms.npz ;;
    colony_cal) corpus=$O/colony_corpus_cr_active40.npz ;;
    song) corpus=$O/song_corpus_koch.npz ;;
  esac
  # shellcheck disable=SC2086
  "$PY" -u -W ignore "$W/scripts/train_sw_encoder.py" --train-corpus "$corpus" $COMMON \
    --out "$O/basic_$c.npz" > "$O/basic_$c.log" 2>&1 &
done
wait
