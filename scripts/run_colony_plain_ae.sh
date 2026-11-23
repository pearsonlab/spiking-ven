#!/bin/bash
# Colony (CR) vs song (Koch): plain autoencoder, same settings, in parallel.
#   untied ReLU autoencoder; STRFs on a Takenaka-Malmquist basis (16 distinct poles, 2-64 ms);
#   linear (non-convolutional) window decoder; uniform reconstruction weighting; normalised
#   input; lr $LR (default 3e-4: at 1e-3 the colony run diverged at epoch 4 -- its heavy
#   level tails make the plain gradient step too large -- while song was stable). No rate
#   constraint, no unit-norm STRFs, no noise, no softplus, and no L1 / L2 / smoothness /
#   min-phase (train_sw_encoder.py turns the last three on by default).
# Usage: scripts/run_colony_plain_ae.sh OUT_DIR  (OUT_DIR holds both corpora)
set -u
O=${1:?output dir}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
MOTIFS=${MOTIFS:-/Users/jmp33/code/spiking-ven/outputs/motifs.npz}
export PYTHONPATH=$W/src
LR=${LR:-3e-4}
COMMON="--motifs $MOTIFS --normalize-input --lr $LR --tm-basis 2,64,16
  --l1 0 --l2 0 --smooth 0 --mp-strength 0 --epochs 150 --patience 0 --log-every 1 --resume"
for c in colony song; do
  corpus=$O/colony_corpus_cr.npz; [ "$c" = song ] && corpus=$O/song_corpus_koch.npz
  # shellcheck disable=SC2086
  "$PY" -u -W ignore "$W/scripts/train_sw_encoder.py" --train-corpus "$corpus" $COMMON \
    --out "$O/plain_$c.npz" > "$O/plain_$c.log" 2>&1 &
done
wait
