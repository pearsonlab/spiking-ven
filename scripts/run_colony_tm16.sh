#!/bin/bash
# Colony (CR) encoder with the exact 11-21 song recipe (koch18_tm16_win_plain_rate0.5):
# Takenaka-Malmquist basis, 16 distinct poles log-spaced 2-64 ms; window autoencoder with a
# linear window decoder, uniform reconstruction weighting; unit-norm STRF x gain, rate target
# 0.5 (plain dual ascent every 20 bouts), input/output noise SNR 10/3, softplus ramped from
# beta 1 and held at 9 (never annealed); no L1/L2/smoothness/min-phase.
# Its song control is diag_A_tm16_exact from scripts/run_basis_diag.sh (identical flags).
# Usage: scripts/run_colony_tm16.sh OUT_DIR  (OUT_DIR holds colony_corpus_cr.npz)
set -u
O=${1:?output dir}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
MOTIFS=${MOTIFS:-/Users/jmp33/code/spiking-ven/outputs/motifs.npz}
export PYTHONPATH=$W/src
# shellcheck disable=SC2086
"$PY" -u -W ignore "$W/scripts/train_sw_encoder.py" --motifs "$MOTIFS" \
  --train-corpus "$O/colony_corpus_cr.npz" --normalize-input --lr 1e-3 \
  --tm-basis 2,64,16 --unit-norm-strf --scale-free-shape --l1 0 --l2 0 --smooth 0 \
  --mp-strength 0 --act-target 0.5 --act-mode rate --al-plain --act-dual-lr 0.1 \
  --act-update-every 20 --input-noise 0.316 --output-noise 0.2887 --softplus-beta0 1 \
  --softplus-hold 9 --softplus-anneal 1 --epochs 150 --patience 0 --log-every 1 --resume \
  --out "$O/enc_colony_tm16.npz" > "$O/train_colony_tm16.log" 2>&1
