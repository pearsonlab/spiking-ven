#!/bin/bash
# Colony (CR) vs Koch song corpus: same encoder, same settings, in parallel.
#   window autoencoder, linear (non-convolutional) window decoder, uniform reconstruction
#   weighting (no --recon-tau-ms); Laguerre basis per channel (16 functions, one pole at
#   tau 8 ms: the most even coverage of the 64 ms window); otherwise the TM16 rate-0.5 recipe
#   (unit-norm shape x gain, rate target 0.5 by plain dual ascent, input/output noise SNR
#   10/3, softplus held at beta 9, no early stopping). No L1, smoothness or min-phase pull:
#   train_sw_encoder.py turns all three on by default, and the min-phase pull both front-loads
#   STRF energy (confounding any lag comparison) and costs ~15x in run time (FFTs per update).
# Usage: scripts/run_colony_laguerre.sh OUT_DIR  (OUT_DIR holds both corpora)
set -u
O=${1:?output dir}
W=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
MOTIFS=${MOTIFS:-/Users/jmp33/code/spiking-ven/outputs/motifs.npz}
export PYTHONPATH=$W/src OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 MKL_NUM_THREADS=6
COMMON="--motifs $MOTIFS --epochs 150 --log-every 1
  --normalize-input --tm-basis 8,8,16 --lr 1e-3 --unit-norm-strf --scale-free-shape
  --act-mode rate --act-target 0.5 --al-plain --act-dual-lr 0.1 --act-update-every 20
  --input-noise 0.316 --output-noise 0.289 --softplus-beta0 9 --softplus-hold 9 --patience 0 --resume
  --l1 0 --smooth 0 --mp-strength 0"
for c in colony koch; do
  corpus=$O/colony_corpus_cr.npz; [ "$c" = koch ] && corpus=$O/song_corpus_koch.npz
  # shellcheck disable=SC2086
  "$PY" -u -W ignore "$W/scripts/train_sw_encoder.py" --train-corpus "$corpus" $COMMON \
    --out "$O/enc_$c.npz" > "$O/train_$c.log" 2>&1 &
done
wait
