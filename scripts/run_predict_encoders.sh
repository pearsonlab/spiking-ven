#!/bin/bash
# Next-1/2/4-frame prediction encoders on the 18-bird corpus. Re-run the same command to
# resume after stopping (checkpoints in outputs/encoders/pen/koch18_<TAG><h>.ckpt.npz).
#
#   bash scripts/run_predict_encoders.sh                                  # L1 2e-4, smooth 0.1
#   TAG=noreg_pred REG="--l1 0 --smooth 0 --mp-strength 0" \
#     bash scripts/run_predict_encoders.sh                                # unregularized
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=${PY:-/Users/jmp33/code/spiking-ven/.venv/bin/python}
TAG=${TAG:-pred}
REG=${REG:---l1 2e-4 --smooth 0.1 --smooth-freq-only}
for h in ${HORIZONS:-1 2 4}; do
  # shellcheck disable=SC2086
  $PY -u -W ignore scripts/train_sw_encoder.py --train-corpus outputs/song_corpus_koch.npz \
    $REG --predict-frames $h --epochs 100 --patience 5 \
    --min-delta 0.002 --log-every 1 --resume --out outputs/encoders/pen/koch18_$TAG$h.npz \
    >> outputs/encoders/pen/koch18_$TAG$h.log 2>&1 &
done
wait
