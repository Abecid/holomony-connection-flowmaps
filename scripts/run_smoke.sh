#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
cd "$(dirname "$0")/.."
python -m holoflow_conn.train_connection \
  --model holonomy_connection \
  --world nonlinear \
  --steps 20 \
  --batch-size 64 \
  --eval-every 20 \
  --eval-batch-size 128 \
  --eval-batches 1 \
  --hidden 32 \
  --depth 2 \
  --model-steps 2 \
  --gt-steps 4 \
  --eval-gt-steps 6 \
  --outdir runs/smoke/holonomy_connection \
  --device auto
python -m holoflow_conn.eval_connection --checkpoint runs/smoke/holonomy_connection/checkpoint.pt --batch-size 128 --batches 1 --gt-steps 6
