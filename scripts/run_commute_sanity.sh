#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
cd "$(dirname "$0")/.."
WORLD=commute STEPS=${STEPS:-1500} BATCH=${BATCH:-256} OUT=${OUT:-runs/commute_sanity} \
  bash scripts/run_synthetic_publishable.sh
