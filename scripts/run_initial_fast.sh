#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
cd "$(dirname "$0")/.."
# Backwards-compatible alias for the one-process quick benchmark.
STEPS=${STEPS:-150} BATCH=${BATCH:-96} HIDDEN=${HIDDEN:-64} DEPTH=${DEPTH:-3} DEVICE=${DEVICE:-auto} WORLD=${WORLD:-nonlinear} OUT=${OUT:-runs/quick_benchmark_${WORLD}}
python -m holoflow_conn.quick_benchmark \
  --world "$WORLD" --steps "$STEPS" --batch-size "$BATCH" --hidden "$HIDDEN" --depth "$DEPTH" \
  --device "$DEVICE" --outdir "$OUT"
