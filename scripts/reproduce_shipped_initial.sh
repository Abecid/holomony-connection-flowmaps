#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
cd "$(dirname "$0")/.."

DEVICE=${DEVICE:-auto}
SEED=${SEED:-0}
OUT_NON=${OUT_NON:-runs/initial_100_all}
OUT_COM=${OUT_COM:-runs/initial_100_commute}

rm -rf "$OUT_NON" "$OUT_COM"

# Noncommuting nonlinear connection: full baseline suite.
for M in independent_cfm shared_cfm local_connection flat_pifm holonomy_connection; do
  python -m holoflow_conn.train_connection \
    --model "$M" --world nonlinear --steps 120 --batch-size 64 \
    --hidden 48 --depth 2 --model-steps 2 --gt-steps 6 --eval-gt-steps 10 \
    --eval-every 120 --eval-batch-size 256 --eval-batches 1 \
    --device "$DEVICE" --seed "$SEED" --outdir "$OUT_NON/$M"
done
python -m holoflow_conn.compare_runs --runs "$OUT_NON"/* --out "$OUT_NON/comparison.csv"

# Commuting negative control: true holonomy zero.
for M in shared_cfm local_connection flat_pifm holonomy_connection; do
  python -m holoflow_conn.train_connection \
    --model "$M" --world commute --steps 80 --batch-size 64 \
    --hidden 48 --depth 2 --model-steps 2 --gt-steps 6 --eval-gt-steps 10 \
    --eval-every 80 --eval-batch-size 256 --eval-batches 1 \
    --device "$DEVICE" --seed "$SEED" --outdir "$OUT_COM/$M"
done
python -m holoflow_conn.compare_runs --runs "$OUT_COM"/* --out "$OUT_COM/comparison.csv"

mkdir -p results/initial
cp "$OUT_NON/comparison.csv" results/initial/nonlinear_quick_comparison.csv
cp "$OUT_COM/comparison.csv" results/initial/commute_quick_comparison.csv
