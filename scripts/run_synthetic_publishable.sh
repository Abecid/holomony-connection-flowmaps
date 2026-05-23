#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
cd "$(dirname "$0")/.."

WORLD=${WORLD:-nonlinear}
STEPS=${STEPS:-3000}
BATCH=${BATCH:-256}
HIDDEN=${HIDDEN:-128}
DEPTH=${DEPTH:-4}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-0}
OUT=${OUT:-runs/synthetic_${WORLD}}
TEST_EVERY=${TEST_EVERY:-500}
EVAL_BATCH=${EVAL_BATCH:-2048}
EVAL_BATCHES=${EVAL_BATCHES:-2}
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-holonomy-connection-flowmaps}
WANDB_GROUP=${WANDB_GROUP:-synthetic_${WORLD}_seed${SEED}}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ARTIFACTS=${WANDB_ARTIFACTS:-0}

MODELS=(independent_cfm shared_cfm local_connection flat_pifm holonomy_connection)
for M in "${MODELS[@]}"; do
  echo "=== Training $M on $WORLD ==="
  python -m holoflow_conn.train_connection \
    --model "$M" \
    --world "$WORLD" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --hidden "$HIDDEN" \
    --depth "$DEPTH" \
    --model-steps 6 \
    --gt-steps 16 \
    --eval-gt-steps 24 \
    --eval-every "$TEST_EVERY" \
    --eval-batch-size "$EVAL_BATCH" \
    --eval-batches "$EVAL_BATCHES" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $(if [ "$WANDB" = "1" ]; then echo "--wandb --wandb-project $WANDB_PROJECT --wandb-group $WANDB_GROUP --wandb-run-name ${M}_${WORLD}_seed${SEED} --wandb-mode $WANDB_MODE"; fi) \
    $(if [ "$WANDB_ARTIFACTS" = "1" ]; then echo "--wandb-log-artifacts"; fi) \
    --outdir "$OUT/$M"
  python -m holoflow_conn.eval_connection \
    --checkpoint "$OUT/$M/checkpoint.pt" \
    --batch-size 4096 \
    --batches 2 \
    --gt-steps 32 \
    --scaling \
    --device "$DEVICE"
done

python -m holoflow_conn.compare_runs \
  --runs "$OUT"/* \
  --out "$OUT/comparison.csv"
