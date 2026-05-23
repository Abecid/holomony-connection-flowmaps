#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run the synthetic benchmark using both GPUs by assigning independent model runs
# to GPU 0/1. This is usually faster and more stable than DataParallel for tiny MLPs.
WORLD=${WORLD:-nonlinear}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-4096}
HIDDEN=${HIDDEN:-256}
DEPTH=${DEPTH:-5}
SEED=${SEED:-0}
OUT=${OUT:-runs/a800_synthetic_${WORLD}_seed${SEED}}
AMP=${AMP:---amp}
COMPILE=${COMPILE:-}
TEST_EVERY=${TEST_EVERY:-500}
EVAL_BATCH=${EVAL_BATCH:-4096}
EVAL_BATCHES=${EVAL_BATCHES:-2}
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-holonomy-connection-flowmaps}
WANDB_GROUP=${WANDB_GROUP:-a800_synthetic_${WORLD}_seed${SEED}}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ARTIFACTS=${WANDB_ARTIFACTS:-0}

mkdir -p "$OUT"
MODELS=(independent_cfm shared_cfm local_connection flat_pifm holonomy_connection)
PIDS=()
for IDX in "${!MODELS[@]}"; do
  M=${MODELS[$IDX]}
  GPU=$((IDX % 2))
  echo "=== Launching $M on cuda:$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m holoflow_conn.train_connection \
    --model "$M" \
    --world "$WORLD" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --flat-batch-size 512 \
    --hidden "$HIDDEN" \
    --depth "$DEPTH" \
    --model-steps 8 \
    --gt-steps 32 \
    --eval-gt-steps 48 \
    --eval-every "$TEST_EVERY" \
    --eval-batch-size "$EVAL_BATCH" \
    --eval-batches "$EVAL_BATCHES" \
    --num-threads 4 \
    --device cuda \
    --seed "$SEED" \
    $AMP $COMPILE \
    $(if [ "$WANDB" = "1" ]; then echo "--wandb --wandb-project $WANDB_PROJECT --wandb-group $WANDB_GROUP --wandb-run-name ${M}_${WORLD}_seed${SEED} --wandb-mode $WANDB_MODE"; fi) \
    $(if [ "$WANDB_ARTIFACTS" = "1" ]; then echo "--wandb-log-artifacts"; fi) \
    --outdir "$OUT/$M" > "$OUT/$M.log" 2>&1 &
  PIDS+=("$!")
  # Keep at most two jobs active.
  if (( ${#PIDS[@]} == 2 )); then
    wait "${PIDS[0]}" "${PIDS[1]}"
    PIDS=()
  fi
done
if (( ${#PIDS[@]} > 0 )); then
  wait "${PIDS[@]}"
fi

for M in "${MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -m holoflow_conn.eval_connection \
    --checkpoint "$OUT/$M/checkpoint.pt" \
    --batch-size 16384 \
    --batches 4 \
    --gt-steps 64 \
    --scaling \
    --device cuda
  done

python -m holoflow_conn.compare_runs --runs "$OUT"/* --out "$OUT/comparison.csv"
echo "Wrote $OUT/comparison.csv"
