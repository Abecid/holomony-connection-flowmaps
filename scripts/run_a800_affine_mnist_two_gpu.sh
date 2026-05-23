#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# GPU visual benchmark. Uses torchvision MNIST and order-sensitive affine controls.
DATA=${DATA:-data}
STEPS=${STEPS:-8000}
BATCH=${BATCH:-1024}
HIDDEN=${HIDDEN:-96}
DEPTH=${DEPTH:-6}
SEED=${SEED:-0}
OUT=${OUT:-runs/a800_affine_mnist_seed${SEED}}
AMP=${AMP:---amp}
COMPILE=${COMPILE:-}
WORKERS=${WORKERS:-8}
TEST_EVERY=${TEST_EVERY:-500}
EVAL_BATCHES=${EVAL_BATCHES:-2}
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-holonomy-connection-flowmaps}
WANDB_GROUP=${WANDB_GROUP:-a800_affine_mnist_seed${SEED}}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ARTIFACTS=${WANDB_ARTIFACTS:-0}

python -m holoflow_conn.affine_mnist --data-dir "$DATA" --download-only
mkdir -p "$OUT"
MODELS=(shared_cfm local_connection flat_pifm holonomy_connection)
PIDS=()
for IDX in "${!MODELS[@]}"; do
  M=${MODELS[$IDX]}
  GPU=$((IDX % 2))
  echo "=== Launching affine-MNIST $M on cuda:$GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU python -m holoflow_conn.affine_mnist \
    --model "$M" \
    --data-dir "$DATA" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --eval-batch-size "$BATCH" \
    --workers "$WORKERS" \
    --hidden "$HIDDEN" \
    --depth "$DEPTH" \
    --model-steps 4 \
    --eval-every "$TEST_EVERY" \
    --eval-batches "$EVAL_BATCHES" \
    --device cuda \
    --seed "$SEED" \
    $AMP $COMPILE \
    $(if [ "$WANDB" = "1" ]; then echo "--wandb --wandb-project $WANDB_PROJECT --wandb-group $WANDB_GROUP --wandb-run-name affine_${M}_seed${SEED} --wandb-mode $WANDB_MODE"; fi) \
    $(if [ "$WANDB_ARTIFACTS" = "1" ]; then echo "--wandb-log-artifacts"; fi) \
    --outdir "$OUT/$M" > "$OUT/$M.log" 2>&1 &
  PIDS+=("$!")
  if (( ${#PIDS[@]} == 2 )); then
    wait "${PIDS[0]}" "${PIDS[1]}"
    PIDS=()
  fi
done
if (( ${#PIDS[@]} > 0 )); then
  wait "${PIDS[@]}"
fi
python -m holoflow_conn.compare_affine_mnist --runs "$OUT"/* --out "$OUT/comparison.csv"
echo "Wrote $OUT/comparison.csv"
