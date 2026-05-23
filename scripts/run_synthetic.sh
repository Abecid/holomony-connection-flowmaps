#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run the synthetic benchmark by assigning independent model runs across the
# visible GPUs. This is usually faster and more stable than DataParallel for
# tiny MLPs.
WORLD=${WORLD:-nonlinear}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-4096}
HIDDEN=${HIDDEN:-256}
DEPTH=${DEPTH:-5}
SEED=${SEED:-0}
OUT=${OUT:-runs/synthetic_${WORLD}_seed${SEED}}
AMP=${AMP:---amp}
COMPILE=${COMPILE:-}
TEST_EVERY=${TEST_EVERY:-500}
EVAL_BATCH=${EVAL_BATCH:-4096}
EVAL_BATCHES=${EVAL_BATCHES:-2}
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-holonomy-connection-flowmaps}
WANDB_GROUP=${WANDB_GROUP:-synthetic_${WORLD}_seed${SEED}}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ARTIFACTS=${WANDB_ARTIFACTS:-0}
FAIL_ON_ERROR=${FAIL_ON_ERROR:-0}
NUM_GPUS=${NUM_GPUS:-}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-}
GPU_IDS=()

if [ -z "$NUM_GPUS" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "-1" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_IDS[@]}
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
  else
    NUM_GPUS=1
  fi
fi
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be a positive integer, got '$NUM_GPUS'" >&2
  exit 2
fi
if [ -z "$MAX_PARALLEL_JOBS" ]; then
  MAX_PARALLEL_JOBS=$NUM_GPUS
fi
if [[ ! "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL_JOBS < 1 )); then
  echo "MAX_PARALLEL_JOBS must be a positive integer, got '$MAX_PARALLEL_JOBS'" >&2
  exit 2
fi
if (( ${#GPU_IDS[@]} == 0 )); then
  for ((I = 0; I < NUM_GPUS; I++)); do
    GPU_IDS+=("$I")
  done
fi

mkdir -p "$OUT"
# Start the main method first so baseline crashes or hangs cannot block it from
# launching. The remaining models are comparison baselines.
MODELS=(holonomy_connection independent_cfm shared_cfm flat_pifm local_connection)
PIDS=()
PID_MODELS=()
TRAIN_FAILURES=()
EVAL_FAILURES=()
COMPLETED_RUNS=()

wait_for_active_jobs() {
  local IDX PID M STATUS
  for IDX in "${!PIDS[@]}"; do
    PID=${PIDS[$IDX]}
    M=${PID_MODELS[$IDX]}
    STATUS=0
    wait "$PID" || STATUS=$?
    if (( STATUS != 0 )); then
      echo "=== $M failed with exit status $STATUS; see $OUT/$M.log ==="
      TRAIN_FAILURES+=("$M:$STATUS")
    fi
  done
  PIDS=()
  PID_MODELS=()
}

for IDX in "${!MODELS[@]}"; do
  M=${MODELS[$IDX]}
  GPU=${GPU_IDS[$((IDX % NUM_GPUS))]}
  echo "=== Launching $M on CUDA_VISIBLE_DEVICES=$GPU ($NUM_GPUS GPUs, max $MAX_PARALLEL_JOBS jobs) ==="
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
  PID_MODELS+=("$M")
  if (( ${#PIDS[@]} == MAX_PARALLEL_JOBS )); then
    wait_for_active_jobs
  fi
done
if (( ${#PIDS[@]} > 0 )); then
  wait_for_active_jobs
fi

for M in "${MODELS[@]}"; do
  if [ ! -f "$OUT/$M/checkpoint.pt" ]; then
    echo "=== Skipping eval for $M: missing $OUT/$M/checkpoint.pt ==="
    continue
  fi
  STATUS=0
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} python -m holoflow_conn.eval_connection \
    --checkpoint "$OUT/$M/checkpoint.pt" \
    --batch-size 16384 \
    --batches 4 \
    --gt-steps 64 \
    --scaling \
    --device cuda || STATUS=$?
  if (( STATUS != 0 )); then
    echo "=== Eval failed for $M with exit status $STATUS ==="
    EVAL_FAILURES+=("$M:$STATUS")
  else
    COMPLETED_RUNS+=("$OUT/$M")
  fi
done

if (( ${#COMPLETED_RUNS[@]} > 0 )); then
  python -m holoflow_conn.compare_runs --runs "${COMPLETED_RUNS[@]}" --out "$OUT/comparison.csv"
  echo "Wrote $OUT/comparison.csv"
else
  echo "=== No completed runs to compare ==="
fi

if (( ${#TRAIN_FAILURES[@]} > 0 || ${#EVAL_FAILURES[@]} > 0 )); then
  {
    printf "Training failures: %s\n" "${TRAIN_FAILURES[*]:-none}"
    printf "Eval failures: %s\n" "${EVAL_FAILURES[*]:-none}"
  } > "$OUT/failures.txt"
  echo "=== Some jobs failed; wrote $OUT/failures.txt ==="
  if [ "$FAIL_ON_ERROR" = "1" ]; then
    exit 1
  fi
fi
