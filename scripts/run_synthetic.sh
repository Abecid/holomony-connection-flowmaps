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
SEEDS=${SEEDS:-$SEED}
SEEDS=${SEEDS//,/ }
OUT=${OUT:-runs/synthetic_${WORLD}_seed${SEED}}
OUT_ROOT=${OUT_ROOT:-}
AMP=${AMP:---amp}
COMPILE=${COMPILE:-}
TEST_EVERY=${TEST_EVERY:-500}
MODEL_STEPS=${MODEL_STEPS:-8}
GT_STEPS=${GT_STEPS:-32}
EVAL_GT_STEPS=${EVAL_GT_STEPS:-48}
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
EVAL_FAILURES=()
read -r -a SEED_VALUES <<< "$SEEDS"
if (( ${#SEED_VALUES[@]} == 0 )); then
  echo "SEEDS must contain at least one seed" >&2
  exit 2
fi

seed_out() {
  local S=$1
  if [ -n "$OUT_ROOT" ]; then
    printf "%s_seed%s" "$OUT_ROOT" "$S"
  elif (( ${#SEED_VALUES[@]} == 1 )); then
    printf "%s" "$OUT"
  else
    printf "runs/synthetic_%s_seed%s" "$WORLD" "$S"
  fi
}

for S in "${SEED_VALUES[@]}"; do
  mkdir -p "$(seed_out "$S")"
done

TRAIN_FAILURES_FILE="$PWD/.run_synthetic_train_failures_$$"
: > "$TRAIN_FAILURES_FILE"
export WORLD STEPS BATCH HIDDEN DEPTH TEST_EVERY MODEL_STEPS GT_STEPS EVAL_GT_STEPS EVAL_BATCH EVAL_BATCHES
export WANDB WANDB_PROJECT WANDB_GROUP WANDB_MODE WANDB_ARTIFACTS
export AMP COMPILE SEEDS OUT OUT_ROOT NUM_GPUS MAX_PARALLEL_JOBS TRAIN_FAILURES_FILE
export GPU_IDS_STR="${GPU_IDS[*]}"

python - <<'PY'
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path


world = os.environ["WORLD"]
seeds = os.environ["SEEDS"].replace(",", " ").split()
models = ["holonomy_connection", "independent_cfm", "shared_cfm", "flat_pifm", "local_connection"]
gpu_ids = os.environ["GPU_IDS_STR"].split()
max_jobs = int(os.environ["MAX_PARALLEL_JOBS"])
failures_file = Path(os.environ["TRAIN_FAILURES_FILE"])


def seed_out(seed: str) -> Path:
    out_root = os.environ.get("OUT_ROOT", "")
    if out_root:
        return Path(f"{out_root}_seed{seed}")
    if len(seeds) == 1:
        return Path(os.environ["OUT"])
    return Path(f"runs/synthetic_{world}_seed{seed}")


def command(model: str, seed: str, out: Path) -> list[str]:
    cmd = [
        "python", "-m", "holoflow_conn.train_connection",
        "--model", model,
        "--world", world,
        "--steps", os.environ["STEPS"],
        "--batch-size", os.environ["BATCH"],
        "--flat-batch-size", "512",
        "--hidden", os.environ["HIDDEN"],
        "--depth", os.environ["DEPTH"],
        "--model-steps", os.environ["MODEL_STEPS"],
        "--gt-steps", os.environ["GT_STEPS"],
        "--eval-gt-steps", os.environ["EVAL_GT_STEPS"],
        "--eval-every", os.environ["TEST_EVERY"],
        "--eval-batch-size", os.environ["EVAL_BATCH"],
        "--eval-batches", os.environ["EVAL_BATCHES"],
        "--num-threads", "4",
        "--device", "cuda",
        "--seed", seed,
    ]
    cmd += shlex.split(os.environ.get("AMP", ""))
    cmd += shlex.split(os.environ.get("COMPILE", ""))
    if os.environ.get("WANDB") == "1":
        group = out.name if len(seeds) > 1 else os.environ.get("WANDB_GROUP", out.name)
        cmd += [
            "--wandb",
            "--wandb-project", os.environ["WANDB_PROJECT"],
            "--wandb-group", group,
            "--wandb-run-name", f"{model}_{world}_seed{seed}",
            "--wandb-mode", os.environ["WANDB_MODE"],
        ]
    if os.environ.get("WANDB_ARTIFACTS") == "1":
        cmd.append("--wandb-log-artifacts")
    cmd += ["--outdir", str(out / model)]
    return cmd


pending = [(seed, model) for model in models for seed in seeds]
free_gpus = gpu_ids[:max_jobs]
active: list[tuple[subprocess.Popen, str, str, str, Path]] = []

while pending or active:
    while pending and free_gpus and len(active) < max_jobs:
        seed, model = pending.pop(0)
        gpu = free_gpus.pop(0)
        out = seed_out(seed)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / f"{model}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        print(
            f"=== Launching {model} seed{seed} on CUDA_VISIBLE_DEVICES={gpu} "
            f"({len(gpu_ids)} GPUs, max {max_jobs} jobs) ===",
            flush=True,
        )
        log = log_path.open("w")
        proc = subprocess.Popen(command(model, seed, out), stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        active.append((proc, gpu, seed, model, log_path))

    for idx, (proc, gpu, seed, model, log_path) in enumerate(active):
        status = proc.poll()
        if status is None:
            continue
        active.pop(idx)
        free_gpus.append(gpu)
        if status != 0:
            print(f"=== {model} seed{seed} failed with exit status {status}; see {log_path} ===", flush=True)
            with failures_file.open("a") as f:
                f.write(f"{model}_seed{seed}:{status}\n")
        break
    else:
        time.sleep(2)
PY

for S in "${SEED_VALUES[@]}"; do
  SEED_OUT=$(seed_out "$S")
  COMPLETED_RUNS=()
  for M in "${MODELS[@]}"; do
    if [ ! -f "$SEED_OUT/$M/checkpoint.pt" ]; then
      echo "=== Skipping eval for $M seed$S: missing $SEED_OUT/$M/checkpoint.pt ==="
      continue
    fi
    STATUS=0
    CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} python -m holoflow_conn.eval_connection \
      --checkpoint "$SEED_OUT/$M/checkpoint.pt" \
      --batch-size 16384 \
      --batches 4 \
      --gt-steps "$EVAL_GT_STEPS" \
      --scaling \
      --device cuda || STATUS=$?
    if (( STATUS != 0 )); then
      echo "=== Eval failed for $M seed$S with exit status $STATUS ==="
      EVAL_FAILURES+=("${M}_seed${S}:$STATUS")
    else
      COMPLETED_RUNS+=("$SEED_OUT/$M")
    fi
  done

  if (( ${#COMPLETED_RUNS[@]} > 0 )); then
    python -m holoflow_conn.compare_runs --runs "${COMPLETED_RUNS[@]}" --out "$SEED_OUT/comparison.csv"
    echo "Wrote $SEED_OUT/comparison.csv"
  else
    echo "=== No completed runs to compare for seed$S ==="
  fi
done

if [ -s "$TRAIN_FAILURES_FILE" ] || (( ${#EVAL_FAILURES[@]} > 0 )); then
  TRAIN_FAILURES=$(tr '\n' ' ' < "$TRAIN_FAILURES_FILE")
  {
    printf "Training failures: %s\n" "${TRAIN_FAILURES:-none}"
    printf "Eval failures: %s\n" "${EVAL_FAILURES[*]:-none}"
  } > "$(seed_out "${SEED_VALUES[0]}")/failures.txt"
  echo "=== Some jobs failed; wrote $(seed_out "${SEED_VALUES[0]}")/failures.txt ==="
  if [ "$FAIL_ON_ERROR" = "1" ]; then
    rm -f "$TRAIN_FAILURES_FILE"
    exit 1
  fi
fi
rm -f "$TRAIN_FAILURES_FILE"
